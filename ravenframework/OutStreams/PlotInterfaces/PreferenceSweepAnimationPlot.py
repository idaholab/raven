# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Animate how the preferred solution changes as user weights shift between
objectives.

The animation highlights the incumbent design as weights sweep from one extreme
to the other, making it ideal for preference what-if analyses. Example
question: "If I favour objective 2 twice as much as objective 1, which sample
becomes best, and how quickly does the choice jump as I relax that bias?"
"""

import io
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class PreferenceSweepAnimationPlot(PlotInterface):
  """
  Sweep preference weights between two objectives and highlight the incumbent
  best solution. Use the produced GIF/HTML frames to answer what-if scenarios
  such as how the preferred design changes when stakeholders gradually shift
  emphasis from one criterion to the other or when weights are pinned to a
  specific value of interest.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    objectives = InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly two objective columns to evaluate under weighted preference.""")
    spec.addSub(objectives)
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> supplied, reduce the animation to a specific generation."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional Pareto rank filter (e.g., 1 to consider only nondominated points)."""))
    spec.addSub(InputData.parameterInputFactory('frames', contentType=InputTypes.IntegerType,
        descr=r"""Number of preference weights to sweep across (default 15)."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for the produced animation (default 2)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combination."""))
    spec.addSub(InputData.parameterInputFactory('save_frames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves selected preference frames as PNG images alongside the animation."""))
    spec.addSub(InputData.parameterInputFactory('frames_max', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <save_frames> is true (default 10)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'PreferenceSweepAnimationPlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
    self.generation = None
    self.rank = None
    self.frames = 15
    self.fps = 2.0
    self.formats = {'gif', 'html'}
    self.save_frames = False
    self.frames_max = 10

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for PreferenceSweepAnimationPlot "{self.name}".')
    self.sourceName = src.value

    objNode = spec.findFirst('objectives')
    if objNode is None or len(objNode.value) != 2:
      self.raiseAnError(IOError, f'PreferenceSweepAnimationPlot "{self.name}" requires exactly two <objectives>.')
    self.objectives = [entry for entry in objNode.value if entry]

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    rankNode = spec.findFirst('rank')
    if rankNode is not None and rankNode.value is not None:
      self.rank = int(rankNode.value)

    frameNode = spec.findFirst('frames')
    if frameNode is not None and frameNode.value is not None:
      self.frames = max(3, int(frameNode.value))

    fpsNode = spec.findFirst('fps')
    if fpsNode is not None and fpsNode.value is not None:
      self.fps = max(0.1, float(fpsNode.value))

    fmtNode = spec.findFirst('format')
    if fmtNode is not None and fmtNode.value is not None:
      parsed = self._parse_formats(fmtNode.value)
      if parsed:
        self.formats = parsed

    saveNode = spec.findFirst('save_frames')
    if saveNode is not None and saveNode.value is not None:
      self.save_frames = bool(saveNode.value)

    framesMaxNode = spec.findFirst('frames_max')
    if framesMaxNode is not None and framesMaxNode.value is not None:
      self.frames_max = max(1, int(framesMaxNode.value))

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for PreferenceSweepAnimationPlot "{self.name}".')
    available = src.getVars()
    needed = set(self.objectives)
    if self.index:
      needed.add(self.index)
    if self.rank is not None:
      needed.add('rank')
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by PreferenceSweepAnimationPlot "{self.name}".')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'PreferenceSweepAnimationPlot "{self.name}" received an empty dataset; nothing to animate.')
      return
    subset = df.copy()
    for obj in self.objectives:
      subset[obj] = subset[obj].astype(float)
    if self.index:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if self.rank is not None and 'rank' in subset.columns:
      subset = subset[subset['rank'].astype(float) == float(self.rank)]
    subset = subset.dropna(subset=self.objectives)
    if subset.empty:
      self.raiseAWarning(f'PreferenceSweepAnimationPlot "{self.name}" has no samples after filtering.')
      return

    weights = np.linspace(0.0, 1.0, num=self.frames)
    objectives_array = subset[self.objectives].to_numpy(dtype=float)
    best_indices = []
    scores = []
    for w in weights:
      coeffs = np.array([w, 1.0 - w], dtype=float)
      scalar = objectives_array @ coeffs
      idx = int(np.argmin(scalar))
      best_indices.append(idx)
      scores.append(float(scalar[idx]))
    best_indices = np.asarray(best_indices, dtype=int)
    scores = np.asarray(scores, dtype=float)

    xy = subset[self.objectives].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if 'gif' in self.formats:
      self._write_gif(weights, xy, best_indices, scores)
    if 'html' in self.formats:
      self._write_html(weights, xy, best_indices, scores)
    if self.save_frames:
      self._write_frames(weights, xy, best_indices, scores)
    plt.close(fig)

  def _write_gif(self, weights, xy, best_indices, scores):
    filename = self._createFilename(defaultName=f'{self.name}.gif')
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=0) as writer:
      for frame_idx, w in enumerate(weights):
        fig = self._render_frame(w, xy, best_indices[frame_idx], scores[frame_idx])
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html(self, weights, xy, best_indices, scores):
    filename = self._createFilename(defaultName=f'{self.name}.html')
    fig, ax = self._create_base_plot()

    def init():
      self._draw_frame(ax, weights[0], xy, best_indices[0], scores[0])
      fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
      return fig.axes

    def update(frame_idx):
      self._draw_frame(ax, weights[frame_idx], xy, best_indices[frame_idx], scores[frame_idx])
      fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
      return fig.axes

    anim = animation.FuncAnimation(fig, update, frames=len(weights),
                                   init_func=init, interval=1000.0 / self.fps,
                                   blit=False)
    html_str = anim.to_jshtml()
    centered_html = f'<div style=\"display:flex;justify-content:center;\">{html_str}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centered_html)
    plt.close(fig)

  def _write_frames(self, weights, xy, best_indices, scores):
    count = min(self.frames_max, len(weights))
    indices = np.linspace(0, len(weights) - 1, count, dtype=int)
    template = os.path.splitext(self._createFilename(defaultName=f'{self.name}_frames'))[0] + '_{idx:04d}.png'
    directory = os.path.dirname(template)
    if directory:
      os.makedirs(directory, exist_ok=True)
    for idx in indices:
      fig = self._render_frame(weights[idx], xy, best_indices[idx], scores[idx])
      fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
      fig.savefig(template.format(idx=idx), dpi=150)
      plt.close(fig)

  def _render_frame(self, weight, xy, best_idx, score):
    fig, ax = self._create_base_plot()
    self._draw_frame(ax, weight, xy, best_idx, score)
    fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
    return fig

  def _draw_frame(self, ax, weight, xy, best_idx, score):
    ax.clear()
    ax.scatter(xy[:, 0], xy[:, 1], c='#bbbbbb', edgecolor='k', linewidths=0.1, alpha=0.6, s=28, label='Samples')
    ax.scatter([xy[best_idx, 0]], [xy[best_idx, 1]], c='tab:red', s=80,
               edgecolors='black', linewidths=0.8, marker='*', label='Preferred')
    ax.set_xlabel(self.objectives[0])
    ax.set_ylabel(self.objectives[1])
    ax.set_title('Preference sweep')
    ax.grid(alpha=0.2)
    ax.legend(loc='best')
    ax.text(0.02, 0.95,
            f'w = {weight:0.2f}\nscore = {score:0.4g}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.8, edgecolor='gray'))

  def _create_base_plot(self):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    return fig, ax

  @staticmethod
  def _parse_formats(raw):
    if not raw:
      return {'gif', 'html'}
    raw = raw.strip().lower()
    if raw == 'both':
      return {'gif', 'html'}
    parts = [frag.strip() for frag in raw.split(',') if frag.strip()]
    valid = set()
    for item in parts:
      if item == 'gif':
        valid.add('gif')
      elif item == 'html':
        valid.add('html')
      elif item == 'both':
        valid.update({'gif', 'html'})
    if not valid:
      valid = {'gif'}
    return valid
