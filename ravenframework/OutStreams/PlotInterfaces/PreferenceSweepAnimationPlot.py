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
    spec.addSub(InputData.parameterInputFactory('space', contentType=InputTypes.StringType,
        descr=r"""Which columns define the preference space.
              Options:
                - objective (default): use the columns named in <objectives>.
                - fitness: use FitnessEvaluation_<objective> columns.
              Note: preference selection is performed on a minimization representation."""))
    spec.addSub(InputData.parameterInputFactory('goals', contentType=InputTypes.StringListType,
        descr=r"""Optional list of goal directions for each objective when <space> is 'objective'.
              Provide as 'min,max' or 'min max'. Length must match <objectives>.
              If omitted, assumes all objectives are minimized."""))
    spec.addSub(InputData.parameterInputFactory('normalize', contentType=InputTypes.BoolType,
        descr=r"""If true (default), min-max normalizes objectives before applying preference weights."""))
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
    self.space = 'objective'
    self.goals = None
    self.normalize = True
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

    spaceNode = spec.findFirst('space')
    if spaceNode is not None and spaceNode.value is not None:
      self.space = str(spaceNode.value).strip().lower()
    if self.space not in ('objective', 'fitness'):
      self.raiseAnError(IOError, f'Unsupported <space> "{self.space}" for PreferenceSweepAnimationPlot "{self.name}".')

    goalsNode = spec.findFirst('goals')
    if goalsNode is not None and goalsNode.value:
      goals = [str(g).strip().lower() for g in goalsNode.value if str(g).strip()]
      if len(goals) != len(self.objectives):
        self.raiseAnError(IOError, f'<goals> must contain {len(self.objectives)} entries for PreferenceSweepAnimationPlot "{self.name}".')
      for g in goals:
        if g not in ('min', 'max'):
          self.raiseAnError(IOError, f'Invalid goal "{g}" in PreferenceSweepAnimationPlot "{self.name}" (use min/max).')
      self.goals = goals

    normNode = spec.findFirst('normalize')
    if normNode is not None and normNode.value is not None:
      self.normalize = bool(normNode.value)

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
    if self.space == 'fitness':
      needed = set(f'FitnessEvaluation_{obj}' for obj in self.objectives)
    else:
      needed = set(self.objectives)
    if self.index:
      needed.add(self.index)
    if self.rank is not None:
      needed.add('rank')
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by PreferenceSweepAnimationPlot "{self.name}".')
    self.source = src

  def _get_objective_columns(self):
    if self.space == 'fitness':
      return [f'FitnessEvaluation_{obj}' for obj in self.objectives]
    return list(self.objectives)

  def _to_minimization_space(self, values):
    values = np.asarray(values, dtype=float).copy()
    if self.space == 'fitness':
      # Interpret fitness as "larger is better"; convert to minimization.
      return -values
    goals = self.goals or ['min'] * len(self.objectives)
    for j, goal in enumerate(goals):
      if goal == 'max':
        values[:, j] = -values[:, j]
    return values

  @staticmethod
  def _minmax_scale(values):
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    ranges = maxs - mins
    safe = np.where(ranges == 0.0, 1.0, ranges)
    return (values - mins) / safe

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'PreferenceSweepAnimationPlot "{self.name}" received an empty dataset; nothing to animate.')
      return
    subset = df.copy()
    obj_cols = self._get_objective_columns()
    for col in obj_cols:
      subset[col] = subset[col].astype(float)
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
    subset = subset.dropna(subset=obj_cols)
    if subset.empty:
      self.raiseAWarning(f'PreferenceSweepAnimationPlot "{self.name}" has no samples after filtering.')
      return

    weights = np.linspace(0.0, 1.0, num=self.frames)
    objectives_array = subset[obj_cols].to_numpy(dtype=float)
    objectives_array = self._to_minimization_space(objectives_array)
    if self.normalize:
      objectives_array = self._minmax_scale(objectives_array)
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

    xy = subset[obj_cols].to_numpy(dtype=float)
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
    ax.set_xlabel(self.objectives[0] if self.space == 'objective' else f'FitnessEvaluation_{self.objectives[0]}')
    ax.set_ylabel(self.objectives[1] if self.space == 'objective' else f'FitnessEvaluation_{self.objectives[1]}')
    ax.set_title('Preference sweep (weighted, normalized)')
    ax.grid(alpha=0.2)
    # Keep legend and annotation from overlapping by separating their anchors.
    legend_x = 1.02
    ax.legend(loc='upper left', bbox_to_anchor=(legend_x, 1.0), borderaxespad=0.0, frameon=True)
    ax.text(legend_x, 0.0,
            f'w = {weight:0.2f}\nscore = {score:0.4g}',
            transform=ax.transAxes, va='bottom', ha='left', fontsize=9, clip_on=False,
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
