# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Animate the cumulative hypervolume progression of a multi-objective optimizer.

Each frame extends the hypervolume time-series up to the current generation and
annotates the latest improvement. This makes sudden drops, plateaus, or surges easy
to pinpoint when reviewing optimization runs.

What-if Scenarios

Hypervolume plateaus early -> the search likely converged; consider tighter termination or
  restarting with new seeds if exploration should continue.
Hypervolume spikes followed by crash -> good points are found then discarded; audit survivor
  selection or constraint enforcement.
Stair-step growth with long flat segments -> exploitation dominates; boost mutation or sample
  injection to regain diversity.
Slow but steady climb -> healthy exploration/exploitation balance; no intervention needed.
"""

import io
import itertools
import math
import os

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import imageio.v2 as imageio

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class HypervolumeMoviePlot(PlotInterface):
  """
  Animated hypervolume progression (gif/html) for multi-objective optimizers.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    objectives = InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Ordered list of objective columns (minimized). If more than two are provided, all pairwise combinations are animated in a subplot layout.""")
    spec.addSub(objectives)
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('reference_point', contentType=InputTypes.StringListType,
        descr=r"""Optional reference point for hypervolume computation. If omitted, the plot uses max objective values (+5%%)."""))
    spec.addSub(InputData.parameterInputFactory('max_frames', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on the number of generations rendered. Defaults to min(total generations, 10)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combinations."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for the generated animations. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('save_frames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves sampled generations as standalone PNG frames alongside the animation outputs."""))
    spec.addSub(InputData.parameterInputFactory('frames_max', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <save_frames> is true. Defaults to 10; generations are sampled evenly."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'HypervolumeMoviePlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.objective_pairs = []
    self.index = None
    self.reference_point = None
    self._reference_points = {}
    self.maxFrames = None
    self.formats = {'gif', 'html'}
    self.fps = 2.0
    self.save_frames = False
    self.frame_max = 10
    self._global_hv_max = 0.0

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for HypervolumeMoviePlot "{self.name}".')
    self.sourceName = sourceNode.value
    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'Missing <objectives> node for HypervolumeMoviePlot "{self.name}".')
    self.objectives = [entry for entry in objNode.value if entry]
    if len(self.objectives) < 2:
      self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" requires at least two objectives.')
    if len(self.objectives) == 2:
      self.objective_pairs = [tuple(self.objectives)]
    else:
      self.objective_pairs = [tuple(pair) for pair in itertools.combinations(self.objectives, 2)]
    idxNode = spec.findFirst('index')
    if idxNode is None or idxNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for HypervolumeMoviePlot "{self.name}".')
    self.index = idxNode.value

    refNode = spec.findFirst('reference_point')
    if refNode is not None and refNode.value:
      try:
        ref_vals = [float(val) for val in refNode.value]
      except ValueError as err:
        self.raiseAnError(IOError, f'Invalid <reference_point> values for HypervolumeMoviePlot "{self.name}": {err}')
      if len(ref_vals) != 2:
        self.raiseAnError(IOError, f'<reference_point> must contain exactly two values for HypervolumeMoviePlot "{self.name}".')
      self.reference_point = np.asarray(ref_vals, dtype=float)

    maxNode = spec.findFirst('max_frames')
    if maxNode is not None and maxNode.value is not None:
      self.maxFrames = int(maxNode.value)
      if self.maxFrames <= 0:
        self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" received non-positive <max_frames>.')

    formatNode = spec.findFirst('format')
    if formatNode is not None and formatNode.value is not None:
      raw = formatNode.value.strip().lower()
      if raw == 'both' or not raw:
        self.formats = {'gif', 'html'}
      else:
        parts = [frag.strip() for frag in raw.split(',') if frag.strip()]
        mapped = set()
        for item in parts:
          if item in ('gif', 'html'):
            mapped.add(item)
          elif item == 'both':
            mapped.update({'gif', 'html'})
          else:
            self.raiseAnError(IOError, f'Unsupported <format> "{item}" for HypervolumeMoviePlot "{self.name}".')
        if not mapped:
          mapped = {'gif'}
        self.formats = mapped

    fpsNode = spec.findFirst('fps')
    if fpsNode is not None and fpsNode.value is not None:
      self.fps = float(fpsNode.value)
      if self.fps <= 0:
        self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" received non-positive <fps>.')

    saveFramesNode = spec.findFirst('save_frames')
    if saveFramesNode is not None and saveFramesNode.value is not None:
      self.save_frames = bool(saveFramesNode.value)

    framesMaxNode = spec.findFirst('frames_max')
    if framesMaxNode is not None and framesMaxNode.value is not None:
      self.frame_max = int(framesMaxNode.value)
      if self.frame_max <= 0:
        self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" received non-positive <frames_max>.')

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for HypervolumeMoviePlot "{self.name}".')
    self.source = src
    available = self.source.getVars()
    needed = set(self.objectives + [self.index])
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by HypervolumeMoviePlot "{self.name}".')

  def run(self):
    df = self.source.asDataset().to_dataframe().copy()
    if df.empty:
      self.raiseAWarning(f'HypervolumeMoviePlot "{self.name}" received an empty dataset; nothing to animate.')
      return
    df[self.index] = df[self.index].astype(float)
    for obj in self.objectives:
      df[obj] = df[obj].astype(float)
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'HypervolumeMoviePlot "{self.name}" found no generations in column "{self.index}".')
      return

    hv_series = self._compute_hypervolume_series(df, generations)
    if hv_series is None or not hv_series:
      self.raiseAWarning(f'HypervolumeMoviePlot "{self.name}" could not compute hypervolume; aborting.')
      return
    self._global_hv_max = 0.0
    for series in hv_series.values():
      if series.size:
        self._global_hv_max = max(self._global_hv_max, float(np.max(series)))

    frame_cap = self.maxFrames if self.maxFrames is not None else min(len(generations), 10)
    frame_cap = max(1, min(frame_cap, len(generations)))
    _, indices = self._sample_generations(generations, frame_cap)

    if 'gif' in self.formats:
      self._write_gif(generations, hv_series, indices)
    if 'html' in self.formats:
      self._write_html(generations, hv_series, indices)
    if self.save_frames:
      self._write_frames(generations, hv_series, indices)

  def _compute_hypervolume_series(self, df, generations):
    hv_by_pair = {}
    self._reference_points = {}
    for pair in self.objective_pairs:
      if self.reference_point is None:
        maxima = df[list(pair)].max().to_numpy(dtype=float)
        delta = np.abs(maxima) * 0.05
        delta[delta == 0.0] = 0.05
        ref_point = maxima + delta
      else:
        ref_point = np.asarray(self.reference_point, dtype=float)
      self._reference_points[pair] = ref_point
      hv_values = []
      for gen in generations:
        subset = df[df[self.index] == gen]
        hv = self._compute_hypervolume(subset[list(pair)].to_numpy(dtype=float), ref_point)
        hv_values.append(hv)
      hv_by_pair[pair] = np.asarray(hv_values, dtype=float)
    return hv_by_pair

  @staticmethod
  def _compute_hypervolume(points, ref):
    if points.size == 0:
      return 0.0
    order = np.argsort(points[:, 0])
    sorted_pts = points[order]
    hv = 0.0
    prev_x = ref[0]
    for x, y in sorted_pts[::-1]:
      width = prev_x - x
      if width < 0:
        width = 0.0
      height = max(0.0, ref[1] - y)
      hv += width * height
      prev_x = x
    return hv

  def _write_gif(self, generations, hv_series, indices):
    filename = self._createFilename(defaultName=f'{self.name}.gif')
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=0) as writer:
      for idx in indices:
        fig = self._render_frame(generations, hv_series, idx)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html(self, generations, hv_series, indices):
    filename = self._createFilename(defaultName=f'{self.name}.html')
    fig, axes = self._create_figure()

    def init():
      for axis, pair in zip(axes, self.objective_pairs):
        self._draw_series(axis, generations, hv_series[pair], indices[0], pair)
      fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
      return fig.axes

    def update(idx):
      for axis, pair in zip(axes, self.objective_pairs):
        self._draw_series(axis, generations, hv_series[pair], idx, pair)
      fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
      return fig.axes

    anim = animation.FuncAnimation(fig, update, frames=indices,
                                   init_func=init, interval=1000.0 / self.fps,
                                   blit=False)
    html_str = anim.to_jshtml()
    centered_html = f'<div style="display:flex;justify-content:center;">{html_str}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centered_html)
    plt.close(fig)

  def _write_frames(self, generations, hv_series, indices):
    frame_idx = self._select_frame_indices(len(indices))
    if not frame_idx:
      return
    base = self._createFilename(defaultName=f'{self.name}_frames')
    template = os.path.splitext(base)[0] + '_{index:04d}.png'
    directory = os.path.dirname(template)
    if directory:
      os.makedirs(directory, exist_ok=True)
    for frame_pos in frame_idx:
      idx = indices[frame_pos]
      fig = self._render_frame(generations, hv_series, idx)
      fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
      fig.savefig(template.format(index=idx), dpi=150)
      plt.close(fig)

  def _render_frame(self, generations, hv_series, idx):
    fig, axes = self._create_figure()
    for axis, pair in zip(axes, self.objective_pairs):
      self._draw_series(axis, generations, hv_series[pair], idx, pair)
    fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
    return fig

  def _draw_series(self, ax, generations, hv_series, idx, pair):
    ax.clear()
    upto_gens = generations[:idx + 1]
    upto_hv = hv_series[:idx + 1]
    ax.plot(upto_gens, upto_hv, color='tab:blue', linewidth=2.0)
    ax.scatter([upto_gens[-1]], [upto_hv[-1]], color='tab:orange', edgecolor='black', s=60, zorder=3)
    ax.set_xlabel(self.index)
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'{pair[0]} vs {pair[1]} (Generation {self._format_generation(upto_gens[-1])})')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(min(generations), max(generations))
    ymax = self._global_hv_max if self._global_hv_max > 0.0 else (np.max(hv_series) if hv_series.size else 1.0)
    ax.set_ylim(0.0, ymax * 1.05 if ymax > 0.0 else 1.0)
    ax.text(0.02, 0.92,
            f'Latest: {upto_hv[-1]:.4g}\nBest: {np.max(upto_hv):.4g}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.8, edgecolor='gray'))

  @staticmethod
  def _format_generation(genID):
    if float(genID).is_integer():
      return int(genID)
    return genID

  def _create_figure(self):
    n_pairs = len(self.objective_pairs) if self.objective_pairs else 1
    cols = max(1, n_pairs)
    fig, axes = plt.subplots(1, cols, figsize=(4.8 * cols, 4.2))
    if not isinstance(axes, np.ndarray):
      axes = [axes]
    else:
      axes = axes.flatten().tolist()
    return fig, axes

  @staticmethod
  def _sample_generations(generations, limit):
    if limit >= len(generations):
      return list(generations), list(range(len(generations)))
    positions = np.linspace(0, len(generations) - 1, limit, dtype=int)
    selected = []
    for idx in positions:
      if idx not in selected:
        selected.append(idx)
    cursor = 0
    while len(selected) < limit and cursor < len(generations):
      if cursor not in selected:
        selected.append(cursor)
      cursor += 1
    selected = sorted(set(selected))
    if len(selected) > limit:
      selected = selected[:limit - 1] + [len(generations) - 1]
    elif selected[-1] != len(generations) - 1:
      selected[-1] = len(generations) - 1
    return [generations[i] for i in selected], selected

  def _select_frame_indices(self, total):
    if not self.save_frames or total <= 0 or self.frame_max <= 0:
      return []
    if total <= self.frame_max:
      return list(range(total))
    stride = int(math.ceil(total / float(self.frame_max)))
    indices = list(range(0, total, stride))
    if indices and indices[-1] != total - 1:
      if len(indices) >= self.frame_max:
        indices[-1] = total - 1
      else:
        indices.append(total - 1)
    return sorted(set(indices))
