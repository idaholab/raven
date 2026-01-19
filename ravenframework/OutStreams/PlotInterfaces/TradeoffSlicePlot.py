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
Visualize pairwise objective slices with density contours and highlighted nondominated samples.
"""

import math
import itertools
import io
import os

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import imageio.v2 as imageio

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class TradeoffSlicePlot(PlotInterface):
  """
  Generate density contours for objective pairs and emphasize nondominated samples per generation.
  Each frame looks at one generation and, for every requested objective pair, shades the joint density of the whole population with a quick Gaussian KDE. Rank-1 (nondominated) samples are overplotted in orange with a thin black edge, while dominated points stay semi-transparent gray. This makes it easy to see where the front is filling the map versus where the population is still exploring.

  What-if Scenarios

  Front looks like a narrow ridge: the nondominated points collapse along a line or tight contour -> your objectives are heavily correlated; consider adding diversity pressure or re-scaling objectives to spread the front.
  Bright contour blobs with few orange points: the population is visiting promising areas but not landing nondominated samples -> mutation/crossover might be jumping too far; tighten step sizes or add elitism.
  Orange points scattered outside dense contours: the algorithm is finding new tradeoffs while most of the population lags -> try increasing survivor selection pressure or seeding elites to pull the crowd forward.
  Contours banded but orange points only near one corner: constraints or penalties may be eliminating wide portions of the space -> review feasibility filters or relax penalties to regain coverage of the other leg of the front.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    objectives = InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""List of objective columns to consider; at least two are required.""")
    spec.addSub(objectives)
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('max_frames', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on the number of generations rendered. Defaults to min(total generations, 20)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combinations."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for generated animations. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('save_frames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves sampled generations as standalone PNG frames alongside the animation outputs."""))
    spec.addSub(InputData.parameterInputFactory('frames_max', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <save_frames> is true. Defaults to 10; generations are sampled evenly."""))
    pairs = InputData.parameterInputFactory('pairs',
        descr=r"""Optional list of <pair> entries selecting specific objective pairs to plot.""")
    pairs.addSub(InputData.parameterInputFactory('pair', contentType=InputTypes.StringListType,
        descr=r"""Two objective names separated by whitespace or commas."""))
    spec.addSub(pairs)
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'TradeoffSlicePlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
    self.pairs = []
    self.maxFrames = None
    self.formats = {'gif', 'html'}
    self.fps = 2.0
    self.save_frames = False
    self.frame_max = 10
    self.gridPoints = 60

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for TradeoffSlicePlot "{self.name}".')
    self.sourceName = sourceNode.value
    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'Missing <objectives> node for TradeoffSlicePlot "{self.name}".')
    self.objectives = [entry for entry in objNode.value if entry]
    if len(self.objectives) < 2:
      self.raiseAnError(IOError, f'TradeoffSlicePlot "{self.name}" requires at least two objectives.')
    indexNode = spec.findFirst('index')
    if indexNode is None or indexNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for TradeoffSlicePlot "{self.name}".')
    self.index = indexNode.value

    maxNode = spec.findFirst('max_frames')
    if maxNode is not None and maxNode.value is not None:
      self.maxFrames = int(maxNode.value)
      if self.maxFrames <= 0:
        self.raiseAnError(IOError, f'TradeoffSlicePlot "{self.name}" received non-positive <max_frames>.')
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
            self.raiseAnError(IOError, f'Unsupported <format> "{item}" for TradeoffSlicePlot "{self.name}".')
        if not mapped:
          mapped = {'gif'}
        self.formats = mapped
    fpsNode = spec.findFirst('fps')
    if fpsNode is not None and fpsNode.value is not None:
      self.fps = float(fpsNode.value)
      if self.fps <= 0:
        self.raiseAnError(IOError, f'TradeoffSlicePlot "{self.name}" received non-positive <fps>.')
    saveFramesNode = spec.findFirst('save_frames')
    if saveFramesNode is not None and saveFramesNode.value is not None:
      self.save_frames = bool(saveFramesNode.value)
    framesMaxNode = spec.findFirst('frames_max')
    if framesMaxNode is not None and framesMaxNode.value is not None:
      self.frame_max = int(framesMaxNode.value)
      if self.frame_max <= 0:
        self.raiseAnError(IOError, f'TradeoffSlicePlot "{self.name}" received non-positive <frames_max>.')

    self.pairs = []
    pairsNode = spec.findFirst('pairs')
    if pairsNode is not None:
      for pairNode in pairsNode.subparts:
        if pairNode.name != 'pair':
          continue
        entries = [entry.strip() for entry in pairNode.value if entry]
        if not entries and isinstance(pairNode.value, str):
          entries = [item.strip() for item in pairNode.value.split(',') if item.strip()]
        if len(entries) != 2:
          self.raiseAnError(IOError, f'Each <pair> in TradeoffSlicePlot "{self.name}" must list exactly two objectives.')
        self.pairs.append(tuple(entries))
    if not self.pairs:
      # default to all unique pairs
      combos = list(itertools.combinations(self.objectives, 2))
      if not combos:
        combos = [tuple(self.objectives[:2])]
      self.pairs = combos

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for TradeoffSlicePlot "{self.name}".')
    self.source = src
    available = self.source.getVars()
    needed = set(self.objectives + [self.index])
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by TradeoffSlicePlot "{self.name}".')

  def run(self):
    df = self.source.asDataset().to_dataframe().copy()
    if df.empty:
      self.raiseAWarning(f'TradeoffSlicePlot "{self.name}" received an empty dataset; nothing to draw.')
      return
    df[self.index] = df[self.index].astype(float)
    for obj in self.objectives:
      df[obj] = df[obj].astype(float)
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'TradeoffSlicePlot "{self.name}" found no generations in column "{self.index}".')
      return
    frame_cap = self.maxFrames if self.maxFrames is not None else min(len(generations), 20)
    frame_cap = max(1, min(frame_cap, len(generations)))
    gens_to_render = self._select_generations(generations, frame_cap)
    if not gens_to_render:
      self.raiseAWarning(f'TradeoffSlicePlot "{self.name}" did not select any generations to render.')
      return

    if 'gif' in self.formats:
      self._write_gif(df, gens_to_render)
    if 'html' in self.formats:
      self._write_html(df, gens_to_render)
    if self.save_frames:
      self._write_frames(df, gens_to_render)

  def _write_gif(self, df, generations):
    filename = self._createFilename(defaultName=f'{self.name}.gif')
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=0) as writer:
      for gen in generations:
        subset = df[df[self.index] == gen]
        if subset.empty:
          continue
        rank_mask = self._prepare_rank_mask(subset)
        fig, axes = self._create_figure()
        self._populate_axes(axes, subset, rank_mask)
        fig.suptitle(f'{self.name}: Generation {self._format_generation(gen)}')
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html(self, df, generations):
    filename = self._createFilename(defaultName=f'{self.name}.html')
    fig, axes = self._create_figure()

    def init():
      subset = df[df[self.index] == generations[0]]
      rank_mask = self._prepare_rank_mask(subset)
      self._populate_axes(axes, subset, rank_mask)
      fig.suptitle(f'{self.name}: Generation {self._format_generation(generations[0])}')
      fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
      return fig.axes

    def update(gen):
      subset = df[df[self.index] == gen]
      rank_mask = self._prepare_rank_mask(subset)
      self._populate_axes(axes, subset, rank_mask)
      fig.suptitle(f'{self.name}: Generation {self._format_generation(gen)}')
      fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
      return fig.axes

    anim = animation.FuncAnimation(fig, update, frames=generations,
                                   init_func=init, interval=1000.0 / self.fps,
                                   blit=False)
    html_str = anim.to_jshtml()
    centered_html = f'<div style="display:flex;justify-content:center;">{html_str}</div>'
    with open(filename, 'w', encoding='utf-8') as output:
      output.write(centered_html)
    plt.close(fig)

  def _write_frames(self, df, generations):
    frame_indices = self._select_frame_indices(len(generations))
    if not frame_indices:
      return
    base = self._createFilename(defaultName=f'{self.name}_frames')
    template = os.path.splitext(base)[0] + '_{index:04d}.png'
    directory = os.path.dirname(template)
    if directory:
      os.makedirs(directory, exist_ok=True)
    for idx in frame_indices:
      gen = generations[idx]
      subset = df[df[self.index] == gen]
      if subset.empty:
        continue
      rank_mask = self._prepare_rank_mask(subset)
      fig, axes = self._create_figure()
      self._populate_axes(axes, subset, rank_mask)
      fig.suptitle(f'{self.name}: Generation {self._format_generation(gen)}')
      fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
      fig.savefig(template.format(index=idx), dpi=150)
      plt.close(fig)

  def _create_figure(self):
    n_pairs = len(self.pairs)
    ncols = min(3, n_pairs)
    nrows = int(math.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    return fig, axes

  def _populate_axes(self, axes, subset, rank_mask):
    nrows, ncols = axes.shape
    total_axes = nrows * ncols
    handles = []
    labels = []
    for idx, pair in enumerate(self.pairs):
      row = idx // ncols
      col = idx % ncols
      ax = axes[row][col]
      ax.cla()
      x = subset[pair[0]].to_numpy(dtype=float)
      y = subset[pair[1]].to_numpy(dtype=float)
      if x.size == 0:
        ax.set_axis_off()
        continue
      self._draw_density(ax, x, y)
      dominated_mask = ~rank_mask if rank_mask is not None else np.zeros_like(x, dtype=bool)
      if dominated_mask.any():
        scatter_pop = ax.scatter(x[dominated_mask], y[dominated_mask], s=20,
                                 color='gray', alpha=0.45,
                                 label='Population' if 'Population' not in labels else None)
        if scatter_pop.get_label():
          handles.append(scatter_pop)
          labels.append(scatter_pop.get_label())
      if rank_mask is not None and rank_mask.any():
        scatter_rank = ax.scatter(x[rank_mask], y[rank_mask], s=40, color='tab:orange',
                                  edgecolor='black', linewidth=0.6,
                                  label='Rank-1' if 'Rank-1' not in labels else None, zorder=3)
        if scatter_rank.get_label():
          handles.append(scatter_rank)
          labels.append(scatter_rank.get_label())
      ax.set_xlabel(pair[0])
      ax.set_ylabel(pair[1])
      ax.set_title(f'{pair[0]} vs {pair[1]}')
      ax.grid(alpha=0.3, linestyle='--')
    for idx in range(len(self.pairs), total_axes):
      row = idx // ncols
      col = idx % ncols
      axes[row][col].set_axis_off()
    if handles:
      axes[0][0].legend(handles, labels, loc='best', frameon=True)

  def _prepare_rank_mask(self, subset):
    rank_mask = None
    if 'rank' in subset.columns:
      try:
        rank_mask = subset['rank'].astype(float).to_numpy() == 1.0
      except (ValueError, TypeError):
        rank_mask = None
    if rank_mask is None:
      rank_mask = self._compute_pareto_mask(subset[self.objectives].to_numpy(dtype=float))
    return rank_mask

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

  @staticmethod
  def _select_generations(all_gens, limit):
    if limit >= len(all_gens):
      return list(all_gens)
    positions = np.linspace(0, len(all_gens) - 1, limit, dtype=int)
    selected = []
    for idx in positions:
      if idx not in selected:
        selected.append(idx)
    cursor = 0
    while len(selected) < limit and cursor < len(all_gens):
      if cursor not in selected:
        selected.append(cursor)
      cursor += 1
    selected = sorted(set(selected))
    if len(selected) > limit:
      selected = selected[:limit - 1] + [len(all_gens) - 1]
    elif selected[-1] != len(all_gens) - 1:
      selected[-1] = len(all_gens) - 1
    return [all_gens[i] for i in sorted(selected)]

  @staticmethod
  def _format_generation(genID):
    if float(genID).is_integer():
      return int(genID)
    return genID

  def _draw_density(self, ax, x, y):
    if x.size < 5 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
      return
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if np.isclose(xmax, xmin) or np.isclose(ymax, ymin):
      return
    xi = np.linspace(xmin, xmax, self.gridPoints)
    yi = np.linspace(ymin, ymax, self.gridPoints)
    X, Y = np.meshgrid(xi, yi)

    range_x = max(xmax - xmin, 1e-8)
    range_y = max(ymax - ymin, 1e-8)
    bw_x = max(np.std(x), 0.15 * range_x, 1e-8)
    bw_y = max(np.std(y), 0.15 * range_y, 1e-8)

    dx = (X[..., np.newaxis] - x[np.newaxis, np.newaxis, :]) / bw_x
    dy = (Y[..., np.newaxis] - y[np.newaxis, np.newaxis, :]) / bw_y
    density = np.exp(-0.5 * (dx ** 2 + dy ** 2))
    density = density.sum(axis=2)
    density /= (x.size * 2.0 * math.pi * bw_x * bw_y)
    levels = np.linspace(np.min(density), np.max(density), 8)
    if np.allclose(levels[0], levels[-1]):
      return
    ax.contourf(xi, yi, density, levels=levels, cmap='Blues', alpha=0.6)

  @staticmethod
  def _compute_pareto_mask(values):
    if values.size == 0:
      return np.array([], dtype=bool)
    n_points = values.shape[0]
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
      if not mask[i]:
        continue
      dominates = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
      mask[dominates] = False
    return mask
