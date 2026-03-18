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
Animate spatial sampling coverage to ensure design variables are thoroughly explored.

The plot overlays a density heatmap (2D histogram) of sampled points with the actual sample scatter
for each generation. Watching the animation exposes gaps, forgotten corners, or collapsing exploration.

What-if scenarios

Heatmap remains concentrated in one region -> sampling failed to explore the domain; increase mutation
  range, switch samplers, or reinitialise populations.
Coverage expands early then retracts -> restarts or survivor selection might be collapsing diversity;
  review elitism settings.
Uniform coverage but scatter shows repeated clusters -> algorithm revisits the same zones; consider
  penalising duplicates or adding local perturbations.
Coverage map gradually fills the domain -> healthy exploration; monitor convergence criteria vs coverage needs.
"""

import io
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


class SamplingCoverageMapPlot(PlotInterface):
  """
  Animated 2D density map coupled with scatter samples across generations.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    variables = InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""Two spatial variables to monitor for coverage.""")
    spec.addSub(variables)
    spec.addSub(InputData.parameterInputFactory('bins', contentType=InputTypes.IntegerType,
        descr=r"""Number of histogram bins per axis (default 40)."""))
    spec.addSub(InputData.parameterInputFactory('max_frames', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on the number of generations rendered. Defaults to min(total generations, 10)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combinations."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for generated animations. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('save_frames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves sampled generations as standalone PNG frames alongside the animation outputs."""))
    spec.addSub(InputData.parameterInputFactory('frames_max', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <save_frames> is true. Defaults to 10; generations are sampled evenly."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'SamplingCoverageMapPlot'
    self.source = None
    self.sourceName = None
    self.index = None
    self.variables = []
    self.bins = 40
    self.maxFrames = None
    self.formats = {'gif', 'html'}
    self.fps = 2.0
    self.save_frames = False
    self.frame_max = 10
    self._ranges = None

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for SamplingCoverageMapPlot "{self.name}".')
    self.sourceName = sourceNode.value

    indexNode = spec.findFirst('index')
    if indexNode is None or indexNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for SamplingCoverageMapPlot "{self.name}".')
    self.index = indexNode.value

    varsNode = spec.findFirst('variables')
    if varsNode is None or not varsNode.value or len(varsNode.value) < 2:
      self.raiseAnError(IOError, f'SamplingCoverageMapPlot "{self.name}" requires at least two <variables>.')
    self.variables = varsNode.value[:2]

    binsNode = spec.findFirst('bins')
    if binsNode is not None and binsNode.value is not None:
      self.bins = max(5, int(binsNode.value))

    maxNode = spec.findFirst('max_frames')
    if maxNode is not None and maxNode.value is not None:
      self.maxFrames = int(maxNode.value)
      if self.maxFrames <= 0:
        self.raiseAnError(IOError, f'SamplingCoverageMapPlot "{self.name}" received non-positive <max_frames>.')

    formatNode = spec.findFirst('format')
    if formatNode is not None and formatNode.value is not None:
      raw = formatNode.value.strip().lower()
      if raw == 'both' or not raw:
        self.formats = {'gif', 'html'}
      else:
        requested = set()
        for frag in raw.replace(';', ',').split(','):
          token = frag.strip()
          if not token:
            continue
          if token in ('gif', 'html'):
            requested.add(token)
          elif token == 'both':
            requested.update({'gif', 'html'})
          else:
            self.raiseAnError(IOError, f'Unsupported <format> "{token}" for SamplingCoverageMapPlot "{self.name}".')
        if not requested:
          requested = {'gif'}
        self.formats = requested

    fpsNode = spec.findFirst('fps')
    if fpsNode is not None and fpsNode.value is not None:
      self.fps = float(fpsNode.value)
      if self.fps <= 0:
        self.raiseAnError(IOError, f'SamplingCoverageMapPlot "{self.name}" received non-positive <fps>.')

    saveNode = spec.findFirst('save_frames')
    if saveNode is not None and saveNode.value is not None:
      self.save_frames = bool(saveNode.value)

    framesNode = spec.findFirst('frames_max')
    if framesNode is not None and framesNode.value is not None:
      self.frame_max = int(framesNode.value)
      if self.frame_max <= 0:
        self.raiseAnError(IOError, f'SamplingCoverageMapPlot "{self.name}" received non-positive <frames_max>.')

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for SamplingCoverageMapPlot "{self.name}".')
    self.source = src
    available = self.source.getVars()
    missing = [var for var in [self.index] + self.variables if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by SamplingCoverageMapPlot "{self.name}".')

  def run(self):
    df = self.source.asDataset().to_dataframe().copy()
    if df.empty:
      self.raiseAWarning(f'SamplingCoverageMapPlot "{self.name}" received an empty dataset; nothing to animate.')
      return
    df[self.index] = df[self.index].astype(float)
    for var in self.variables:
      df[var] = df[var].astype(float)
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'SamplingCoverageMapPlot "{self.name}" found no generations in column "{self.index}".')
      return
    self._ranges = self._compute_ranges(df)

    limit = self.maxFrames if self.maxFrames is not None else min(len(generations), 10)
    limit = max(1, min(limit, len(generations)))
    selected_generations, indices = self._sample_generations(generations, limit)

    if 'gif' in self.formats:
      self._write_gif(df, selected_generations)
    if 'html' in self.formats:
      self._write_html(df, selected_generations)
    if self.save_frames:
      self._write_frames(df, selected_generations)

  def _write_gif(self, df, generations):
    filename = self._createFilename(defaultName=f'{self.name}.gif')
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=0) as writer:
      for generation in generations:
        fig = self._render_frame(df, generation)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html(self, df, generations):
    filename = self._createFilename(defaultName=f'{self.name}.html')
    fig, ax, im, scatter = self._setup_axes()

    def init():
      im.set_data(np.zeros((self.bins, self.bins)))
      scatter.set_offsets(np.empty((0, 2)))
      ax.set_title('')
      return im, scatter

    def update(idx):
      generation = generations[idx]
      density, xedges, yedges, samples = self._compute_density(df, generation)
      im.set_extent([xedges[0], xedges[-1], yedges[0], yedges[-1]])
      im.set_data(density.T)
      if np.isfinite(density).any():
        im.set_clim(0.0, np.nanmax(density) or 1.0)
      scatter.set_offsets(samples)
      ax.set_title(f'Generation {self._format_generation(generation)}')
      return im, scatter

    anim = animation.FuncAnimation(fig, update, frames=range(len(generations)),
                                   init_func=init, interval=1000.0 / self.fps,
                                   blit=False)
    html_str = anim.to_jshtml()
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(f'<div style="display:flex;justify-content:center;">{html_str}</div>')
    plt.close(fig)

  def _write_frames(self, df, generations):
    frame_positions = self._select_frame_indices(len(generations))
    if not frame_positions:
      return
    base = self._createFilename(defaultName=f'{self.name}_frames')
    template = os.path.splitext(base)[0] + '_{index:04d}.png'
    directory = os.path.dirname(template)
    if directory:
      os.makedirs(directory, exist_ok=True)
    for pos in frame_positions:
      generation = generations[pos]
      fig = self._render_frame(df, generation)
      fig.savefig(template.format(index=int(generation)), dpi=150)
      plt.close(fig)

  def _render_frame(self, df, generation):
    density, xedges, yedges, samples = self._compute_density(df, generation)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(density.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   origin='lower', aspect='auto', cmap='viridis')
    if np.isfinite(density).any():
      im.set_clim(0.0, np.nanmax(density) or 1.0)
    ax.scatter(samples[:, 0], samples[:, 1], s=20, c='white', edgecolors='k', linewidths=0.2, alpha=0.8)
    ax.set_xlim(self._ranges[0])
    ax.set_ylim(self._ranges[1])
    ax.set_xlabel(self.variables[0])
    ax.set_ylabel(self.variables[1])
    ax.set_title(f'Sampling coverage (Generation {self._format_generation(generation)})')
    ax.grid(alpha=0.2, linestyle='--')
    fig.colorbar(im, ax=ax, label='Sample density')
    fig.tight_layout()
    return fig

  def _setup_axes(self):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(np.zeros((self.bins, self.bins)), origin='lower', aspect='auto',
                   cmap='viridis', extent=[self._ranges[0][0], self._ranges[0][1],
                                           self._ranges[1][0], self._ranges[1][1]])
    scatter = ax.scatter([], [], s=20, c='white', edgecolors='k', linewidths=0.2, alpha=0.8)
    ax.set_xlim(self._ranges[0])
    ax.set_ylim(self._ranges[1])
    ax.set_xlabel(self.variables[0])
    ax.set_ylabel(self.variables[1])
    fig.colorbar(im, ax=ax, label='Sample density')
    fig.tight_layout()
    return fig, ax, im, scatter

  def _compute_density(self, df, generation):
    subset = df[df[self.index] == generation]
    samples = subset[self.variables].to_numpy(dtype=float)
    samples = samples[np.isfinite(samples).all(axis=1)]
    if samples.size == 0:
      density = np.zeros((self.bins, self.bins), dtype=float)
      xedges = np.linspace(self._ranges[0][0], self._ranges[0][1], self.bins + 1)
      yedges = np.linspace(self._ranges[1][0], self._ranges[1][1], self.bins + 1)
      samples = np.empty((0, 2))
      return density, xedges, yedges, samples
    xedges = np.linspace(self._ranges[0][0], self._ranges[0][1], self.bins + 1)
    yedges = np.linspace(self._ranges[1][0], self._ranges[1][1], self.bins + 1)
    density, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1],
                                             bins=[xedges, yedges])
    density = density.astype(float)
    return density, xedges, yedges, samples

  def _compute_ranges(self, df):
    ranges = []
    for var in self.variables:
      data = df[var].to_numpy(dtype=float)
      data = data[np.isfinite(data)]
      if data.size == 0:
        ranges.append((0.0, 1.0))
        continue
      low = float(np.min(data))
      high = float(np.max(data))
      if math.isclose(low, high):
        delta = abs(low) if low != 0 else 1.0
        low -= 0.5 * delta
        high += 0.5 * delta
      padding = 0.05 * (high - low)
      ranges.append((low - padding, high + padding))
    return tuple(ranges)

  @staticmethod
  def _sample_generations(generations, limit):
    if limit >= len(generations):
      return generations, list(range(len(generations)))
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
    if selected[-1] != len(generations) - 1:
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

  @staticmethod
  def _format_generation(genID):
    if float(genID).is_integer():
      return int(genID)
    return genID
