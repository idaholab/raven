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
Animation of the Pareto front evolution over optimizer generations.
"""

import io
import hashlib
import re

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class NSGAFrontAnimation(PlotInterface):
  """
  Produces a GIF showing how the Pareto front evolves across generations.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Two objective names describing the axes of the scatter plot."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Name of the variable containing the generation identifier (typically batchId)."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Pareto rank to display. Defaults to 1."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second to use for the animation GIF. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif" (default) or "html" for an interactive animation with play/pause controls."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA Front Animation'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
    self.rank = 1
    self.fps = 2.0
    self.format = 'gif'

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    objectives = spec.findFirst('objectives')
    if objectives is None:
      self.raiseAnError(IOError, f'Missing <objectives> node for NSGAFrontAnimation "{self.name}".')
    self.objectives = objectives.value
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, f'Missing <index> node for NSGAFrontAnimation "{self.name}".')
    self.index = idxNode.value
    rankNode = spec.findFirst('rank')
    if rankNode is not None:
      self.rank = rankNode.value
    fpsNode = spec.findFirst('fps')
    if fpsNode is not None:
      self.fps = max(fpsNode.value, 0.1)
    fmtNode = spec.findFirst('format')
    if fmtNode is not None:
      fmt = fmtNode.value.strip().lower()
      if fmt not in {'gif', 'html'}:
        self.raiseAnError(IOError, f'Unsupported format "{fmt}" for NSGAFrontAnimation "{self.name}". Use "gif" or "html".')
      self.format = fmt

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for NSGAFrontAnimation "{self.name}".')
    dataVars = self.source.getVars()
    missing = [var for var in self.objectives + [self.index] if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing variable(s) {} required by NSGAFrontAnimation "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)
    if len(self.objectives) != 2:
      self.raiseAnError(IOError, 'NSGAFrontAnimation "{}" currently supports exactly two objectives.'.format(self.name))

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if self.index not in df.columns:
      self.raiseAnError(IOError, f'Index variable "{self.index}" not present for NSGAFrontAnimation "{self.name}".')
    if 'rank' in df.columns:
      df = df[df['rank'] == self.rank]
      if df.empty:
        self.raiseAWarning(f'No samples with rank == {self.rank}; falling back to all samples.')
        df = self.source.asDataset().to_dataframe()

    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'No generations found for "{self.name}". Nothing to animate.')
      return

    xVar, yVar = self.objectives
    xMin, xMax = self._scaled_bounds(df[xVar].min(), df[xVar].max())
    yMin, yMax = self._scaled_bounds(df[yVar].min(), df[yVar].max())
    cd_limits = self._determine_color_limits(df)

    if self.format == 'html':
      self._write_html_animation(df, generations, xVar, yVar, xMin, xMax, yMin, yMax, cd_limits,
                                 filename_default=f'{self.name}.html')
    else:
      self._write_gif_animation(df, generations, xVar, yVar, xMin, xMax, yMin, yMax, cd_limits,
                                filename_default=f'{self.name}.gif')

  def _write_gif_animation(self, df, generations, xVar, yVar, xMin, xMax, yMin, yMax, cd_limits, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=1) as writer:
      for gen in generations:
        subset = df[df[self.index] == gen]
        fig, ax = plt.subplots()
        scatterArgs, color_data = self._build_scatter_args(subset, cd_limits)
        sc = ax.scatter(subset[xVar], subset[yVar], **scatterArgs)
        if len(subset) > 1:
          ordered = subset.sort_values(by=xVar)
          ax.plot(ordered[xVar], ordered[yVar], color='k', linewidth=0.8, alpha=0.6)
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        ax.set_xlabel(xVar)
        ax.set_ylabel(yVar)
        ax.set_title(f'Generation {gen}')
        if color_data is not None and np.size(color_data) > 0:
          cbar = fig.colorbar(sc, ax=ax)
          cbar.set_label('CD')
          if cd_limits is not None:
            sc.set_clim(*cd_limits)
        fig.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html_animation(self, df, generations, xVar, yVar, xMin, xMax, yMin, yMax, cd_limits, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    fig, ax = plt.subplots()
    init_subset = df[df[self.index] == generations[0]]
    scatterArgs, color_data = self._build_scatter_args(init_subset, cd_limits)
    scatterArgs.pop('c', None)
    scatterArgs.pop('cmap', None)
    scatterArgs.pop('vmin', None)
    scatterArgs.pop('vmax', None)
    sc = ax.scatter([], [], **scatterArgs)
    line, = ax.plot([], [], color='k', linewidth=0.8, alpha=0.6)
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_xlabel(xVar)
    ax.set_ylabel(yVar)
    title = ax.set_title('')
    cbar = None
    if color_data is not None and np.size(color_data) > 0:
      cbar = fig.colorbar(sc, ax=ax)
      cbar.set_label('CD')

    def init():
      sc.set_offsets(np.empty((0, 2)))
      if color_data is not None:
        sc.set_array(np.array([]))
      line.set_data([], [])
      title.set_text('')
      return sc, line

    def update(gen):
      subset = df[df[self.index] == gen]
      offsets = np.column_stack((subset[xVar].to_numpy(), subset[yVar].to_numpy()))
      sc.set_offsets(offsets)
      if len(subset) > 1:
        ordered = subset.sort_values(by=xVar)
        line.set_data(ordered[xVar].to_numpy(), ordered[yVar].to_numpy())
      else:
        line.set_data([], [])
      _, frame_colors = self._build_scatter_args(subset, cd_limits)
      if frame_colors is not None and np.size(frame_colors) > 0:
        sc.set_array(np.asarray(frame_colors))
        sc.set_cmap('viridis')
        if cd_limits is not None:
          sc.set_clim(*cd_limits)
        if cbar is not None:
          cbar.update_normal(sc)
      else:
        sc.set_array(np.array([]))
      title.set_text(f'Generation {gen}')
      return sc, line

    anim = animation.FuncAnimation(fig, update, frames=generations, init_func=init,
                                   interval=1000.0 / self.fps, blit=False)
    html_str = anim.to_jshtml()
    html_str = self._normalize_animation_ids(html_str)
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(html_str)
    plt.close(fig)

  def _build_scatter_args(self, subset, cd_limits):
    scatterArgs = {'edgecolors': 'k', 'linewidths': 0.3}
    color_data = None
    if 'CD' in subset.columns and not subset.empty:
      cdValues = subset['CD'].replace([np.inf, -np.inf], np.nan)
      if cdValues.notna().any():
        fillValue = cdValues[cdValues.notna()].mean()
        color_data = cdValues.fillna(fillValue if np.isfinite(fillValue) else 0.0).to_numpy()
      else:
        color_data = np.zeros(len(cdValues))
      if cd_limits is not None:
        vmin, vmax = cd_limits
      else:
        vmin = np.nanmin(color_data)
        vmax = np.nanmax(color_data)
      scatterArgs.update({'c': color_data, 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax})
    return scatterArgs, color_data

  @staticmethod
  def _scaled_bounds(min_val, max_val):
    if np.isclose(min_val, max_val):
      delta = abs(min_val) if min_val != 0 else 1.0
      return min_val - 0.1 * delta, max_val + 0.1 * delta
    low = min_val * 0.9 if min_val >= 0 else min_val * 1.1
    high = max_val * 1.1 if max_val >= 0 else max_val * 0.9
    if np.isclose(low, high):
      delta = abs(low) if low != 0 else 1.0
      low -= 0.1 * delta
      high += 0.1 * delta
    return low, high

  @staticmethod
  def _determine_color_limits(df):
    if 'CD' not in df.columns or df['CD'].empty:
      return None
    cdValues = df['CD'].replace([np.inf, -np.inf], np.nan)
    if cdValues.notna().any():
      vmin = cdValues.min()
      vmax = cdValues.max()
      if np.isfinite(vmin) and np.isfinite(vmax) and not np.isclose(vmin, vmax):
        return (vmin, vmax)
    return None

  def _normalize_animation_ids(self, html_str):
    """
    Replace randomly generated Matplotlib animation element ids with deterministic ones.
    """
    match = re.search(r'_anim_img([0-9a-f]+)', html_str)
    if not match:
      return html_str
    random_suffix = match.group(1)
    base_name = self.name if getattr(self, 'name', None) else 'animation'
    safe_name = ''.join(ch if ch.isalnum() else '_' for ch in base_name)
    seed = f'{self.__class__.__name__}:{safe_name}:{self.rank}'
    deterministic = hashlib.md5(seed.encode('utf-8')).hexdigest()
    return html_str.replace(random_suffix, deterministic)
