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
import math
import re
from itertools import combinations

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
try:
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 needed for 3D projections
except ImportError:  # pragma: no cover
  Axes3D = None

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
        descr=r"""Objective names describing the scatter axes. Provide two or more to plot every pairwise combination."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Name of the variable containing the generation identifier (typically batchId)."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Pareto rank to display. Defaults to 1."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second to use for the animation GIF. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or a comma-separated list of formats."""))
    spec.addSub(InputData.parameterInputFactory('connect', contentType=InputTypes.StringType,
        descr=r"""How to connect Pareto front samples: "lines" (default), "none", or "surface" (3-objective only)."""))
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
    self.formats = {'gif', 'html'}
    self.connect_mode = 'lines'
    self._connect_warning_emitted = False
    self._surface_failure_warned = False

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
      self.formats = self._parse_formats(fmtNode.value)
    else:
      self.formats = {'gif', 'html'}
    connectNode = spec.findFirst('connect')
    if connectNode is not None:
      self.connect_mode = self._parse_connect_mode(connectNode.value)

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
    if len(self.objectives) < 2:
      self.raiseAnError(IOError, f'NSGAFrontAnimation "{self.name}" requires at least two objectives to form scatter axes.')

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

    axis_limits = self._compute_axis_limits(df)
    cd_limits = self._determine_color_limits(df)
    views = self._build_views()

    for fmt in self.formats:
      if fmt == 'html':
        self._write_html_animation(df, generations, views, axis_limits, cd_limits,
                                   filename_default=f'{self.name}.html')
      elif fmt == 'gif':
        self._write_gif_animation(df, generations, views, axis_limits, cd_limits,
                                  filename_default=f'{self.name}.gif')

  def _write_gif_animation(self, df, generations, views, axis_limits, cd_limits, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=1) as writer:
      for gen in generations:
        subset = df[df[self.index] == gen]
        fig, axes = self._make_axes_for_views(views)
        scatterArgs, color_data = self._build_scatter_args(subset, cd_limits)
        scatter_handles = []
        for ax, view in zip(axes, views):
          if view['dim'] == 3:
            xVar, yVar, zVar = view['vars']
            sc = ax.scatter(subset[xVar], subset[yVar], subset[zVar], **scatterArgs)
            self._maybe_draw_connection_static(ax, view, subset)
            xMin, xMax = axis_limits[xVar]
            yMin, yMax = axis_limits[yVar]
            zMin, zMax = axis_limits[zVar]
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
            ax.set_zlim(zMin, zMax)
            ax.set_xlabel(xVar)
            ax.set_ylabel(yVar)
            ax.set_zlabel(zVar)
            ax.set_title(f'{xVar} vs {yVar} vs {zVar}')
          else:
            xVar, yVar = view['vars']
            sc = ax.scatter(subset[xVar], subset[yVar], **scatterArgs)
            self._maybe_draw_connection_static(ax, view, subset)
            xMin, xMax = axis_limits[xVar]
            yMin, yMax = axis_limits[yVar]
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
            ax.set_xlabel(xVar)
            ax.set_ylabel(yVar)
            ax.set_title(f'{xVar} vs {yVar}')
          scatter_handles.append(sc)
        fig.suptitle(f'Generation {gen}')
        fig.tight_layout(rect=[0, 0, 0.86, 0.95])
        if scatter_handles and color_data is not None and np.size(color_data) > 0:
          cbar = self._add_colorbar(fig, axes, scatter_handles[0], cd_limits)
          if cbar is not None and cd_limits is not None:
            scatter_handles[0].set_clim(*cd_limits)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html_animation(self, df, generations, views, axis_limits, cd_limits, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    fig, axes = self._make_axes_for_views(views)
    init_subset = df[df[self.index] == generations[0]]
    scatterArgs, color_data = self._build_scatter_args(init_subset, cd_limits)
    color_payload = scatterArgs.pop('c', None)
    scatterArgs.pop('vmin', None)
    scatterArgs.pop('vmax', None)
    if color_payload is None:
      scatterArgs.pop('cmap', None)
    scatters = []
    connections = []
    for ax, view in zip(axes, views):
      if view['dim'] == 3:
        xVar, yVar, zVar = view['vars']
        sc = ax.scatter([], [], [], **scatterArgs)
        xMin, xMax = axis_limits[xVar]
        yMin, yMax = axis_limits[yVar]
        zMin, zMax = axis_limits[zVar]
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        ax.set_zlim(zMin, zMax)
        ax.set_xlabel(xVar)
        ax.set_ylabel(yVar)
        ax.set_zlabel(zVar)
        ax.set_title(f'{xVar} vs {yVar} vs {zVar}')
      else:
        xVar, yVar = view['vars']
        sc = ax.scatter([], [], **scatterArgs)
        xMin, xMax = axis_limits[xVar]
        yMin, yMax = axis_limits[yVar]
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        ax.set_xlabel(xVar)
        ax.set_ylabel(yVar)
        ax.set_title(f'{xVar} vs {yVar}')
      scatters.append(sc)
      connections.append(self._init_connection_artist(ax, view))
    suptitle = fig.suptitle('')
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    cbar = None
    if color_data is not None and np.size(color_data) > 0 and scatters:
      cbar = self._add_colorbar(fig, axes, scatters[0], cd_limits)

    def init():
      artists = []
      for sc, conn, view in zip(scatters, connections, views):
        if view['dim'] == 3:
          sc._offsets3d = (np.array([]), np.array([]), np.array([]))
          sc.set_array(np.array([]))
        else:
          sc.set_offsets(np.empty((0, 2)))
          sc.set_array(np.array([]))
        self._reset_connection_artist(conn, view)
        if conn['artist'] is not None:
          artists.append(conn['artist'])
      suptitle.set_text('')
      return tuple(list(scatters) + artists)

    def update(gen):
      subset = df[df[self.index] == gen]
      _, frame_colors = self._build_scatter_args(subset, cd_limits)
      artists = []
      for ax, sc, conn, view in zip(axes, scatters, connections, views):
        if view['dim'] == 3:
          xVar, yVar, zVar = view['vars']
          xs = subset[xVar].to_numpy()
          ys = subset[yVar].to_numpy()
          zs = subset[zVar].to_numpy()
          sc._offsets3d = (xs, ys, zs)
        else:
          xVar, yVar = view['vars']
          offsets = np.column_stack((subset[xVar].to_numpy(), subset[yVar].to_numpy()))
          sc.set_offsets(offsets)
        if frame_colors is not None and np.size(frame_colors) > 0:
          sc.set_array(np.asarray(frame_colors))
          sc.set_cmap('viridis')
          if cd_limits is not None:
            sc.set_clim(*cd_limits)
        else:
          sc.set_array(np.array([]))
        conn_artist = self._update_connection_artist(conn, ax, view, subset)
        if conn_artist is not None:
          artists.append(conn_artist)
      if cbar is not None and scatters:
        cbar.update_normal(scatters[0])
      suptitle.set_text(f'Generation {gen}')
      return tuple(list(scatters) + artists)

    anim = animation.FuncAnimation(fig, update, frames=generations, init_func=init,
                                   interval=1000.0 / self.fps, blit=False)
    html_str = anim.to_jshtml()
    html_str = self._normalize_animation_ids(html_str)
    centered_html = f'<div style="display:flex;justify-content:center;">{html_str}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centered_html)
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

  def _build_views(self):
    if len(self.objectives) == 2:
      return [{'vars': tuple(self.objectives[:2]), 'dim': 2, 'connect': self._resolve_connect_mode(2)}]
    if len(self.objectives) == 3:
      return [{'vars': tuple(self.objectives[:3]), 'dim': 3, 'connect': self._resolve_connect_mode(3)}]
    pairs = list(combinations(self.objectives, 2))
    return [{'vars': pair, 'dim': 2, 'connect': self._resolve_connect_mode(2)} for pair in pairs]

  def _maybe_draw_connection_static(self, ax, view, subset):
    mode = view.get('connect', 'lines')
    if mode == 'lines':
      self._draw_static_line(ax, view, subset)
    elif mode == 'surface' and view['dim'] == 3:
      self._build_surface(ax, subset, view['vars'], color='#555555', alpha=0.35)

  @staticmethod
  def _draw_static_line(ax, view, subset):
    if len(subset) <= 1:
      return
    ordered = subset.sort_values(by=view['vars'][0])
    if view['dim'] == 3:
      xVar, yVar, zVar = view['vars']
      ax.plot(ordered[xVar], ordered[yVar], ordered[zVar], color='k', linewidth=0.8, alpha=0.6)
    else:
      xVar, yVar = view['vars']
      ax.plot(ordered[xVar], ordered[yVar], color='k', linewidth=0.8, alpha=0.6)

  def _init_connection_artist(self, ax, view):
    entry = {'mode': view.get('connect', 'lines'), 'artist': None}
    if entry['mode'] == 'lines':
      if view['dim'] == 3:
        entry['artist'], = ax.plot([], [], [], color='k', linewidth=0.8, alpha=0.6)
      else:
        entry['artist'], = ax.plot([], [], color='k', linewidth=0.8, alpha=0.6)
    return entry

  def _reset_connection_artist(self, entry, view):
    mode = entry['mode']
    artist = entry.get('artist')
    if mode == 'lines' and artist is not None:
      artist.set_data([], [])
      if view['dim'] == 3:
        artist.set_3d_properties([])
    elif mode == 'surface' and artist is not None:
      artist.remove()
      entry['artist'] = None

  def _update_connection_artist(self, entry, ax, view, subset):
    mode = entry['mode']
    if mode == 'lines':
      artist = entry.get('artist')
      if artist is None:
        return None
      if len(subset) > 1:
        ordered = subset.sort_values(by=view['vars'][0])
        if view['dim'] == 3:
          xVar, yVar, zVar = view['vars']
          artist.set_data(ordered[xVar].to_numpy(), ordered[yVar].to_numpy())
          artist.set_3d_properties(ordered[zVar].to_numpy())
        else:
          xVar, yVar = view['vars']
          artist.set_data(ordered[xVar].to_numpy(), ordered[yVar].to_numpy())
      else:
        artist.set_data([], [])
        if view['dim'] == 3:
          artist.set_3d_properties([])
      return artist
    if mode == 'surface':
      current = entry.get('artist')
      if current is not None:
        current.remove()
        entry['artist'] = None
      if view['dim'] != 3:
        return None
      surface = self._build_surface(ax, subset, view['vars'], color='#555555', alpha=0.35)
      entry['artist'] = surface
      return surface
    return None

  def _build_surface(self, ax, subset, variables, color, alpha=0.35):
    if len(subset) < 3:
      return None
    xVals = subset[variables[0]].to_numpy()
    yVals = subset[variables[1]].to_numpy()
    zVals = subset[variables[2]].to_numpy()
    try:
      surface = ax.plot_trisurf(xVals, yVals, zVals, color=color, alpha=alpha, linewidth=0.2, edgecolor='none')
    except (ValueError, RuntimeError) as err:
      if not self._surface_failure_warned:
        self.raiseAWarning(f'Unable to build trisurface for view {variables}: {err}. Falling back to scatter only.')
        self._surface_failure_warned = True
      return None
    return surface

  def _compute_axis_limits(self, df):
    limits = {}
    for obj in self.objectives:
      column = df[obj]
      limits[obj] = self._scaled_bounds(column.min(), column.max())
    return limits

  def _make_axes(self, panels):
    panels = max(1, panels)
    if panels == 1:
      fig, ax = plt.subplots(figsize=(6, 6))
      return fig, [ax]
    # arrange panels in a grid that keeps layout balanced and centered
    max_cols = 3
    if panels == 4:
      ncols = 2
    else:
      ncols = min(max_cols, panels)
    nrows = int(math.ceil(panels / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes_grid).flatten()
    for idx in range(panels, len(axes)):
      fig.delaxes(axes[idx])
    return fig, list(axes[:panels])

  def _make_axes_for_views(self, views):
    if len(views) == 1 and views[0]['dim'] == 3:
      if Axes3D is None:
        self.raiseAnError(RuntimeError, 'mpl_toolkits.mplot3d is not available but 3 objectives were requested for {}.'.format(self.name))
      fig = plt.figure(figsize=(7, 6))
      ax = fig.add_subplot(111, projection='3d')
      return fig, [ax]
    return self._make_axes(len(views))

  def _parse_formats(self, raw):
    if raw is None:
      return {'gif', 'html'}
    text = raw.strip().lower()
    if text == 'both':
      return {'gif', 'html'}
    tokens = {token.strip() for token in text.replace(';', ',').split(',') if token.strip()}
    if not tokens:
      return {'gif', 'html'}
    allowed = {'gif', 'html'}
    invalid = tokens - allowed
    if invalid:
      bad = ', '.join(sorted(invalid))
      self.raiseAnError(IOError, f'Unsupported format(s) "{bad}" for NSGAFrontAnimation "{self.name}". Use "gif", "html", or "both".')
    return tokens

  def _parse_connect_mode(self, raw):
    if raw is None:
      return 'lines'
    text = raw.strip().lower()
    allowed = {'lines', 'none', 'surface'}
    if text not in allowed:
      options = ', '.join(sorted(allowed))
      self.raiseAnError(IOError, f'Unsupported connect mode "{raw}" for NSGAFrontAnimation "{self.name}". Choose from: {options}.')
    return text

  def _resolve_connect_mode(self, dim):
    if self.connect_mode == 'surface' and dim != 3:
      if not self._connect_warning_emitted:
        self.raiseAWarning('connect="surface" is only available for three-objective views; disabling connections for lower-dimensional projections.')
        self._connect_warning_emitted = True
      return 'none'
    return self.connect_mode

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

  @staticmethod
  def _add_colorbar(fig, axes, mappable, cd_limits):
    """
    Place a shared colorbar just outside the subplot grid so it never overlaps a panel.
    """
    if mappable is None:
      return None
    axes_array = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    axes_list = list(axes_array)
    if not axes_list:
      return None
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes_list]
    max_right = max(pos.x1 for pos in positions)
    min_bottom = min(pos.y0 for pos in positions)
    max_top = max(pos.y1 for pos in positions)
    pad = 0.025
    width = 0.02
    left = max_right + pad
    if left + width > 0.98:
      width = max(0.01, 0.98 - left)
    if width <= 0:
      left = 0.96
      width = 0.02
    cax = fig.add_axes([left, min_bottom, width, max_top - min_bottom])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label('CD')
    if cd_limits is not None:
      mappable.set_clim(*cd_limits)
    return cbar
