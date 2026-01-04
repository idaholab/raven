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
Animation showing all Pareto fronts (by rank) evolving through generations.
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
from matplotlib.cm import get_cmap
try:
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for projection registration)
except ImportError:  # pragma: no cover
  Axes3D = None

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class NSGAFrontRankAnimation(PlotInterface):
  """
  Visualizes the evolution of all Pareto fronts coloured by rank, with per-frame annotations.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Objective names to use for the scatter plot axes. At least two are required."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for the animation. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or a comma-separated combination."""))
    spec.addSub(InputData.parameterInputFactory('connect', contentType=InputTypes.StringType,
        descr=r"""Connection style for samples: "lines" (default), "none", or "surface" (three objectives only)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA Rank Animation'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
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
      self.raiseAnError(IOError, f'Missing <objectives> node for NSGAFrontRankAnimation "{self.name}".')
    self.objectives = objectives.value
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, f'Missing <index> node for NSGAFrontRankAnimation "{self.name}".')
    self.index = idxNode.value
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
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for NSGAFrontRankAnimation "{self.name}".')
    dataVars = self.source.getVars()
    required = self.objectives + [self.index, 'rank']
    missing = [var for var in required if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing variable(s) {} required by NSGAFrontRankAnimation "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)
    if len(self.objectives) < 2:
      self.raiseAnError(IOError, f'NSGAFrontRankAnimation "{self.name}" requires at least two objectives.')

  def run(self):
    df = self.source.asDataset().to_dataframe()
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'No generations found for NSGAFrontRankAnimation "{self.name}".')
      return
    axis_limits = self._compute_axis_limits(df)
    ranks = sorted(df['rank'].unique())
    cmap = get_cmap('tab10', len(ranks))
    color_lookup = {rank: cmap(idx) for idx, rank in enumerate(ranks)}
    views = self._build_views()

    for fmt in self.formats:
      if fmt == 'html':
        self._write_html(df, generations, views, axis_limits, ranks, color_lookup,
                         filename_default=f'{self.name}_rank_animation.html')
      elif fmt == 'gif':
        self._write_gif(df, generations, views, axis_limits, ranks, color_lookup,
                        filename_default=f'{self.name}_rank_animation.gif')

  def _write_gif(self, df, generations, views, axis_limits, ranks, color_lookup, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=1) as writer:
      for gen in generations:
        subset = df[df[self.index] == gen]
        fig, axes = self._make_axes_for_views(views)
        for ax_idx, (ax, view) in enumerate(zip(axes, views)):
          vars_tuple = view['vars']
          for rank in ranks:
            front = subset[subset['rank'] == rank]
            if front.empty:
              continue
            if view['dim'] == 3:
              xVar, yVar, zVar = vars_tuple
              ax.scatter(front[xVar], front[yVar], front[zVar], color=color_lookup[rank], label=f'Rank {rank}',
                         edgecolors='k', linewidths=0.3)
            else:
              xVar, yVar = vars_tuple
              ax.scatter(front[xVar], front[yVar], color=color_lookup[rank], label=f'Rank {rank}',
                         edgecolors='k', linewidths=0.3)
            self._maybe_draw_rank_connection_static(ax, view, front, color_lookup[rank])
          if view['dim'] == 3:
            xVar, yVar, zVar = vars_tuple
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
            xVar, yVar = vars_tuple
            xMin, xMax = axis_limits[xVar]
            yMin, yMax = axis_limits[yVar]
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
            ax.set_xlabel(xVar)
            ax.set_ylabel(yVar)
            ax.set_title(f'{xVar} vs {yVar}')
          if ax_idx == 0:
            self._add_rank_annotation(ax, subset, ranks)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
              ax.legend(loc='best')
        fig.suptitle(f'Generation {gen}')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html(self, df, generations, views, axis_limits, ranks, color_lookup, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    fig, axes = self._make_axes_for_views(views)
    scatters = []
    connections = []
    annotation_texts = []
    for idx, (ax, view) in enumerate(zip(axes, views)):
      scatter_handles = {}
      for rank in ranks:
        if view['dim'] == 3:
          scatter_handles[rank] = ax.scatter([], [], [], color=color_lookup[rank], label=f'Rank {rank}',
                                             edgecolors='k', linewidths=0.3)
        else:
          scatter_handles[rank] = ax.scatter([], [], color=color_lookup[rank], label=f'Rank {rank}',
                                             edgecolors='k', linewidths=0.3)
      connection_handles = {rank: self._init_rank_connection_artist(ax, view, color_lookup[rank]) for rank in ranks}
      if view['dim'] == 3:
        xVar, yVar, zVar = view['vars']
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
        xMin, xMax = axis_limits[xVar]
        yMin, yMax = axis_limits[yVar]
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        ax.set_xlabel(xVar)
        ax.set_ylabel(yVar)
        ax.set_title(f'{xVar} vs {yVar}')
      if idx == 0 and ranks:
        ax.legend(loc='best')
        annotation_texts = []
        for r_idx in range(len(ranks)):
          y = 0.95 - r_idx * 0.05
          if view['dim'] == 3:
            txt = ax.text2D(0.02, y, '', transform=ax.transAxes, fontsize=9, va='top')
          else:
            txt = ax.text(0.02, y, '', transform=ax.transAxes, fontsize=9, va='top')
          annotation_texts.append(txt)
      scatters.append(scatter_handles)
      connections.append(connection_handles)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    suptitle = fig.suptitle('')

    def init():
      artists = []
      for view, scatter_handles, connection_handles in zip(views, scatters, connections):
        for sc in scatter_handles.values():
          if view['dim'] == 3:
            sc._offsets3d = (np.array([]), np.array([]), np.array([]))
          else:
            sc.set_offsets(np.empty((0, 2)))
        for conn in connection_handles.values():
          self._reset_rank_connection_artist(conn, view)
          if conn['artist'] is not None:
            artists.append(conn['artist'])
        artists.extend(scatter_handles.values())
      for txt in annotation_texts:
        txt.set_text('')
      suptitle.set_text('')
      artists.extend(annotation_texts)
      return artists

    def update(gen):
      subset = df[df[self.index] == gen]
      artists = []
      for axis_idx, (view, scatter_handles, connection_handles) in enumerate(zip(views, scatters, connections)):
        for r_idx, rank in enumerate(ranks):
          front = subset[subset['rank'] == rank]
          if view['dim'] == 3:
            xVar, yVar, zVar = view['vars']
            xs = front[xVar].to_numpy()
            ys = front[yVar].to_numpy()
            zs = front[zVar].to_numpy()
            scatter_handles[rank]._offsets3d = (xs, ys, zs)
          else:
            xVar, yVar = view['vars']
            offsets = np.column_stack((front[xVar].to_numpy(), front[yVar].to_numpy()))
            scatter_handles[rank].set_offsets(offsets)
          conn_artist = self._update_rank_connection_artist(connection_handles[rank], axes[axis_idx], view, front)
          if conn_artist is not None:
            artists.append(conn_artist)
          if axis_idx == 0 and annotation_texts:
            annotation_texts[r_idx].set_text(f'Rank {rank}: {len(front)} pts')
        artists.extend(scatter_handles.values())
      suptitle.set_text(f'Generation {gen}')
      artists.extend(annotation_texts)
      return artists

    anim = animation.FuncAnimation(fig, update, frames=generations, init_func=init,
                                   interval=1000.0 / self.fps, blit=False)
    html_str = anim.to_jshtml()
    html_str = self._normalize_animation_ids(html_str)
    centered_html = f'<div style="display:flex;justify-content:center;">{html_str}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centered_html)
    plt.close(fig)

  @staticmethod
  def _add_rank_annotation(ax, subset, ranks):
    y = 0.95
    for rank in ranks:
      count = (subset['rank'] == rank).sum()
      if hasattr(ax, 'name') and getattr(ax, 'name', '').lower() == '3d':
        ax.text2D(0.02, y, f'Rank {rank}: {count} pts', transform=ax.transAxes, fontsize=9, va='top')
      else:
        ax.text(0.02, y, f'Rank {rank}: {count} pts', transform=ax.transAxes, fontsize=9, va='top')
      y -= 0.05

  def _compute_axis_limits(self, df):
    limits = {}
    for obj in self.objectives:
      column = df[obj]
      limits[obj] = self._scaled_bounds(column.min(), column.max())
    return limits

  def _build_views(self):
    if len(self.objectives) == 2:
      return [{'vars': tuple(self.objectives[:2]), 'dim': 2, 'connect': self._resolve_connect_mode(2)}]
    if len(self.objectives) == 3:
      return [{'vars': tuple(self.objectives[:3]), 'dim': 3, 'connect': self._resolve_connect_mode(3)}]
    pairs = list(combinations(self.objectives, 2))
    return [{'vars': pair, 'dim': 2, 'connect': self._resolve_connect_mode(2)} for pair in pairs]

  def _maybe_draw_rank_connection_static(self, ax, view, front, color):
    mode = view.get('connect', 'lines')
    if mode == 'lines':
      self._draw_rank_line(ax, view, front, color)
    elif mode == 'surface' and view['dim'] == 3:
      self._build_surface(ax, front, view['vars'], color, alpha=0.35)

  @staticmethod
  def _draw_rank_line(ax, view, front, color):
    if len(front) <= 1:
      return
    ordered = front.sort_values(by=view['vars'][0])
    if view['dim'] == 3:
      xVar, yVar, zVar = view['vars']
      ax.plot(ordered[xVar], ordered[yVar], ordered[zVar], color=color, linewidth=0.8, alpha=0.6)
    else:
      xVar, yVar = view['vars']
      ax.plot(ordered[xVar], ordered[yVar], color=color, linewidth=0.8, alpha=0.6)

  def _init_rank_connection_artist(self, ax, view, color):
    mode = view.get('connect', 'lines')
    entry = {'mode': mode, 'artist': None, 'color': color}
    if mode == 'lines':
      if view['dim'] == 3:
        entry['artist'], = ax.plot([], [], [], color=color, linewidth=0.8, alpha=0.6)
      else:
        entry['artist'], = ax.plot([], [], color=color, linewidth=0.8, alpha=0.6)
    return entry

  def _reset_rank_connection_artist(self, entry, view):
    artist = entry.get('artist')
    if entry['mode'] == 'lines' and artist is not None:
      artist.set_data([], [])
      if view['dim'] == 3:
        artist.set_3d_properties([])
    elif entry['mode'] == 'surface' and artist is not None:
      artist.remove()
      entry['artist'] = None

  def _update_rank_connection_artist(self, entry, ax, view, front):
    mode = entry['mode']
    color = entry.get('color', 'k')
    if mode == 'lines':
      artist = entry.get('artist')
      if artist is None:
        return None
      if len(front) > 1:
        ordered = front.sort_values(by=view['vars'][0])
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
      surface = self._build_surface(ax, front, view['vars'], color, alpha=0.35)
      entry['artist'] = surface
      return surface
    return None

  def _build_surface(self, ax, data, variables, color, alpha=0.35):
    if data is None or len(data) < 3:
      return None
    xVals = data[variables[0]].to_numpy()
    yVals = data[variables[1]].to_numpy()
    zVals = data[variables[2]].to_numpy()
    try:
      surface = ax.plot_trisurf(xVals, yVals, zVals, color=color, alpha=alpha, linewidth=0.2, edgecolor='none')
    except (ValueError, RuntimeError) as err:
      if not self._surface_failure_warned:
        self.raiseAWarning(f'Unable to build trisurface for view {variables}: {err}. Falling back to scatter only.')
        self._surface_failure_warned = True
      return None
    return surface

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

  def _make_axes(self, panels):
    panels = max(1, panels)
    if panels == 1:
      fig, ax = plt.subplots(figsize=(6, 6))
      return fig, [ax]
    if panels == 4:
      ncols = 2
    else:
      ncols = min(3, panels)
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
      self.raiseAnError(IOError, f'Unsupported format(s) "{bad}" for NSGAFrontRankAnimation "{self.name}". Use "gif", "html", or "both".')
    return tokens

  def _parse_connect_mode(self, raw):
    if raw is None:
      return 'lines'
    text = raw.strip().lower()
    allowed = {'lines', 'none', 'surface'}
    if text not in allowed:
      options = ', '.join(sorted(allowed))
      self.raiseAnError(IOError, f'Unsupported connect mode "{raw}" for NSGAFrontRankAnimation "{self.name}". Choose from: {options}.')
    return text

  def _resolve_connect_mode(self, dim):
    if self.connect_mode == 'surface' and dim != 3:
      if not self._connect_warning_emitted:
        self.raiseAWarning('connect="surface" is only available for three-objective views; disabling connections for other projections.')
        self._connect_warning_emitted = True
      return 'none'
    return self.connect_mode

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
    seed = f'{self.__class__.__name__}:{safe_name}'
    deterministic = hashlib.md5(seed.encode('utf-8')).hexdigest()
    return html_str.replace(random_suffix, deterministic)
