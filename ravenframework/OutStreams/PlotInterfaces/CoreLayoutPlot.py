#!/usr/bin/env python
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
Plot the SIMULATE core layout colored by locXX values from a PRLO optimizer export.

Intended to mirror the milestone report core plots (A.. columns, 1.. rows) and
supports symmetry views (full, quarter, eighth).
"""

import re
import io

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
from matplotlib.patches import Patch

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class CoreLayoutPlot(PlotInterface):
  """
  Render a core map using locXX variables from a PRLO optimizer export and a SIMULATE template.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject containing locXX variables (e.g., opt_export)."""))
    spec.addSub(InputData.parameterInputFactory('template', contentType=InputTypes.StringType,
        descr=r"""Path to the SIMULATE template (EQinput.inp/firstInput.inp) containing locXX placeholders."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation/index column used to pick a row."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""Optional explicit generation value; otherwise the last available is used."""))
    spec.addSub(InputData.parameterInputFactory('accept_column', contentType=InputTypes.StringType,
        descr=r"""Optional column used to filter accepted rows (default 'accepted')."""))
    spec.addSub(InputData.parameterInputFactory('accept_value', contentType=InputTypes.StringType,
        descr=r"""Value in <accept_column> to keep (default 'final')."""))
    spec.addSub(InputData.parameterInputFactory('symmetry', contentType=InputTypes.StringType,
        descr=r"""One of: full (default), quarter, eighth."""))
    spec.addSub(InputData.parameterInputFactory('quadrant', contentType=InputTypes.StringType,
        descr=r"""For quarter symmetry, one of NE, NW, SE, SW (default NE)."""))
    spec.addSub(InputData.parameterInputFactory('octant', contentType=InputTypes.StringType,
        descr=r"""For eighth symmetry, one of NE, NW, SE, SW (default NE)."""))
    spec.addSub(InputData.parameterInputFactory('x_axis_position', contentType=InputTypes.StringType,
        descr=r"""Optional x-axis placement: 'bottom' (default) or 'top'."""))
    spec.addSub(InputData.parameterInputFactory('y_axis_position', contentType=InputTypes.StringType,
        descr=r"""Optional y-axis placement: 'left' (default) or 'right'."""))
    spec.addSub(InputData.parameterInputFactory('categorical_legend', contentType=InputTypes.BoolType,
        descr=r"""When plotting burn tiers, use a discrete legend instead of a gradient colorbar (default False)."""))
    spec.addSub(InputData.parameterInputFactory('surface3d', contentType=InputTypes.BoolType,
        descr=r"""Render a 3D surface view of the grid (matplotlib)."""))
    spec.addSub(InputData.parameterInputFactory('bar3d', contentType=InputTypes.BoolType,
        descr=r"""Render a 3D bar view of the grid (matplotlib)."""))
    spec.addSub(InputData.parameterInputFactory('interactive_surface', contentType=InputTypes.BoolType,
        descr=r"""When surface3d is true, emit an interactive HTML Plotly surface (default False)."""))
    spec.addSub(InputData.parameterInputFactory('interactive_bar', contentType=InputTypes.BoolType,
        descr=r"""When bar3d is true, emit an interactive HTML Plotly bar chart (default False)."""))
    spec.addSub(InputData.parameterInputFactory('limit_plane', contentType=InputTypes.FloatType,
        descr=r"""Optional horizontal limit/constraint line (drawn as a translucent plane in 3-D bars)."""))
    spec.addSub(InputData.parameterInputFactory('animate_bar', contentType=InputTypes.BoolType,
        descr=r"""Animate bar3d over available stage columns (e.g., exp_fresh_, exp_cycle_1_stepN_, exp_cycle_1_)."""))
    spec.addSub(InputData.parameterInputFactory('animate_bar_interp', contentType=InputTypes.IntegerType,
        descr=r"""Optional number of interpolated frames inserted between stages when animating bars (default 0)."""))
    spec.addSub(InputData.parameterInputFactory('animate_bar_fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for animated bars (default 2.0)."""))
    spec.addSub(InputData.parameterInputFactory('animate_bar_format', contentType=InputTypes.StringType,
        descr=r"""Which bar animation outputs to emit: html, gif, or both (default both)."""))
    spec.addSub(InputData.parameterInputFactory('label_mode', contentType=InputTypes.StringType,
        descr=r"""How to label cells: 'value' (default), 'loc', or 'both'."""))
    spec.addSub(InputData.parameterInputFactory('assembly_table', contentType=InputTypes.StringType,
        descr=r"""Optional CSV path with assembly metadata (columns like Assembly Type, FA ID, Fresh Fuel Label, Enrichment (wt%), IFBA Rods, Pyrex Rods)."""))
    spec.addSub(InputData.parameterInputFactory('color_prefix', contentType=InputTypes.StringType,
        descr=r"""Optional prefix for columns that should drive the color scale (e.g., burnup_locXX)."""))
    spec.addSub(InputData.parameterInputFactory('colorbar_label', contentType=InputTypes.StringType,
        descr=r"""Override the colorbar label (defaults to 'Fuel ID / value' or 'Burnup / color metric' when color_prefix is used)."""))
    spec.addSub(InputData.parameterInputFactory('cmap', contentType=InputTypes.StringType,
        descr=r"""Matplotlib colormap name (e.g., viridis, coolwarm). Defaults to viridis."""))
    spec.addSub(InputData.parameterInputFactory('color_metric', contentType=InputTypes.StringType,
        descr=r"""Optional shorthand metric selector; currently supports 'power' (uses RPF maps) when <color_prefix> is not provided."""))
    spec.addSub(InputData.parameterInputFactory('power_stage', contentType=InputTypes.StringType,
        descr=r"""When color_metric is 'power', which stage to plot: fresh (default) or cycle_1."""))
    spec.addSub(InputData.parameterInputFactory('surface3d', contentType=InputTypes.BoolType,
        descr=r"""If true, also emit a 3-D surface view of the grid alongside the 2-D map."""))
    spec.addSub(InputData.parameterInputFactory('bar3d', contentType=InputTypes.BoolType,
        descr=r"""If true, also emit a 3-D bar view (x,y at assembly location, height = value)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'CoreLayoutPlot'
    self.sourceName = None
    self.templatePath = None
    self.index = None
    self.generation = None
    self.acceptColumn = 'accepted'
    self.acceptValue = 'final'
    self.symmetry = 'full'
    self.quadrant = 'NE'
    self.octant = 'NE'
    self.xAxisPosition = 'bottom'
    self.yAxisPosition = 'left'
    self.surface3d = False
    self.bar3d = False
    self.interactiveSurface = False
    self.interactiveBar = False
    self.limitPlane = None
    self.animateBar = False
    self.animateBarInterp = 0
    self.animateBarFps = 2.0
    self.animateBarFormat = 'both'
    self.categoricalLegend = False
    self.labelMode = 'value'
    self.assemblyTable = None
    self.colorPrefix = None
    self.colorbarLabel = None
    self.cmapName = None
    self.colorMetric = None
    self.powerStage = 'fresh'
    self.surface3d = False
    self.bar3d = False
    self.source = None
    self.locToCoords = {}
    self.locLabels = {}
    self.size = 0

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'CoreLayoutPlot "{self.name}" missing <source>.')
    self.sourceName = src.value

    tmpl = spec.findFirst('template')
    if tmpl is None or not tmpl.value:
      self.raiseAnError(IOError, f'CoreLayoutPlot "{self.name}" missing <template>.')
    self.templatePath = tmpl.value

    idx = spec.findFirst('index')
    if idx is not None and idx.value:
      self.index = idx.value
    gen = spec.findFirst('generation')
    if gen is not None and gen.value is not None:
      self.generation = float(gen.value)

    accCol = spec.findFirst('accept_column')
    if accCol is not None and accCol.value:
      self.acceptColumn = accCol.value
    accVal = spec.findFirst('accept_value')
    if accVal is not None and accVal.value is not None:
      self.acceptValue = accVal.value

    sym = spec.findFirst('symmetry')
    if sym is not None and sym.value:
      self.symmetry = sym.value.lower()
    quad = spec.findFirst('quadrant')
    if quad is not None and quad.value:
      self.quadrant = quad.value.upper()
    octant = spec.findFirst('octant')
    if octant is not None and octant.value:
      cand = octant.value.upper()
      if cand not in {'NE', 'NW', 'SE', 'SW'}:
        self.raiseAnError(IOError, f'Invalid <octant> "{octant.value}" for CoreLayoutPlot "{self.name}".')
      self.octant = cand
    lbl = spec.findFirst('label_mode')
    if lbl is not None and lbl.value:
      mode = lbl.value.lower()
      if mode not in {'value', 'loc', 'both'}:
        self.raiseAnError(IOError, f'Invalid <label_mode> "{lbl.value}" for CoreLayoutPlot "{self.name}".')
      self.labelMode = mode
    xaxis = spec.findFirst('x_axis_position')
    if xaxis is not None and xaxis.value:
      cand = xaxis.value.lower()
      if cand not in {'top', 'bottom'}:
        self.raiseAnError(IOError, f'Invalid <x_axis_position> "{xaxis.value}" for CoreLayoutPlot "{self.name}".')
      self.xAxisPosition = cand
    yaxis = spec.findFirst('y_axis_position')
    if yaxis is not None and yaxis.value:
      cand = yaxis.value.lower()
      if cand not in {'left', 'right'}:
        self.raiseAnError(IOError, f'Invalid <y_axis_position> "{yaxis.value}" for CoreLayoutPlot "{self.name}".')
      self.yAxisPosition = cand
    surf3d = spec.findFirst('surface3d')
    if surf3d is not None and surf3d.value is not None:
      self.surface3d = bool(surf3d.value)
    bar3d = spec.findFirst('bar3d')
    if bar3d is not None and bar3d.value is not None:
      self.bar3d = bool(bar3d.value)
    inter_surf = spec.findFirst('interactive_surface')
    if inter_surf is not None and inter_surf.value is not None:
      self.interactiveSurface = bool(inter_surf.value)
    inter_bar = spec.findFirst('interactive_bar')
    if inter_bar is not None and inter_bar.value is not None:
      self.interactiveBar = bool(inter_bar.value)
    limit_plane = spec.findFirst('limit_plane')
    if limit_plane is not None and limit_plane.value is not None:
      self.limitPlane = float(limit_plane.value)
    anim_bar = spec.findFirst('animate_bar')
    if anim_bar is not None and anim_bar.value is not None:
      self.animateBar = bool(anim_bar.value)
    anim_interp = spec.findFirst('animate_bar_interp')
    if anim_interp is not None and anim_interp.value is not None:
      self.animateBarInterp = int(anim_interp.value)
    anim_fps = spec.findFirst('animate_bar_fps')
    if anim_fps is not None and anim_fps.value is not None:
      self.animateBarFps = float(anim_fps.value)
    anim_fmt = spec.findFirst('animate_bar_format')
    if anim_fmt is not None and anim_fmt.value:
      self.animateBarFormat = anim_fmt.value.lower()
    cat_legend = spec.findFirst('categorical_legend')
    if cat_legend is not None and cat_legend.value is not None:
      self.categoricalLegend = bool(cat_legend.value)
    asm = spec.findFirst('assembly_table')
    if asm is not None and asm.value:
      self.assemblyTable = asm.value
    colorPrefix = spec.findFirst('color_prefix')
    if colorPrefix is not None and colorPrefix.value:
      self.colorPrefix = colorPrefix.value
    metricNode = spec.findFirst('color_metric')
    if metricNode is not None and metricNode.value:
      self.colorMetric = metricNode.value.lower()
    powerStageNode = spec.findFirst('power_stage')
    if powerStageNode is not None and powerStageNode.value:
      stage_cand = powerStageNode.value.lower()
      if stage_cand not in {'fresh', 'cycle_1'}:
        self.raiseAnError(IOError, f'Invalid <power_stage> "{powerStageNode.value}" for CoreLayoutPlot "{self.name}".')
      self.powerStage = stage_cand
    colorbarLabel = spec.findFirst('colorbar_label')
    if colorbarLabel is not None and colorbarLabel.value:
      self.colorbarLabel = colorbarLabel.value
    cmapNode = spec.findFirst('cmap')
    if cmapNode is not None and cmapNode.value:
      self.cmapName = cmapNode.value
    surfNode = spec.findFirst('surface3d')
    if surfNode is not None and surfNode.value is not None:
      self.surface3d = bool(surfNode.value)
    barNode = spec.findFirst('bar3d')
    if barNode is not None and barNode.value is not None:
      self.bar3d = bool(barNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'Source "{self.sourceName}" not found for CoreLayoutPlot "{self.name}".')
    self.source = src
    # parse template once
    self._parseTemplate(self.templatePath)
    if not self.locToCoords:
      self.raiseAnError(IOError, f'No locXX placeholders parsed from template "{self.templatePath}".')

  def _parseTemplate(self, path):
    self.locToCoords = {}
    self.locLabels = {}
    self.size = 0
    try:
      with open(path, 'r') as fh:
        lines = fh.readlines()
    except IOError as err:
      self.raiseAnError(IOError, f'Failed to read template "{path}": {err}')

    for line in lines:
      if self.size == 0:
        m = re.search(r"'DIM\.PWR'\s+(\d+)", line, re.IGNORECASE)
        if m:
          self.size = int(m.group(1))
      mtyp = re.match(r"\s*'FUE\.TYP'\s*[, ]\s*(\d+)\s*,?(.*)$", line, re.IGNORECASE)
      if mtyp:
        row = int(mtyp.group(1)) - 1
        tokens = re.split(r"[,\s]+", mtyp.group(2).strip().rstrip('/'))
        col = 0
        for tok in tokens:
          if not tok:
            continue
          ml = re.search(r"loc[^0-9]*([0-9]+)", tok, re.IGNORECASE)
          if ml:
            norm_loc = self._normalizeLoc(f"loc{ml.group(1)}")
            self.locToCoords.setdefault(norm_loc, []).append((row, col))
            self.locLabels.setdefault(norm_loc, f"loc{ml.group(1)}")
          col += 1
        self.size = max(self.size, row + 1, col, self.size)
        continue
      mfull = re.search(r"^\s*(\d+)\s+(\d+).*loc[^0-9]*([0-9]+)", line, re.IGNORECASE)
      if mfull:
        r = int(mfull.group(1)) - 1
        c = int(mfull.group(2)) - 1
        raw_loc = f'loc{mfull.group(3)}'.lower()
        norm_loc = self._normalizeLoc(raw_loc)
        self.locToCoords.setdefault(norm_loc, []).append((r, c))
        self.locLabels.setdefault(norm_loc, raw_loc)
    if self.size == 0 and self.locToCoords:
      self.size = max(max(r, c) for coords in self.locToCoords.values() for r, c in coords) + 1
    if self.size == 0:
      self.size = 15  # fall back

  def _normalizeLoc(self, name):
    m = re.search(r'loc0*(\d+)', name.lower())
    return f'loc{int(m.group(1))}' if m else name.lower()

  def _extractLocValues(self, row, prefix=''):
    values = {}
    prefix_lower = prefix.lower() if prefix else ''
    for col in row.index:
      col_lower = col.lower()
      if prefix_lower:
        if not col_lower.startswith(prefix_lower):
          continue
        base = col_lower[len(prefix_lower):]
      else:
        base = col_lower
      if not base.startswith('loc'):
        continue
      norm = self._normalizeLoc(base)
      try:
        values[norm] = float(row[col])
      except Exception:
        values[norm] = float(hash(row[col]) % 500)
    return values

  def _filterRow(self, df):
    subset = df.copy()
    if self.acceptColumn in subset.columns and self.acceptValue is not None:
      subset = subset[subset[self.acceptColumn] == self.acceptValue]
    if subset.empty:
      subset = df.copy()
    if self.index and self.index in subset.columns:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        subset = subset[np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if subset.empty:
      return None
    return subset.iloc[0]

  def _applySymmetry(self, grid):
    if self.symmetry == 'full':
      return grid, self.locToCoords, self.size, 0, 0
    half = self.size // 2 + self.size % 2
    if self.symmetry == 'quarter':
      if self.quadrant == 'NE':
        rs = slice(0, half)
        cs = slice(self.size - half, self.size)
      elif self.quadrant == 'NW':
        rs = slice(0, half)
        cs = slice(0, half)
      elif self.quadrant == 'SE':
        rs = slice(self.size - half, self.size)
        cs = slice(self.size - half, self.size)
      else:
        rs = slice(self.size - half, self.size)
        cs = slice(0, half)
      sub = grid[rs, cs]
      locs = {}
      for loc, coords in self.locToCoords.items():
        filtered = []
        for r, c in coords:
          if rs.start <= r < rs.stop and cs.start <= c < cs.stop:
            filtered.append((r - rs.start, c - cs.start))
        if filtered:
          locs[loc] = filtered
      return sub, locs, sub.shape[0], rs.start, cs.start
    if self.symmetry == 'eighth':
      north = self.octant in {'NE', 'NW'}
      east = self.octant in {'NE', 'SE'}
      rs = slice(0, half) if north else slice(self.size - half, self.size)
      cs = slice(self.size - half, self.size) if east else slice(0, half)
      sub = grid[rs, cs].copy()
      n = sub.shape[0]
      if n != sub.shape[1]:
        n = min(sub.shape[0], sub.shape[1])
        sub = sub[:n, :n]
      keep_fn = (lambda rr, cc: cc >= n - rr - 1) if east else (lambda rr, cc: cc <= rr)
      for r in range(n):
        for c in range(n):
          if not keep_fn(r, c):
            sub[r, c] = np.nan
      locs = {}
      for loc, coords in self.locToCoords.items():
        filtered = []
        for r, c in coords:
          if rs.start <= r < rs.stop and cs.start <= c < cs.stop:
            rr = r - rs.start
            cc = c - cs.start
            if rr >= n or cc >= n or not keep_fn(rr, cc):
              continue
            filtered.append((rr, cc))
        if filtered:
          locs[loc] = filtered
      return sub, locs, n, rs.start, cs.start
    return grid, self.locToCoords, self.size, 0, 0

  def _buildGrid(self, values):
    grid = np.full((self.size, self.size), np.nan)
    for loc, coords in self.locToCoords.items():
      if loc not in values:
        continue
      for r, c in coords:
        grid[r, c] = values[loc]
    return grid

  def _basePrefix(self):
    """
      Derive a base prefix for stage-aware columns (strip stage tags like fresh_, cycle_1_, cycle_1_stepN_).
    """
    if not self.colorPrefix:
      return None
    pref = self.colorPrefix.lower()
    stage_tags = ('fresh_', 'cycle_1_', 'cycle_1_step')
    for tag in stage_tags:
      idx = pref.find(tag)
      if idx > -1:
        return pref[:idx]
    return pref

  def _collectStageGrids(self, row, base_prefix):
    """
      Collect stage-labeled grids from the data row.
      @ In, row, pandas.Series
      @ In, base_prefix, str or None
      @ Out, list(tuple(stage, grid))
    """
    if not base_prefix:
      return []
    stage_vals = {}
    pref_len = len(base_prefix)
    for col in row.index:
      col_lower = col.lower()
      if not col_lower.startswith(base_prefix):
        continue
      suffix = col_lower[pref_len:]
      m = re.match(r'(fresh|cycle_1|cycle_1_step\d+)_?(loc\d+)$', suffix)
      if not m:
        continue
      stage, loc = m.groups()
      norm = self._normalizeLoc(loc)
      try:
        val = float(row[col])
      except Exception:
        val = np.nan
      stage_vals.setdefault(stage, {})[norm] = val
    if not stage_vals:
      return []
    def _stage_key(name):
      if name == 'fresh':
        return (0, 0)
      m = re.match(r'cycle_1_step(\d+)', name)
      if m:
        return (1, int(m.group(1)))
      if name == 'cycle_1':
        return (2, 0)
      return (3, name)
    ordered = []
    for stage in sorted(stage_vals, key=_stage_key):
      grid = self._buildGrid(stage_vals[stage])
      ordered.append((stage, grid))
    return ordered

  def _emitPlotlySurface(self, grid_view, row_start, col_start, use_burn_tier):
    try:
      import plotly.graph_objects as go
      import plotly.io as pio
    except Exception:
      self.raiseAnError(IOError,
                        f'CoreLayoutPlot "{self.name}" requested interactive_surface '
                        'but Plotly is not installed. Install plotly to emit interactive surfaces '
                        '(e.g., pip install plotly).')
    try:
      z_float = np.asarray(grid_view, dtype=float)
    except Exception as err:
      self.raiseAnError(IOError,
                        f'CoreLayoutPlot "{self.name}" interactive_surface expects numeric scalars; '
                        f'failed to cast grid values to float: {err}')
    if not np.isfinite(z_float).any():
      return False
    z = z_float.astype(object)
    z[np.isnan(z_float)] = None
    x_labels = [chr(ord('A') + col_start + i) for i in range(z.shape[1])]
    y_labels = [row_start + i + 1 for i in range(z.shape[0])]
    surface_kwargs = {}
    if use_burn_tier:
      tier_colors = ['#2ca02c', '#ffbf00', '#d62728']
      surface_kwargs.update(dict(colorscale=[(0, tier_colors[0]), (0.5/2.0, tier_colors[0]),
                                             (0.5/2.0, tier_colors[1]), (1.5/2.0, tier_colors[1]),
                                             (1.5/2.0, tier_colors[2]), (1.0, tier_colors[2])],
                                 zmin=-0.5, zmax=2.5, showscale=not self.categoricalLegend,
                                 colorbar=dict(title=dict(text=self.colorbarLabel or 'Burn tier',
                                                          font=dict(color='#e8f0ff')),
                                               tickvals=[0, 1, 2],
                                               ticktext=['0 fresh', '1 once', '2 twice+'],
                                               tickfont=dict(color='#e8f0ff'))))
    else:
      cmap = plt.cm.get_cmap(self.cmapName or 'viridis')
      finite_vals = z_float[np.isfinite(z_float)]
      vmin = float(np.min(finite_vals))
      vmax = float(np.max(finite_vals))
      colors = [matplotlib.colors.rgb2hex(cmap(i / 256.0)) for i in range(cmap.N)]
      surface_kwargs.update(dict(colorscale=[[i / (len(colors) - 1), col] for i, col in enumerate(colors)],
                                 cmin=vmin, cmax=vmax,
                                 showscale=True,
                                 colorbar=dict(title=dict(text=self.colorbarLabel or 'Value',
                                                          font=dict(color='#e8f0ff')),
                                               tickfont=dict(color='#e8f0ff'))))
    fig = go.Figure(data=[go.Surface(z=z, x=x_labels, y=y_labels, hovertemplate='Row %{y}<br>Col %{x}<br>Val %{z}<extra></extra>',
                                     **surface_kwargs)])
    fig.update_layout(scene=dict(
      xaxis_title='Column',
      yaxis_title='Row',
      zaxis_title=self.colorbarLabel or 'Value',
      xaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff'),
      yaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff'),
      zaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff'),
      bgcolor='#0a0f1c'
    ),
      font=dict(color='#e8f0ff'),
      paper_bgcolor='#0a0f1c',
      plot_bgcolor='#0a0f1c',
      title=f'Core layout: {self.name}')
    filename = self._createFilename(defaultName=f'{self.name}.html')
    pio.write_html(fig, file=filename, include_plotlyjs='cdn', auto_open=False)
    return True

  def _emitPlotlyBars(self, grid_view, row_start, col_start, use_burn_tier):
    try:
      import plotly.graph_objects as go
      import plotly.io as pio
    except Exception:
      self.raiseAnError(IOError,
                        f'CoreLayoutPlot "{self.name}" requested interactive_bar '
                        'but Plotly is not installed. Install plotly to emit interactive bars '
                        '(e.g., pip install plotly).')
    try:
      z_float = np.asarray(grid_view, dtype=float)
    except Exception as err:
      self.raiseAnError(IOError,
                        f'CoreLayoutPlot "{self.name}" interactive_bar expects numeric scalars; '
                        f'failed to cast grid values to float: {err}')
    mask = np.isfinite(z_float)
    if not mask.any():
      return False
    coords = np.argwhere(mask)
    heights = z_float[mask]
    # use numeric axes for stability, then relabel ticks
    x_ticktext = [chr(ord('A') + col_start + i) for i in range(z_float.shape[1])]
    y_ticktext = [row_start + i + 1 for i in range(z_float.shape[0])]
    xs_num = [c for r, c in coords]
    ys_num = [r for r, c in coords]
    # build per-bar colors and optional colorbar proxy
    bar_traces = []
    colorbar_trace = None
    if use_burn_tier:
      tier_colors = ['#2ca02c', '#ffbf00', '#d62728']
      colorscale = [(0, tier_colors[0]), (0.5/2.0, tier_colors[0]),
                    (0.5/2.0, tier_colors[1]), (1.5/2.0, tier_colors[1]),
                    (1.5/2.0, tier_colors[2]), (1.0, tier_colors[2])]
      cmin, cmax = -0.5, 2.5
      colorbar_kwargs = dict(title=dict(text=self.colorbarLabel or 'Burn tier', font=dict(color='#e8f0ff')),
                             tickvals=[0, 1, 2],
                             ticktext=['0 fresh', '1 once', '2 twice+'],
                             tickfont=dict(color='#e8f0ff'))
      if not self.categoricalLegend:
        colorbar_trace = dict(colorscale=colorscale, cmin=cmin, cmax=cmax, colorbar=colorbar_kwargs)
      def color_for(val):
        if np.isnan(val):
          return '#444444'
        if val < 0.5:
          return tier_colors[0]
        if val < 1.5:
          return tier_colors[1]
        return tier_colors[2]
      colors = [color_for(v) for v in heights]
    else:
      cmap = plt.cm.get_cmap(self.cmapName or 'viridis')
      finite_vals = heights[np.isfinite(heights)]
      vmin = float(np.min(finite_vals))
      vmax = float(np.max(finite_vals))
      colors = [matplotlib.colors.rgb2hex(cmap(i / 256.0)) for i in range(cmap.N)]
      colorscale = [[i / (len(colors) - 1), col] for i, col in enumerate(colors)]
      colorbar_kwargs = dict(title=dict(text=self.colorbarLabel or 'Value', font=dict(color='#e8f0ff')),
                             tickfont=dict(color='#e8f0ff'))
      colorbar_trace = dict(colorscale=colorscale, cmin=vmin, cmax=vmax, colorbar=colorbar_kwargs)
      # map each height to hex using same cmap
      normed = (heights - vmin) / (vmax - vmin + 1e-12)
      colors = [matplotlib.colors.rgb2hex(cmap(np.clip(v, 0.0, 1.0))) for v in normed]
    # build bar meshes (extruded squares)
    dx = dy = 0.8
    half_dx = dx / 2.0
    half_dy = dy / 2.0
    for x, y, h, col in zip(xs_num, ys_num, heights, colors):
      if np.isnan(h):
        continue
      verts = [
        (x - half_dx, y - half_dy, 0.0),  # 0
        (x + half_dx, y - half_dy, 0.0),  # 1
        (x + half_dx, y + half_dy, 0.0),  # 2
        (x - half_dx, y + half_dy, 0.0),  # 3
        (x - half_dx, y - half_dy, h),    # 4
        (x + half_dx, y - half_dy, h),    # 5
        (x + half_dx, y + half_dy, h),    # 6
        (x - half_dx, y + half_dy, h),    # 7
      ]
      i = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # triangles
      j = [1, 2, 2, 5, 3, 6, 0, 4, 5, 7, 6, 4]
      k = [2, 3, 3, 6, 7, 7, 4, 7, 6, 6, 7, 5]
      vx, vy, vz = zip(*verts)
      bar_traces.append(go.Mesh3d(
        x=list(vx), y=list(vy), z=list(vz),
        i=i, j=j, k=k,
        color=col,
        opacity=0.92,
        flatshading=True,
        hovertemplate=f'Row {y_ticktext[y]}<br>Col {x_ticktext[x]}<br>Val {h:.4g}<extra></extra>'
      ))
    # optional constraint plane
    if self.limitPlane is not None:
      plane_z = float(self.limitPlane)
      xgrid, ygrid = np.meshgrid(range(len(x_ticktext)), range(len(y_ticktext)))
      bar_traces.append(go.Surface(
        x=xgrid, y=ygrid, z=np.full_like(xgrid, plane_z, dtype=float),
        showscale=False,
        opacity=0.18,
        colorscale=[[0, '#ff8800'], [1, '#ff8800']],
        hoverinfo='skip'
      ))
    # add hidden colorbar proxy if available
    if colorbar_trace:
      cb = colorbar_trace
      bar_traces.append(go.Scatter3d(x=[-1, -1], y=[-1, -1], z=[0, 0],
                                     mode='markers',
                                     marker=dict(size=0.0001,
                                                 color=[cb.get('cmin', 0), cb.get('cmax', 1)],
                                                 colorscale=cb['colorscale'],
                                                 cmin=cb.get('cmin', None),
                                                 cmax=cb.get('cmax', None),
                                                 showscale=True,
                                                 colorbar=cb.get('colorbar', {})),
                                     hoverinfo='skip'))
    fig = go.Figure(data=bar_traces)
    fig.update_layout(scene=dict(
      xaxis_title='Column',
      yaxis_title='Row',
      zaxis_title=self.colorbarLabel or 'Value',
      xaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff',
                 tickmode='array', tickvals=list(range(len(x_ticktext))), ticktext=x_ticktext),
      yaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff',
                 tickmode='array', tickvals=list(range(len(y_ticktext))), ticktext=y_ticktext),
      zaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff'),
      bgcolor='#0a0f1c'
    ),
      font=dict(color='#e8f0ff'),
      paper_bgcolor='#0a0f1c',
      plot_bgcolor='#0a0f1c',
      title=f'Core layout: {self.name} (bars)')
    filename = self._createFilename(defaultName=f'{self.name}_bars.html')
    pio.write_html(fig, file=filename, include_plotlyjs='cdn', auto_open=False)
    return True

  def _barTracesFromGrid(self, grid_view, x_ticktext, y_ticktext, use_burn_tier, vmin=None, vmax=None):
    import plotly.graph_objects as go
    mask = np.isfinite(grid_view)
    if not mask.any():
      return [], None
    coords = np.argwhere(mask)
    heights = grid_view[mask]
    xs_num = [c for r, c in coords]
    ys_num = [r for r, c in coords]
    bar_traces = []
    colorbar_trace = None
    if use_burn_tier:
      tier_colors = ['#2ca02c', '#ffbf00', '#d62728']
      colorscale = [(0, tier_colors[0]), (0.5/2.0, tier_colors[0]),
                    (0.5/2.0, tier_colors[1]), (1.5/2.0, tier_colors[1]),
                    (1.5/2.0, tier_colors[2]), (1.0, tier_colors[2])]
      cmin, cmax = -0.5, 2.5
      colorbar_kwargs = dict(title=dict(text=self.colorbarLabel or 'Burn tier', font=dict(color='#e8f0ff')),
                             tickvals=[0, 1, 2],
                             ticktext=['0 fresh', '1 once', '2 twice+'],
                             tickfont=dict(color='#e8f0ff'))
      colorbar_trace = dict(colorscale=colorscale, cmin=cmin, cmax=cmax, colorbar=colorbar_kwargs)
      def color_for(val):
        if np.isnan(val):
          return '#444444'
        if val < 0.5:
          return tier_colors[0]
        if val < 1.5:
          return tier_colors[1]
        return tier_colors[2]
      colors = [color_for(v) for v in heights]
    else:
      cmap = plt.cm.get_cmap(self.cmapName or 'viridis')
      if vmin is None or vmax is None:
        finite_vals = heights[np.isfinite(heights)]
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
      sampled = getattr(cmap, 'colors', None)
      if sampled is None:
        sampled = cmap(np.linspace(0.0, 1.0, 256))
      colorscale = [[i / (len(sampled) - 1), matplotlib.colors.to_hex(c)] for i, c in enumerate(sampled)]
      colorbar_kwargs = dict(title=dict(text=self.colorbarLabel or 'Value', font=dict(color='#e8f0ff')),
                             tickfont=dict(color='#e8f0ff'))
      colorbar_trace = dict(colorscale=colorscale, cmin=vmin, cmax=vmax, colorbar=colorbar_kwargs)
      normed = (heights - vmin) / (vmax - vmin + 1e-12)
      colors = [matplotlib.colors.rgb2hex(cmap(np.clip(v, 0.0, 1.0))) for v in normed]
    dx = dy = 0.8
    half_dx = dx / 2.0
    half_dy = dy / 2.0
    for x, y, h, col in zip(xs_num, ys_num, heights, colors):
      if np.isnan(h):
        continue
      verts = [
        (x - half_dx, y - half_dy, 0.0),
        (x + half_dx, y - half_dy, 0.0),
        (x + half_dx, y + half_dy, 0.0),
        (x - half_dx, y + half_dy, 0.0),
        (x - half_dx, y - half_dy, h),
        (x + half_dx, y - half_dy, h),
        (x + half_dx, y + half_dy, h),
        (x - half_dx, y + half_dy, h),
      ]
      i = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
      j = [1, 2, 2, 5, 3, 6, 0, 4, 5, 7, 6, 4]
      k = [2, 3, 3, 6, 7, 7, 4, 7, 6, 6, 7, 5]
      vx, vy, vz = zip(*verts)
      bar_traces.append(go.Mesh3d(
        x=list(vx), y=list(vy), z=list(vz),
        i=i, j=j, k=k,
        color=col,
        opacity=0.92,
        flatshading=True,
        hovertemplate=f'Row {y_ticktext[y]}<br>Col {x_ticktext[x]}<br>Val {h:.4g}<extra></extra>'
      ))
    return bar_traces, colorbar_trace

  def _matplotlibBarFrame(self, grid_view, x_ticktext, y_ticktext, use_burn_tier, vmin=None, vmax=None, limit_plane=None):
    """
      Render a single bar frame with Matplotlib and return PNG bytes (used as GIF fallback).
    """
    mask = np.isfinite(grid_view)
    if not mask.any():
      return None
    xs, ys = np.meshgrid(np.arange(grid_view.shape[1]), np.arange(grid_view.shape[0]))
    xs_flat = xs[mask]
    ys_flat = ys[mask]
    heights = grid_view[mask]

    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0a0f1c')
    ax.set_facecolor('#0a0f1c')

    if use_burn_tier:
      tier_colors = ['#2ca02c', '#ffbf00', '#d62728']
      cmap = matplotlib.colors.ListedColormap(tier_colors)
      norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    else:
      cmap = plt.get_cmap(self.cmapName or 'viridis')
      if vmin is None or vmax is None:
        finite_vals = heights[np.isfinite(heights)]
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
      norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    width = depth = 0.85
    colors = cmap(norm(heights))
    ax.bar3d(xs_flat - 0.425, ys_flat - 0.425, np.zeros_like(heights),
             width, depth, heights, shade=True, color=colors,
             edgecolor='k', linewidth=0.15)

    if limit_plane is not None:
      plane_z = float(limit_plane)
      xgrid, ygrid = np.meshgrid(np.arange(grid_view.shape[1]), np.arange(grid_view.shape[0]))
      ax.plot_surface(xgrid, ygrid, np.full_like(xgrid, plane_z, dtype=float),
                      color='#ff8800', alpha=0.18, linewidth=0, antialiased=False)

    ax.set_xlabel('Column (A..)', color='#e8f0ff')
    ax.set_ylabel('Row (1..)', color='#e8f0ff')
    ax.set_zlabel(self.colorbarLabel or 'Value', color='#e8f0ff')
    ax.tick_params(colors='#e8f0ff')
    ax.set_xticks(range(len(x_ticktext)))
    ax.set_yticks(range(len(y_ticktext)))
    ax.set_xticklabels(x_ticktext, color='#e8f0ff')
    ax.set_yticklabels(y_ticktext, color='#e8f0ff')
    ax.view_init(elev=35, azim=-135)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

  def _emitPlotlyBarsAnimated(self, stage_grids, stage_labels, row_start, col_start, use_burn_tier):
    try:
      import plotly.graph_objects as go
      import plotly.io as pio
      import copy
    except Exception:
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" could not import Plotly for animated bars.')
      return False
    x_ticktext = [chr(ord('A') + col_start + i) for i in range(stage_grids[0].shape[1])]
    y_ticktext = [row_start + i + 1 for i in range(stage_grids[0].shape[0])]
    vmin = vmax = None
    if not use_burn_tier:
      finite_all = np.concatenate([g[np.isfinite(g)] for g in stage_grids if np.isfinite(g).any()]) if stage_grids else np.array([])
      if finite_all.size:
        vmin = float(np.min(finite_all))
        vmax = float(np.max(finite_all))
    # build traces for first frame
    base_traces, colorbar_trace = self._barTracesFromGrid(stage_grids[0], x_ticktext, y_ticktext, use_burn_tier, vmin=vmin, vmax=vmax)
    if self.limitPlane is not None:
      plane_z = float(self.limitPlane)
      xgrid, ygrid = np.meshgrid(range(len(x_ticktext)), range(len(y_ticktext)))
      base_traces.append(go.Surface(
        x=xgrid, y=ygrid, z=np.full_like(xgrid, plane_z, dtype=float),
        showscale=False,
        opacity=0.18,
        colorscale=[[0, '#ff8800'], [1, '#ff8800']],
        hoverinfo='skip'
      ))
    if colorbar_trace:
      cb = colorbar_trace
      base_traces.append(go.Scatter3d(x=[-1, -1], y=[-1, -1], z=[0, 0],
                                      mode='markers',
                                      marker=dict(size=0.0001,
                                                  color=[cb.get('cmin', 0), cb.get('cmax', 1)],
                                                  colorscale=cb['colorscale'],
                                                  cmin=cb.get('cmin', None),
                                                  cmax=cb.get('cmax', None),
                                                  showscale=True,
                                                  colorbar=cb.get('colorbar', {})),
                                      hoverinfo='skip'))
    frames = []
    for idx, grid in enumerate(stage_grids):
      traces, _ = self._barTracesFromGrid(grid, x_ticktext, y_ticktext, use_burn_tier, vmin=vmin, vmax=vmax)
      if self.limitPlane is not None:
        plane_z = float(self.limitPlane)
        xgrid, ygrid = np.meshgrid(range(len(x_ticktext)), range(len(y_ticktext)))
        traces.append(go.Surface(
          x=xgrid, y=ygrid, z=np.full_like(xgrid, plane_z, dtype=float),
          showscale=False,
          opacity=0.18,
          colorscale=[[0, '#ff8800'], [1, '#ff8800']],
          hoverinfo='skip'
        ))
      frames.append(go.Frame(data=traces, name=str(idx)))
    fig = go.Figure(data=base_traces, frames=frames)
    fig.update_layout(
      scene=dict(
        xaxis_title='Column',
        yaxis_title='Row',
        zaxis_title=self.colorbarLabel or 'Value',
        xaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff',
                   tickmode='array', tickvals=list(range(len(x_ticktext))), ticktext=x_ticktext),
        yaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff',
                   tickmode='array', tickvals=list(range(len(y_ticktext))), ticktext=y_ticktext),
        zaxis=dict(backgroundcolor='#0a0f1c', color='#e8f0ff'),
        bgcolor='#0a0f1c'
      ),
      font=dict(color='#e8f0ff'),
      paper_bgcolor='#0a0f1c',
      plot_bgcolor='#0a0f1c',
      title=f'Core layout: {self.name} (bar animation)',
      updatemenus=[dict(type='buttons',
                        buttons=[dict(label='Play',
                                      method='animate',
                                      args=[None, dict(frame=dict(duration=1000.0 / max(self.animateBarFps, 0.1), redraw=True),
                                                       fromcurrent=True, transition=dict(duration=0))]),
                                 dict(label='Pause',
                                      method='animate',
                                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])],
                        showactive=False,
                        x=0.05, y=0.05)]
    )
    fig.update_layout(sliders=[dict(
      steps=[dict(method='animate', args=[[fr.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                  label=lbl) for fr, lbl in zip(frames, stage_labels)],
      x=0.1, xanchor='left', y=0.0, yanchor='top'
    )])
    html_name = self._createFilename(defaultName=f'{self.name}_bars_animation.html')
    pio.write_html(fig, file=html_name, include_plotlyjs='cdn', auto_open=False)
    if self.animateBarFormat in {'both', 'gif'}:
      try:
        import imageio.v2 as imageio
        imgs = []
        for fr in frames:
          tmp_fig = go.Figure(data=fr.data, layout=fig.layout)
          img_bytes = pio.to_image(tmp_fig, format='png')
          imgs.append(imageio.imread(img_bytes))
        gif_name = self._createFilename(defaultName=f'{self.name}_bars_animation.gif')
        imageio.mimsave(gif_name, imgs, duration=1.0 / max(self.animateBarFps, 0.1))
      except Exception as err:
        self.raiseAWarning(f'CoreLayoutPlot \"{self.name}\" could not write GIF animation via Plotly/Kaleido: {err}; falling back to Matplotlib.')
        try:
          import imageio.v2 as imageio
          imgs = []
          for grid in stage_grids:
            frame_bytes = self._matplotlibBarFrame(grid, x_ticktext, y_ticktext, use_burn_tier,
                                                   vmin=vmin, vmax=vmax, limit_plane=self.limitPlane)
            if frame_bytes:
              imgs.append(imageio.imread(frame_bytes))
          if imgs:
            gif_name = self._createFilename(defaultName=f'{self.name}_bars_animation.gif')
            imageio.mimsave(gif_name, imgs, duration=1.0 / max(self.animateBarFps, 0.1))
          else:
            self.raiseAWarning(f'CoreLayoutPlot \"{self.name}\" Matplotlib fallback found no frames for GIF.')
        except Exception as err2:
          self.raiseAWarning(f'CoreLayoutPlot \"{self.name}\" Matplotlib GIF fallback failed: {err2}')
    return True

  def run(self):
    df = self.source.asDataset().to_dataframe()
    row = self._filterRow(df)
    if row is None:
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" found no data to plot.')
      return
    # If user selected a shorthand metric, derive the prefix.
    if self.colorMetric and not self.colorPrefix:
      if self.colorMetric == 'power':
        self.colorPrefix = f"rpf_{self.powerStage}_"
        if self.colorbarLabel is None:
          stage_lbl = 'BOC' if self.powerStage == 'fresh' else 'EOC'
          self.colorbarLabel = f'Assembly power (relative, {stage_lbl})'
      else:
        self.raiseAWarning(f'CoreLayoutPlot "{self.name}" received unsupported color_metric "{self.colorMetric}".')
    base_vals = self._extractLocValues(row)
    color_vals = {}
    if self.colorPrefix:
      # Try the requested prefix first.
      color_vals = self._extractLocValues(row, prefix=self.colorPrefix)
      # For burn_tier, also try reasonable fallbacks so we don't silently plot loc ids.
      if (not color_vals) and self.colorPrefix.lower().startswith('burn_tier'):
        fallback_prefixes = []
        if self.colorPrefix.lower().startswith('burn_tier_fresh'):
          fallback_prefixes = ['burn_tier_', 'burn_tier_fresh_']
        elif self.colorPrefix.lower().startswith('burn_tier_cycle_1'):
          fallback_prefixes = ['burn_tier_cycle_1_', 'burn_tier_']
        else:
          fallback_prefixes = ['burn_tier_']
        for fpref in fallback_prefixes:
          color_vals = self._extractLocValues(row, prefix=fpref)
          if color_vals:
            break
      if not color_vals:
        self.raiseAWarning(f'CoreLayoutPlot "{self.name}" did not find columns with prefix "{self.colorPrefix}"; skipping plot.')
        return
    active_vals = color_vals or base_vals
    if self.colorPrefix and self.colorPrefix.lower().startswith('burn_tier'):
      # normalize burn tier values to discrete bins
      for k, v in list(active_vals.items()):
        if np.isnan(v):
          active_vals[k] = np.nan
        else:
          active_vals[k] = float(round(v))
    if not active_vals:
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" found no locXX-compatible columns to plot.')
      return

    grid = self._buildGrid(active_vals)
    grid_view, locs_view, size_view, row_start, col_start = self._applySymmetry(grid)
    if np.all(np.isnan(grid_view)):
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" grid is empty after mapping values.')
      return

    use_burn_tier = self.colorPrefix and self.colorPrefix.lower().startswith('burn_tier')
    stage_grids = []
    stage_labels = []
    if self.animateBar and self.bar3d and self.interactiveBar:
      base_prefix = self._basePrefix() or (self.colorPrefix.lower() if self.colorPrefix else None)
      collected = self._collectStageGrids(row, base_prefix)
      if collected:
        for lbl, g in collected:
          gv, _, _, _, _ = self._applySymmetry(g)
          stage_grids.append(gv)
          stage_labels.append(lbl)
        if self.animateBarInterp > 0 and len(stage_grids) >= 2:
          interp_grids = []
          interp_labels = []
          for idx in range(len(stage_grids) - 1):
            g0 = stage_grids[idx]
            g1 = stage_grids[idx + 1]
            interp_grids.append(g0)
            interp_labels.append(stage_labels[idx])
            for k in range(1, self.animateBarInterp + 1):
              alpha = k / float(self.animateBarInterp + 1)
              blended = np.where(np.isfinite(g0) & np.isfinite(g1),
                                 g0 * (1 - alpha) + g1 * alpha,
                                 np.where(np.isfinite(g0), g0, g1))
              interp_grids.append(blended)
              interp_labels.append(f'{stage_labels[idx]}→{stage_labels[idx+1]} {k}/{self.animateBarInterp}')
          interp_grids.append(stage_grids[-1])
          interp_labels.append(stage_labels[-1])
          stage_grids, stage_labels = interp_grids, interp_labels
        # ensure first stage aligns with current grid to keep color scales consistent
        if stage_grids and stage_grids[0].shape != grid_view.shape:
          self.raiseAWarning(f'CoreLayoutPlot \"{self.name}\" found mismatched stage grid shape; skipping animation.')
          stage_grids = []
          stage_labels = []
    surface_done = False
    bar_done = False
    if self.surface3d and self.interactiveSurface:
      surface_done = self._emitPlotlySurface(grid_view, row_start, col_start, use_burn_tier)
    if self.bar3d and self.interactiveBar:
      if self.animateBar and stage_grids:
        bar_done = self._emitPlotlyBarsAnimated(stage_grids, stage_labels, row_start, col_start, use_burn_tier)
      else:
        bar_done = self._emitPlotlyBars(grid_view, row_start, col_start, use_burn_tier)
    if ((not self.surface3d or not self.interactiveSurface or surface_done) and
        (not self.bar3d or not self.interactiveBar or bar_done) and
        (surface_done or bar_done)):
      # If we animated bars, still fall through to emit static PNGs.
      if not (self.animateBar and self.bar3d and self.interactiveBar):
        return

    masked = np.ma.masked_invalid(grid_view)
    if self.surface3d or self.bar3d:
      from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
      fig = plt.figure(figsize=(7.2, 5.6))
      ax = fig.add_subplot(111, projection='3d')
    else:
      fig, ax = plt.subplots(figsize=(6.8, 6.8))
    fig.patch.set_facecolor('#0a0f1c')
    if not (self.surface3d or self.bar3d):
      ax.set_facecolor('#0a0f1c')
    img_kwargs = {}
    cbar = None
    if self.surface3d or self.bar3d:
      xs, ys = np.meshgrid(np.arange(size_view), np.arange(size_view))
      vals = masked.filled(np.nan)
      if use_burn_tier:
        tier_colors = ['#2ca02c', '#ffbf00', '#d62728']
        cmap = matplotlib.colors.ListedColormap(tier_colors)
        norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
      else:
        cmap = plt.cm.get_cmap(self.cmapName or 'viridis')
        norm = None
      if self.surface3d:
        ax.plot_surface(xs, ys, vals, cmap=cmap, norm=norm, linewidth=0, antialiased=False, shade=True)
        if self.limitPlane is not None:
          ax.plot_surface(xs, ys, np.full_like(vals, float(self.limitPlane)), color='#ff8800', alpha=0.18, linewidth=0, antialiased=False)
      if self.bar3d:
        dz = vals.flatten()
        dz[np.isnan(dz)] = 0.0
        dx = dy = 0.8
        ax.bar3d(xs.flatten() - 0.4, ys.flatten() - 0.4, np.zeros_like(dz), dx, dy, dz, shade=True, color=cmap(norm(dz)) if norm else cmap((dz - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-12)))
      ax.set_xlabel('Column')
      ax.set_ylabel('Row')
      ax.set_zlabel(self.colorbarLabel or 'Value')
      ax.set_xticks(range(size_view))
      ax.set_yticks(range(size_view))
      ax.set_xticklabels([chr(ord('A') + col_start + i) for i in range(size_view)], color='#e8f0ff')
      ax.set_yticklabels([row_start + i + 1 for i in range(size_view)], color='#e8f0ff')
      ax.tick_params(colors='#e8f0ff')
      ax.set_title(f'Core layout: {self.name}', color='#e8f0ff')
      fig.tight_layout()
      filename = self._createFilename(defaultName=f'{self.name}.png')
      fig.savefig(filename, dpi=220)
      plt.close(fig)
      return
    if use_burn_tier:
      tier_colors = ['#2ca02c', '#ffbf00', '#d62728']  # fresh, once, twice+
      cmap = matplotlib.colors.ListedColormap(tier_colors)
      bounds = [-0.5, 0.5, 1.5, 2.5]
      norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
      img = ax.imshow(masked, cmap=cmap, norm=norm, origin='upper', interpolation='none')
      if not self.categoricalLegend:
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2], boundaries=bounds)
        cbar_label = self.colorbarLabel or 'Burn tier'
        cbar.ax.set_ylabel(cbar_label, color='#e8f0ff')
        cbar.ax.yaxis.set_tick_params(color='#e8f0ff')
        cbar.ax.set_yticklabels(['0 fresh', '1 once', '2 twice+'])
        for lab in cbar.ax.get_yticklabels():
          lab.set_color('#e8f0ff')
      else:
        legend_handles = [
          Patch(facecolor=tier_colors[0], edgecolor='none', label='0 fresh'),
          Patch(facecolor=tier_colors[1], edgecolor='none', label='1 once'),
          Patch(facecolor=tier_colors[2], edgecolor='none', label='2 twice+')
        ]
        ax.legend(handles=legend_handles, title=self.colorbarLabel or 'Burn tier',
                  loc='upper right', frameon=True, facecolor='#0a0f1c', edgecolor='#e8f0ff',
                  fontsize=7, title_fontsize=8)
    else:
      cmap = plt.cm.get_cmap(self.cmapName or 'viridis')
      img = ax.imshow(masked, cmap=cmap, origin='upper', interpolation='none', **img_kwargs)
      cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
      cbar_label = self.colorbarLabel or ('Burnup / color metric' if color_vals else 'Fuel ID / value')
      cbar.ax.set_ylabel(cbar_label, color='#e8f0ff')
      cbar.ax.yaxis.set_tick_params(color='#e8f0ff')
      for lab in cbar.ax.get_yticklabels():
        lab.set_color('#e8f0ff')

    for loc, coords in locs_view.items():
      val = active_vals.get(loc, np.nan)
      display_loc = self.locLabels.get(loc, loc)
      # Choose text color to keep contrast on bright categories (e.g., yellow tier 1).
      text_color = '#e8f0ff'
      if use_burn_tier and np.isfinite(val):
        if round(val) == 1:
          text_color = '#000000'
      for r, c in coords:
        txt = ''
        if self.labelMode == 'value':
          txt = f'{val:.0f}' if np.isfinite(val) else ''
        elif self.labelMode == 'loc':
          txt = display_loc
        else:
          txt_val = f'{val:.0f}' if np.isfinite(val) else ''
          txt = f'{display_loc}\n{txt_val}' if txt_val else display_loc
        ax.text(c, r, txt, ha='center', va='center', fontsize=7, color=text_color)

    ax.set_xticks(range(size_view))
    ax.set_yticks(range(size_view))
    ax.set_xticklabels([chr(ord('A') + col_start + i) for i in range(size_view)], color='#e8f0ff')
    ax.set_yticklabels([row_start + i + 1 for i in range(size_view)], color='#e8f0ff')
    if self.xAxisPosition == 'top':
      ax.xaxis.tick_top()
      ax.xaxis.set_label_position('top')
    else:
      ax.xaxis.tick_bottom()
      ax.xaxis.set_label_position('bottom')
    if self.yAxisPosition == 'right':
      ax.yaxis.tick_right()
      ax.yaxis.set_label_position('right')
    else:
      ax.yaxis.tick_left()
      ax.yaxis.set_label_position('left')
    ax.tick_params(colors='#e8f0ff')
    ax.set_xlabel('Column (A..)', color='#e8f0ff')
    ax.set_ylabel('Row (1..)', color='#e8f0ff')
    ax.set_title(f'Core layout: {self.name}', color='#e8f0ff')
    ax.grid(color='#1e2a42', linewidth=0.6)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=220)
    plt.close(fig)

    if self.bar3d:
      try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
      except Exception:
        Axes3D = None
      xs, ys = np.meshgrid(np.arange(size_view), np.arange(size_view))
      xs_flat = xs.flatten()
      ys_flat = ys.flatten()
      vals_flat = grid_view.flatten()
      mask_flat = np.isnan(vals_flat)
      xs_draw = xs_flat[~mask_flat]
      ys_draw = ys_flat[~mask_flat]
      zs_draw = np.zeros_like(xs_draw, dtype=float)
      heights = vals_flat[~mask_flat]
      figbar = plt.figure(figsize=(7.2, 5.6))
      axbar = figbar.add_subplot(111, projection='3d')
      axbar.set_facecolor('#0a0f1c')
      figbar.patch.set_facecolor('#0a0f1c')
      width = depth = 0.85
      if use_burn_tier:
        tier_colors = ['#2ca02c', '#ffbf00', '#d62728']
        cmap_bar = matplotlib.colors.ListedColormap(tier_colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm_bar = matplotlib.colors.BoundaryNorm(bounds, cmap_bar.N)
        colors = cmap_bar(norm_bar(heights))
      else:
        cmap_bar = plt.cm.get_cmap(self.cmapName or 'viridis')
        norm_bar = matplotlib.colors.Normalize(vmin=np.nanmin(grid_view), vmax=np.nanmax(grid_view))
        colors = cmap_bar(norm_bar(heights))
      axbar.bar3d(xs_draw, ys_draw, zs_draw, width, depth, heights,
                  shade=True, color=colors, edgecolor='k', linewidth=0.15)
      axbar.set_xlabel('Column (A..)', color='#e8f0ff')
      axbar.set_ylabel('Row (1..)', color='#e8f0ff')
      axbar.set_zlabel(self.colorbarLabel or 'Value', color='#e8f0ff')
      axbar.tick_params(colors='#e8f0ff')
      axbar.view_init(elev=35, azim=-135)
      figbar.tight_layout()
      filename_bar = self._createFilename(defaultName=f'{self.name}_bars.png')
      figbar.savefig(filename_bar, dpi=220)
      plt.close(figbar)

    if self.surface3d:
      try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
      except Exception:
        Axes3D = None
      zdata = np.array(grid_view, dtype=float)
      mask = np.isnan(zdata)
      zplot = np.ma.array(zdata, mask=mask)
      xs, ys = np.meshgrid(np.arange(size_view), np.arange(size_view))
      fig3d = plt.figure(figsize=(7.2, 5.6))
      ax3d = fig3d.add_subplot(111, projection='3d')
      ax3d.set_facecolor('#0a0f1c')
      fig3d.patch.set_facecolor('#0a0f1c')
      if use_burn_tier:
        tier_colors = ['#2ca02c', '#ffbf00', '#d62728']
        cmap3d = matplotlib.colors.ListedColormap(tier_colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm3d = matplotlib.colors.BoundaryNorm(bounds, cmap3d.N)
        color_array = cmap3d(norm3d(np.where(mask, 0, zdata)))
        color_array[..., 3] = np.where(mask, 0, 1)
        surf = ax3d.plot_surface(xs, ys, zplot, facecolors=color_array, rstride=1, cstride=1,
                                 linewidth=0.4, antialiased=True)
        if not self.categoricalLegend:
          cbar = fig3d.colorbar(matplotlib.cm.ScalarMappable(norm=norm3d, cmap=cmap3d),
                                ax=ax3d, fraction=0.046, pad=0.08, ticks=[0, 1, 2], boundaries=bounds)
          cbar.ax.set_ylabel(self.colorbarLabel or 'Burn tier', color='#e8f0ff')
          cbar.ax.set_yticklabels(['0 fresh', '1 once', '2 twice+'])
          cbar.ax.yaxis.set_tick_params(color='#e8f0ff')
          for lab in cbar.ax.get_yticklabels():
            lab.set_color('#e8f0ff')
        else:
          legend_handles = [
            Patch(facecolor=tier_colors[0], edgecolor='none', label='0 fresh'),
            Patch(facecolor=tier_colors[1], edgecolor='none', label='1 once'),
            Patch(facecolor=tier_colors[2], edgecolor='none', label='2 twice+')
          ]
          ax3d.legend(handles=legend_handles, title=self.colorbarLabel or 'Burn tier',
                      loc='upper right', frameon=True, facecolor='#0a0f1c',
                      edgecolor='#e8f0ff', fontsize=7, title_fontsize=8)
      else:
        cmap3d = plt.cm.get_cmap(self.cmapName or 'viridis')
        surf = ax3d.plot_surface(xs, ys, zplot, cmap=cmap3d, rstride=1, cstride=1,
                                 linewidth=0.4, antialiased=True)
        cbar = fig3d.colorbar(surf, ax=ax3d, fraction=0.046, pad=0.08)
        cbar_label = self.colorbarLabel or ('Burnup / color metric' if color_vals else 'Fuel ID / value')
        cbar.ax.set_ylabel(cbar_label, color='#e8f0ff')
        cbar.ax.yaxis.set_tick_params(color='#e8f0ff')
        for lab in cbar.ax.get_yticklabels():
          lab.set_color('#e8f0ff')
      ax3d.set_xlabel('Column (A..)', color='#e8f0ff')
      ax3d.set_ylabel('Row (1..)', color='#e8f0ff')
      ax3d.set_zlabel(self.colorbarLabel or 'Value', color='#e8f0ff')
      ax3d.tick_params(colors='#e8f0ff')
      ax3d.view_init(elev=35, azim=-135)
      fig3d.tight_layout()
      filename3d = self._createFilename(defaultName=f'{self.name}_surface.png')
      fig3d.savefig(filename3d, dpi=220)
      plt.close(fig3d)

    # Optional assembly metadata table
    if self.assemblyTable:
      try:
        import pandas as pd
        meta_df = pd.read_csv(self.assemblyTable)
        fig2, ax2 = plt.subplots(figsize=(8, 1 + 0.3 * len(meta_df)))
        ax2.axis('off')
        table = ax2.table(cellText=meta_df.values, colLabels=meta_df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        fig2.tight_layout()
        table_name = self._createFilename(defaultName=f'{self.name}_assembly_table.png')
        fig2.savefig(table_name, dpi=220, bbox_inches='tight')
        plt.close(fig2)
      except Exception as err:
        self.raiseAWarning(f'CoreLayoutPlot "{self.name}" failed to render assembly table: {err}')
