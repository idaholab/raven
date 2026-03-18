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
Adjusted epsilon-optimal set visualization (2 objectives).

This plot illustrates the idea behind epsilon-dominance / epsilon-elitism:
objective space is partitioned into epsilon-sized boxes. A point is considered
epsilon-dominant if it lies in a box that is no worse (in all objectives) than
another point's box and strictly better in at least one objective box index.

The "adjusted" view used here matches common pedagogical figures:
- normalize objectives to [0, 1] (optional)
- choose epsilon either directly (<epsilon>) or as 1/<search_extent>
- show the epsilon grid, the occupied/selected epsilon boxes, and the
  resulting epsilon-efficient (non-epsilon-dominated) representatives.

Notes:
- This is a visualization tool; it does not change optimizer behavior.
- When multiple samples fall in the same epsilon box, one representative is
  chosen (lowest sum of minimization-space objectives).
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import textwrap

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class AdjustedEpsilonOptimalPlot(PlotInterface):
  """Plot epsilon grid + epsilon-efficient set for two objectives."""

  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly two objective variables to plot."""))
    spec.addSub(InputData.parameterInputFactory('goals', contentType=InputTypes.StringListType,
        descr=r"""Optional goal directions for objectives: min/max. If omitted, assumes min,min."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional constraint evaluation columns used to define feasibility. Values > 0 are feasible; values <= 0 violated.
                   Use "all" to include every column named ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional Pareto rank filter (e.g., 1 to highlight only rank-1)."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> provided, limits the plot to the specified generation. If omitted, uses the last generation."""))
    spec.addSub(InputData.parameterInputFactory('use_all_generations', contentType=InputTypes.BoolType,
        descr=r"""When <index> is provided, use all generations instead of only the last generation (or <generation>). Default true."""))
    spec.addSub(InputData.parameterInputFactory('include_failed', contentType=InputTypes.BoolType,
        descr=r"""If true, includes samples marked as failed (if the source contains a 'failed' column).
                   If false (default), failed samples are excluded to avoid invalid sentinel values (e.g., -1) distorting the plot."""))
    spec.addSub(InputData.parameterInputFactory('normalize', contentType=InputTypes.BoolType,
        descr=r"""If true (default), min-max normalizes objectives before applying epsilon boxes."""))
    spec.addSub(InputData.parameterInputFactory('normalize_mode', contentType=InputTypes.StringType,
        descr=r"""When <normalize> is true, choose the normalization method: "minmax" (default) or "quantile".
                   Quantile normalization uses <normalize_quantiles> to reduce outlier compression."""))
    spec.addSub(InputData.parameterInputFactory('normalize_quantiles', contentType=InputTypes.FloatListType,
        descr=r"""When <normalize_mode> is "quantile", two quantiles (e.g., 0.02,0.98) used to scale objectives.
                   Values are clipped to [0,1] after scaling."""))
    spec.addSub(InputData.parameterInputFactory('search_extent', contentType=InputTypes.IntegerType,
        descr=r"""If provided, uses epsilon = 1/search_extent in normalized space and draws a search_extent x search_extent grid."""))
    spec.addSub(InputData.parameterInputFactory('epsilon', contentType=InputTypes.FloatType,
        descr=r"""Optional epsilon size in normalized space. Overrides <search_extent> when provided."""))
    spec.addSub(InputData.parameterInputFactory('show_grid', contentType=InputTypes.BoolType,
        descr=r"""If true (default), draw epsilon grid lines."""))
    spec.addSub(InputData.parameterInputFactory('show_boxes', contentType=InputTypes.BoolType,
        descr=r"""If true (default), shade epsilon boxes occupied by the epsilon-efficient set."""))
    spec.addSub(InputData.parameterInputFactory('show_all', contentType=InputTypes.BoolType,
        descr=r"""If true (default), plot all samples as background points."""))
    spec.addSub(InputData.parameterInputFactory('show_infeasible', contentType=InputTypes.BoolType,
        descr=r"""If true, also plot infeasible points (as red x) when <constraints> are provided. Default true."""))
    spec.addSub(InputData.parameterInputFactory('auto_limits', contentType=InputTypes.BoolType,
        descr=r"""If true (default), zoom axes to the occupied region (all samples and epsilon-efficient points)."""))
    spec.addSub(InputData.parameterInputFactory('limits_padding', contentType=InputTypes.FloatType,
        descr=r"""Padding fraction added around auto-limits (default 0.05)."""))
    spec.addSub(InputData.parameterInputFactory('xlim', contentType=InputTypes.FloatListType,
        descr=r"""Optional explicit x-axis limits [xmin, xmax] in plotted coordinates."""))
    spec.addSub(InputData.parameterInputFactory('ylim', contentType=InputTypes.FloatListType,
        descr=r"""Optional explicit y-axis limits [ymin, ymax] in plotted coordinates."""))
    spec.addSub(InputData.parameterInputFactory('aspect', contentType=InputTypes.StringType,
        descr=r"""Axis aspect ratio: "equal" (default, keeps epsilon boxes square) or "auto"."""))
    spec.addSub(InputData.parameterInputFactory('axis_space', contentType=InputTypes.StringType,
        descr=r"""Axis coordinate system: "normalized" (default) or "objective".
                   If "objective", axes show original objective values (with goals applied for direction),
                   and epsilon grid/boxes are mapped from normalized space to objective space."""))
    spec.addSub(InputData.parameterInputFactory('layout_top', contentType=InputTypes.FloatType,
        descr=r"""Top boundary (0-1] passed to matplotlib tight_layout(rect=...). Smaller values leave more headroom above the title. Default 0.96."""))
    spec.addSub(InputData.parameterInputFactory('show_info', contentType=InputTypes.BoolType,
        descr=r"""If true (default), prints a small "info cues" box under the legend summarizing key plot settings
                   (epsilon/extent, normalization, axis space, filters, and constraint usage)."""))
    spec.addSub(InputData.parameterInputFactory('info_fontsize', contentType=InputTypes.FloatType,
        descr=r"""Font size for the info box (default 9)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'AdjustedEpsilonOptimalPlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.goals = None
    self.constraints = []
    self.useAllConstraints = False
    self.rank = None
    self.index = None
    self.generation = None
    self.useAllGenerations = True
    self.includeFailed = False
    self.normalize = True
    self.normalizeMode = 'minmax'
    self.normalizeQuantiles = (0.02, 0.98)
    self.searchExtent = None
    self.epsilon = None
    self.showGrid = True
    self.showBoxes = True
    self.showAll = True
    self.showInfeasible = True
    self.autoLimits = True
    self.limitsPadding = 0.05
    self.xlim = None
    self.ylim = None
    self.aspect = 'equal'
    self._aspectFromInput = False
    self.axisSpace = 'normalized'
    self.layoutTop = 0.96
    self.showInfo = True
    self.infoFontsize = 9.0

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'Missing <source> node for AdjustedEpsilonOptimalPlot "{self.name}".')
    self.sourceName = src.value

    objNode = spec.findFirst('objectives')
    if objNode is None or len(objNode.value) != 2:
      self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" requires exactly two <objectives>.')
    self.objectives = [entry for entry in objNode.value if entry]

    goalsNode = spec.findFirst('goals')
    if goalsNode is not None and goalsNode.value:
      goals = [str(v).strip().lower() for v in goalsNode.value if str(v).strip()]
      if len(goals) != 2 or any(g not in {'min', 'max'} for g in goals):
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" requires <goals> as two entries min/max.')
      self.goals = goals

    consNode = spec.findFirst('constraints')
    if consNode is not None and consNode.value:
      entries = [str(entry).strip() for entry in consNode.value if str(entry).strip()]
      if any(entry.lower() == 'all' for entry in entries):
        self.useAllConstraints = True
      else:
        self.constraints = entries

    rankNode = spec.findFirst('rank')
    if rankNode is not None and rankNode.value is not None:
      self.rank = int(rankNode.value)

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = str(idxNode.value)
    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)
    uagNode = spec.findFirst('use_all_generations')
    if uagNode is not None and uagNode.value is not None:
      self.useAllGenerations = bool(uagNode.value)

    ifNode = spec.findFirst('include_failed')
    if ifNode is not None and ifNode.value is not None:
      self.includeFailed = bool(ifNode.value)

    normNode = spec.findFirst('normalize')
    if normNode is not None and normNode.value is not None:
      self.normalize = bool(normNode.value)

    nmNode = spec.findFirst('normalize_mode')
    if nmNode is not None and nmNode.value:
      self.normalizeMode = str(nmNode.value).strip().lower()
    if self.normalizeMode not in {'minmax', 'quantile'}:
      self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received unsupported <normalize_mode> "{self.normalizeMode}".')
    nqNode = spec.findFirst('normalize_quantiles')
    if nqNode is not None and nqNode.value:
      vals = [float(v) for v in nqNode.value]
      if len(vals) != 2:
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" requires two values in <normalize_quantiles>.')
      qlo, qhi = min(vals[0], vals[1]), max(vals[0], vals[1])
      if not (0.0 <= qlo <= 1.0 and 0.0 <= qhi <= 1.0 and qlo < qhi):
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received invalid <normalize_quantiles> ({qlo}, {qhi}).')
      self.normalizeQuantiles = (qlo, qhi)

    seNode = spec.findFirst('search_extent')
    if seNode is not None and seNode.value is not None:
      self.searchExtent = int(seNode.value)
      if self.searchExtent <= 0:
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received non-positive <search_extent>.')
    epsNode = spec.findFirst('epsilon')
    if epsNode is not None and epsNode.value is not None:
      self.epsilon = float(epsNode.value)
      if self.epsilon <= 0.0:
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received non-positive <epsilon>.')

    sgNode = spec.findFirst('show_grid')
    if sgNode is not None and sgNode.value is not None:
      self.showGrid = bool(sgNode.value)
    sbNode = spec.findFirst('show_boxes')
    if sbNode is not None and sbNode.value is not None:
      self.showBoxes = bool(sbNode.value)
    saNode = spec.findFirst('show_all')
    if saNode is not None and saNode.value is not None:
      self.showAll = bool(saNode.value)
    sifNode = spec.findFirst('show_infeasible')
    if sifNode is not None and sifNode.value is not None:
      self.showInfeasible = bool(sifNode.value)

    alNode = spec.findFirst('auto_limits')
    if alNode is not None and alNode.value is not None:
      self.autoLimits = bool(alNode.value)
    lpNode = spec.findFirst('limits_padding')
    if lpNode is not None and lpNode.value is not None:
      self.limitsPadding = float(lpNode.value)
    xlimNode = spec.findFirst('xlim')
    if xlimNode is not None and xlimNode.value:
      vals = [float(v) for v in xlimNode.value]
      if len(vals) != 2:
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" requires two values in <xlim>.')
      self.xlim = (min(vals[0], vals[1]), max(vals[0], vals[1]))
    ylimNode = spec.findFirst('ylim')
    if ylimNode is not None and ylimNode.value:
      vals = [float(v) for v in ylimNode.value]
      if len(vals) != 2:
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" requires two values in <ylim>.')
      self.ylim = (min(vals[0], vals[1]), max(vals[0], vals[1]))
    aspNode = spec.findFirst('aspect')
    if aspNode is not None and aspNode.value:
      self.aspect = str(aspNode.value).strip().lower()
      self._aspectFromInput = True
    if self.aspect not in {'equal', 'auto'}:
      self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received unsupported <aspect> "{self.aspect}".')

    asNode = spec.findFirst('axis_space')
    if asNode is not None and asNode.value:
      self.axisSpace = str(asNode.value).strip().lower()
    if self.axisSpace not in {'normalized', 'objective'}:
      self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received unsupported <axis_space> "{self.axisSpace}".')
    # "equal" aspect is great for normalized space (0-1 box), but it can collapse the plot in objective space
    # when objective magnitudes differ by orders (e.g., eta ~ 0.3 vs cost ~ 2.5e6). Default to "auto" in that case.
    if self.axisSpace == 'objective' and not self._aspectFromInput and self.aspect == 'equal':
      self.aspect = 'auto'

    ltNode = spec.findFirst('layout_top')
    if ltNode is not None and ltNode.value is not None:
      self.layoutTop = float(ltNode.value)
      if not (0.0 < self.layoutTop <= 1.0):
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received invalid <layout_top> {self.layoutTop}; expected (0, 1].')

    siNode = spec.findFirst('show_info')
    if siNode is not None and siNode.value is not None:
      self.showInfo = bool(siNode.value)
    fsNode = spec.findFirst('info_fontsize')
    if fsNode is not None and fsNode.value is not None:
      self.infoFontsize = float(fsNode.value)
      if self.infoFontsize <= 0.0:
        self.raiseAnError(IOError, f'AdjustedEpsilonOptimalPlot "{self.name}" received non-positive <info_fontsize>.')

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for AdjustedEpsilonOptimalPlot "{self.name}".')
    available = src.getVars()
    if self.useAllConstraints:
      self.constraints = sorted(var for var in available if var.startswith('ConstraintEvaluation_'))
    needed = list(self.objectives)
    if self.constraints:
      needed += list(self.constraints)
    if self.rank is not None:
      needed.append('rank')
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by AdjustedEpsilonOptimalPlot "{self.name}".')
    self.source = src

  @staticmethod
  def _is_feasible(df, constraints):
    if df is None or df.empty or not constraints:
      return np.ones(0 if df is None else len(df), dtype=bool)
    feasible = np.ones(len(df), dtype=bool)
    for var in constraints:
      vals = df[var].astype(float).to_numpy()
      feasible &= vals > 0.0
    return feasible

  def _to_min_space(self, values):
    vals = np.asarray(values, dtype=float).copy()
    goals = self.goals or ['min', 'min']
    for j, g in enumerate(goals):
      if g == 'max':
        vals[:, j] = -vals[:, j]
    return vals

  def _from_min_space(self, values):
    """Inverse of _to_min_space for display (min-space -> objective-space with goals applied)."""
    vals = np.asarray(values, dtype=float).copy()
    goals = self.goals or ['min', 'min']
    for j, g in enumerate(goals):
      if g == 'max':
        vals[:, j] = -vals[:, j]
    return vals

  @staticmethod
  def _minmax(values):
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    span = maxs - mins
    span = np.where(span == 0.0, 1.0, span)
    return mins, maxs, span

  def _quantile_scale(self, values):
    qlo, qhi = self.normalizeQuantiles
    lo = np.nanquantile(values, qlo, axis=0)
    hi = np.nanquantile(values, qhi, axis=0)
    span = hi - lo
    span = np.where(span == 0.0, 1.0, span)
    return lo, hi, span

  @staticmethod
  def _epsilon_nondominated(cells):
    """cells: (N,2) non-negative ints. Returns boolean mask of epsilon-nondominated samples."""
    n = cells.shape[0]
    keep = np.ones(n, dtype=bool)
    # O(N^2) is fine for plotting-scale N; if needed, optimize later.
    for i in range(n):
      if not keep[i]:
        continue
      ci = cells[i]
      for j in range(n):
        if i == j or not keep[i]:
          continue
        cj = cells[j]
        if np.all(cj <= ci) and np.any(cj < ci):
          keep[i] = False
          break
    return keep

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'AdjustedEpsilonOptimalPlot "{self.name}" received an empty dataset; skipping.')
      return

    subset = df.copy()
    if not self.includeFailed and 'failed' in subset.columns:
      # Many workflows use 'failed' to mark invalid rows (often with sentinel objective values).
      subset = subset[pd.to_numeric(subset['failed'], errors='coerce').fillna(0.0) == 0.0]
    if self.index and not self.useAllGenerations:
      subset[self.index] = pd.to_numeric(subset[self.index], errors='coerce')
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if self.rank is not None and 'rank' in subset.columns:
      subset = subset[subset['rank'].astype(float) == float(self.rank)]
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=self.objectives)
    if subset.empty:
      self.raiseAWarning(f'AdjustedEpsilonOptimalPlot "{self.name}" has no samples after filtering.')
      return

    objs = subset[self.objectives].astype(float).to_numpy()
    objs_min = self._to_min_space(objs)

    if self.normalize:
      if self.normalizeMode == 'quantile':
        mins, maxs, span = self._quantile_scale(objs_min)
      else:
        mins, maxs, span = self._minmax(objs_min)
      norm = (objs_min - mins) / span
      norm = np.clip(norm, 0.0, 1.0)
    else:
      # shift to non-negative for epsilon boxing
      mins, maxs, span = self._minmax(objs_min)
      norm = objs_min - mins

    epsilon = None
    if self.epsilon is not None:
      epsilon = float(self.epsilon)
    elif self.searchExtent is not None:
      epsilon = 1.0 / float(self.searchExtent)
    else:
      epsilon = 0.1

    # box indices
    cells = np.floor(norm / epsilon).astype(int)
    cells = np.maximum(cells, 0)

    # choose one representative per cell (lowest sum in minimization space)
    cell_keys = [tuple(c) for c in cells]
    df_rep = subset.copy()
    df_rep['_cell0'] = cells[:, 0]
    df_rep['_cell1'] = cells[:, 1]
    df_rep['_score'] = objs_min.sum(axis=1)
    grouped = df_rep.sort_values('_score').groupby(['_cell0', '_cell1'], as_index=False).first()
    rep_cells = grouped[['_cell0', '_cell1']].to_numpy(dtype=int)
    rep_vals = grouped[self.objectives].astype(float).to_numpy()

    # epsilon-nondominated in cell space among representatives
    nd_mask = self._epsilon_nondominated(rep_cells)
    eps_front = grouped.loc[nd_mask].copy()

    # Prepare plot coordinates (always show normalized plane for pedagogy)
    if self.normalize:
      plot_coords_all = norm
      rep_min = self._to_min_space(eps_front[self.objectives].astype(float).to_numpy())
      rep_norm_coords = (rep_min - mins) / span
      rep_norm_coords = np.clip(rep_norm_coords, 0.0, 1.0)
    else:
      plot_coords_all = norm
      rep_min = self._to_min_space(eps_front[self.objectives].astype(float).to_numpy())
      rep_norm_coords = rep_min - mins

    def _norm_to_display(coords):
      coords = np.asarray(coords, dtype=float)
      if self.axisSpace == 'normalized' or not self.normalize:
        return coords
      # coords in [0,1] normalized min-space -> min-space -> objective-space
      min_space = coords * span + mins
      return self._from_min_space(min_space)

    display_all = _norm_to_display(plot_coords_all)
    display_rep = _norm_to_display(rep_norm_coords) if rep_norm_coords is not None else None

    feasible_mask = self._is_feasible(subset, self.constraints) if self.constraints else np.ones(len(subset), dtype=bool)
    infeasible_mask = ~feasible_mask

    fig_width = 10.0 if self.showInfo else 7.0
    fig, ax = plt.subplots(figsize=(fig_width, 6.3))

    if self.showGrid:
      extent = self.searchExtent if self.searchExtent is not None else int(np.ceil(1.0 / epsilon))
      extent = max(1, int(extent))
      ticks = np.arange(0, extent + 1) * epsilon
      for t in ticks:
        if self.axisSpace == 'normalized' or not self.normalize:
          xv, yv = t, t
        else:
          xv = float(self._from_min_space(np.array([[t * span[0] + mins[0], 0.0]], dtype=float))[0, 0])
          yv = float(self._from_min_space(np.array([[0.0, t * span[1] + mins[1]]], dtype=float))[0, 1])
        ax.axvline(xv, color='#cccccc', linewidth=0.6, alpha=0.6, zorder=0)
        ax.axhline(yv, color='#cccccc', linewidth=0.6, alpha=0.6, zorder=0)

    if self.showAll:
      ax.scatter(display_all[feasible_mask, 0], display_all[feasible_mask, 1],
                 c='#6abf69', s=16, alpha=0.35, edgecolors='none', label='Samples')
      if self.constraints and self.showInfeasible and infeasible_mask.any():
        ax.scatter(display_all[infeasible_mask, 0], display_all[infeasible_mask, 1],
                   c='#d32f2f', s=24, alpha=0.55, marker='x', linewidths=0.8, label='Infeasible')

    # epsilon-efficient representatives
    rep_cells_front = eps_front[['_cell0', '_cell1']].to_numpy(dtype=int)
    ax.scatter(display_rep[:, 0], display_rep[:, 1],
               c='#1565c0', s=36, alpha=0.95, edgecolors='white', linewidths=0.4, label='Epsilon-efficient')

    if self.showBoxes and rep_cells_front.size > 0:
      for c0, c1 in rep_cells_front:
        x0n = c0 * epsilon
        y0n = c1 * epsilon
        if self.axisSpace == 'normalized' or not self.normalize:
          x0, y0 = x0n, y0n
          w, h = epsilon, epsilon
        else:
          # map rectangle corners from normalized to objective display coordinates
          p00 = _norm_to_display(np.array([[x0n, y0n]], dtype=float))[0]
          p10 = _norm_to_display(np.array([[x0n + epsilon, y0n]], dtype=float))[0]
          p01 = _norm_to_display(np.array([[x0n, y0n + epsilon]], dtype=float))[0]
          x0, y0 = float(p00[0]), float(p00[1])
          w = float(p10[0] - p00[0])
          h = float(p01[1] - p00[1])
        rect = Rectangle((x0, y0), w, h,
                         facecolor='#f6c453', edgecolor='#b88700', linewidth=1.0, alpha=0.45)
        ax.add_patch(rect)

    # Axis limits: explicit overrides auto; otherwise zoom to occupied region for readability.
    if self.xlim is not None:
      ax.set_xlim(self.xlim[0], self.xlim[1])
    if self.ylim is not None:
      ax.set_ylim(self.ylim[0], self.ylim[1])
    if self.xlim is None or self.ylim is None:
      if self.autoLimits:
        clouds = [display_all]
        if display_rep is not None and len(display_rep):
          clouds.append(display_rep)
        pts = np.vstack(clouds)
        xmin, xmax = float(np.nanmin(pts[:, 0])), float(np.nanmax(pts[:, 0]))
        ymin, ymax = float(np.nanmin(pts[:, 1])), float(np.nanmax(pts[:, 1]))
        spanx = xmax - xmin
        spany = ymax - ymin
        padx = max(epsilon * 2.0, float(self.limitsPadding) * (spanx if spanx > 0 else 1.0))
        pady = max(epsilon * 2.0, float(self.limitsPadding) * (spany if spany > 0 else 1.0))
        xmin -= padx
        xmax += padx
        ymin -= pady
        ymax += pady
        if self.normalize and self.axisSpace == 'normalized':
          xmin = max(0.0, xmin)
          xmax = min(1.0, xmax)
          ymin = max(0.0, ymin)
          ymax = min(1.0, ymax)
        if self.xlim is None:
          ax.set_xlim(xmin, xmax)
        if self.ylim is None:
          ax.set_ylim(ymin, ymax)
      else:
        ax.set_xlim(0.0, 1.0 if self.normalize else float(np.nanmax(plot_coords_all[:, 0]) * 1.05))
        ax.set_ylim(0.0, 1.0 if self.normalize else float(np.nanmax(plot_coords_all[:, 1]) * 1.05))

    if self.aspect == 'equal':
      if self.axisSpace == 'objective':
        # In objective space, units can differ dramatically (e.g., O(1) vs O(1e6)).
        # Enforcing equal data scaling collapses the axes into a near-1D strip; ignore in that case.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        spanx = abs(float(x1) - float(x0))
        spany = abs(float(y1) - float(y0))
        if spanx > 0.0 and spany > 0.0:
          ratio = max(spany / spanx, spanx / spany)
        else:
          ratio = np.inf
        if ratio > 50.0:
          self.raiseAWarning(
            f'AdjustedEpsilonOptimalPlot "{self.name}": <aspect>=equal with <axis_space>=objective '
            f'produces an unusable plot (axis span ratio ~ {ratio:.3g}); using aspect=auto instead. '
            f'Set explicit <xlim>/<ylim> if you need square scaling.'
          )
        else:
          ax.set_aspect('equal', adjustable='box')
      else:
        ax.set_aspect('equal', adjustable='box')
    if self.axisSpace == 'objective':
      ax.set_xlabel(self.objectives[0])
      ax.set_ylabel(self.objectives[1])
    else:
      ax.set_xlabel(self.objectives[0] + (' (normalized)' if self.normalize else ''))
      ax.set_ylabel(self.objectives[1] + (' (normalized)' if self.normalize else ''))

    extent = self.searchExtent if self.searchExtent is not None else int(np.ceil(1.0 / epsilon))
    title = f'Adjusted epsilon-optimal set (extent={extent}, eps={epsilon:g})'
    ax.set_title(title)
    ax.grid(alpha=0.15)

    # Use a right-side panel for legend + if-cues so long text doesn't shrink the plot area.
    if self.showInfo:
      right_panel_left = 0.76
      plot_right = right_panel_left - 0.02
    else:
      right_panel_left = None
      plot_right = 0.96

    fig.tight_layout(rect=[0.08, 0.0, float(plot_right), float(self.layoutTop)])

    if self.showInfo:
      panel = fig.add_axes([float(right_panel_left), 0.05, 0.98 - float(right_panel_left), float(self.layoutTop) - 0.05])
      panel.set_axis_off()
      handles, labels = ax.get_legend_handles_labels()
      legend = panel.legend(handles, labels, loc='upper left', frameon=True)

      def _wrap_bullet(text, width_chars):
        wrapped = textwrap.wrap(text, width=max(12, int(width_chars)))
        if not wrapped:
          return ["•"]
        out = ["• " + wrapped[0]]
        out.extend(["  " + w for w in wrapped[1:]])
        return out

      # "If-cues": actionable interpretation guidance.
      good_cues = [
        ("blue points span most of the diagonal", "you have good Pareto coverage (diverse tradeoffs)."),
        ("blue points are spaced across many boxes", "you have many meaningfully different options at this eps."),
        ("the cloud is a smooth monotone trend", "objectives are consistently conflicting (expected tradeoff)."),
      ]
      bad_cues = [
        ("there are gaps in blue", "increase exploration (more gens/pop size, higher mutation) or relax constraints."),
        ("many green points fall in the same box", "decrease eps (increase extent) to resolve finer differences."),
        ("many infeasible X appear", "revisit constraints/penalties or add feasibility-first selection."),
        ("points pin near 0 or 1 after normalization", "use quantile normalization or inspect outliers/failed rows."),
      ]

      cue_width = 32
      good_lines = ["Good signs:"]
      for a, b in good_cues:
        good_lines += _wrap_bullet(f"If {a}, then {b}", cue_width)
      bad_lines = ["Needs attention:"]
      for a, b in bad_cues:
        bad_lines += _wrap_bullet(f"If {a}, then {b}", cue_width)

      fig.canvas.draw()
      renderer = fig.canvas.get_renderer()
      lb = legend.get_window_extent(renderer=renderer).transformed(panel.transAxes.inverted())
      x_left = float(lb.x0)
      y_top = float(lb.y0) - 0.02

      t_good = panel.text(
        x_left, y_top, "\n".join(good_lines),
        transform=panel.transAxes,
        ha='left', va='top',
        fontsize=float(self.infoFontsize),
        color='#2e7d32',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#c8e6c9', alpha=0.98),
      )
      fig.canvas.draw()
      gb = t_good.get_window_extent(renderer=renderer).transformed(panel.transAxes.inverted())
      y_top2 = float(gb.y0) - 0.03
      panel.text(
        x_left, y_top2, "\n".join(bad_lines),
        transform=panel.transAxes,
        ha='left', va='top',
        fontsize=float(self.infoFontsize),
        color='#c62828',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#ffcdd2', alpha=0.98),
      )

    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=170)
    plt.close(fig)
