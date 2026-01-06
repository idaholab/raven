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
Visual utility to render the rank-1 Pareto front produced by NSGA-II style optimizers.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

try:
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for side-effects)
except ImportError:
  Axes3D = None


class NSGAParetoFrontPlot(PlotInterface):
  """
  Static scatter plot of the dominant (rank-1) Pareto front.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject produced by the optimizer."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""List of objective variables to plot. Two objectives renders a 2D scatter while three
                   objectives renders a 3D scatter. More than three objectives are not supported."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional list of constraint evaluation columns. Values > 0 are treated as feasible;
                   values <= 0 indicate violation. Use "all" to include every column named
                   ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('color', contentType=InputTypes.StringType,
        descr=r"""Optional variable used to color points. Defaults to crowding distance if available."""))
    spec.addSub(InputData.parameterInputFactory('color_mode', contentType=InputTypes.StringType,
        descr=r"""How to color highlighted points. Options: "variable" (default, uses <color> or CD),
                   "violation" (total constraint violation magnitude), or "none"."""))
    spec.addSub(InputData.parameterInputFactory('violation_metric', contentType=InputTypes.StringType,
        descr=r"""When <color_mode> is "violation", reduce multiple constraints into a scalar using:
                   "sum" (default), "max", or "l2"."""))
    spec.addSub(InputData.parameterInputFactory('infeasible_style', contentType=InputTypes.StringType,
        descr=r"""How to render infeasible points when <constraints> are provided. Options: "fade" (default),
                   "cross", or "hide"."""))
    spec.addSub(InputData.parameterInputFactory('show_all', contentType=InputTypes.BoolType,
        descr=r"""If true, also draws non-highlighted samples (e.g., ranks != <rank>) in the background."""))
    spec.addSub(InputData.parameterInputFactory('active_constraints', contentType=InputTypes.BoolType,
        descr=r"""If true, draws a red outline around feasible points that are near-active for any constraint."""))
    spec.addSub(InputData.parameterInputFactory('active_tol', contentType=InputTypes.FloatType,
        descr=r"""Threshold for "near-active" constraints (default 1e-3). A constraint is considered active
                   when 0 < ConstraintEvaluation_* <= active_tol."""))
    spec.addSub(InputData.parameterInputFactory('cmap', contentType=InputTypes.StringType,
        descr=r"""Matplotlib colormap name used for numeric <color_mode> (default viridis)."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Pareto rank to display. Defaults to 1 (non-dominated front)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA Pareto Front Plot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.constraints = []
    self.useAllConstraints = False
    self.colorVar = None
    self.colorMode = 'variable'
    self.violationMetric = 'sum'
    self.infeasibleStyle = 'fade'
    self.showAll = False
    self.showActiveConstraints = False
    self.activeTol = 1e-3
    self.cmap = 'viridis'
    self.rank = 1

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    objectives = spec.findFirst('objectives')
    if objectives is None:
      self.raiseAnError(IOError, 'Missing <objectives> node for NSGAParetoFrontPlot "{}".'.format(self.name))
    self.objectives = objectives.value
    consNode = spec.findFirst('constraints')
    if consNode is not None:
      entries = consNode.value
      if entries:
        normalized = [str(val).strip() for val in entries if str(val).strip()]
        if len(normalized) == 1 and normalized[0].lower() == 'all':
          self.useAllConstraints = True
        else:
          self.constraints = normalized
    colorMode = spec.findFirst('color_mode')
    if colorMode is not None and colorMode.value is not None:
      self.colorMode = str(colorMode.value).strip().lower()
    violationMetric = spec.findFirst('violation_metric')
    if violationMetric is not None and violationMetric.value is not None:
      self.violationMetric = str(violationMetric.value).strip().lower()
    infeasibleStyle = spec.findFirst('infeasible_style')
    if infeasibleStyle is not None and infeasibleStyle.value is not None:
      self.infeasibleStyle = str(infeasibleStyle.value).strip().lower()
    showAll = spec.findFirst('show_all')
    if showAll is not None:
      self.showAll = bool(showAll.value)
    activeConstraints = spec.findFirst('active_constraints')
    if activeConstraints is not None:
      self.showActiveConstraints = bool(activeConstraints.value)
    activeTol = spec.findFirst('active_tol')
    if activeTol is not None and activeTol.value is not None:
      self.activeTol = float(activeTol.value)
    cmap = spec.findFirst('cmap')
    if cmap is not None and cmap.value is not None:
      self.cmap = str(cmap.value).strip()
    color = spec.findFirst('color')
    if color is not None:
      self.colorVar = color.value
    rank = spec.findFirst('rank')
    if rank is not None:
      self.rank = rank.value

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for NSGAParetoFrontPlot "{self.name}".')
    dataVars = self.source.getVars()
    missing = [var for var in self.objectives if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing required objective(s) {} for NSGAParetoFrontPlot "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)
    if len(self.objectives) not in {2, 3}:
      self.raiseAnError(IOError, 'NSGAParetoFrontPlot "{}" requires 2 or 3 objectives; got {}.'.format(
          self.name, len(self.objectives)))
    if self.useAllConstraints:
      detected = sorted(var for var in dataVars if var.startswith('ConstraintEvaluation_'))
      self.constraints = detected
      if not self.constraints:
        self.raiseAWarning(f'NSGAParetoFrontPlot "{self.name}" requested <constraints>all</constraints> but no ConstraintEvaluation_* columns were found; proceeding without constraint styling.')
    elif self.constraints:
      missing_constraints = [var for var in self.constraints if var not in dataVars]
      if missing_constraints:
        self.raiseAWarning(f'NSGAParetoFrontPlot "{self.name}" could not find constraint column(s) {missing_constraints}; proceeding with available constraints only.')
        self.constraints = [var for var in self.constraints if var in dataVars]

    if self.colorMode not in {'variable', 'violation', 'none'}:
      self.raiseAnError(IOError, f'NSGAParetoFrontPlot "{self.name}" received unsupported <color_mode> "{self.colorMode}".')
    if self.violationMetric not in {'sum', 'max', 'l2'}:
      self.raiseAnError(IOError, f'NSGAParetoFrontPlot "{self.name}" received unsupported <violation_metric> "{self.violationMetric}".')
    if self.infeasibleStyle not in {'fade', 'cross', 'hide'}:
      self.raiseAnError(IOError, f'NSGAParetoFrontPlot "{self.name}" received unsupported <infeasible_style> "{self.infeasibleStyle}".')
    if self.activeTol <= 0.0:
      self.raiseAnError(IOError, f'NSGAParetoFrontPlot "{self.name}" received non-positive <active_tol> {self.activeTol}.')

  @staticmethod
  def _is_feasible(df, constraints):
    if not constraints or df.empty:
      return np.ones(len(df), dtype=bool)
    feasible = np.ones(len(df), dtype=bool)
    for var in constraints:
      if var not in df.columns:
        continue
      vals = df[var].astype(float).to_numpy()
      feasible &= vals > 0.0
    return feasible

  def _constraint_violation(self, df):
    if not self.constraints or df.empty:
      return np.zeros(len(df), dtype=float)
    values = []
    for var in self.constraints:
      if var not in df.columns:
        continue
      vals = df[var].astype(float).to_numpy()
      values.append(np.maximum(0.0, -vals))
    if not values:
      return np.zeros(len(df), dtype=float)
    stacked = np.vstack(values)
    if self.violationMetric == 'max':
      return np.max(stacked, axis=0)
    if self.violationMetric == 'l2':
      return np.sqrt(np.sum(stacked * stacked, axis=0))
    return np.sum(stacked, axis=0)

  def _active_mask(self, df, feasible_mask):
    if not self.constraints or df.empty:
      return np.zeros(len(df), dtype=bool)
    if not feasible_mask.any():
      return np.zeros(len(df), dtype=bool)
    active_tol = float(self.activeTol)
    mins = np.full(len(df), np.inf, dtype=float)
    for var in self.constraints:
      if var not in df.columns:
        continue
      vals = df[var].astype(float).to_numpy()
      pos = np.where(vals > 0.0, vals, np.inf)
      mins = np.minimum(mins, pos)
    return feasible_mask & np.isfinite(mins) & (mins <= active_tol)

  def _color_payload(self, df):
    if df.empty:
      return None, False, None
    if self.colorMode == 'none':
      return None, False, None
    if self.colorMode == 'violation':
      colors = self._constraint_violation(df)
      return colors, True, 'constraint violation'

    colorVar = self.colorVar
    if colorVar is None and 'CD' in df.columns:
      colorVar = 'CD'
    if colorVar is None:
      return None, False, None
    if colorVar not in df.columns:
      self.raiseAWarning('Color variable "{}" not found; using uniform color.'.format(colorVar))
      return None, False, None

    series = df[colorVar]
    raw_values = series.to_numpy() if hasattr(series, 'to_numpy') else np.asarray(series)
    flat = np.asarray(raw_values).ravel()
    if flat.size == 0:
      self.raiseAWarning('Color variable "{}" contains no samples; using uniform color.'.format(colorVar))
      return None, False, None

    try:
      numeric_values = flat.astype(float)
    except (ValueError, TypeError):
      numeric_values = None

    if numeric_values is not None:
      finite_mask = np.isfinite(numeric_values)
      if not finite_mask.any():
        self.raiseAWarning('Color variable "{}" has no finite numeric values; using uniform color.'.format(colorVar))
        return None, False, None
      numeric_values = numeric_values.astype(float, copy=False)
      numeric_values[~finite_mask] = np.nan
      return numeric_values, True, colorVar

    string_values = np.array([str(val).strip() for val in flat], dtype=object)
    if not string_values.size:
      self.raiseAWarning('Color variable "{}" has no usable categorical values; using uniform color.'.format(colorVar))
      return None, False, None
    if not all(val and val.lower() not in {'nan', 'none'} and is_color_like(val) for val in string_values):
      self.raiseAWarning('Color variable "{}" cannot be interpreted as numeric or named colors; using uniform color.'.format(colorVar))
      return None, False, None
    return string_values, False, colorVar

  def run(self):
    df = self.source.asDataset().to_dataframe()
    hasRank = 'rank' in df.columns
    if not hasRank:
      self.raiseAWarning('DataObject "{}" lacks a "rank" column; highlighting all points (requested rank {}).'.format(
          self.source.name, self.rank))
    highlight = df.copy() if not hasRank else df[df['rank'] == self.rank]
    if highlight.empty:
      self.raiseAWarning('No samples with rank == {} for "{}"; highlighting all points.'.format(self.rank, self.name))
      highlight = df.copy()

    background = None
    if self.showAll and hasRank:
      background = df[df['rank'] != self.rank]

    constraints = self.constraints if self.constraints else []
    highlight_feasible = self._is_feasible(highlight, constraints)
    background_feasible = self._is_feasible(background, constraints) if background is not None else None

    good = highlight[highlight_feasible] if constraints else highlight
    if constraints and good.empty:
      self.raiseAWarning(f'NSGAParetoFrontPlot "{self.name}" found no feasible points to highlight; plotting infeasible points only.')
      good = highlight.copy()

    colors, useColorbar, colorLabel = self._color_payload(good)
    scatterKwargs = {}
    if colors is not None:
      scatterKwargs['c'] = colors
      if useColorbar:
        scatterKwargs['cmap'] = self.cmap

    fig = plt.figure()
    if len(self.objectives) == 2:
      ax = fig.add_subplot(111)
      if background is not None and not background.empty:
        ax.scatter(background[self.objectives[0]], background[self.objectives[1]],
                   c='#bdbdbd', s=22, alpha=0.20, linewidths=0.0)
        if constraints and self.infeasibleStyle != 'hide' and not background_feasible.all():
          bad = background[~background_feasible]
          if not bad.empty:
            if self.infeasibleStyle == 'cross':
              ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]],
                         c='#7f7f7f', s=28, alpha=0.30, marker='x', linewidths=1.0)
            else:
              ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]],
                         c='#7f7f7f', s=22, alpha=0.12, linewidths=0.0)

      if constraints and self.infeasibleStyle != 'hide' and not highlight_feasible.all():
        bad = highlight[~highlight_feasible]
        if not bad.empty:
          if self.infeasibleStyle == 'cross':
            ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]],
                       c='#7f7f7f', s=42, alpha=0.65, marker='x', linewidths=1.2)
          else:
            ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]],
                       c='#7f7f7f', s=36, alpha=0.25, linewidths=0.0)

      scatterKwargs2D = dict(scatterKwargs)
      scatterKwargs2D.update({'edgecolors': 'k', 'linewidths': 0.5, 's': 42, 'alpha': 0.9})
      sc = ax.scatter(good[self.objectives[0]], good[self.objectives[1]], **scatterKwargs2D)
      ax.set_xlabel(self.objectives[0])
      ax.set_ylabel(self.objectives[1])
      ax.set_title(f'Pareto Front (rank={self.rank})')
      if useColorbar and colorLabel is not None and colors is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(colorLabel)
      if constraints and self.showActiveConstraints:
        active = self._active_mask(highlight, highlight_feasible)
        if active.any():
          ax.scatter(highlight.loc[active, self.objectives[0]], highlight.loc[active, self.objectives[1]],
                     facecolors='none', edgecolors='tab:red', s=110, linewidths=1.2)
    else:
      if Axes3D is None:
        self.raiseAnError(RuntimeError, 'mpl_toolkits.mplot3d is not available but 3 objectives were requested.')
      ax = fig.add_subplot(111, projection='3d')
      if background is not None and not background.empty:
        ax.scatter(background[self.objectives[0]], background[self.objectives[1]], background[self.objectives[2]],
                   c='#bdbdbd', s=18, alpha=0.18, depthshade=True)
        if constraints and self.infeasibleStyle != 'hide' and not background_feasible.all():
          bad = background[~background_feasible]
          if not bad.empty:
            if self.infeasibleStyle == 'cross':
              ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]], bad[self.objectives[2]],
                         c='#7f7f7f', s=26, alpha=0.30, marker='x', depthshade=False)
            else:
              ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]], bad[self.objectives[2]],
                         c='#7f7f7f', s=18, alpha=0.12, depthshade=True)

      if constraints and self.infeasibleStyle != 'hide' and not highlight_feasible.all():
        bad = highlight[~highlight_feasible]
        if not bad.empty:
          if self.infeasibleStyle == 'cross':
            ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]], bad[self.objectives[2]],
                       c='#7f7f7f', s=46, alpha=0.65, marker='x', depthshade=False)
          else:
            ax.scatter(bad[self.objectives[0]], bad[self.objectives[1]], bad[self.objectives[2]],
                       c='#7f7f7f', s=28, alpha=0.25, depthshade=True)

      scatterKwargs3D = dict(scatterKwargs)
      scatterKwargs3D.update({'depthshade': True, 's': 40, 'alpha': 0.9})
      sc = ax.scatter(good[self.objectives[0]], good[self.objectives[1]], good[self.objectives[2]], **scatterKwargs3D)
      ax.set_xlabel(self.objectives[0])
      ax.set_ylabel(self.objectives[1])
      ax.set_zlabel(self.objectives[2])
      ax.set_title(f'Pareto Front (rank={self.rank})')
      if useColorbar and colorLabel is not None and colors is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=12, pad=0.1)
        cbar.set_label(colorLabel)
      if constraints and self.showActiveConstraints:
        active = self._active_mask(highlight, highlight_feasible)
        if active.any():
          ax.scatter(highlight.loc[active, self.objectives[0]], highlight.loc[active, self.objectives[1]],
                     highlight.loc[active, self.objectives[2]],
                     facecolors='none', edgecolors='tab:red', s=110, linewidths=1.0, depthshade=False)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    plt.savefig(filename)
    plt.close(fig)
