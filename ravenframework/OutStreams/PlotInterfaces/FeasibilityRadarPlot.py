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
Radar/spider summary comparing feasible vs infeasible samples.

This plot aggregates a set of variables for two populations:
  - feasible: all selected constraints > 0
  - infeasible: any selected constraint <= 0

Each variable is min-max normalized across the full dataset (or the filtered
generation), and the chosen aggregate statistic (median by default) is drawn as
a polygon on a polar axis. This provides an at-a-glance view of how constraint
violations shift the typical decision-variable/objective pattern.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class FeasibilityRadarPlot(PlotInterface):
  """Radar/spider chart comparing feasible vs infeasible aggregates."""

  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing the samples."""))
    spec.addSub(InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""Variables to include as radar axes."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Constraint evaluation columns defining feasibility. Values > 0 are feasible; values <= 0 indicate violation.
                   Use "all" to include every column named ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> provided, limits the plot to the specified generation. If omitted, uses the last generation."""))
    spec.addSub(InputData.parameterInputFactory('aggregate', contentType=InputTypes.StringType,
        descr=r"""Aggregate statistic per group: "median" (default) or "mean"."""))
    spec.addSub(InputData.parameterInputFactory('include_all', contentType=InputTypes.BoolType,
        descr=r"""If true, also draws an 'all samples' polygon. Default false."""))
    spec.addSub(InputData.parameterInputFactory('use_all_generations', contentType=InputTypes.BoolType,
        descr=r"""When <index> is provided, use all generations instead of only the last generation (or <generation>). Default false."""))
    spec.addSub(InputData.parameterInputFactory('quantiles', contentType=InputTypes.FloatListType,
        descr=r"""Optional two quantiles (e.g., 0.1, 0.9) used to draw per-group bands around the aggregate profile."""))
    spec.addSub(InputData.parameterInputFactory('bands', contentType=InputTypes.StringListType,
        descr=r"""Which groups should have quantile bands when <quantiles> are provided. Options: feasible, infeasible, all. Default feasible,infeasible."""))
    spec.addSub(InputData.parameterInputFactory('band_alpha', contentType=InputTypes.FloatType,
        descr=r"""Fill alpha for quantile bands (default 0.10)."""))
    spec.addSub(InputData.parameterInputFactory('show_empty_groups', contentType=InputTypes.BoolType,
        descr=r"""If true, include legend entries even when a group has zero samples. Default true."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'FeasibilityRadarPlot'
    self.source = None
    self.sourceName = None
    self.variables = []
    self.constraints = []
    self.useAllConstraints = False
    self.index = None
    self.generation = None
    self.aggregate = 'median'
    self.includeAll = False
    self.useAllGenerations = False
    self.quantiles = None
    self.bandGroups = {'feasible', 'infeasible'}
    self.bandAlpha = 0.10
    self.showEmptyGroups = True

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'Missing <source> node for FeasibilityRadarPlot "{self.name}".')
    self.sourceName = src.value

    varNode = spec.findFirst('variables')
    if varNode is None or not varNode.value:
      self.raiseAnError(IOError, f'FeasibilityRadarPlot "{self.name}" requires non-empty <variables>.')
    self.variables = [entry for entry in varNode.value if entry]

    consNode = spec.findFirst('constraints')
    if consNode is None or not consNode.value:
      self.raiseAnError(IOError, f'FeasibilityRadarPlot "{self.name}" requires non-empty <constraints>.')
    entries = [str(entry).strip() for entry in consNode.value if str(entry).strip()]
    if any(entry.lower() == 'all' for entry in entries):
      self.useAllConstraints = True
    else:
      self.constraints = entries

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value
    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    aggNode = spec.findFirst('aggregate')
    if aggNode is not None and aggNode.value:
      self.aggregate = str(aggNode.value).strip().lower()
    if self.aggregate not in {'median', 'mean'}:
      self.raiseAnError(IOError, f'FeasibilityRadarPlot "{self.name}" received unsupported <aggregate> "{self.aggregate}".')

    allNode = spec.findFirst('include_all')
    if allNode is not None and allNode.value is not None:
      self.includeAll = bool(allNode.value)

    uagNode = spec.findFirst('use_all_generations')
    if uagNode is not None and uagNode.value is not None:
      self.useAllGenerations = bool(uagNode.value)

    qNode = spec.findFirst('quantiles')
    if qNode is not None and qNode.value:
      try:
        qvals = [float(v) for v in qNode.value]
      except (TypeError, ValueError):
        self.raiseAnError(IOError, f'FeasibilityRadarPlot "{self.name}" received invalid <quantiles>.')
      if len(qvals) != 2:
        self.raiseAnError(IOError, f'FeasibilityRadarPlot "{self.name}" requires two values in <quantiles>.')
      qlo, qhi = min(qvals[0], qvals[1]), max(qvals[0], qvals[1])
      if not (0.0 <= qlo <= 1.0 and 0.0 <= qhi <= 1.0 and qlo < qhi):
        self.raiseAnError(IOError, f'FeasibilityRadarPlot "{self.name}" received invalid <quantiles> ({qlo}, {qhi}).')
      self.quantiles = (qlo, qhi)

    bandsNode = spec.findFirst('bands')
    if bandsNode is not None and bandsNode.value:
      groups = {str(v).strip().lower() for v in bandsNode.value if str(v).strip()}
      allowed = {'feasible', 'infeasible', 'all'}
      bad = sorted(groups - allowed)
      if bad:
        self.raiseAnError(IOError, f'FeasibilityRadarPlot "{self.name}" received invalid <bands> entries: {bad}.')
      self.bandGroups = groups if groups else {'feasible', 'infeasible'}

    baNode = spec.findFirst('band_alpha')
    if baNode is not None and baNode.value is not None:
      self.bandAlpha = float(baNode.value)

    segNode = spec.findFirst('show_empty_groups')
    if segNode is not None and segNode.value is not None:
      self.showEmptyGroups = bool(segNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for FeasibilityRadarPlot "{self.name}".')
    available = src.getVars()
    if self.useAllConstraints:
      self.constraints = sorted(var for var in available if var.startswith('ConstraintEvaluation_'))
    needed = list(self.variables) + list(self.constraints)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by FeasibilityRadarPlot "{self.name}".')
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

  def _aggregate_row(self, df):
    if df.empty:
      return None
    if self.aggregate == 'mean':
      return df.mean(axis=0)
    return df.median(axis=0)

  def _quantile_rows(self, df):
    if df.empty or self.quantiles is None:
      return None, None
    qlo, qhi = self.quantiles
    return df.quantile(qlo, axis=0), df.quantile(qhi, axis=0)

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'FeasibilityRadarPlot "{self.name}" received an empty dataset; skipping.')
      return

    subset = df.copy()
    if self.index and not self.useAllGenerations:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if subset.empty:
      self.raiseAWarning(f'FeasibilityRadarPlot "{self.name}" had no samples after filtering.')
      return

    cols = list(self.variables) + list(self.constraints)
    data = subset[cols].copy()
    for col in cols:
      data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=self.variables + self.constraints)
    if data.empty:
      self.raiseAWarning(f'FeasibilityRadarPlot "{self.name}" has no finite samples after coercion.')
      return

    feasible_mask = self._is_feasible(data, self.constraints)
    feas = data.loc[feasible_mask, self.variables]
    infeas = data.loc[~feasible_mask, self.variables]

    # Normalize variables to [0,1] across all samples shown.
    mins = data[self.variables].min(axis=0)
    maxs = data[self.variables].max(axis=0)
    span = (maxs - mins).replace(0.0, np.nan)
    norm_all = (data[self.variables] - mins) / span
    norm_all = norm_all.fillna(0.5)
    norm_feas = norm_all.loc[feas.index] if not feas.empty else pd.DataFrame(columns=self.variables)
    norm_infeas = norm_all.loc[infeas.index] if not infeas.empty else pd.DataFrame(columns=self.variables)

    agg_feas = self._aggregate_row(norm_feas)
    agg_infeas = self._aggregate_row(norm_infeas)
    agg_all = self._aggregate_row(norm_all) if self.includeAll else None
    q_feas_lo, q_feas_hi = self._quantile_rows(norm_feas)
    q_infeas_lo, q_infeas_hi = self._quantile_rows(norm_infeas)
    q_all_lo, q_all_hi = self._quantile_rows(norm_all) if self.includeAll else (None, None)

    if agg_feas is None and agg_infeas is None:
      self.raiseAWarning(f'FeasibilityRadarPlot "{self.name}" had no feasible or infeasible samples to plot.')
      return

    labels = list(self.variables)
    n = len(labels)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.grid(alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    def _series_values(series):
      values = np.asarray([float(series[var]) for var in labels], dtype=float)
      values = np.clip(values, 0.0, 1.0)
      return np.concatenate([values, values[:1]])

    def _plot_band(lo, hi, color):
      if lo is None or hi is None:
        return
      r_lo = _series_values(lo)
      r_hi = _series_values(hi)
      # build a closed polygon between the two curves
      poly_theta = np.concatenate([angles, angles[::-1]])
      poly_r = np.concatenate([r_hi, r_lo[::-1]])
      ax.fill(poly_theta, poly_r, color=color, alpha=float(self.bandAlpha), edgecolor='none')

    def _plot_poly(series, color, label):
      if series is None:
        return
      values = _series_values(series)
      ax.plot(angles, values, color=color, linewidth=2.2, label=label)
      ax.fill(angles, values, color=color, alpha=0.10)

    if agg_all is not None:
      if 'all' in self.bandGroups:
        _plot_band(q_all_lo, q_all_hi, '#1565c0')
      _plot_poly(agg_all, '#1565c0', f'All (n={len(norm_all)})')
    if agg_feas is not None or self.showEmptyGroups:
      if 'feasible' in self.bandGroups:
        _plot_band(q_feas_lo, q_feas_hi, '#2e7d32')
      _plot_poly(agg_feas, '#2e7d32', f'Feasible (n={len(norm_feas)})')
    if agg_infeas is not None or self.showEmptyGroups:
      if 'infeasible' in self.bandGroups:
        _plot_band(q_infeas_lo, q_infeas_hi, '#d32f2f')
      _plot_poly(agg_infeas, '#d32f2f', f'Infeasible (n={len(norm_infeas)})')

    ax.set_title(f'Feasibility radar ({self.aggregate})', va='bottom')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=True)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)
