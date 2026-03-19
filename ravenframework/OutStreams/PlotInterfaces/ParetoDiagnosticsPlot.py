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
Plot diagnostic metrics (hypervolume, dominance counts) for multi-objective optimizers.
two time-series views from the same optimizer run:

Hypervolume progression (top panel). Each dot is the dominated space covered by the population at a generation.
Interpreting this curve depends on the space and objective directions:
  - Hypervolume is computed on a *minimization* representation of the objectives.
  - If your optimization contains maximization objectives, this plot must either (a) be told the goal
    directions (min/max) or (b) use the optimizer's fitness columns and convert them back to a minimization
    space.
  - In "best-so-far archive" space, hypervolume should be non-decreasing; in "current generation" space it
    can decrease (e.g., if elites are not preserved or constraints tighten).

Dominance statistics (lower panel). The solid green line is the count of rank‑1 samples; the gray dashed line shows total evaluated samples that generation. If Rank‑1 climbs toward the population size, the front is densifying—good for exploitation, but watch for premature convergence. A shrinking total population combined with stable Rank‑1 suggests aggressive pruning that may hurt diversity. If Rank‑1 collapses suddenly, constraints or objective scaling likely shifted, pushing most samples off the front.

Typical “if you see this, consider…” scenarios:

Hypervolume flat after a few generations → You’ve probably converged. Lower mutation or more elites may be wasting evaluations; consider tightening termination criteria or seeding a restart if you still need exploration.
Hypervolume climbs but Rank‑1 stays tiny → The population keeps finding isolated dominant points while diversity stays low. Increase population size or tweak crowding/selection pressure.
Rank‑1 spikes to nearly full population, hypervolume barely moves → Everyone is mutually nondominated but clustered; objectives are weakly conflicting or dominated by noise. Check normalization, rescale objectives, or add constraints to reintroduce trade-offs.
Both metrics crash together → Likely due to constraint enforcement or a bad survivor step that discarded valuable solutions. Audit constraint evaluations and acceptance logic or revert to a previous generation checkpoint.
Hypervolume jumps upward while total samples drop → You may have over-aggressive feasibility filters, keeping only the best few. Relax constraint tolerances or diversify survivor selection to avoid losing edge coverage.

"""

import math
import textwrap

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ParetoDiagnosticsPlot(PlotInterface):
  """
  Generates line plots of hypervolume and dominance statistics across generations.
  Currently supports two-objective problems.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    objectives = InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Ordered list of objective columns (exactly two are currently supported).""")
    spec.addSub(objectives)
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Name of the generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('space', contentType=InputTypes.StringType,
        descr=r"""Which columns define the Pareto space and hypervolume computation.
              Options:
                - objective (default): use the columns named in <objectives>.
                - fitness: use FitnessEvaluation_<objective> columns (and convert to a minimization space).
              Note: Hypervolume is always computed on a minimization representation."""))
    spec.addSub(InputData.parameterInputFactory('goals', contentType=InputTypes.StringListType,
        descr=r"""Optional list of goal directions for each objective when <space> is 'objective'.
              Provide as 'min,max' or 'min max'. Length must match <objectives>.
              If omitted, assumes all objectives are minimized."""))
    spec.addSub(InputData.parameterInputFactory('reference_point', contentType=InputTypes.StringListType,
        descr=r"""Optional comma-separated reference point used for hypervolume computation, expressed in the same space as selected by <space>.
              If omitted, the plot auto-selects a conservative (worse-than-observed) reference point consistent with the minimization
              representation used for hypervolume:
                - for minimized objectives: max(value) + margin
                - for maximized objectives: min(value) - margin
                - for fitness space (larger is better): min(fitness) - margin
              """))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ParetoDiagnostics'
    self.source = None
    self.sourceName = None
    self.index = None
    self.objectives = []          # user-provided objective names
    self._space_objectives = []   # actual dataframe columns used for hypervolume
    self.reference_point = None
    self.space = 'objective'      # 'objective' or 'fitness'
    self.goals = None             # list of 'min'/'max' or None

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    objNode = spec.findFirst('objectives')
    if objNode is None:
      self.raiseAnError(IOError, 'Missing <objectives> node in ParetoDiagnosticsPlot "{}".'.format(self.name))
    self.objectives = [entry for entry in objNode.value if entry]
    if len(self.objectives) != 2:
      self.raiseAnError(IOError, 'ParetoDiagnosticsPlot "{}" currently supports exactly two objectives.'.format(self.name))
    spaceNode = spec.findFirst('space')
    if spaceNode is not None and spaceNode.value is not None:
      self.space = str(spaceNode.value).strip().lower()
    if self.space not in ('objective', 'fitness'):
      self.raiseAnError(IOError, f'Unsupported <space> "{self.space}" in ParetoDiagnosticsPlot "{self.name}".')
    goalsNode = spec.findFirst('goals')
    if goalsNode is not None and goalsNode.value:
      goals = [str(g).strip().lower() for g in goalsNode.value if str(g).strip()]
      if len(goals) != len(self.objectives):
        self.raiseAnError(IOError, f'<goals> must contain {len(self.objectives)} entries for ParetoDiagnosticsPlot "{self.name}".')
      for g in goals:
        if g not in ('min', 'max'):
          self.raiseAnError(IOError, f'Invalid goal "{g}" in ParetoDiagnosticsPlot "{self.name}" (use min/max).')
      self.goals = goals
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, 'Missing <index> node in ParetoDiagnosticsPlot "{}".'.format(self.name))
    self.index = idxNode.value
    refNode = spec.findFirst('reference_point')
    if refNode is not None and refNode.value:
      try:
        ref_vals = [float(val) for val in refNode.value]
      except ValueError as err:
        self.raiseAnError(IOError, f'Invalid <reference_point> values for ParetoDiagnosticsPlot "{self.name}": {err}')
      if len(ref_vals) != len(self.objectives):
        self.raiseAnError(IOError, f'<reference_point> must contain {len(self.objectives)} entries for ParetoDiagnosticsPlot "{self.name}".')
      self.reference_point = np.asarray(ref_vals, dtype=float)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, 'Source "{}" not found for ParetoDiagnosticsPlot "{}".'.format(self.sourceName, self.name))
    available = self.source.getVars()
    if self.space == 'fitness':
      self._space_objectives = [f'FitnessEvaluation_{name}' for name in self.objectives]
      needed = list(self._space_objectives) + [self.index]
    else:
      self._space_objectives = list(self.objectives)
      needed = list(self.objectives) + [self.index]
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing required variable(s) {missing} for ParetoDiagnosticsPlot "{self.name}".')
    if self.reference_point is None:
      df = self.source.asDataset().to_dataframe()
      ref_vals = []
      for j, obj in enumerate(self._space_objectives):
        series = df[obj].astype(float)
        series_max = float(series.max())
        series_min = float(series.min())
        span = series_max - series_min
        margin = 0.05 * abs(span) if span != 0.0 else 0.05 * max(abs(series_max), abs(series_min), 1.0)
        margin = max(margin, 0.05)
        if self.space == 'fitness':
          ref_vals.append(series_min - margin)
          continue
        goal = (self.goals[j] if self.goals is not None else 'min')
        if goal == 'max':
          ref_vals.append(series_min - margin)
        else:
          ref_vals.append(series_max + margin)
      self.reference_point = np.asarray(ref_vals, dtype=float)

  def run(self):
    df = self.source.asDataset().to_dataframe()
    df = df.copy()
    df[self.index] = df[self.index].astype(float)
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'No generations found for ParetoDiagnosticsPlot "{self.name}".')
      return
    hv_series = []
    pareto_counts = []
    total_counts = []
    for gen in generations:
      subset = df[df[self.index] == gen]
      total_counts.append(len(subset))
      hv = self._compute_hypervolume(subset)
      hv_series.append(hv)
      pareto_counts.append(self._count_rank_one(subset))
    fig = plt.figure(figsize=(9.0, 6.8))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.55], hspace=0.35, wspace=0.25)
    axes = [fig.add_subplot(grid[0, :])]
    axes.append(fig.add_subplot(grid[1, :], sharex=axes[0]))
    tip_axes = [fig.add_subplot(grid[2, 0]),
                fig.add_subplot(grid[2, 1])]

    axes[0].plot(generations, hv_series, marker='o', color='tab:blue', linewidth=1.5)
    axes[0].set_ylabel('Hypervolume')
    space_label = 'fitness' if self.space == 'fitness' else 'objective'
    axes[0].set_title(f'Hypervolume progression ({space_label}: {", ".join(self.objectives)})')
    axes[0].grid(alpha=0.3)

    axes[1].plot(generations, pareto_counts, marker='o', color='tab:green', linewidth=1.5, label='Rank 1 count')
    axes[1].plot(generations, total_counts, marker='o', color='tab:gray', linewidth=1.0, linestyle='--', label='Total samples')
    axes[1].set_xlabel(self.index)
    axes[1].set_ylabel('Population')
    axes[1].set_title('Dominance statistics')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    def _write_tips(ax, title, messages):
      ax.set_axis_off()
      ax.set_xlim(0.0, 1.0)
      ax.set_ylim(0.0, 1.0)
      y = 1.0
      if title:
        ax.text(0.0, y, title, fontsize=9, va='top', ha='left', fontweight='bold', color='black')
        y -= 0.18
      for msg, color in messages:
        wrapped = textwrap.fill(msg, width=48)
        ax.text(0.0, y, wrapped, fontsize=9, va='top', ha='left', color=color)
        lines = wrapped.count('\n') + 1
        y -= 0.18 * lines
        y -= 0.05

    _write_tips(tip_axes[0], 'Hypervolume notes:', [
        ('Steady rise -> Pareto set expanding (often good exploration/exploitation balance).', 'green'),
        ('Sharp drop -> can indicate lost elites OR changing feasibility/penalties; verify objective directions.', 'red'),
        ('Large oscillation -> mutation/restart/constraint oscillations; check scaling and survivor selection.', 'red'),
    ])

    _write_tips(tip_axes[1], 'Dominance notes:', [
        ('Rank-1 ~= population -> front densifying (exploit, monitor diversity).', 'green'),
        ('Shrinking population with flat Rank-1 -> pruning may erode diversity.', 'red'),
        ('Rank-1 collapse -> constraints or scaling likely rejected most samples.', 'red'),
    ])

    plt.setp(axes[0].get_xticklabels(), visible=False)
    fig.align_ylabels(axes)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=144)
    plt.close(fig)

  def _compute_hypervolume(self, subset):
    """
    Hypervolume computation for bi-objective problems on a minimization representation.

    - If <space> is "objective": uses <goals> (min/max) if provided, otherwise assumes all minimized.
      Maximization objectives are multiplied by -1 before hypervolume computation.
    - If <space> is "fitness": uses FitnessEvaluation_* columns and multiplies by -1 since fitness is
      "larger is better" but hypervolume is computed on a minimization space.
    """
    if subset.empty:
      return 0.0
    objs = subset[self._space_objectives].astype(float).to_numpy()
    ref = np.asarray(self.reference_point, dtype=float).copy()
    if self.space == 'fitness':
      objs = -objs
      ref = -ref
    else:
      goals = self.goals or ['min'] * len(self.objectives)
      for j, g in enumerate(goals):
        if g == 'max':
          objs[:, j] = -objs[:, j]
          ref[j] = -ref[j]
    # Sort by first objective ascending
    order = np.argsort(objs[:, 0])
    sorted_objs = objs[order]
    hv = 0.0
    prev_x = ref[0]
    for x, y in sorted_objs[::-1]:
      width = prev_x - x
      if width < 0:
        width = 0
      height = max(0.0, ref[1] - y)
      hv += width * height
      prev_x = x
    return hv

  @staticmethod
  def _count_rank_one(subset):
    if subset.empty or 'rank' not in subset.columns:
      return 0
    try:
      ranks = subset['rank'].astype(float)
    except (ValueError, TypeError):
      return 0
    return int((ranks == 1).sum())
