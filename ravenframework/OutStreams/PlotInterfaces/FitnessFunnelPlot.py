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
Visualise generational convergence in single-objective runs via best/mean fitness bands.

Each generation is represented by three elements:
  * A line joining the best (min or max) fitness, highlighting improvement plateaus.
  * A line for the mean fitness.
  * A shaded band spanning mean ± one standard deviation to emphasise narrowing variability.
  * Individual samples are plotted as semi-transparent scatters to expose outliers.

What-if scenarios

Best line flat while mean drops slowly -> convergence has stalled; consider increasing mutation or
  restarts to escape local minima.
Mean remains far from best with wide variance band -> population is diverse but not converging; tighten
  selection pressure or reduce exploration.
Variance collapses early yet best keeps improving -> exploitation dominates successfully; optionally
  reduce early elitism to avoid missing alternate basins.
Best oscillates while variance spikes -> potential instability (e.g., repair operators or penalty swings);
  inspect constraint handling or fitness scaling.
"""

import os

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class FitnessFunnelPlot(PlotInterface):
  """
  Static line/scatter plot summarising best and mean fitness per generation.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringType,
        descr=r"""Column containing the single-objective fitness to evaluate."""))
    spec.addSub(InputData.parameterInputFactory('goal', contentType=InputTypes.StringType,
        descr=r"""Optimisation goal for the metric: "min" (default) or "max"."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'FitnessFunnelPlot'
    self.source = None
    self.sourceName = None
    self.index = None
    self.metric = None
    self.goal = 'min'

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for FitnessFunnelPlot "{self.name}".')
    self.sourceName = sourceNode.value

    indexNode = spec.findFirst('index')
    if indexNode is None or indexNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for FitnessFunnelPlot "{self.name}".')
    self.index = indexNode.value

    metricNode = spec.findFirst('metric')
    if metricNode is None or metricNode.value is None:
      self.raiseAnError(IOError, f'Missing <metric> node for FitnessFunnelPlot "{self.name}".')
    self.metric = metricNode.value

    goalNode = spec.findFirst('goal')
    if goalNode is not None and goalNode.value:
      goal = goalNode.value.strip().lower()
      if goal not in ('min', 'max'):
        self.raiseAnError(IOError, f'Unsupported <goal> "{goal}" for FitnessFunnelPlot "{self.name}". Use "min" or "max".')
      self.goal = goal

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for FitnessFunnelPlot "{self.name}".')
    self.source = src
    variables = self.source.getVars()
    missing = [var for var in (self.index, self.metric) if var not in variables]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by FitnessFunnelPlot "{self.name}".')

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'FitnessFunnelPlot "{self.name}" received an empty dataset; skipping.')
      return

    df[self.index] = df[self.index].astype(float)
    df[self.metric] = df[self.metric].astype(float)
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'FitnessFunnelPlot "{self.name}" found no generations in column "{self.index}".')
      return

    stats = self._compute_statistics(df, generations)
    if stats is None:
      self.raiseAWarning(f'FitnessFunnelPlot "{self.name}" could not compute statistics for metric "{self.metric}".')
      return

    best, mean, std = stats
    filename = self._createFilename(defaultName=f'{self.name}.png')
    directory = os.path.dirname(filename)
    if directory:
      os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(df[self.index], df[self.metric], s=18, alpha=0.35, c='#1f77b4', linewidths=0)
    ax.plot(generations, mean, color='#ff7f0e', linewidth=2.0, label='Mean fitness')
    lower = mean - std
    upper = mean + std
    ax.fill_between(generations, lower, upper, color='#ffbb78', alpha=0.35, label='±1σ')
    ax.plot(generations, best, color='#2ca02c', linewidth=2.4, label=f'Best ({self.goal})')

    ax.set_xlabel(self.index)
    ax.set_ylabel(self.metric)
    ax.set_title('Fitness Funnel')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

  def _compute_statistics(self, df, generations):
    best = []
    mean = []
    std = []
    for gen in generations:
      values = df[df[self.index] == gen][self.metric].to_numpy(dtype=float)
      values = values[np.isfinite(values)]
      if values.size == 0:
        best.append(np.nan)
        mean.append(np.nan)
        std.append(np.nan)
        continue
      mean.append(float(np.mean(values)))
      std.append(float(np.std(values)))
      if self.goal == 'max':
        best.append(float(np.max(values)))
      else:
        best.append(float(np.min(values)))
    best = np.asarray(best, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if not np.isfinite(best).any():
      return None
    return best, mean, std
