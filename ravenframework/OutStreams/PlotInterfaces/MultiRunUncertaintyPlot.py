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
Plot mean and spread of an objective across multiple optimisation runs.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class MultiRunUncertaintyPlot(PlotInterface):
  """
  Aggregate repeated runs by generation and display mean plus confidence band.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject containing samples from multiple runs."""))
    spec.addSub(InputData.parameterInputFactory('run_id', contentType=InputTypes.StringType,
        descr=r"""Column identifying independent runs. If omitted, all samples treated as one run."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringType,
        descr=r"""Objective or metric column to summarise."""))
    spec.addSub(InputData.parameterInputFactory('quantiles', contentType=InputTypes.FloatListType,
        descr=r"""Optional lower and upper quantiles for shading (default 0.1,0.9)."""))
    spec.addSub(InputData.parameterInputFactory('goal', contentType=InputTypes.StringType,
        descr=r"""Optional goal ('min' or 'max') to annotate best-so-far trajectory."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'MultiRunUncertaintyPlot'
    self.source = None
    self.sourceName = None
    self.runColumn = None
    self.index = None
    self.metric = None
    self.quantiles = (0.1, 0.9)
    self.goal = None

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for MultiRunUncertaintyPlot "{self.name}".')
    self.sourceName = src.value

    runNode = spec.findFirst('run_id')
    if runNode is not None and runNode.value:
      self.runColumn = runNode.value

    idxNode = spec.findFirst('index')
    if idxNode is None or not idxNode.value:
      self.raiseAnError(IOError, f'Missing <index> node for MultiRunUncertaintyPlot "{self.name}".')
    self.index = idxNode.value

    metricNode = spec.findFirst('metric')
    if metricNode is None or not metricNode.value:
      self.raiseAnError(IOError, f'Missing <metric> node for MultiRunUncertaintyPlot "{self.name}".')
    self.metric = metricNode.value

    quantNode = spec.findFirst('quantiles')
    if quantNode is not None and quantNode.value:
      values = [float(val) for val in quantNode.value]
      if len(values) != 2:
        self.raiseAnError(IOError, f'<quantiles> must supply exactly two values for MultiRunUncertaintyPlot "{self.name}".')
      lower, upper = values
      if not (0 < lower < upper < 1):
        self.raiseAnError(IOError, f'Invalid quantiles {values} for MultiRunUncertaintyPlot "{self.name}".')
      self.quantiles = (lower, upper)

    goalNode = spec.findFirst('goal')
    if goalNode is not None and goalNode.value:
      goal = goalNode.value.strip().lower()
      if goal not in ('min', 'max'):
        self.raiseAnError(IOError, f'Unsupported goal "{goal}" for MultiRunUncertaintyPlot "{self.name}".')
      self.goal = goal

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for MultiRunUncertaintyPlot "{self.name}".')
    available = src.getVars()
    needed = [self.index, self.metric]
    if self.runColumn:
      needed.append(self.runColumn)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by MultiRunUncertaintyPlot "{self.name}".')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; MultiRunUncertaintyPlot "{self.name}" skipped.')
      return
    df = df.copy()
    df[self.index] = df[self.index].astype(float)
    df[self.metric] = df[self.metric].astype(float)
    if self.runColumn:
      df[self.runColumn] = df[self.runColumn].astype(str)
    else:
      df['_single_run'] = 'run0'
      self.runColumn = '_single_run'

    grouped = df.groupby([self.runColumn, self.index])[self.metric].agg(['mean']).reset_index()
    pivot = grouped.pivot(index=self.index, columns=self.runColumn, values='mean').sort_index()
    if pivot.empty:
      self.raiseAWarning(f'MultiRunUncertaintyPlot "{self.name}" found no aggregated data.')
      return
    generations = pivot.index.to_numpy(dtype=float)
    run_values = pivot.to_numpy(dtype=float)

    mean = np.nanmean(run_values, axis=1)
    lower = np.nanquantile(run_values, q=self.quantiles[0], axis=1)
    upper = np.nanquantile(run_values, q=self.quantiles[1], axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(generations, mean, color='tab:blue', label='Mean')
    ax.fill_between(generations, lower, upper, color='tab:blue', alpha=0.25,
                    label=f'Quantiles {self.quantiles[0]:.2f}-{self.quantiles[1]:.2f}')

    if self.goal:
      if self.goal == 'min':
        best_line = np.minimum.accumulate(mean)
      else:
        best_line = np.maximum.accumulate(mean)
      ax.plot(generations, best_line, color='tab:orange', linestyle='--',
              label=f'Best-so-far ({self.goal})')

    ax.set_xlabel(self.index)
    ax.set_ylabel(self.metric)
    ax.set_title('Multi-run objective summary')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
