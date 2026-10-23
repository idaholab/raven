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
Stacked area chart for the rank composition through NSGA-II generations.
"""

import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class NSGARankHistoryPlot(PlotInterface):
  """
  Visualizes the fraction of the population belonging to each Pareto rank over time.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Variable denoting the generation identifier (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('normalize', contentType=InputTypes.BoolType,
        descr=r"""If true (default) plot rank proportions, otherwise plot absolute counts."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA Rank History'
    self.source = None
    self.sourceName = None
    self.index = None
    self.normalize = True

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, f'Missing <index> node for NSGARankHistoryPlot "{self.name}".')
    self.index = idxNode.value
    normalizeNode = spec.findFirst('normalize')
    if normalizeNode is not None:
      self.normalize = bool(normalizeNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for NSGARankHistoryPlot "{self.name}".')
    dataVars = self.source.getVars()
    required = [self.index, 'rank']
    missing = [var for var in required if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing variable(s) {} required by NSGARankHistoryPlot "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)

  def run(self):
    df = self.source.asDataset().to_dataframe()
    grouped = df.groupby([self.index, 'rank']).size().rename('count').reset_index()
    pivot = grouped.pivot(index=self.index, columns='rank', values='count').fillna(0.0)
    pivot = pivot.sort_index()

    if self.normalize:
      totals = pivot.sum(axis=1).replace(0, np.nan)
      pivot = pivot.div(totals, axis=0).fillna(0.0)
      ylabel = 'Population Fraction'
    else:
      ylabel = 'Population Count'

    ranks = pivot.columns.tolist()
    colors = plt.cm.get_cmap('tab10', len(ranks))
    fig, ax = plt.subplots()
    ax.stackplot(pivot.index, pivot.values.T, labels=[f'Rank {r}' for r in ranks],
                 colors=[colors(i) for i in range(len(ranks))])
    ax.set_xlabel(self.index)
    ax.set_ylabel(ylabel)
    ax.set_title('Rank Composition Over Time')
    ax.legend(loc='upper right')
    fig.tight_layout()

    filename = self._createFilename(defaultName=f'{self.name}.png')
    plt.savefig(filename)
    plt.close(fig)
