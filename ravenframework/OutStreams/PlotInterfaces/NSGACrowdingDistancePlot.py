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
Diagnostic plot for crowding distance evolution per generation.
"""

import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class NSGACrowdingDistancePlot(PlotInterface):
  """
  Tracks distribution statistics for the crowding distance metric across generations.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier (typically batchId)."""))
    spec.addSub(InputData.parameterInputFactory('percentiles', contentType=InputTypes.StringListType,
        descr=r"""Optional list of percentile strings (0-100) to report. Defaults to 10,50,90."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA Crowding Distance Plot'
    self.source = None
    self.sourceName = None
    self.index = None
    self.percentiles = [10, 50, 90]

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, f'Missing <index> node for NSGACrowdingDistancePlot "{self.name}".')
    self.index = idxNode.value
    percNode = spec.findFirst('percentiles')
    if percNode is not None:
      try:
        self.percentiles = [float(p) for p in percNode.value]
      except ValueError as err:
        self.raiseAnError(IOError, f'Percentiles must be numeric for NSGACrowdingDistancePlot "{self.name}".', err)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for NSGACrowdingDistancePlot "{self.name}".')
    dataVars = self.source.getVars()
    required = [self.index, 'CD']
    missing = [var for var in required if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing variable(s) {} required by NSGACrowdingDistancePlot "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)

  def run(self):
    df = self.source.asDataset().to_dataframe()
    grouped = df.groupby(self.index)['CD']
    gens = sorted(grouped.groups.keys())
    if not gens:
      self.raiseAWarning(f'No generations found for NSGACrowdingDistancePlot "{self.name}".')
      return

    stats = {'min': [], 'max': [], 'mean': []}
    percentileSeries = {p: [] for p in self.percentiles}
    for gen in gens:
      values = grouped.get_group(gen).replace([np.inf, -np.inf], np.nan).dropna()
      if values.empty:
        stats['min'].append(np.nan)
        stats['max'].append(np.nan)
        stats['mean'].append(np.nan)
        for p in self.percentiles:
          percentileSeries[p].append(np.nan)
        continue
      stats['min'].append(values.min())
      stats['max'].append(values.max())
      stats['mean'].append(values.mean())
      for p in self.percentiles:
        percentileSeries[p].append(np.percentile(values, p))

    fig, ax = plt.subplots()
    ax.plot(gens, stats['mean'], label='Mean', color='C0')
    ax.fill_between(gens, stats['min'], stats['max'], color='C0', alpha=0.2, label='Min/Max Range')
    for idx, p in enumerate(self.percentiles):
      ax.plot(gens, percentileSeries[p], linestyle='--', color=f'C{idx+1}', label=f'P{int(p)}')
    ax.set_xlabel(self.index)
    ax.set_ylabel('Crowding Distance')
    ax.set_title('Crowding Distance Statistics')
    ax.legend(loc='best')
    fig.tight_layout()

    filename = self._createFilename(defaultName=f'{self.name}.png')
    plt.savefig(filename)
    plt.close(fig)

