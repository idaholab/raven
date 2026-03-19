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
Visualize the density of dominated vs non-dominated samples for two-objective optimizers.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class DominanceHeatMapPlot(PlotInterface):
  """
  Generates a 2D heatmap comparing dominated and non-dominated samples of the final population.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly two objective columns to visualize in the heatmap."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Name of the generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('bins', contentType=InputTypes.IntegerType,
        descr=r"""Number of bins per objective axis (default 40)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'DominanceHeatMap'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
    self.bins = 40

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    objNode = spec.findFirst('objectives')
    if objNode is None:
      self.raiseAnError(IOError, f'Missing <objectives> node for DominanceHeatMapPlot "{self.name}".')
    self.objectives = [entry for entry in objNode.value if entry]
    if len(self.objectives) != 2:
      self.raiseAnError(IOError, f'DominanceHeatMapPlot "{self.name}" requires exactly two objectives.')
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, f'Missing <index> node for DominanceHeatMapPlot "{self.name}".')
    self.index = idxNode.value
    binsNode = spec.findFirst('bins')
    if binsNode is not None:
      self.bins = max(5, int(binsNode.value))

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'Source "{self.sourceName}" not found for DominanceHeatMapPlot "{self.name}".')
    available = self.source.getVars()
    missing = [var for var in list(self.objectives) + [self.index] if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" missing variable(s) {missing} required by DominanceHeatMapPlot "{self.name}".')

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; skipping DominanceHeatMapPlot "{self.name}".')
      return
    df[self.index] = df[self.index].astype(float)
    final_gen = df[self.index].max()
    subset = df[df[self.index] == final_gen]
    if subset.empty:
      self.raiseAWarning(f'No data found for final generation in DominanceHeatMapPlot "{self.name}".')
      return
    obj_vals = subset[self.objectives].astype(float).to_numpy()
    dominated_mask = self._dominated_mask(subset)
    heat_data, xedges, yedges = np.histogram2d(obj_vals[:, 0], obj_vals[:, 1],
                                               bins=self.bins)
    heat_data = heat_data.T  # align with imshow expectation

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    cmap = plt.get_cmap('Blues')
    im = ax.imshow(heat_data, extent=extent, origin='lower', cmap=cmap, aspect='auto')

    dominated_points = obj_vals[dominated_mask]
    nondominated_points = obj_vals[~dominated_mask]
    if len(dominated_points):
      ax.scatter(dominated_points[:, 0], dominated_points[:, 1],
                 s=20, c='#d62728', alpha=0.6, label='Dominated')
    if len(nondominated_points):
      ax.scatter(nondominated_points[:, 0], nondominated_points[:, 1],
                 s=28, edgecolors='k', linewidths=0.4,
                 c='#2ca02c', alpha=0.9, label='Rank 1')

    ax.set_xlabel(self.objectives[0])
    ax.set_ylabel(self.objectives[1])
    ax.set_title(f'Dominance heatmap (generation {int(final_gen)})')
    ax.grid(alpha=0.2)
    ax.legend()

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Sample density')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)

  @staticmethod
  def _dominated_mask(subset):
    if subset.empty:
      return np.zeros(0, dtype=bool)
    if 'rank' in subset.columns:
      try:
        ranks = subset['rank'].astype(float).to_numpy()
        return ~(ranks == 1)
      except (ValueError, TypeError):
        pass
    if 'accepted' in subset.columns:
      accepted = subset['accepted'].astype(str).str.lower().to_numpy()
      return ~(accepted == 'final')
    return np.zeros(len(subset), dtype=bool)
