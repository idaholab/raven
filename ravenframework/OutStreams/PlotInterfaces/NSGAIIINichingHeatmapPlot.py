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
"""Heatmap of NSGA-III niche occupancy over generations."""

import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from .NSGAIIIPlotUtils import associatePoints, generateReferenceDirections, normalizeObjectives
from ...utils import InputData, InputTypes


class NSGAIIINichingHeatmapPlot(PlotInterface):
  """Tracks how many samples map to each reference direction throughout an NSGA-III run."""

  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable input for this plot.
      @ In, cls, class, the class for which we are retrieving the specification
      @ Out, spec, InputData.ParameterInput, input specification
    """
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""SolutionExport DataObject produced by NSGA-III."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Objectives handled by the optimizer (>= 3 recommended)."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional rank filter before computing niche occupancy."""))
    spec.addSub(InputData.parameterInputFactory('populationSize', contentType=InputTypes.IntegerType,
        descr=r"""Override for the NSGA-III population size used to build reference directions."""))
    spec.addSub(InputData.parameterInputFactory('maxGenerations', contentType=InputTypes.IntegerType,
        descr=r"""Optional limit on the number of generations to plot (most recent are kept)."""))
    spec.addSub(InputData.parameterInputFactory('normalizeRows', contentType=InputTypes.StringType,
        descr=r"""Set to 'true' to convert counts into per-generation fractions."""))
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'NSGA-III Niching Heatmap'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
    self.rank = None
    self.population = None
    self.maxGenerations = None
    self.normalizeRows = False

  def handleInput(self, spec):
    """
      Read the plot's input specification.
      @ In, spec, InputData.ParameterInput, the parsed input for this plot
      @ Out, None
    """
    super().handleInput(spec)
    source = spec.findFirst('source')
    if source is None:
      self.raiseAnError(IOError, f'Missing <source> for {self.name}.')
    self.sourceName = source.value
    objectives = spec.findFirst('objectives')
    if objectives is None or not objectives.value:
      self.raiseAnError(IOError, f'{self.name} requires <objectives>.')
    self.objectives = objectives.value
    index = spec.findFirst('index')
    if index is None:
      self.raiseAnError(IOError, f'Missing <index> node for {self.name}.')
    self.index = index.value
    rank = spec.findFirst('rank')
    if rank is not None:
      self.rank = int(rank.value)
    populationSize = spec.findFirst('populationSize')
    if populationSize is not None:
      self.population = max(1, int(populationSize.value))
    maxGenerations = spec.findFirst('maxGenerations')
    if maxGenerations is not None and maxGenerations.value:
      self.maxGenerations = max(1, int(maxGenerations.value))
    normalizeRows = spec.findFirst('normalizeRows')
    if normalizeRows is not None and isinstance(normalizeRows.value, str):
      self.normalizeRows = normalizeRows.value.strip().lower() in {'true', '1', 'yes'}

  def initialize(self, stepEntities):
    """
      Resolve the source DataObject and validate the requested variables are present.
      @ In, stepEntities, dict, entities available to this OutStream step
      @ Out, None
    """
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" for {self.name}.')
    availableVars = self.source.getVars()
    missing = [var for var in self.objectives + [self.index] if var not in availableVars]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by {self.name}.')

  def run(self):
    """
      Build the niche-occupancy heatmap: for each generation, associate the (normalized) samples with
      the NSGA-III reference directions and count how many map to each, then render the counts (or
      per-generation fractions) as a generations-by-directions heatmap with an active-directions trace.
      @ In, None
      @ Out, None
    """
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'{self.name} received an empty data set.')
      return
    if self.index not in df.columns:
      self.raiseAnError(IOError, f'Index column "{self.index}" not present for {self.name}.')
    generations = sorted(df[self.index].unique())
    if self.maxGenerations is not None and len(generations) > self.maxGenerations:
      generations = generations[-self.maxGenerations:]

    population = self.population or int(df[df[self.index] == generations[-1]].shape[0])
    population = max(population, 1)
    referenceDirs, _ = generateReferenceDirections(len(self.objectives), population)

    occupancy = []
    labels = []
    for generation in generations:
      subset = df[df[self.index] == generation]
      if subset.empty:
        continue
      if self.rank is not None and 'rank' in subset.columns:
        subset = subset[subset['rank'] == self.rank]
        if subset.empty:
          continue
      normalized = normalizeObjectives(subset[self.objectives].to_numpy(dtype=float))
      assoc, _ = associatePoints(normalized, referenceDirs)
      counts = np.bincount(assoc, minlength=referenceDirs.shape[0])
      occupancy.append(counts)
      labels.append(generation)

    if not occupancy:
      self.raiseAWarning(f'{self.name} found no generations with usable samples.')
      return

    matrix = np.vstack(occupancy)
    if self.normalizeRows:
      rowSums = matrix.sum(axis=1, keepdims=True)
      rowSums[rowSums == 0.0] = 1.0
      matrixToPlot = matrix / rowSums
      cbarLabel = 'Fraction of population'
    else:
      matrixToPlot = matrix
      cbarLabel = '# samples'

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrixToPlot, aspect='auto', cmap='viridis', origin='lower')
    ax.set_ylabel(self.index)
    ax.set_xlabel('Reference direction index')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    if matrix.shape[1] <= 15:
      ax.set_xticks(np.arange(matrix.shape[1]))
    else:
      ax.set_xticks(np.linspace(0, matrix.shape[1] - 1, 6))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbarLabel)

    coverage = np.count_nonzero(matrix > 0, axis=1)
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(labels)), coverage, color='white', linewidth=1.2, marker='o', markersize=4)
    ax2.set_ylabel('Active directions', color='white')
    ax2.set_ylim(0, referenceDirs.shape[0])
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(False)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
