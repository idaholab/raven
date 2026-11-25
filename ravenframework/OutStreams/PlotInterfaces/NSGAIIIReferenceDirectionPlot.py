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
"""Visualize NSGA-III reference-direction coverage for three-objective studies."""

import math

import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from .NSGAIIIPlotUtils import associatePoints, generateReferenceDirections, normalizeObjectives
from ...utils import InputData, InputTypes


class NSGAIIIReferenceDirectionPlot(PlotInterface):
  """Displays the final-generation Pareto samples against the NSGA-III reference simplex."""

  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable input for this plot.
      @ In, cls, class, the class for which we are retrieving the specification
      @ Out, spec, InputData.ParameterInput, input specification
    """
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""SolutionExport DataObject produced by the NSGA-III optimizer."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly three objectives to project onto the ternary simplex."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Name of the generation column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""Optional explicit generation to analyse. Defaults to the latest."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional rank filter (applied when the data set provides a 'rank' column)."""))
    spec.addSub(InputData.parameterInputFactory('populationSize', contentType=InputTypes.IntegerType,
        descr=r"""Override for the NSGA-III population size used to generate reference directions."""))
    spec.addSub(InputData.parameterInputFactory('topDirections', contentType=InputTypes.IntegerType,
        descr=r"""Number of niches to highlight in the occupancy bar chart (default 15)."""))
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'NSGA-III Reference Plot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
    self.rank = None
    self.generation = None
    self.population = None
    self.topDirections = 15

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
    if objectives is None or len(objectives.value) != 3:
      self.raiseAnError(IOError, f'{self.name} requires exactly three <objectives>.')
    self.objectives = objectives.value
    index = spec.findFirst('index')
    if index is None:
      self.raiseAnError(IOError, f'Missing <index> node for {self.name}.')
    self.index = index.value
    generation = spec.findFirst('generation')
    if generation is not None:
      self.generation = generation.value
    rank = spec.findFirst('rank')
    if rank is not None:
      self.rank = int(rank.value)
    populationSize = spec.findFirst('populationSize')
    if populationSize is not None:
      self.population = max(1, int(populationSize.value))
    topDirections = spec.findFirst('topDirections')
    if topDirections is not None and topDirections.value:
      self.topDirections = max(1, int(topDirections.value))

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
      Project the selected generation's (normalized) samples onto the ternary simplex, overlay the
      NSGA-III reference directions, color each sample by its niche occupancy, and draw a companion
      bar chart of the most populated niches.
      @ In, None
      @ Out, None
    """
    df = self.source.asDataset().to_dataframe()
    if self.index not in df.columns:
      self.raiseAnError(IOError, f'Index column "{self.index}" not found for {self.name}.')
    generation = self._selectGeneration(df)
    subset = df[df[self.index] == generation]
    if subset.empty:
      self.raiseAWarning(f'No samples for generation {generation}; skipping {self.name}.')
      return
    if self.rank is not None and 'rank' in subset.columns:
      subset = subset[subset['rank'] == self.rank]
      if subset.empty:
        self.raiseAWarning(f'No samples with rank {self.rank} in generation {generation}; falling back to full generation.')
        subset = df[df[self.index] == generation]

    values = subset[self.objectives].to_numpy(dtype=float)
    normalized = normalizeObjectives(values)
    population = self.population or int(df[df[self.index] == generation].shape[0])
    population = max(population, normalized.shape[0])
    referenceDirs, simplexDirs = generateReferenceDirections(len(self.objectives), population)
    assoc, _ = associatePoints(normalized, referenceDirs)
    counts = np.bincount(assoc, minlength=referenceDirs.shape[0])
    coverage = int(np.count_nonzero(counts))

    pointXY = self._toSimplexCoordinates(normalized)
    refXY = self._toSimplexCoordinates(simplexDirs)

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])
    axSimplex = fig.add_subplot(gs[0, 0])
    axBar = fig.add_subplot(gs[0, 1])

    self._drawSimplex(axSimplex)
    if pointXY[0].size:
      scatter = axSimplex.scatter(pointXY[0], pointXY[1], c=counts[assoc], cmap='viridis', s=45,
                                  edgecolors='k', linewidths=0.3)
      cbar = fig.colorbar(scatter, ax=axSimplex, fraction=0.046, pad=0.04)
      cbar.set_label('# samples in niche')
    axSimplex.scatter(refXY[0], refXY[1], marker='^', s=30, facecolors='none', edgecolors='tab:red', linewidths=0.8)
    axSimplex.set_title(f'Generation {generation} (rank {self.rank or "all"})')
    axSimplex.text(0.02, -0.08, f'Active directions: {coverage}/{len(referenceDirs)}', transform=axSimplex.transAxes)

    topIdx = np.argsort(counts)[::-1][:min(self.topDirections, len(counts))]
    topCounts = counts[topIdx]
    if topIdx.size:
      yPos = np.arange(topIdx.size)
      axBar.barh(yPos, topCounts, color='tab:blue')
      axBar.set_yticks(yPos)
      axBar.set_yticklabels([f'Ref {i}' for i in topIdx])
      axBar.invert_yaxis()
    axBar.set_xlabel('# samples')
    axBar.set_title('Most populated niches')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)

  def _selectGeneration(self, df):
    """
      Pick the generation to plot: the explicit <generation> if given, else the latest.
      @ In, df, pandas.DataFrame, the source data
      @ Out, generation, object, the generation index value to plot
    """
    if self.generation is not None:
      return self.generation
    try:
      return df[self.index].max()
    except (ValueError, TypeError):
      return df[self.index].iloc[-1]

  @staticmethod
  def _toSimplexCoordinates(matrix):
    """
      Convert nonnegative 3-objective rows to barycentric (x, y) coordinates on the unit triangle.
      @ In, matrix, np.ndarray, (nPoints, 3) nonnegative values
      @ Out, (x, y), tuple(np.ndarray, np.ndarray), simplex coordinates (empty arrays if no points)
    """
    if matrix.size == 0:
      return np.array([]), np.array([])
    matrix = np.clip(matrix, 0.0, None)
    sums = matrix.sum(axis=1, keepdims=True)
    sums[sums == 0.0] = 1.0
    bary = matrix / sums
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = bary[:, 2] * (math.sqrt(3.0) / 2.0)
    return x, y

  @staticmethod
  def _drawSimplex(ax):
    """
      Draw the bounding ternary-simplex triangle on the given axis.
      @ In, ax, matplotlib.axes.Axes, the axis to draw on
      @ Out, None
    """
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0], [0.0, 0.0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black', linewidth=1.0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3.0) / 2.0 + 0.05)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
