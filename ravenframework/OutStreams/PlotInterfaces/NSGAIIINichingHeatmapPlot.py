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
from .NSGAIIIPlotUtils import associate_points, generate_reference_directions, normalize_objectives
from ...utils import InputData, InputTypes


class NSGAIIINichingHeatmapPlot(PlotInterface):
  """Tracks how many samples map to each reference direction throughout an NSGA-III run."""

  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""SolutionExport DataObject produced by NSGA-III."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Objectives handled by the optimizer (>= 3 recommended)."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional rank filter before computing niche occupancy."""))
    spec.addSub(InputData.parameterInputFactory('population_size', contentType=InputTypes.IntegerType,
        descr=r"""Override for the NSGA-III population size used to build reference directions."""))
    spec.addSub(InputData.parameterInputFactory('max_generations', contentType=InputTypes.IntegerType,
        descr=r"""Optional limit on the number of generations to plot (most recent are kept)."""))
    spec.addSub(InputData.parameterInputFactory('normalize_rows', contentType=InputTypes.StringType,
        descr=r"""Set to 'true' to convert counts into per-generation fractions."""))
    return spec

  def __init__(self):
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
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None:
      self.raiseAnError(IOError, f'Missing <source> for {self.name}.')
    self.sourceName = src.value
    objectives = spec.findFirst('objectives')
    if objectives is None or not objectives.value:
      self.raiseAnError(IOError, f'{self.name} requires <objectives>.')
    self.objectives = objectives.value
    idx = spec.findFirst('index')
    if idx is None:
      self.raiseAnError(IOError, f'Missing <index> node for {self.name}.')
    self.index = idx.value
    rank = spec.findFirst('rank')
    if rank is not None:
      self.rank = int(rank.value)
    pop = spec.findFirst('population_size')
    if pop is not None:
      self.population = max(1, int(pop.value))
    max_gen = spec.findFirst('max_generations')
    if max_gen is not None and max_gen.value:
      self.maxGenerations = max(1, int(max_gen.value))
    norm = spec.findFirst('normalize_rows')
    if norm is not None and isinstance(norm.value, str):
      self.normalizeRows = norm.value.strip().lower() in {'true', '1', 'yes'}

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" for {self.name}.')
    vars_available = self.source.getVars()
    missing = [var for var in self.objectives + [self.index] if var not in vars_available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by {self.name}.')

  def run(self):
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
    ref_dirs, _ = generate_reference_directions(len(self.objectives), population)

    occupancy = []
    labels = []
    for gen in generations:
      subset = df[df[self.index] == gen]
      if subset.empty:
        continue
      if self.rank is not None and 'rank' in subset.columns:
        subset = subset[subset['rank'] == self.rank]
        if subset.empty:
          continue
      normalized = normalize_objectives(subset[self.objectives].to_numpy(dtype=float))
      assoc, _ = associate_points(normalized, ref_dirs)
      counts = np.bincount(assoc, minlength=ref_dirs.shape[0])
      occupancy.append(counts)
      labels.append(gen)

    if not occupancy:
      self.raiseAWarning(f'{self.name} found no generations with usable samples.')
      return

    matrix = np.vstack(occupancy)
    if self.normalizeRows:
      row_sums = matrix.sum(axis=1, keepdims=True)
      row_sums[row_sums == 0.0] = 1.0
      matrix_to_plot = matrix / row_sums
      cbar_label = 'Fraction of population'
    else:
      matrix_to_plot = matrix
      cbar_label = '# samples'

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrix_to_plot, aspect='auto', cmap='viridis', origin='lower')
    ax.set_ylabel(self.index)
    ax.set_xlabel('Reference direction index')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    if matrix.shape[1] <= 15:
      ax.set_xticks(np.arange(matrix.shape[1]))
    else:
      ax.set_xticks(np.linspace(0, matrix.shape[1] - 1, 6))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    coverage = np.count_nonzero(matrix > 0, axis=1)
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(labels)), coverage, color='white', linewidth=1.2, marker='o', markersize=4)
    ax2.set_ylabel('Active directions', color='white')
    ax2.set_ylim(0, ref_dirs.shape[0])
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(False)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
