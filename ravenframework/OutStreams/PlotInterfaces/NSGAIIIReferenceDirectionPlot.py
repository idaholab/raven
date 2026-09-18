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
"""Visualise NSGA-III reference direction coverage for three-objective studies."""

import math

import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from .NSGAIIIPlotUtils import associate_points, generate_reference_directions, normalize_objectives
from ...utils import InputData, InputTypes


class NSGAIIIReferenceDirectionPlot(PlotInterface):
  """Displays the final-generation Pareto samples against the NSGA-III reference simplex."""

  @classmethod
  def getInputSpecification(cls):
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
    spec.addSub(InputData.parameterInputFactory('population_size', contentType=InputTypes.IntegerType,
        descr=r"""Override for the NSGA-III population size used to generate reference directions."""))
    spec.addSub(InputData.parameterInputFactory('top_directions', contentType=InputTypes.IntegerType,
        descr=r"""Number of niches to highlight in the occupancy bar chart (default 15)."""))
    return spec

  def __init__(self):
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
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None:
      self.raiseAnError(IOError, f'Missing <source> for {self.name}.')
    self.sourceName = src.value
    objectives = spec.findFirst('objectives')
    if objectives is None or len(objectives.value) != 3:
      self.raiseAnError(IOError, f'{self.name} requires exactly three <objectives>.')
    self.objectives = objectives.value
    idx = spec.findFirst('index')
    if idx is None:
      self.raiseAnError(IOError, f'Missing <index> node for {self.name}.')
    self.index = idx.value
    gen = spec.findFirst('generation')
    if gen is not None:
      self.generation = gen.value
    rank = spec.findFirst('rank')
    if rank is not None:
      self.rank = int(rank.value)
    pop = spec.findFirst('population_size')
    if pop is not None:
      self.population = max(1, int(pop.value))
    top = spec.findFirst('top_directions')
    if top is not None and top.value:
      self.topDirections = max(1, int(top.value))

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
    if self.index not in df.columns:
      self.raiseAnError(IOError, f'Index column "{self.index}" not found for {self.name}.')
    generation = self._select_generation(df)
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
    normalized = normalize_objectives(values)
    population = self.population or int(df[df[self.index] == generation].shape[0])
    population = max(population, normalized.shape[0])
    ref_dirs, simplex_dirs = generate_reference_directions(len(self.objectives), population)
    assoc, distances = associate_points(normalized, ref_dirs)
    counts = np.bincount(assoc, minlength=ref_dirs.shape[0])
    coverage = int(np.count_nonzero(counts))

    point_xy = self._to_simplex_coordinates(normalized)
    ref_xy = self._to_simplex_coordinates(simplex_dirs)

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])
    ax_simplex = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    self._draw_simplex(ax_simplex)
    if point_xy[0].size:
      scatter = ax_simplex.scatter(point_xy[0], point_xy[1], c=counts[assoc], cmap='viridis', s=45,
                                   edgecolors='k', linewidths=0.3)
      cbar = fig.colorbar(scatter, ax=ax_simplex, fraction=0.046, pad=0.04)
      cbar.set_label('# samples in niche')
    ax_simplex.scatter(ref_xy[0], ref_xy[1], marker='^', s=30, facecolors='none', edgecolors='tab:red', linewidths=0.8)
    ax_simplex.set_title(f'Generation {generation} (rank {self.rank or "all"})')
    ax_simplex.text(0.02, -0.08, f'Active directions: {coverage}/{len(ref_dirs)}', transform=ax_simplex.transAxes)

    top_idx = np.argsort(counts)[::-1][:min(self.topDirections, len(counts))]
    top_counts = counts[top_idx]
    if top_idx.size:
      y_pos = np.arange(top_idx.size)
      ax_bar.barh(y_pos, top_counts, color='tab:blue')
      ax_bar.set_yticks(y_pos)
      ax_bar.set_yticklabels([f'Ref {i}' for i in top_idx])
      ax_bar.invert_yaxis()
    ax_bar.set_xlabel('# samples')
    ax_bar.set_title('Most populated niches')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)

  def _select_generation(self, df):
    if self.generation is not None:
      return self.generation
    try:
      return df[self.index].max()
    except Exception:
      return df[self.index].iloc[-1]

  @staticmethod
  def _to_simplex_coordinates(matrix):
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
  def _draw_simplex(ax):
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0], [0.0, 0.0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black', linewidth=1.0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3.0) / 2.0 + 0.05)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
