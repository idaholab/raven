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
Probability map showing attainment surfaces aggregated across optimisation runs.
"""

import itertools

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class AttainmentSurfacePlot(PlotInterface):
  """
  Estimate empirical attainment probabilities for objective pairs using multiple runs.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject containing samples from one or more runs."""))
    objectives = InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Objective columns to evaluate. If more than two are provided, all pairwise combinations are plotted in a shared figure.""")
    spec.addSub(objectives)
    spec.addSub(InputData.parameterInputFactory('run_id', contentType=InputTypes.StringType,
        descr=r"""Optional column that distinguishes independent optimisation runs."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId). If provided, only the last
                  generation per run is considered unless <generation> is supplied."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> is supplied, optional explicit generation to analyse."""))
    spec.addSub(InputData.parameterInputFactory('levels', contentType=InputTypes.FloatListType,
        descr=r"""Optional probability contour levels (0-1). Defaults to 0.25,0.5,0.75."""))
    spec.addSub(InputData.parameterInputFactory('grid_size', contentType=InputTypes.IntegerType,
        descr=r"""Resolution of the attainment grid per axis (default 80)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'AttainmentSurfacePlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.objective_pairs = []
    self.runColumn = None
    self.index = None
    self.generation = None
    self.levels = (0.25, 0.5, 0.75)
    self.gridSize = 80

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for AttainmentSurfacePlot "{self.name}".')
    self.sourceName = src.value

    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'AttainmentSurfacePlot "{self.name}" requires at least two <objectives>.')
    objectives = [entry for entry in objNode.value if entry]
    if len(objectives) < 2:
      self.raiseAnError(IOError, f'AttainmentSurfacePlot "{self.name}" requires at least two objectives; got {len(objectives)}.')
    self.objectives = objectives
    if len(objectives) == 2:
      self.objective_pairs = [tuple(objectives)]
    else:
      self.objective_pairs = [tuple(pair) for pair in itertools.combinations(objectives, 2)]

    runNode = spec.findFirst('run_id')
    if runNode is not None and runNode.value:
      self.runColumn = runNode.value

    indexNode = spec.findFirst('index')
    if indexNode is not None and indexNode.value:
      self.index = indexNode.value
    generationNode = spec.findFirst('generation')
    if generationNode is not None and generationNode.value is not None:
      self.generation = float(generationNode.value)

    levelNode = spec.findFirst('levels')
    if levelNode is not None and levelNode.value:
      levels = [float(val) for val in levelNode.value]
      if not levels:
        self.raiseAnError(IOError, f'Empty <levels> specified for AttainmentSurfacePlot "{self.name}".')
      for lvl in levels:
        if lvl <= 0.0 or lvl >= 1.0:
          self.raiseAnError(IOError, f'Invalid attainment level "{lvl}" for AttainmentSurfacePlot "{self.name}".')
      self.levels = tuple(levels)

    gridNode = spec.findFirst('grid_size')
    if gridNode is not None and gridNode.value is not None:
      size = int(gridNode.value)
      if size < 10:
        self.raiseAnError(IOError, f'grid_size must be >= 10 for AttainmentSurfacePlot "{self.name}".')
      self.gridSize = size

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for AttainmentSurfacePlot "{self.name}".')
    available = src.getVars()
    needed = list(self.objectives)
    if self.runColumn:
      needed.append(self.runColumn)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by AttainmentSurfacePlot "{self.name}".')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; AttainmentSurfacePlot "{self.name}" skipped.')
      return
    subset = df.copy()
    if self.index:
      subset[self.index] = subset[self.index].astype(float)
    if self.runColumn:
      runs = sorted(subset[self.runColumn].unique())
    else:
      runs = [None]

    samples_per_pair = {pair: [] for pair in self.objective_pairs}
    for run_id in runs:
      if self.runColumn:
        run_subset = subset[subset[self.runColumn] == run_id]
      else:
        run_subset = subset
      if run_subset.empty:
        continue
      if self.index:
        if self.generation is not None:
          mask = np.isclose(run_subset[self.index].to_numpy(dtype=float), self.generation)
          run_subset = run_subset[mask]
        else:
          max_gen = run_subset[self.index].max()
          run_subset = run_subset[run_subset[self.index] == max_gen]
      if run_subset.empty:
        continue
      for pair in self.objective_pairs:
        arr = run_subset[list(pair)].astype(float).to_numpy()
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size == 0:
          continue
        samples_per_pair[pair].append(arr)

    valid_pairs = [pair for pair, arrays in samples_per_pair.items() if arrays]
    if not valid_pairs:
      self.raiseAWarning(f'AttainmentSurfacePlot "{self.name}" found no usable samples.')
      return

    fig, axes = self._create_figure(len(valid_pairs))
    contour_levels = np.append([0.0], list(self.levels) + [1.0])
    contour_mappable = None
    axes_with_data = []

    for ax, pair in zip(axes, valid_pairs):
      runs_for_pair = samples_per_pair[pair]
      all_samples = np.vstack(runs_for_pair)
      mins = np.nanmin(all_samples, axis=0)
      maxs = np.nanmax(all_samples, axis=0)
      if np.any(~np.isfinite(mins)) or np.any(~np.isfinite(maxs)):
        self.raiseAWarning(f'Non-finite objective bounds for AttainmentSurfacePlot "{self.name}" on pair {pair}; skipping subplot.')
        ax.set_visible(False)
        continue

      padding = 0.05 * (maxs - mins)
      padding[padding == 0.0] = 0.05
      x_vals = np.linspace(mins[0] - padding[0], maxs[0] + padding[0], self.gridSize)
      y_vals = np.linspace(mins[1] - padding[1], maxs[1] + padding[1], self.gridSize)
      x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')

      prob = np.zeros_like(x_mesh, dtype=float)
      for arr in runs_for_pair:
        dom_x = arr[:, 0][:, None, None] <= x_mesh
        dom_y = arr[:, 1][:, None, None] <= y_mesh
        attained = np.logical_and(dom_x, dom_y).any(axis=0)
        prob += attained.astype(float)
      prob /= len(runs_for_pair)

      contour = ax.contourf(x_mesh, y_mesh, prob, levels=contour_levels,
                            cmap='Blues', alpha=0.85, vmin=0.0, vmax=1.0)
      ax.contour(x_mesh, y_mesh, prob, levels=list(self.levels), colors='k', linewidths=0.8)
      ax.set_xlabel(pair[0])
      ax.set_ylabel(pair[1])
      ax.set_title(f'Attainment: {pair[0]} vs {pair[1]}')
      ax.grid(alpha=0.25)
      contour_mappable = contour if contour_mappable is None else contour_mappable
      axes_with_data.append(ax)

    if not axes_with_data:
      self.raiseAWarning(f'AttainmentSurfacePlot "{self.name}" could not render any subplots due to invalid data.')
      plt.close(fig)
      return

    if contour_mappable is not None:
      cbar = fig.colorbar(contour_mappable, ax=axes_with_data, fraction=0.035, pad=0.04)
      cbar.set_label('P(attained)')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)

  @staticmethod
  def _create_figure(num_panels):
    fig, axes = plt.subplots(1, num_panels, figsize=(6.4 * num_panels, 5.2))
    if not isinstance(axes, np.ndarray):
      axes = [axes]
    else:
      axes = axes.flatten().tolist()
    return fig, axes
