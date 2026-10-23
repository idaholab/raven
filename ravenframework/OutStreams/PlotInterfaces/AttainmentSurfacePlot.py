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
    spec.addSub(InputData.parameterInputFactory('runId', contentType=InputTypes.StringType,
        descr=r"""Optional column that distinguishes independent optimisation runs."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId). If provided, only the last
                  generation per run is considered unless <generation> is supplied."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> is supplied, optional explicit generation to analyse."""))
    spec.addSub(InputData.parameterInputFactory('levels', contentType=InputTypes.FloatListType,
        descr=r"""Optional probability contour levels (0-1). Defaults to 0.25,0.5,0.75."""))
    spec.addSub(InputData.parameterInputFactory('gridSize', contentType=InputTypes.IntegerType,
        descr=r"""Resolution of the attainment grid per axis (default 80)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'AttainmentSurfacePlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.objectivePairs = []
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
      self.objectivePairs = [tuple(objectives)]
    else:
      self.objectivePairs = [tuple(pair) for pair in itertools.combinations(objectives, 2)]

    runNode = spec.findFirst('runId')
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

    gridNode = spec.findFirst('gridSize')
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

    samplesPerPair = {pair: [] for pair in self.objectivePairs}
    for runId in runs:
      if self.runColumn:
        runSubset = subset[subset[self.runColumn] == runId]
      else:
        runSubset = subset
      if runSubset.empty:
        continue
      if self.index:
        if self.generation is not None:
          mask = np.isclose(runSubset[self.index].to_numpy(dtype=float), self.generation)
          runSubset = runSubset[mask]
        else:
          maxGen = runSubset[self.index].max()
          runSubset = runSubset[runSubset[self.index] == maxGen]
      if runSubset.empty:
        continue
      for pair in self.objectivePairs:
        arr = runSubset[list(pair)].astype(float).to_numpy()
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size == 0:
          continue
        samplesPerPair[pair].append(arr)

    validPairs = [pair for pair, arrays in samplesPerPair.items() if arrays]
    if not validPairs:
      self.raiseAWarning(f'AttainmentSurfacePlot "{self.name}" found no usable samples.')
      return

    fig, axes = self._createFigure(len(validPairs))
    contourLevels = np.append([0.0], list(self.levels) + [1.0])
    contourMappable = None
    axesWithData = []

    for ax, pair in zip(axes, validPairs):
      runsForPair = samplesPerPair[pair]
      allSamples = np.vstack(runsForPair)
      mins = np.nanmin(allSamples, axis=0)
      maxs = np.nanmax(allSamples, axis=0)
      if np.any(~np.isfinite(mins)) or np.any(~np.isfinite(maxs)):
        self.raiseAWarning(f'Non-finite objective bounds for AttainmentSurfacePlot "{self.name}" on pair {pair}; skipping subplot.')
        ax.set_visible(False)
        continue

      padding = 0.05 * (maxs - mins)
      padding[padding == 0.0] = 0.05
      xVals = np.linspace(mins[0] - padding[0], maxs[0] + padding[0], self.gridSize)
      yVals = np.linspace(mins[1] - padding[1], maxs[1] + padding[1], self.gridSize)
      xMesh, yMesh = np.meshgrid(xVals, yVals, indexing='xy')

      prob = np.zeros_like(xMesh, dtype=float)
      for arr in runsForPair:
        domX = arr[:, 0][:, None, None] <= xMesh
        domY = arr[:, 1][:, None, None] <= yMesh
        attained = np.logical_and(domX, domY).any(axis=0)
        prob += attained.astype(float)
      prob /= len(runsForPair)

      contour = ax.contourf(xMesh, yMesh, prob, levels=contourLevels,
                            cmap='Blues', alpha=0.85, vmin=0.0, vmax=1.0)
      ax.contour(xMesh, yMesh, prob, levels=list(self.levels), colors='k', linewidths=0.8)
      ax.set_xlabel(pair[0])
      ax.set_ylabel(pair[1])
      ax.set_title(f'Attainment: {pair[0]} vs {pair[1]}')
      ax.grid(alpha=0.25)
      contourMappable = contour if contourMappable is None else contourMappable
      axesWithData.append(ax)

    if not axesWithData:
      self.raiseAWarning(f'AttainmentSurfacePlot "{self.name}" could not render any subplots due to invalid data.')
      plt.close(fig)
      return

    fig.tight_layout(rect=[0, 0, 0.94, 1])
    if contourMappable is not None:
      cbar = self._addSharedColorbar(fig, axesWithData, contourMappable)
      if cbar is not None:
        cbar.set_label('P(attained)')
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)

  @staticmethod
  def _createFigure(numPanels):
    fig, axes = plt.subplots(1, numPanels, figsize=(6.4 * numPanels, 5.2))
    if not isinstance(axes, np.ndarray):
      axes = [axes]
    else:
      axes = axes.flatten().tolist()
    return fig, axes

  @staticmethod
  def _addSharedColorbar(fig, axes, mappable):
    if fig is None or mappable is None or not axes:
      return None
    axesList = [ax for ax in axes if ax.get_visible()]
    if not axesList:
      return None
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axesList]
    maxRight = max(pos.x1 for pos in positions)
    minBottom = min(pos.y0 for pos in positions)
    maxTop = max(pos.y1 for pos in positions)
    pad = 0.025
    width = 0.02
    left = maxRight + pad
    if left + width > 0.98:
      width = max(0.01, 0.98 - left)
      left = 0.98 - width
    if width <= 0:
      return None
    cax = fig.add_axes([left, minBottom, width, maxTop - minBottom])
    return fig.colorbar(mappable, cax=cax)
