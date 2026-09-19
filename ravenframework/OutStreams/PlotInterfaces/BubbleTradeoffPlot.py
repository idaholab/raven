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
Static two- or three-objective scatter plot that encodes an additional metric as bubble size.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class BubbleTradeoffPlot(PlotInterface):
  """
  Render a two-objective trade-off scatter in which each sample's bubble area
  represents an additional metric (for example, constraint slack or a third objective).
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject produced by the optimizer."""))
    objectives = InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Two or three objective variable names to use for the scatter axes. Three objectives trigger a 3D scatter view.""")
    spec.addSub(objectives)
    spec.addSub(InputData.parameterInputFactory('size', contentType=InputTypes.StringType,
        descr=r"""Variable whose magnitude should be mapped to bubble size."""))
    spec.addSub(InputData.parameterInputFactory('color', contentType=InputTypes.StringType,
        descr=r"""Optional variable used to color points (numeric or named colors)."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId). If provided, the plot
                  uses the maximum value observed unless <generation> is also supplied."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> is provided, optional explicit generation value to display."""))
    spec.addSub(InputData.parameterInputFactory('sizeBounds', contentType=InputTypes.FloatListType,
        descr=r"""Optional minimum and maximum marker areas (in points^2). Defaults to 50,500."""))
    spec.addSub(InputData.parameterInputFactory('normalize', contentType=InputTypes.BoolType,
        descr=r"""If true (default) bubble areas are normalized between the provided <sizeBounds>."""))
    spec.addSub(InputData.parameterInputFactory('viewAngles', contentType=InputTypes.FloatListType,
        descr=r"""Optional elevation and azimuth (degrees) for 3D mode. Defaults to 25,-60."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'BubbleTradeoffPlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.sizeVar = None
    self.colorVar = None
    self.index = None
    self.generation = None
    self.sizeBounds = (50.0, 500.0)
    self.normalize = True
    self.viewAngles = (25.0, -60.0)

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for BubbleTradeoffPlot "{self.name}".')
    self.sourceName = src.value

    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'BubbleTradeoffPlot "{self.name}" requires two or three <objectives>.')
    objectives = [entry for entry in objNode.value if entry]
    if len(objectives) not in (2, 3):
      self.raiseAnError(IOError, f'BubbleTradeoffPlot "{self.name}" expected two or three objectives; got {len(objectives)}.')
    self.objectives = objectives

    sizeNode = spec.findFirst('size')
    if sizeNode is None or not sizeNode.value:
      self.raiseAnError(IOError, f'Missing <size> node for BubbleTradeoffPlot "{self.name}".')
    self.sizeVar = sizeNode.value

    colorNode = spec.findFirst('color')
    if colorNode is not None and colorNode.value:
      self.colorVar = colorNode.value

    indexNode = spec.findFirst('index')
    if indexNode is not None and indexNode.value:
      self.index = indexNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    boundsNode = spec.findFirst('sizeBounds')
    if boundsNode is not None and boundsNode.value:
      values = [float(val) for val in boundsNode.value]
      if len(values) != 2:
        self.raiseAnError(IOError, f'<sizeBounds> for BubbleTradeoffPlot "{self.name}" must provide exactly two entries.')
      low, high = values
      if low <= 0 or high <= 0 or low >= high:
        self.raiseAnError(IOError, f'Invalid <sizeBounds> {values} for BubbleTradeoffPlot "{self.name}".')
      self.sizeBounds = (low, high)

    normalizeNode = spec.findFirst('normalize')
    if normalizeNode is not None and normalizeNode.value is not None:
      self.normalize = bool(normalizeNode.value)

    viewNode = spec.findFirst('viewAngles')
    if viewNode is not None and viewNode.value:
      values = [float(val) for val in viewNode.value]
      if len(values) != 2:
        self.raiseAnError(IOError, f'<viewAngles> for BubbleTradeoffPlot "{self.name}" expects two floats (elevation, azimuth).')
      self.viewAngles = (values[0], values[1])

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for BubbleTradeoffPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.objectives) + [self.sizeVar]
    if self.colorVar:
      needed.append(self.colorVar)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by BubbleTradeoffPlot "{self.name}".')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; BubbleTradeoffPlot "{self.name}" skipped.')
      return
    subset = df.copy()
    if self.index:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
        if subset.empty:
          self.raiseAWarning(f'No rows matched generation {self.generation} in BubbleTradeoffPlot "{self.name}".')
      else:
        maxGen = subset[self.index].max()
        subset = subset[subset[self.index] == maxGen]
        if subset.empty:
          self.raiseAWarning(f'No rows found for generation {max_gen} in BubbleTradeoffPlot "{self.name}".')
    if subset.empty:
      return

    x = subset[self.objectives[0]].astype(float).to_numpy()
    y = subset[self.objectives[1]].astype(float).to_numpy()
    z = None
    is3d = len(self.objectives) == 3
    if is3d:
      z = subset[self.objectives[2]].astype(float).to_numpy()
    sizeRaw = subset[self.sizeVar].astype(float).to_numpy()
    finiteMask = np.isfinite(sizeRaw)
    if not finiteMask.any():
      self.raiseAWarning(f'No finite values in <size> variable "{self.sizeVar}" for BubbleTradeoffPlot "{self.name}".')
      return
    sizeClean = np.zeros_like(sizeRaw, dtype=float)
    sizeClean[finiteMask] = sizeRaw[finiteMask]
    if self.normalize:
      minVal = sizeClean[finiteMask].min()
      maxVal = sizeClean[finiteMask].max()
      if math.isclose(minVal, maxVal):
        norm = np.ones_like(sizeClean) * 0.5
      else:
        norm = (sizeClean - minVal) / (maxVal - minVal)
        norm = np.clip(norm, 0.0, 1.0)
      minArea, maxArea = self.sizeBounds
      sizes = minArea + norm * (maxArea - minArea)
    else:
      sizes = np.clip(sizeClean, self.sizeBounds[0], self.sizeBounds[1])

    colorArg = None
    cmap = None
    if self.colorVar:
      series = subset[self.colorVar]
      try:
        colorVals = series.astype(float).to_numpy()
        colorArg = colorVals
        cmap = 'viridis'
      except (ValueError, TypeError):
        colorArg = series.astype(str).str.strip().replace('', np.nan).to_numpy()

    if is3d:
      fig = plt.figure(figsize=(6.4, 5.2))
      ax = fig.add_subplot(111, projection='3d')
      ax.view_init(elev=self.viewAngles[0], azim=self.viewAngles[1])
    else:
      fig, ax = plt.subplots(figsize=(6.4, 5.2))
    scatterKwargs = {'s': sizes, 'edgecolors': 'k', 'linewidths': 0.4, 'alpha': 0.85}
    if colorArg is not None:
      scatterKwargs['c'] = colorArg
      if cmap:
        scatterKwargs['cmap'] = cmap
    if is3d:
      sc = ax.scatter(x, y, z, **scatterKwargs)
    else:
      sc = ax.scatter(x, y, **scatterKwargs)
    ax.set_xlabel(self.objectives[0])
    ax.set_ylabel(self.objectives[1])
    if is3d:
      ax.set_zlabel(self.objectives[2])
    ax.set_title('Bubble trade-off')
    ax.grid(alpha=0.25)

    if self.colorVar and cmap and np.isfinite(colorArg).any():
      cbar = fig.colorbar(sc, ax=ax)
      cbar.set_label(self.colorVar)

    legendHandle = plt.Line2D([], [], marker='o', color='w', markerfacecolor='#1f77b4',
                               markeredgecolor='k', markersize=8, label=self.sizeVar)
    ax.legend(handles=[legendHandle], title='Bubble encodes', loc='best')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
