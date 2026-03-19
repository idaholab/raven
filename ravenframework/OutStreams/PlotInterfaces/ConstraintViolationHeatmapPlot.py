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
Visualise spatial distribution of constraint violations for two variables.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ConstraintViolationHeatmapPlot(PlotInterface):
  """
  Heatmap showing average constraint violation magnitude across a 2D variable grid.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    axes = InputData.parameterInputFactory('axes', contentType=InputTypes.StringListType,
        descr=r"""Exactly two variable names to define the heatmap axes.""")
    spec.addSub(axes)
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional list of constraint evaluation columns (values <= 0 mean violation). Defaults to all columns starting with ConstraintEvaluation_."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> supplied, optionally select a specific generation to plot."""))
    spec.addSub(InputData.parameterInputFactory('bins', contentType=InputTypes.IntegerType,
        descr=r"""Number of bins per axis (default 40)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ConstraintViolationHeatmapPlot'
    self.source = None
    self.sourceName = None
    self.axes = []
    self.constraints = []
    self.useAll = True
    self.index = None
    self.generation = None
    self.bins = 40

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for ConstraintViolationHeatmapPlot "{self.name}".')
    self.sourceName = src.value

    axesNode = spec.findFirst('axes')
    if axesNode is None or len(axesNode.value) != 2:
      self.raiseAnError(IOError, f'ConstraintViolationHeatmapPlot "{self.name}" requires exactly two <axes>.')
    self.axes = [entry for entry in axesNode.value if entry]

    consNode = spec.findFirst('constraints')
    if consNode is not None and consNode.value:
      self.constraints = [entry for entry in consNode.value if entry]
      self.useAll = False

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    binsNode = spec.findFirst('bins')
    if binsNode is not None and binsNode.value is not None:
      self.bins = max(5, int(binsNode.value))

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for ConstraintViolationHeatmapPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.axes)
    if self.index:
      needed.append(self.index)
    if self.useAll:
      self.constraints = [var for var in available if var.startswith('ConstraintEvaluation_')]
    needed.extend(self.constraints)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by ConstraintViolationHeatmapPlot "{self.name}".')
    if not self.constraints:
      self.raiseAnError(IOError, f'ConstraintViolationHeatmapPlot "{self.name}" found no constraint columns to evaluate.')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; ConstraintViolationHeatmapPlot "{self.name}" skipped.')
      return
    subset = df.copy()
    if self.index:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if subset.empty:
      self.raiseAWarning(f'ConstraintViolationHeatmapPlot "{self.name}" had no rows after filtering.')
      return

    x = subset[self.axes[0]].astype(float).to_numpy()
    y = subset[self.axes[1]].astype(float).to_numpy()
    violation = np.zeros(len(subset), dtype=float)
    for cons in self.constraints:
      vals = subset[cons].astype(float).to_numpy()
      violation += np.clip(-vals, a_min=0.0, a_max=None)
    if not np.isfinite(violation).any():
      self.raiseAWarning(f'ConstraintViolationHeatmapPlot "{self.name}" found no finite violations.')
      return

    heat, xedges, yedges = np.histogram2d(x, y, bins=self.bins)
    sumViol, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=violation)
    with np.errstate(invalid='ignore', divide='ignore'):
      meanViolation = np.where(heat > 0, sumViol / heat, 0.0)
    meanViolation = meanViolation.T

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    cmap = plt.get_cmap('Reds')
    im = ax.imshow(meanViolation, extent=extent, origin='lower', cmap=cmap, aspect='auto')
    ax.set_xlabel(self.axes[0])
    ax.set_ylabel(self.axes[1])
    ax.set_title('Constraint violation intensity')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean violation magnitude')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
