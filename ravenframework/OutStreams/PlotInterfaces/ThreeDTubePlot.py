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
Three-dimensional tube plot that highlights an optimizer's best-performing
trajectory through objective space.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ThreeDTubePlot(PlotInterface):
  """
  Selects a representative point per generation and renders a thick path (tube)
  through the three-objective space to convey convergence trends.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the history DataObject."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly three objective variable names used as X/Y/Z axes."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column used to order the path."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional rank filter applied before selecting the generation representative."""))
    spec.addSub(InputData.parameterInputFactory('selection_metric', contentType=InputTypes.StringType,
        descr=r"""Optional variable whose minimum determines the representative sample per generation. Defaults to sum(objectives)."""))
    spec.addSub(InputData.parameterInputFactory('line_width', contentType=InputTypes.FloatType,
        descr=r"""Base line width for the tube (default 3.0)."""))
    spec.addSub(InputData.parameterInputFactory('max_points', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on how many points to retain along the path (default 200)."""))
    spec.addSub(InputData.parameterInputFactory('view_angles', contentType=InputTypes.FloatListType,
        descr=r"""Optional elevation and azimuth (degrees)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ThreeDTubePlot'
    self.sourceName = None
    self.source = None
    self.objectives = []
    self.index = None
    self.rankFilter = None
    self.selectionMetric = None
    self.lineWidth = 3.0
    self.maxPoints = 200
    self.viewAngles = (25.0, -60.0)

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'ThreeDTubePlot "{self.name}" missing <source>.')
    self.sourceName = src.value

    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'ThreeDTubePlot "{self.name}" requires exactly three <objectives>.')
    objectives = [entry for entry in objNode.value if entry]
    if len(objectives) != 3:
      self.raiseAnError(IOError, f'ThreeDTubePlot "{self.name}" expected three objectives; got {len(objectives)}.')
    self.objectives = objectives

    indexNode = spec.findFirst('index')
    if indexNode is None or not indexNode.value:
      self.raiseAnError(IOError, f'ThreeDTubePlot "{self.name}" requires an <index> column.')
    self.index = indexNode.value

    rankNode = spec.findFirst('rank')
    if rankNode is not None and rankNode.value is not None:
      self.rankFilter = int(rankNode.value)

    metricNode = spec.findFirst('selection_metric')
    if metricNode is not None and metricNode.value:
      self.selectionMetric = metricNode.value

    widthNode = spec.findFirst('line_width')
    if widthNode is not None and widthNode.value:
      self.lineWidth = max(0.5, float(widthNode.value))

    maxNode = spec.findFirst('max_points')
    if maxNode is not None and maxNode.value:
      self.maxPoints = max(2, int(maxNode.value))

    viewNode = spec.findFirst('view_angles')
    if viewNode is not None and viewNode.value:
      values = [float(val) for val in viewNode.value]
      if len(values) != 2:
        self.raiseAnError(IOError, f'<view_angles> for ThreeDTubePlot "{self.name}" expects two floats.')
      self.viewAngles = (values[0], values[1])

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'Source "{self.sourceName}" not found for ThreeDTubePlot "{self.name}".')
    required = set(self.objectives + [self.index])
    if self.rankFilter is not None:
      required.add('rank')
    if self.selectionMetric:
      required.add(self.selectionMetric)
    missing = [var for var in required if var not in src.getVars()]
    if missing:
      self.raiseAnError(IOError, f'Source "{src.name}" missing variables {missing} required by ThreeDTubePlot "{self.name}".')
    self.source = src

  def _pickRepresentative(self, frame):
    """
      Return the representative row for a generation block.
    """
    candidates = frame[self.objectives].astype(float)
    if candidates.empty:
      return None
    if self.selectionMetric:
      metricVals = frame[self.selectionMetric].astype(float)
      idx = metricVals.idxmin()
    else:
      sums = candidates.sum(axis=1)
      idx = sums.idxmin()
    return frame.loc[idx, self.objectives].astype(float).to_numpy()

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source "{self.source.name}" empty; skipping ThreeDTubePlot "{self.name}".')
      return
    subset = df.copy()
    subset[self.index] = subset[self.index].astype(float)
    if self.rankFilter is not None and 'rank' in subset.columns:
      subset['rank'] = subset['rank'].astype(float)
      subset = subset[np.isclose(subset['rank'].to_numpy(dtype=float), float(self.rankFilter))]
    if subset.empty:
      self.raiseAWarning(f'No rows remained after filtering in ThreeDTubePlot "{self.name}".')
      return

    grouped = subset.groupby(self.index, sort=True)
    points = []
    generations = []
    for gen, frame in grouped:
      rep = self._pickRepresentative(frame)
      if rep is not None and np.all(np.isfinite(rep)):
        points.append(rep)
        generations.append(gen)
    if len(points) < 2:
      self.raiseAWarning(f'Need at least two generations with finite representatives for ThreeDTubePlot "{self.name}".')
      return

    points = np.array(points, dtype=float)
    generations = np.array(generations, dtype=float)
    if len(points) > self.maxPoints:
      idx = np.linspace(0, len(points) - 1, self.maxPoints, dtype=int)
      points = points[idx]
      generations = generations[idx]

    colors = (generations - generations.min()) / max(generations.ptp(), 1e-9)
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=self.viewAngles[0], azim=self.viewAngles[1])

    for i in range(len(points) - 1):
      seg = points[i:i+2]
      ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
              color=cmap(colors[i]),
              linewidth=self.lineWidth,
              solid_capstyle='round',
              alpha=0.95)

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=colors, cmap=cmap, s=36, depthshade=True, alpha=0.9)

    ax.set_xlabel(self.objectives[0])
    ax.set_ylabel(self.objectives[1])
    ax.set_zlabel(self.objectives[2])
    ax.set_title('3D tube trajectory (best-of-generation)')
    ax.grid(alpha=0.2)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label('Normalized generation')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)
