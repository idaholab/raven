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
Three-dimensional vector/arrow plot that traces generation-to-generation motion
of an optimizer inside objective space.
"""

import math

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ThreeDVectorPlot(PlotInterface):
  """
  Aggregate consecutive generations into a centroid trajectory and visualize the
  direction of travel as 3D arrows.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject produced by the optimizer history export."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly three objective variable names used as X/Y/Z axes."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Monotonic generation identifier (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional integer Rank filter (e.g., 1 for Pareto front)."""))
    spec.addSub(InputData.parameterInputFactory('max_vectors', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on how many arrows to draw (default 50)."""))
    spec.addSub(InputData.parameterInputFactory('normalize_vectors', contentType=InputTypes.BoolType,
        descr=r"""If true (default), normalize arrow direction before scaling by <scale>."""))
    spec.addSub(InputData.parameterInputFactory('scale', contentType=InputTypes.FloatType,
        descr=r"""Length scaling factor applied after normalization (default 1.0)."""))
    spec.addSub(InputData.parameterInputFactory('view_angles', contentType=InputTypes.FloatListType,
        descr=r"""Optional elevation and azimuth (degrees) for the 3D camera view."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ThreeDVectorPlot'
    self.sourceName = None
    self.source = None
    self.objectives = []
    self.index = None
    self.rankFilter = None
    self.maxVectors = 50
    self.normalizeVectors = True
    self.scale = 1.0
    self.viewAngles = (25.0, -60.0)

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'ThreeDVectorPlot "{self.name}" missing <source>.')
    self.sourceName = src.value

    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'ThreeDVectorPlot "{self.name}" requires exactly three <objectives>.')
    objectives = [entry for entry in objNode.value if entry]
    if len(objectives) != 3:
      self.raiseAnError(IOError, f'ThreeDVectorPlot "{self.name}" expected three objectives; got {len(objectives)}.')
    self.objectives = objectives

    indexNode = spec.findFirst('index')
    if indexNode is None or not indexNode.value:
      self.raiseAnError(IOError, f'ThreeDVectorPlot "{self.name}" requires an <index> column.')
    self.index = indexNode.value

    rankNode = spec.findFirst('rank')
    if rankNode is not None and rankNode.value is not None:
      self.rankFilter = int(rankNode.value)

    maxNode = spec.findFirst('max_vectors')
    if maxNode is not None and maxNode.value:
      value = int(maxNode.value)
      if value < 1:
        self.raiseAnError(IOError, f'<max_vectors> for ThreeDVectorPlot "{self.name}" must be >= 1.')
      self.maxVectors = value

    normalizeNode = spec.findFirst('normalize_vectors')
    if normalizeNode is not None and normalizeNode.value is not None:
      self.normalizeVectors = bool(normalizeNode.value)

    scaleNode = spec.findFirst('scale')
    if scaleNode is not None and scaleNode.value:
      self.scale = float(scaleNode.value)

    viewNode = spec.findFirst('view_angles')
    if viewNode is not None and viewNode.value:
      values = [float(val) for val in viewNode.value]
      if len(values) != 2:
        self.raiseAnError(IOError, f'<view_angles> for ThreeDVectorPlot "{self.name}" expects two floats (elev, azim).')
      self.viewAngles = (values[0], values[1])

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'Source "{self.sourceName}" not found for ThreeDVectorPlot "{self.name}".')
    required = set(self.objectives + [self.index])
    if self.rankFilter is not None:
      required.add('rank')
    missing = [var for var in required if var not in src.getVars()]
    if missing:
      self.raiseAnError(IOError, f'Source "{src.name}" missing variables {missing} required by ThreeDVectorPlot "{self.name}".')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source "{self.source.name}" empty; skipping ThreeDVectorPlot "{self.name}".')
      return
    subset = df.copy()
    subset[self.index] = subset[self.index].astype(float)
    if self.rankFilter is not None and 'rank' in subset.columns:
      subset['rank'] = subset['rank'].astype(float)
      subset = subset[np.isclose(subset['rank'].to_numpy(dtype=float), float(self.rankFilter))]
    if subset.empty:
      self.raiseAWarning(f'No rows remained after filtering in ThreeDVectorPlot "{self.name}".')
      return

    grouped = subset.groupby(self.index, sort=True)
    centroids = grouped[self.objectives].mean()
    centroids = centroids.dropna()
    if len(centroids) < 2:
      self.raiseAWarning(f'Need at least two generations for ThreeDVectorPlot "{self.name}".')
      return

    points = centroids.to_numpy(dtype=float)
    generations = centroids.index.to_numpy(dtype=float)
    origins = points[:-1]
    vectors = points[1:] - points[:-1]

    if self.normalizeVectors:
      norms = np.linalg.norm(vectors, axis=1, keepdims=True)
      norms[norms == 0] = 1.0
      vectors = vectors / norms
    vectors = vectors * self.scale

    gen_mid = 0.5 * (generations[1:] + generations[:-1])
    if len(vectors) > self.maxVectors:
      step = math.ceil(len(vectors) / self.maxVectors)
      origins = origins[::step]
      vectors = vectors[::step]
      gen_mid = gen_mid[::step]

    fig = plt.figure(figsize=(6.6, 5.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=self.viewAngles[0], azim=self.viewAngles[1])

    cmap = plt.cm.plasma
    norm_colors = None
    if len(gen_mid) > 0:
      norm_colors = (gen_mid - gen_mid.min()) / max(gen_mid.ptp(), 1e-9)
    else:
      norm_colors = np.zeros(len(origins))
    quiv = ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
                     vectors[:, 0], vectors[:, 1], vectors[:, 2],
                     color=cmap(norm_colors), arrow_length_ratio=0.12, linewidth=1.5, alpha=0.95)

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=np.linspace(0.0, 1.0, len(points)),
                         cmap=cmap, s=24, depthshade=True, alpha=0.9)

    ax.set_xlabel(self.objectives[0])
    ax.set_ylabel(self.objectives[1])
    ax.set_zlabel(self.objectives[2])
    ax.set_title('3D vector field of generation drift')
    ax.grid(alpha=0.2)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label('Normalized generation')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)
