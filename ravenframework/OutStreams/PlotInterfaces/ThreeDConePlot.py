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
Three-dimensional cone plot for visualizing dominance cones or reference spreads
originating from a selected apex (typically the utopia point).
"""

import math

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ThreeDConePlot(PlotInterface):
  """
  Render cones from an apex toward top-performing samples to highlight their
  dominance/coverage directions in the three-objective space.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject containing objective evaluations."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly three objective variable names used as X/Y/Z axes."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier used to pick the final generation."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""Optional explicit generation value. Defaults to the maximum <index>. """))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional rank filter before selecting cones (e.g., 1)."""))
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringType,
        descr=r"""Optional variable used to sort cones (descending). Defaults to vector norm."""))
    spec.addSub(InputData.parameterInputFactory('top_k', contentType=InputTypes.IntegerType,
        descr=r"""Number of cones to draw (default 8)."""))
    spec.addSub(InputData.parameterInputFactory('apex', contentType=InputTypes.FloatListType,
        descr=r"""Optional apex coordinates (three floats). Defaults to the origin."""))
    spec.addSub(InputData.parameterInputFactory('angle', contentType=InputTypes.FloatType,
        descr=r"""Cone half-angle in degrees (default 12)."""))
    spec.addSub(InputData.parameterInputFactory('height_scale', contentType=InputTypes.FloatType,
        descr=r"""Scalar applied to each cone height (default 1.0)."""))
    spec.addSub(InputData.parameterInputFactory('view_angles', contentType=InputTypes.FloatListType,
        descr=r"""Optional elevation and azimuth (degrees)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ThreeDConePlot'
    self.sourceName = None
    self.source = None
    self.objectives = []
    self.index = None
    self.generation = None
    self.rankFilter = None
    self.metric = None
    self.topK = 8
    self.apex = np.zeros(3, dtype=float)
    self.angle = 12.0
    self.heightScale = 1.0
    self.viewAngles = (25.0, -60.0)

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'ThreeDConePlot "{self.name}" missing <source>.')
    self.sourceName = src.value

    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'ThreeDConePlot "{self.name}" requires exactly three <objectives>.')
    objectives = [entry for entry in objNode.value if entry]
    if len(objectives) != 3:
      self.raiseAnError(IOError, f'ThreeDConePlot "{self.name}" expected three objectives; got {len(objectives)}.')
    self.objectives = objectives

    indexNode = spec.findFirst('index')
    if indexNode is not None and indexNode.value:
      self.index = indexNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    rankNode = spec.findFirst('rank')
    if rankNode is not None and rankNode.value is not None:
      self.rankFilter = int(rankNode.value)

    metricNode = spec.findFirst('metric')
    if metricNode is not None and metricNode.value:
      self.metric = metricNode.value

    topNode = spec.findFirst('top_k')
    if topNode is not None and topNode.value:
      value = int(topNode.value)
      if value < 1:
        self.raiseAnError(IOError, f'<top_k> for ThreeDConePlot "{self.name}" must be >= 1.')
      self.topK = value

    apexNode = spec.findFirst('apex')
    if apexNode is not None and apexNode.value:
      values = [float(val) for val in apexNode.value]
      if len(values) != 3:
        self.raiseAnError(IOError, f'<apex> for ThreeDConePlot "{self.name}" expects three floats.')
      self.apex = np.array(values, dtype=float)

    angleNode = spec.findFirst('angle')
    if angleNode is not None and angleNode.value:
      self.angle = max(0.1, float(angleNode.value))

    scaleNode = spec.findFirst('height_scale')
    if scaleNode is not None and scaleNode.value:
      self.heightScale = max(0.01, float(scaleNode.value))

    viewNode = spec.findFirst('view_angles')
    if viewNode is not None and viewNode.value:
      values = [float(val) for val in viewNode.value]
      if len(values) != 2:
        self.raiseAnError(IOError, f'<view_angles> for ThreeDConePlot "{self.name}" expects two floats.')
      self.viewAngles = (values[0], values[1])

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'Source "{self.sourceName}" not found for ThreeDConePlot "{self.name}".')
    required = set(self.objectives)
    if self.index:
      required.add(self.index)
    if self.metric:
      required.add(self.metric)
    if self.rankFilter is not None:
      required.add('rank')
    missing = [var for var in required if var not in src.getVars()]
    if missing:
      self.raiseAnError(IOError, f'Source "{src.name}" missing variables {missing} required by ThreeDConePlot "{self.name}".')
    self.source = src

  def _filterLatestGeneration(self, df):
    """
      Select rows for the requested generation (explicit or max index).
    """
    if self.index is None:
      return df
    df[self.index] = df[self.index].astype(float)
    if self.generation is not None:
      mask = np.isclose(df[self.index].to_numpy(dtype=float), self.generation)
      subset = df[mask]
    else:
      max_gen = df[self.index].max()
      subset = df[df[self.index] == max_gen]
    return subset

  def _makeConeFaces(self, apex, tip):
    """
      Construct triangular faces for a cone surface.
    """
    direction = tip - apex
    length = np.linalg.norm(direction)
    if length <= 0:
      return []
    height = length * self.heightScale
    dir_unit = direction / length
    # choose vector not parallel to dir_unit
    if abs(dir_unit[0]) < 0.9:
      ref = np.array([1.0, 0.0, 0.0])
    else:
      ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(dir_unit, ref)
    if np.linalg.norm(u) == 0:
      u = np.array([0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u)
    v = np.cross(dir_unit, u)
    v = v / np.linalg.norm(v)
    angle_rad = math.radians(self.angle)
    radius = math.tan(angle_rad) * height
    center = apex + dir_unit * height
    segments = 32
    rim = []
    for theta in np.linspace(0.0, 2.0 * math.pi, num=segments, endpoint=False):
      rim_point = center + radius * (math.cos(theta) * u + math.sin(theta) * v)
      rim.append(rim_point)
    faces = []
    for i in range(len(rim)):
      faces.append([apex, rim[i], rim[(i + 1) % len(rim)]])
    faces.append(rim)
    return faces

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source "{self.source.name}" empty; skipping ThreeDConePlot "{self.name}".')
      return
    subset = df.copy()
    if self.rankFilter is not None and 'rank' in subset.columns:
      subset['rank'] = subset['rank'].astype(float)
      subset = subset[np.isclose(subset['rank'].to_numpy(dtype=float), float(self.rankFilter))]
    if subset.empty:
      self.raiseAWarning(f'No rows remained after <rank> filtering in ThreeDConePlot "{self.name}".')
      return
    subset = self._filterLatestGeneration(subset)
    if subset.empty:
      self.raiseAWarning(f'Generation filtering removed all rows in ThreeDConePlot "{self.name}".')
      return

    objVals = subset[self.objectives].astype(float)
    metricVals = None
    if self.metric:
      metricVals = subset[self.metric].astype(float).to_numpy()
    else:
      metricVals = np.linalg.norm(objVals.to_numpy(dtype=float) - self.apex, axis=1)
    order = np.argsort(metricVals)[::-1]  # descending
    order = order[:min(self.topK, len(order))]
    if len(order) == 0:
      self.raiseAWarning(f'Unable to identify cones to draw for ThreeDConePlot "{self.name}".')
      return

    selected = objVals.to_numpy(dtype=float)[order]
    colors = np.linspace(0.1, 1.0, len(selected))
    cmap = plt.cm.inferno

    fig = plt.figure(figsize=(6.8, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=self.viewAngles[0], azim=self.viewAngles[1])

    for idx, point in enumerate(selected):
      faces = self._makeConeFaces(self.apex, point)
      if not faces:
        continue
      poly = Poly3DCollection(faces, facecolor=cmap(colors[idx]), alpha=0.35, linewidths=0.5)
      poly.set_edgecolor((0.2, 0.2, 0.2, 0.4))
      ax.add_collection3d(poly)

    ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2],
               c=cmap(colors), s=40, depthshade=True, label='Selected points')
    ax.scatter([self.apex[0]], [self.apex[1]], [self.apex[2]],
               c='k', s=50, marker='*', label='Apex')

    ax.set_xlabel(self.objectives[0])
    ax.set_ylabel(self.objectives[1])
    ax.set_zlabel(self.objectives[2])
    ax.set_title('3D dominance cones')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)
