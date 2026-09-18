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
"""Triangulated Pareto surface plot for three objectives.

RAVEN provides 3D objective-space plots for optimizers (e.g., scatter/tubes/cones).
This PlotInterface adds a lightweight *surface* rendering for rank-1 samples when
three objectives are present.

Notes:
- The "surface" is computed via a 2D triangulation on the first two objectives
  and plotting the third objective as height. This is an approximation intended
  for quick visualization (not a robust manifold reconstruction).
- Use <constraints> to filter to feasible rank-1 points.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

try:
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
  Axes3D = None


class ParetoSurfacePlot(PlotInterface):
  """Triangulated surface + scatter for three objectives."""

  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly three objective variables to plot as (x, y, z)."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional constraint evaluation columns used to define feasibility.
                   Values > 0 are feasible; values <= 0 indicate violation. Use "all" to include every
                   column named ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Pareto rank to display (default 1)."""))
    spec.addSub(InputData.parameterInputFactory('elev', contentType=InputTypes.FloatType,
        descr=r"""3-D camera elevation angle (default 25)."""))
    spec.addSub(InputData.parameterInputFactory('azim', contentType=InputTypes.FloatType,
        descr=r"""3-D camera azimuth angle (default -60)."""))
    spec.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.FloatType,
        descr=r"""Surface alpha (default 0.35)."""))
    spec.addSub(InputData.parameterInputFactory('cmap', contentType=InputTypes.StringType,
        descr=r"""Matplotlib colormap name (default viridis)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ParetoSurfacePlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.constraints = []
    self.useAllConstraints = False
    self.rank = 1
    self.elev = 25.0
    self.azim = -60.0
    self.alpha = 0.35
    self.cmap = 'viridis'

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'Missing <source> node for ParetoSurfacePlot "{self.name}".')
    self.sourceName = src.value

    obj = spec.findFirst('objectives')
    if obj is None or len(obj.value) != 3:
      self.raiseAnError(IOError, f'ParetoSurfacePlot "{self.name}" requires exactly three <objectives>.')
    self.objectives = [entry for entry in obj.value if entry]

    consNode = spec.findFirst('constraints')
    if consNode is not None and consNode.value:
      entries = [str(entry).strip() for entry in consNode.value if str(entry).strip()]
      if any(entry.lower() == 'all' for entry in entries):
        self.useAllConstraints = True
      else:
        self.constraints = entries

    rankNode = spec.findFirst('rank')
    if rankNode is not None:
      self.rank = int(rankNode.value)

    elevNode = spec.findFirst('elev')
    if elevNode is not None:
      self.elev = float(elevNode.value)

    azimNode = spec.findFirst('azim')
    if azimNode is not None:
      self.azim = float(azimNode.value)

    alphaNode = spec.findFirst('alpha')
    if alphaNode is not None:
      self.alpha = float(alphaNode.value)

    cmapNode = spec.findFirst('cmap')
    if cmapNode is not None and cmapNode.value:
      self.cmap = str(cmapNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for ParetoSurfacePlot "{self.name}".')

    available = self.source.getVars()
    if self.useAllConstraints:
      self.constraints = sorted(var for var in available if var.startswith('ConstraintEvaluation_'))

    needed = list(self.objectives) + list(self.constraints)
    if 'rank' in available:
      needed.append('rank')
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variable(s) {missing} required by ParetoSurfacePlot "{self.name}".')

  @staticmethod
  def _is_feasible(df, constraints):
    if df is None or df.empty or not constraints:
      return np.ones(0 if df is None else len(df), dtype=bool)
    feasible = np.ones(len(df), dtype=bool)
    for var in constraints:
      vals = df[var].astype(float).to_numpy()
      feasible &= vals > 0.0
    return feasible

  @staticmethod
  def _triangulation(x, y):
    coords = np.vstack((x, y)).T
    centered = coords - coords.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
      return None
    try:
      triang = mtri.Triangulation(x, y)
    except RuntimeError:
      triang = None
    return triang

  def run(self):
    if Axes3D is None:
      self.raiseAnError(RuntimeError, 'mpl_toolkits.mplot3d is not available; ParetoSurfacePlot requires 3D plotting.')

    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'ParetoSurfacePlot "{self.name}" received an empty dataset; skipping.')
      return

    cols = list(self.objectives)
    if 'rank' in df.columns:
      cols.append('rank')
    cols += list(self.constraints)
    data = df[cols].copy()
    for col in cols:
      data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=self.objectives)

    if 'rank' in data.columns and np.isfinite(data['rank']).any():
      data = data[data['rank'].astype(float) == float(self.rank)]

    if self.constraints and not data.empty:
      feasibleMask = self._is_feasible(data, self.constraints)
      data = data.loc[feasibleMask]

    if data.shape[0] < 3:
      self.raiseAWarning(f'ParetoSurfacePlot "{self.name}" needs >= 3 finite samples after filtering; skipping.')
      return

    x = data[self.objectives[0]].to_numpy(dtype=float)
    y = data[self.objectives[1]].to_numpy(dtype=float)
    z = data[self.objectives[2]].to_numpy(dtype=float)

    triang = self._triangulation(x, y)
    if triang is None or triang.triangles.size == 0:
      self.raiseAWarning(f'ParetoSurfacePlot "{self.name}" could not triangulate objective-space samples; falling back to scatter.')
      triang = None

    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=self.elev, azim=self.azim)

    if triang is not None:
      surf = ax.plot_trisurf(triang, z, cmap=self.cmap, alpha=float(self.alpha), linewidth=0.2, edgecolor='0.35')
      cbar = fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.08)
      cbar.set_label(self.objectives[2])

    ax.scatter(x, y, z, c='k', s=14, alpha=0.8, depthshade=True)

    ax.set_xlabel(self.objectives[0])
    ax.set_ylabel(self.objectives[1])
    ax.set_zlabel(self.objectives[2])
    ax.set_title(f'Pareto surface (rank={self.rank})')

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)
