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
Feasible-region + objective/constraint contours in decision space.

This plot mirrors classic constrained optimization visuals:
- two decision variables on x/y axes
- a 3-D wireframe surface for each requested metric (objective or constraint)
- contour level-sets projected to the base plane
- feasible region shaded (based on ConstraintEvaluation_* columns)
- constraint boundaries (zero level-sets) drawn on the base plane
- Pareto points projected onto the base plane
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

try:
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for side-effects)
except ImportError:
  Axes3D = None


class FeasibleRegionObjectiveContourPlot(PlotInterface):
  """
  Builds 3-D surfaces from scattered samples and overlays feasibility + contour projections.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    axes = InputData.parameterInputFactory('axes', contentType=InputTypes.StringListType,
        descr=r"""Exactly two decision-variable names defining the x/y plane.""")
    spec.addSub(axes)
    spec.addSub(InputData.parameterInputFactory('surfaces', contentType=InputTypes.StringListType,
        descr=r"""List of variables to render as 3-D surfaces (e.g., objectives and/or constraint evaluations).
                   One variable -> single panel; four variables -> 2x2 panels."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional constraint evaluation columns used to define feasibility and boundary lines.
                   Values > 0 are feasible; values <= 0 indicate violation. Use "all" to include every
                   column named ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Pareto rank to display as projected points. Defaults to 1."""))
    spec.addSub(InputData.parameterInputFactory('levels', contentType=InputTypes.IntegerType,
        descr=r"""Number of contour levels to project onto the base plane (default 12)."""))
    spec.addSub(InputData.parameterInputFactory('elev', contentType=InputTypes.FloatType,
        descr=r"""3-D camera elevation angle (default 25)."""))
    spec.addSub(InputData.parameterInputFactory('azim', contentType=InputTypes.FloatType,
        descr=r"""3-D camera azimuth angle (default -60)."""))
    spec.addSub(InputData.parameterInputFactory('base_plane', contentType=InputTypes.FloatType,
        descr=r"""Optional z-offset to use for the base plane. If omitted, computed per panel."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'FeasibleRegionObjectiveContourPlot'
    self.source = None
    self.sourceName = None
    self.axes = []
    self.surfaces = []
    self.constraints = []
    self.useAllConstraints = False
    self.rank = 1
    self.levels = 12
    self.elev = 25.0
    self.azim = -60.0
    self.basePlane = None

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for FeasibleRegionObjectiveContourPlot "{self.name}".')
    self.sourceName = src.value

    axesNode = spec.findFirst('axes')
    if axesNode is None or len(axesNode.value) != 2:
      self.raiseAnError(IOError, f'FeasibleRegionObjectiveContourPlot "{self.name}" requires exactly two <axes>.')
    self.axes = [entry for entry in axesNode.value if entry]

    surfacesNode = spec.findFirst('surfaces')
    if surfacesNode is None or not surfacesNode.value:
      self.raiseAnError(IOError, f'FeasibleRegionObjectiveContourPlot "{self.name}" requires a non-empty <surfaces> list.')
    self.surfaces = [entry for entry in surfacesNode.value if entry]

    consNode = spec.findFirst('constraints')
    if consNode is not None and consNode.value:
      entries = [str(entry).strip() for entry in consNode.value if str(entry).strip()]
      if len(entries) == 1 and entries[0].lower() == 'all':
        self.useAllConstraints = True
      else:
        self.constraints = entries

    rankNode = spec.findFirst('rank')
    if rankNode is not None and rankNode.value is not None:
      self.rank = int(rankNode.value)

    levelsNode = spec.findFirst('levels')
    if levelsNode is not None and levelsNode.value is not None:
      self.levels = max(3, int(levelsNode.value))

    elevNode = spec.findFirst('elev')
    if elevNode is not None and elevNode.value is not None:
      self.elev = float(elevNode.value)
    azimNode = spec.findFirst('azim')
    if azimNode is not None and azimNode.value is not None:
      self.azim = float(azimNode.value)

    baseNode = spec.findFirst('base_plane')
    if baseNode is not None and baseNode.value is not None:
      self.basePlane = float(baseNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    if Axes3D is None:
      self.raiseAnError(RuntimeError, 'mpl_toolkits.mplot3d is not available; FeasibleRegionObjectiveContourPlot requires 3D plotting.')
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for FeasibleRegionObjectiveContourPlot "{self.name}".')
    available = src.getVars()
    if self.useAllConstraints:
      self.constraints = sorted(var for var in available if var.startswith('ConstraintEvaluation_'))
    needed = list(self.axes) + list(self.surfaces)
    if self.constraints:
      needed.extend(self.constraints)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by FeasibleRegionObjectiveContourPlot "{self.name}".')
    self.source = src

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

  @staticmethod
  def _panel_shape(count):
    if count <= 1:
      return (1, 1)
    if count == 2:
      return (1, 2)
    if count == 3:
      return (2, 2)
    return (2, 2)

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'FeasibleRegionObjectiveContourPlot "{self.name}" received an empty dataset; skipping.')
      return

    nPanels = len(self.surfaces)
    nRows, nCols = self._panel_shape(nPanels)
    fig = plt.figure(figsize=(6.2 * nCols, 5.0 * nRows))

    anyFeasible = False

    for idx, var in enumerate(self.surfaces):
      ax = fig.add_subplot(nRows, nCols, idx + 1, projection='3d')
      ax.view_init(elev=self.elev, azim=self.azim)

      cols = [self.axes[0], self.axes[1], var] + list(self.constraints)
      points = df[cols].copy()

      def _coerce_numeric_column(frame, key):
        """Ensure frame[key] is a numeric Series even if selection yields a DataFrame."""
        try:
          selection = frame[key]
        except KeyError:
          return
        if isinstance(selection, pd.DataFrame):
          if selection.shape[1] < 1:
            return
          if selection.shape[1] > 1:
            self.raiseAWarning(f'FeasibleRegionObjectiveContourPlot "{self.name}" received non-scalar variable "{key}" with {selection.shape[1]} columns; using the first column.')
          series = selection.iloc[:, 0]
          frame.drop(columns=selection.columns, inplace=True)
          frame[key] = series
        frame[key] = pd.to_numeric(frame[key], errors='coerce')

      _coerce_numeric_column(points, self.axes[0])
      _coerce_numeric_column(points, self.axes[1])
      _coerce_numeric_column(points, var)
      for cons in self.constraints:
        _coerce_numeric_column(points, cons)
      points = points.dropna(subset=[self.axes[0], self.axes[1], var])
      if points.shape[0] < 3:
        self.raiseAWarning(f'FeasibleRegionObjectiveContourPlot "{self.name}" has too few points for "{var}"; skipping panel.')
        ax.set_axis_off()
        continue

      points = points.groupby([self.axes[0], self.axes[1]], as_index=False).mean(numeric_only=True)
      if points.shape[0] < 3:
        self.raiseAWarning(f'FeasibleRegionObjectiveContourPlot "{self.name}" collapsed to <3 unique points for "{var}".')
        ax.set_axis_off()
        continue

      x = points[self.axes[0]].to_numpy(dtype=float)
      y = points[self.axes[1]].to_numpy(dtype=float)
      z = points[var].to_numpy(dtype=float)

      triang = self._triangulation(x, y)
      if triang is None or triang.triangles.size == 0:
        self.raiseAWarning(f'FeasibleRegionObjectiveContourPlot "{self.name}" could not triangulate "{var}"; skipping panel.')
        ax.set_axis_off()
        continue

      zmin = float(np.nanmin(z))
      zmax = float(np.nanmax(z))
      span = zmax - zmin
      span = span if np.isfinite(span) and span != 0.0 else max(abs(zmax), abs(zmin), 1.0)
      zOffset = float(self.basePlane) if self.basePlane is not None else (zmin - 0.18 * span)

      # Base wireframe (red mesh)
      ax.plot_trisurf(triang, z, color='white', edgecolor='#cc0000', linewidth=0.35, alpha=0.06, shade=False)

      # Feasible region shading (green) on the surface + base plane
      feasibleMask = None
      if self.constraints:
        feasibleMask = self._is_feasible(points, self.constraints)
      if feasibleMask is None:
        feasibleMask = np.ones(len(points), dtype=bool)

      tri = triang.triangles
      triFeasible = feasibleMask[tri].all(axis=1)
      if triFeasible.any():
        anyFeasible = True
        triangFeasible = mtri.Triangulation(x, y, triangles=tri[triFeasible])
        ax.plot_trisurf(triangFeasible, z, color='#2e7d32', alpha=0.22, linewidth=0.0, shade=True)

        verts = [np.column_stack((x[t], y[t])) for t in tri[triFeasible]]
        poly = PolyCollection(verts, facecolors='#2e7d32', edgecolors='none', alpha=0.18)
        ax.add_collection3d(poly, zs=zOffset, zdir='z')

      # Contours projected to base plane
      contourSet = None
      try:
        contourSet = ax.tricontour(triang, z, levels=self.levels, zdir='z', offset=zOffset, cmap='viridis', linewidths=0.9)
      except Exception as err:
        self.raiseAWarning(f'FeasibleRegionObjectiveContourPlot "{self.name}" failed contour projection for "{var}": {err}')

      # Constraint boundaries (0-level sets) on base plane
      for cons in self.constraints:
        consVals = points[cons].to_numpy(dtype=float)
        if not np.isfinite(consVals).any():
          continue
        try:
          ax.tricontour(triang, consVals, levels=[0.0], zdir='z', offset=zOffset, colors='k', linewidths=1.25)
        except Exception:
          continue

      # Pareto points projected to base plane (rank + feasibility)
      pointCols = [self.axes[0], self.axes[1]] + list(self.constraints)
      if 'rank' in df.columns:
        pointCols.append('rank')
      p = df[pointCols].copy()
      _coerce_numeric_column(p, self.axes[0])
      _coerce_numeric_column(p, self.axes[1])
      for cons in self.constraints:
        _coerce_numeric_column(p, cons)
      if 'rank' in p.columns:
        _coerce_numeric_column(p, 'rank')
      p = p.dropna(subset=[self.axes[0], self.axes[1]])
      if 'rank' in p.columns and np.isfinite(p['rank']).any():
        p = p[p['rank'].astype(float) == float(self.rank)]
      if self.constraints and not p.empty:
        feasP = self._is_feasible(p, self.constraints)
        p = p[feasP]
      if not p.empty:
        ax.scatter(p[self.axes[0]].to_numpy(dtype=float),
                   p[self.axes[1]].to_numpy(dtype=float),
                   zs=zOffset, zdir='z', c='k', s=18, alpha=0.85, depthshade=False)

      # Colorbar: contour line colors correspond to the plotted surface variable values.
      if contourSet is not None:
        contourLevels = np.asarray(getattr(contourSet, 'levels', []), dtype=float)
        if contourLevels.size > 1 and np.isfinite(contourLevels).all() and np.nanmax(contourLevels) > np.nanmin(contourLevels):
          cbar = fig.colorbar(contourSet, ax=ax, shrink=0.72, pad=0.06)
          cbar.set_label(f'{var} contour level')
        else:
          self.raiseAWarning(f'FeasibleRegionObjectiveContourPlot "{self.name}" skipped colorbar for "{var}" because the contour projection collapsed to a single level.')

      ax.set_xlabel(self.axes[0])
      ax.set_ylabel(self.axes[1])
      ax.set_zlabel(var)
      ax.set_title(var)
      ax.set_zlim(zOffset, zmax + 0.10 * span)

    # Single legend shared by all panels (colors/markers are consistent across panels).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legendHandles = [
      Line2D([0], [0], color='#cc0000', lw=1.0, label='Triangulated surface (all samples)'),
    ]
    if self.constraints or anyFeasible:
      legendHandles.append(Patch(facecolor='#2e7d32', edgecolor='none', alpha=0.22, label='Feasible region (all constraints > 0)'))
    if self.constraints:
      legendHandles.append(Line2D([0], [0], color='k', lw=1.25, label='Constraint boundary (g=0)'))
    legendHandles.append(Line2D([0], [0], marker='o', color='k', linestyle='None', markersize=5, label=f'Pareto rank {self.rank} (feasible)'))

    fig.legend(handles=legendHandles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(legendHandles),
               fontsize=9, frameon=True)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)

