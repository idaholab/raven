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
"""Decision-space to objective-space mapping plot.

This plot renders paired views of the same samples:
- left: decision variables (x-space)
- right: objectives (f-space)

Optionally, it draws linking segments between each decision point and its
objective point (conceptually illustrating the mapping x -> f(x)).

Optional envelopes (2D hulls) can be drawn in each panel:
- all-sample envelope: boundary of all displayed samples
- infeasible envelope: boundary of infeasible samples only (constraint violations)

Typical use cases:
- explaining multi-objective optimization concepts to stakeholders
- sanity-checking whether clusters in decision space map to clusters in objective space
- visualizing how feasibility / Pareto rank in objective space relates to decision variables
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class DecisionObjectiveMappingPlot(PlotInterface):
  """Side-by-side decision/objective scatter with optional linking lines."""

  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject to plot."""))
    spec.addSub(InputData.parameterInputFactory('decisions', contentType=InputTypes.StringListType,
        descr=r"""Exactly two decision-variable names (x-axis, y-axis) for decision space."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Exactly two objective names (x-axis, y-axis) for objective space."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional constraint evaluation columns used to define feasibility.
                   Values > 0 are feasible; values <= 0 indicate violation. Use "all" to include every
                   column named ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Optional Pareto rank to filter on if a 'rank' column exists (default: no filtering)."""))
    spec.addSub(InputData.parameterInputFactory('max_points', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of points to display and (if enabled) link. Default 250."""))
    spec.addSub(InputData.parameterInputFactory('draw_links', contentType=InputTypes.BoolType,
        descr=r"""If true, draw linking segments for displayed points. Default true."""))
    spec.addSub(InputData.parameterInputFactory('link_alpha', contentType=InputTypes.FloatType,
        descr=r"""Alpha for linking segments. Default 0.15."""))
    spec.addSub(InputData.parameterInputFactory('boundary_all', contentType=InputTypes.BoolType,
        descr=r"""If true, draw a 2D envelope (convex hull) around all displayed samples in both panels. Default false."""))
    spec.addSub(InputData.parameterInputFactory('boundary_all_fill', contentType=InputTypes.BoolType,
        descr=r"""If true, fill the all-sample envelope with <boundary_all_alpha>. Default true."""))
    spec.addSub(InputData.parameterInputFactory('boundary_infeasible', contentType=InputTypes.BoolType,
        descr=r"""If true and <constraints> are provided, draw a 2D envelope (convex hull) around infeasible samples. Default false."""))
    spec.addSub(InputData.parameterInputFactory('boundary_smooth_iters', contentType=InputTypes.IntegerType,
        descr=r"""Optional number of smoothing iterations applied to hull polygons (Chaikin). Default 2."""))
    spec.addSub(InputData.parameterInputFactory('boundary_all_color', contentType=InputTypes.StringType,
        descr=r"""Color for the all-sample envelope outline/fill (default '#1565c0')."""))
    spec.addSub(InputData.parameterInputFactory('boundary_infeasible_color', contentType=InputTypes.StringType,
        descr=r"""Color for the infeasible envelope outline (default '#616161')."""))
    spec.addSub(InputData.parameterInputFactory('boundary_all_alpha', contentType=InputTypes.FloatType,
        descr=r"""Fill alpha for the all-sample envelope (default 0.10)."""))
    spec.addSub(InputData.parameterInputFactory('boundary_linewidth', contentType=InputTypes.FloatType,
        descr=r"""Outline linewidth for envelopes (default 2.0)."""))
    spec.addSub(InputData.parameterInputFactory('title', contentType=InputTypes.StringType,
        descr=r"""Optional figure title. If omitted, a default title is generated."""))
    spec.addSub(InputData.parameterInputFactory('legend_y', contentType=InputTypes.FloatType,
        descr=r"""Vertical placement of the shared legend (figure coordinates). Default 1.02."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'DecisionObjectiveMappingPlot'
    self.source = None
    self.sourceName = None
    self.decisions = []
    self.objectives = []
    self.constraints = []
    self.useAllConstraints = False
    self.rank = None
    self.maxPoints = 250
    self.drawLinks = True
    self.linkAlpha = 0.15
    self.boundaryAll = False
    self.boundaryAllFill = True
    self.boundaryInfeasible = False
    self.boundarySmoothIters = 2
    self.boundaryAllColor = '#1565c0'
    self.boundaryInfeasibleColor = '#616161'
    self.boundaryAllAlpha = 0.10
    self.boundaryLinewidth = 2.0
    self.title = None
    self.legendY = 1.02

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'Missing <source> node for DecisionObjectiveMappingPlot "{self.name}".')
    self.sourceName = src.value

    dec = spec.findFirst('decisions')
    if dec is None or len(dec.value) != 2:
      self.raiseAnError(IOError, f'DecisionObjectiveMappingPlot "{self.name}" requires exactly two <decisions>.')
    self.decisions = [entry for entry in dec.value if entry]

    obj = spec.findFirst('objectives')
    if obj is None or len(obj.value) != 2:
      self.raiseAnError(IOError, f'DecisionObjectiveMappingPlot "{self.name}" requires exactly two <objectives>.')
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

    mp = spec.findFirst('max_points')
    if mp is not None:
      self.maxPoints = int(mp.value)

    dl = spec.findFirst('draw_links')
    if dl is not None:
      self.drawLinks = bool(dl.value)

    la = spec.findFirst('link_alpha')
    if la is not None:
      self.linkAlpha = float(la.value)

    ba = spec.findFirst('boundary_all')
    if ba is not None and ba.value is not None:
      self.boundaryAll = bool(ba.value)
    baf = spec.findFirst('boundary_all_fill')
    if baf is not None and baf.value is not None:
      self.boundaryAllFill = bool(baf.value)
    bi = spec.findFirst('boundary_infeasible')
    if bi is not None and bi.value is not None:
      self.boundaryInfeasible = bool(bi.value)
    bs = spec.findFirst('boundary_smooth_iters')
    if bs is not None and bs.value is not None:
      self.boundarySmoothIters = max(0, int(bs.value))
    bac = spec.findFirst('boundary_all_color')
    if bac is not None and bac.value:
      self.boundaryAllColor = str(bac.value).strip()
    bic = spec.findFirst('boundary_infeasible_color')
    if bic is not None and bic.value:
      self.boundaryInfeasibleColor = str(bic.value).strip()
    baa = spec.findFirst('boundary_all_alpha')
    if baa is not None and baa.value is not None:
      self.boundaryAllAlpha = float(baa.value)
    blw = spec.findFirst('boundary_linewidth')
    if blw is not None and blw.value is not None:
      self.boundaryLinewidth = float(blw.value)

    titleNode = spec.findFirst('title')
    if titleNode is not None and titleNode.value:
      self.title = str(titleNode.value)
    legendNode = spec.findFirst('legend_y')
    if legendNode is not None and legendNode.value is not None:
      self.legendY = float(legendNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for DecisionObjectiveMappingPlot "{self.name}".')

    available = self.source.getVars()
    if self.useAllConstraints:
      self.constraints = sorted(var for var in available if var.startswith('ConstraintEvaluation_'))

    needed = list(self.decisions) + list(self.objectives) + list(self.constraints)
    if self.rank is not None and 'rank' in available:
      needed.append('rank')
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variable(s) {missing} required by DecisionObjectiveMappingPlot "{self.name}".')

  @staticmethod
  def _convex_hull(points):
    """Monotonic chain convex hull. Returns hull vertices in CCW order."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
      return None
    # unique points
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
      return None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
      return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
      while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
        lower.pop()
      lower.append(p)
    upper = []
    for p in reversed(pts):
      while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
        upper.pop()
      upper.append(p)
    hull = np.vstack((lower[:-1], upper[:-1]))
    if hull.shape[0] < 3:
      return None
    return hull

  @staticmethod
  def _chaikin_smooth(poly, iters):
    """Chaikin corner cutting for a closed polygon."""
    if poly is None:
      return None
    pts = np.asarray(poly, dtype=float)
    if pts.shape[0] < 3 or iters <= 0:
      return pts
    for _ in range(iters):
      new_pts = []
      for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        q = 0.75 * p0 + 0.25 * p1
        r = 0.25 * p0 + 0.75 * p1
        new_pts.extend([q, r])
      pts = np.asarray(new_pts, dtype=float)
    return pts

  def _draw_envelope(self, ax, xy, color, fill=False, alpha=0.10, linewidth=2.0, linestyle='-'):
    hull = self._convex_hull(xy)
    if hull is None:
      return False
    hull = self._chaikin_smooth(hull, self.boundarySmoothIters)
    if hull is None or hull.shape[0] < 3:
      return False
    if fill:
      ax.add_patch(Polygon(hull, closed=True, facecolor=color, edgecolor='none', alpha=float(alpha), zorder=0))
    ax.plot(hull[:, 0], hull[:, 1], color=color, linewidth=float(linewidth), linestyle=linestyle, zorder=1)
    ax.plot([hull[-1, 0], hull[0, 0]], [hull[-1, 1], hull[0, 1]], color=color, linewidth=float(linewidth), linestyle=linestyle, zorder=1)
    return True

  @staticmethod
  def _is_feasible(df, constraints):
    if df is None or df.empty or not constraints:
      return np.ones(0 if df is None else len(df), dtype=bool)
    feasible = np.ones(len(df), dtype=bool)
    for var in constraints:
      vals = df[var].astype(float).to_numpy()
      feasible &= vals > 0.0
    return feasible

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'DecisionObjectiveMappingPlot "{self.name}" received an empty dataset; skipping.')
      return

    cols = list(self.decisions) + list(self.objectives) + list(self.constraints)
    if self.rank is not None and 'rank' in df.columns:
      cols.append('rank')
    data_all = df[cols].copy()
    for col in cols:
      data_all[col] = pd.to_numeric(data_all[col], errors='coerce')
    data_all = data_all.dropna(subset=self.decisions + self.objectives)
    if data_all.empty:
      self.raiseAWarning(f'DecisionObjectiveMappingPlot "{self.name}" has no finite samples after coercion; skipping.')
      return

    if self.rank is not None and 'rank' in data_all.columns:
      data_all = data_all[data_all['rank'].astype(float) == float(self.rank)]

    feasibleMaskAll = self._is_feasible(data_all, self.constraints) if self.constraints else np.ones(len(data_all), dtype=bool)

    # Display subset (optionally downsampled for readability), but keep envelopes based on all samples.
    data = data_all.reset_index(drop=True)
    feasibleMask = feasibleMaskAll

    n = len(data)
    if self.maxPoints is not None and n > int(self.maxPoints):
      rng = np.random.default_rng(42)
      keep = rng.choice(n, size=int(self.maxPoints), replace=False)
      keep = np.sort(keep)
      data = data.iloc[keep].reset_index(drop=True)
      feasibleMask = feasibleMask[keep]

    fig, (axDec, axObj) = plt.subplots(1, 2, figsize=(12.0, 5.2))

    if self.constraints:
      feas = feasibleMask
      infeas = ~feas
      if feas.any():
        axDec.scatter(data.loc[feas, self.decisions[0]], data.loc[feas, self.decisions[1]],
                      c='#2e7d32', s=22, alpha=0.85, edgecolors='none')
        axObj.scatter(data.loc[feas, self.objectives[0]], data.loc[feas, self.objectives[1]],
                      c='#2e7d32', s=22, alpha=0.85, edgecolors='none')
      if infeas.any():
        axDec.scatter(data.loc[infeas, self.decisions[0]], data.loc[infeas, self.decisions[1]],
                      c='#d32f2f', s=28, alpha=0.9, marker='x')
        axObj.scatter(data.loc[infeas, self.objectives[0]], data.loc[infeas, self.objectives[1]],
                      c='#d32f2f', s=28, alpha=0.9, marker='x')
    else:
      axDec.scatter(data[self.decisions[0]], data[self.decisions[1]], c='tab:blue', s=22, alpha=0.8, edgecolors='none')
      axObj.scatter(data[self.objectives[0]], data[self.objectives[1]], c='tab:blue', s=22, alpha=0.8, edgecolors='none')

    if self.drawLinks:
      for i in range(len(data)):
        con = ConnectionPatch(xyA=(data.iloc[i][self.decisions[0]], data.iloc[i][self.decisions[1]]), coordsA=axDec.transData,
                              xyB=(data.iloc[i][self.objectives[0]], data.iloc[i][self.objectives[1]]), coordsB=axObj.transData,
                              arrowstyle='-', lw=0.8, alpha=float(self.linkAlpha), color='0.25')
        fig.add_artist(con)

    axDec.set_xlabel(self.decisions[0])
    axDec.set_ylabel(self.decisions[1])
    axDec.set_title('Decision space')
    axDec.grid(alpha=0.25)

    axObj.set_xlabel(self.objectives[0])
    axObj.set_ylabel(self.objectives[1])
    title = 'Objective space'
    if self.rank is not None:
      title += f' (rank={self.rank})'
    axObj.set_title(title)
    axObj.grid(alpha=0.25)

    # Optional envelopes (in both panels)
    legend_boundary_all = False
    legend_boundary_infeas = False
    if self.boundaryAll:
      xy_dec = data_all[[self.decisions[0], self.decisions[1]]].to_numpy(dtype=float)
      xy_obj = data_all[[self.objectives[0], self.objectives[1]]].to_numpy(dtype=float)
      filled = bool(self.boundaryAllFill)
      legend_boundary_all |= self._draw_envelope(axDec, xy_dec, self.boundaryAllColor, fill=filled,
                                                alpha=self.boundaryAllAlpha, linewidth=self.boundaryLinewidth, linestyle='-')
      legend_boundary_all |= self._draw_envelope(axObj, xy_obj, self.boundaryAllColor, fill=filled,
                                                alpha=self.boundaryAllAlpha, linewidth=self.boundaryLinewidth, linestyle='-')
    if self.boundaryInfeasible and self.constraints and (~feasibleMaskAll).any():
      infeas_df = data_all.loc[~feasibleMaskAll]
      xy_dec = infeas_df[[self.decisions[0], self.decisions[1]]].to_numpy(dtype=float)
      xy_obj = infeas_df[[self.objectives[0], self.objectives[1]]].to_numpy(dtype=float)
      legend_boundary_infeas |= self._draw_envelope(axDec, xy_dec, self.boundaryInfeasibleColor, fill=False,
                                                   alpha=0.0, linewidth=self.boundaryLinewidth, linestyle='--')
      legend_boundary_infeas |= self._draw_envelope(axObj, xy_obj, self.boundaryInfeasibleColor, fill=False,
                                                   alpha=0.0, linewidth=self.boundaryLinewidth, linestyle='--')

    handles = []
    if self.constraints:
      handles.append(Patch(facecolor='#2e7d32', edgecolor='none', alpha=0.85, label='Feasible (all constraints > 0)'))
      handles.append(Line2D([0], [0], marker='x', color='#d32f2f', linestyle='None', markersize=7, label='Infeasible'))
    else:
      handles.append(Patch(facecolor='tab:blue', edgecolor='none', alpha=0.85, label='Samples'))
    if self.drawLinks:
      handles.append(Line2D([0], [0], color='0.25', lw=1.0, alpha=float(self.linkAlpha), label='Mapping link'))
    if legend_boundary_all:
      handles.append(Line2D([0], [0], color=self.boundaryAllColor, lw=float(self.boundaryLinewidth), label='All-sample envelope'))
    if legend_boundary_infeas:
      handles.append(Line2D([0], [0], color=self.boundaryInfeasibleColor, lw=float(self.boundaryLinewidth), linestyle='--', label='Infeasible envelope'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, float(self.legendY)), ncol=len(handles), frameon=True, fontsize=9)

    default_title = f'Decision ↔ Objective mapping ({self.decisions[0]}, {self.decisions[1]}) → ({self.objectives[0]}, {self.objectives[1]})'
    fig.suptitle(self.title if self.title else default_title, y=min(0.995, float(self.legendY) + 0.06), fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)
