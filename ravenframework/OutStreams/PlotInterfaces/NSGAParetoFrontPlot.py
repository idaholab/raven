# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Visual utility to render the rank-1 Pareto front produced by NSGA-II style optimizers.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

try:
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for side-effects)
except ImportError:
  Axes3D = None


class NSGAParetoFrontPlot(PlotInterface):
  """
  Static scatter plot of the dominant (rank-1) Pareto front.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject produced by the optimizer."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""List of objective variables to plot. Two objectives renders a 2D scatter while three
                   objectives renders a 3D scatter. More than three objectives are not supported."""))
    spec.addSub(InputData.parameterInputFactory('color', contentType=InputTypes.StringType,
        descr=r"""Optional variable used to color points. Defaults to crowding distance if available."""))
    spec.addSub(InputData.parameterInputFactory('rank', contentType=InputTypes.IntegerType,
        descr=r"""Pareto rank to display. Defaults to 1 (non-dominated front)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA Pareto Front Plot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.colorVar = None
    self.rank = 1

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    objectives = spec.findFirst('objectives')
    if objectives is None:
      self.raiseAnError(IOError, 'Missing <objectives> node for NSGAParetoFrontPlot "{}".'.format(self.name))
    self.objectives = objectives.value
    color = spec.findFirst('color')
    if color is not None:
      self.colorVar = color.value
    rank = spec.findFirst('rank')
    if rank is not None:
      self.rank = rank.value

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for NSGAParetoFrontPlot "{self.name}".')
    dataVars = self.source.getVars()
    missing = [var for var in self.objectives if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing required objective(s) {} for NSGAParetoFrontPlot "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)
    if len(self.objectives) not in {2, 3}:
      self.raiseAnError(IOError, 'NSGAParetoFrontPlot "{}" requires 2 or 3 objectives; got {}.'.format(
          self.name, len(self.objectives)))

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if 'rank' not in df.columns:
      self.raiseAWarning('DataObject "{}" lacks a "rank" column; plotting all points instead of rank {}.'.format(
          self.source.name, self.rank))
      filtered = df.copy()
    else:
      filtered = df[df['rank'] == self.rank]
    if filtered.empty:
      self.raiseAWarning('No samples with rank == {} for "{}"; plot will be empty.'.format(self.rank, self.name))
      filtered = df.copy()

    colorVar = self.colorVar
    if colorVar is None and 'CD' in filtered.columns:
      colorVar = 'CD'

    colors = None
    useColorbar = False
    if colorVar is not None:
      if colorVar not in filtered.columns:
        self.raiseAWarning('Color variable "{}" not found; using uniform color.'.format(colorVar))
        colorVar = None
      else:
        series = filtered[colorVar]
        if hasattr(series, 'to_numpy'):
          raw_values = series.to_numpy()
        else:
          raw_values = np.asarray(series)
        flat = np.asarray(raw_values).ravel()
        if flat.size == 0:
          self.raiseAWarning('Color variable "{}" contains no samples; using uniform color.'.format(colorVar))
          colorVar = None
        else:
          numeric_values = None
          try:
            numeric_values = flat.astype(float)
          except (ValueError, TypeError):
            numeric_values = None

          if numeric_values is not None:
            finite_mask = np.isfinite(numeric_values)
            if not finite_mask.any():
              self.raiseAWarning('Color variable "{}" has no finite numeric values; using uniform color.'.format(colorVar))
              colorVar = None
            else:
              numeric_values = numeric_values.astype(float, copy=False)
              numeric_values[~finite_mask] = np.nan
              colors = numeric_values
              useColorbar = True
          else:
            string_values = np.array([str(val).strip() for val in flat], dtype=object)
            if not string_values.size:
              self.raiseAWarning('Color variable "{}" has no usable categorical values; using uniform color.'.format(colorVar))
              colorVar = None
            elif not all(val and val.lower() not in {'nan', 'none'} and is_color_like(val) for val in string_values):
              self.raiseAWarning('Color variable "{}" cannot be interpreted as numeric or named colors; using uniform color.'.format(colorVar))
              colorVar = None
            else:
              colors = string_values

    scatterKwargs = {}
    if colors is not None:
      scatterKwargs['c'] = colors
      if useColorbar:
        scatterKwargs['cmap'] = 'viridis'

    fig = plt.figure()
    if len(self.objectives) == 2:
      ax = fig.add_subplot(111)
      scatterKwargs2D = dict(scatterKwargs)
      scatterKwargs2D.update({'edgecolors': 'k', 'linewidths': 0.5})
      sc = ax.scatter(filtered[self.objectives[0]], filtered[self.objectives[1]], **scatterKwargs2D)
      ax.set_xlabel(self.objectives[0])
      ax.set_ylabel(self.objectives[1])
      ax.set_title(f'Pareto Front (rank={self.rank})')
      if useColorbar and colorVar is not None and colors is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(colorVar)
    else:
      if Axes3D is None:
        self.raiseAnError(RuntimeError, 'mpl_toolkits.mplot3d is not available but 3 objectives were requested.')
      ax = fig.add_subplot(111, projection='3d')
      scatterKwargs3D = dict(scatterKwargs)
      scatterKwargs3D.update({'depthshade': True})
      sc = ax.scatter(filtered[self.objectives[0]], filtered[self.objectives[1]], filtered[self.objectives[2]],
                      **scatterKwargs3D)
      ax.set_xlabel(self.objectives[0])
      ax.set_ylabel(self.objectives[1])
      ax.set_zlabel(self.objectives[2])
      ax.set_title(f'Pareto Front (rank={self.rank})')
      if useColorbar and colorVar is not None and colors is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=12, pad=0.1)
        cbar.set_label(colorVar)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    plt.savefig(filename)
    plt.close(fig)
