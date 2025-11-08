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
Generate a prosection matrix that slices high-dimensional variables near their
median values.

This diagnostic emphasises pairwise relationships while holding the remaining
dimensions close to their centre, allowing quick what-if checks. Example:
"What if I pin all other variables near their medians—do `x2` and `obj2`
still look correlated?" Adjust `<tolerance>` or target a specific generation
to see whether early vs late populations obey the same local structure.
"""

import itertools

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ProsectionMatrixPlot(PlotInterface):
  """
  Select pairwise variable slices while constraining remaining dimensions near
  their median values. Use it for what-if analysis by tightening `<tolerance>`
  or isolating a generation to confirm whether emergent correlations persist
  when the search focuses on different regions of the design space.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing the samples."""))
    variables = InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""List of variables to consider (at least three).""")
    spec.addSub(variables)
    spec.addSub(InputData.parameterInputFactory('tolerance', contentType=InputTypes.FloatType,
        descr=r"""Normalised tolerance (0-1) used to select samples close to the median of non-plotted dimensions. Defaults to 0.1."""))
    spec.addSub(InputData.parameterInputFactory('color', contentType=InputTypes.StringType,
        descr=r"""Optional variable mapped to point colour."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> provided, limits the plot to the specified generation."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ProsectionMatrixPlot'
    self.source = None
    self.sourceName = None
    self.variables = []
    self.tolerance = 0.1
    self.colorVar = None
    self.index = None
    self.generation = None

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for ProsectionMatrixPlot "{self.name}".')
    self.sourceName = src.value

    varNode = spec.findFirst('variables')
    if varNode is None or len(varNode.value) < 3:
      self.raiseAnError(IOError, f'ProsectionMatrixPlot "{self.name}" requires at least three <variables>.')
    self.variables = [entry for entry in varNode.value if entry]

    tolNode = spec.findFirst('tolerance')
    if tolNode is not None and tolNode.value is not None:
      tol = float(tolNode.value)
      if tol <= 0 or tol > 1:
        self.raiseAnError(IOError, f'Invalid <tolerance> {tol} for ProsectionMatrixPlot "{self.name}". Expected (0,1].')
      self.tolerance = tol

    colorNode = spec.findFirst('color')
    if colorNode is not None and colorNode.value:
      self.colorVar = colorNode.value

    indexNode = spec.findFirst('index')
    if indexNode is not None and indexNode.value:
      self.index = indexNode.value
    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for ProsectionMatrixPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.variables)
    if self.colorVar:
      needed.append(self.colorVar)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by ProsectionMatrixPlot "{self.name}".')
    self.source = src

  def _select_slice(self, df, xVar, yVar, other_vars):
    """
    Returns indices of samples close to median in other variables (normalised by range).
    """
    if not other_vars:
      return df.index
    subset = df[other_vars].copy()
    # Normalise to [0,1] using min/max; protect against constant columns.
    mins = subset.min(axis=0)
    maxs = subset.max(axis=0)
    ranges = (maxs - mins).replace(0.0, np.nan)
    norm = (subset - mins) / ranges
    norm = norm.dropna()
    if norm.empty:
      return pd.Index([])
    median = norm.median(axis=0)
    distances = (norm - median).abs()
    mask = (distances <= self.tolerance).all(axis=1)
    return norm[mask].index

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; ProsectionMatrixPlot "{self.name}" skipped.')
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
      self.raiseAWarning(f'ProsectionMatrixPlot "{self.name}" had no samples after filtering.')
      return

    numeric = subset[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      self.raiseAWarning(f'ProsectionMatrixPlot "{self.name}" found no finite samples for {self.variables}.')
      return

    combos = list(itertools.combinations(self.variables, 2))
    ncols = min(3, len(combos))
    nrows = int(np.ceil(len(combos) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    color_series = None
    cmap = None
    if self.colorVar and self.colorVar in subset.columns:
      color_series = subset.loc[numeric.index, self.colorVar]
      if np.issubdtype(color_series.dtype, np.number):
        cmap = 'viridis'

    for ax, (xVar, yVar) in zip(axes.flat, combos):
      others = [var for var in self.variables if var not in (xVar, yVar)]
      slice_idx = self._select_slice(numeric, xVar, yVar, others)
      if slice_idx.empty:
        ax.text(0.5, 0.5, 'No slice\nselected', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(xVar)
        ax.set_ylabel(yVar)
        continue
      x = numeric.loc[slice_idx, xVar]
      y = numeric.loc[slice_idx, yVar]
      color_values = None
      if color_series is not None:
        color_values = color_series.loc[slice_idx]
        if not np.issubdtype(color_values.dtype, np.number):
          color_values = color_values.astype(str)

      scatter_kwargs = dict(alpha=0.8, edgecolors='k', linewidths=0.2, s=30)
      if color_values is not None:
        scatter_kwargs['c'] = color_values
        if cmap and np.issubdtype(color_series.dtype, np.number):
          scatter_kwargs['cmap'] = cmap
      ax.scatter(x, y, **scatter_kwargs)
      ax.set_xlabel(xVar)
      ax.set_ylabel(yVar)
      ax.grid(alpha=0.2)

    # Hide unused axes if combos < grid size
    total_axes = nrows * ncols
    for idx in range(len(combos), total_axes):
      axes.flat[idx].set_visible(False)

    fig.suptitle('Prosection matrix')
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
