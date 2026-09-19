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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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
    spec.addSub(InputData.parameterInputFactory('rowsPerFig', contentType=InputTypes.IntegerType,
        descr=r"""Optional maximum number of subplot rows per output figure. If set, the matrix is split into
                   multiple images named <filename>_1.png, <filename>_2.png, etc. This improves readability when
                   many variable pairs are requested."""))
    spec.addSub(InputData.parameterInputFactory('minSlicePoints', contentType=InputTypes.IntegerType,
        descr=r"""Minimum number of samples required to plot a slice. If the median-based slice contains fewer
                   samples, the plot falls back to a nearest-to-median selection. Default 15."""))
    spec.addSub(InputData.parameterInputFactory('fallbackPoints', contentType=InputTypes.IntegerType,
        descr=r"""Number of nearest-to-median samples to plot when the median-based slice is empty or too small.
                   Default 50."""))
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
    self.rowsPerFig = None
    self.minSlicePoints = 15
    self.fallbackPoints = 50

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

    rpfNode = spec.findFirst('rowsPerFig')
    if rpfNode is not None and rpfNode.value is not None:
      rpf = int(rpfNode.value)
      if rpf <= 0:
        self.raiseAnError(IOError, f'ProsectionMatrixPlot "{self.name}" received non-positive <rowsPerFig>.')
      self.rowsPerFig = rpf

    mspNode = spec.findFirst('minSlicePoints')
    if mspNode is not None and mspNode.value is not None:
      msp = int(mspNode.value)
      if msp <= 0:
        self.raiseAnError(IOError, f'ProsectionMatrixPlot "{self.name}" received non-positive <minSlicePoints>.')
      self.minSlicePoints = msp

    fpNode = spec.findFirst('fallbackPoints')
    if fpNode is not None and fpNode.value is not None:
      fp = int(fpNode.value)
      if fp <= 0:
        self.raiseAnError(IOError, f'ProsectionMatrixPlot "{self.name}" received non-positive <fallbackPoints>.')
      self.fallbackPoints = fp

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

  def _selectSlice(self, df, xVar, yVar, otherVars):
    """
    Returns indices of samples close to median in other variables (normalised by range).
    If the slice is empty or too small, falls back to the nearest-to-median samples.
    """
    if not otherVars:
      return df.index
    subset = df[otherVars].copy()
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
    selected = norm[mask].index
    if selected.size >= self.minSlicePoints:
      return selected

    linf = distances.max(axis=1)
    linf = linf.replace([np.inf, -np.inf], np.nan).dropna()
    if linf.empty:
      return pd.Index([])
    k = min(int(self.fallbackPoints), len(linf))
    return linf.nsmallest(k).index

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
        maxGen = subset[self.index].max()
        subset = subset[subset[self.index] == maxGen]
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
    totalRows = int(np.ceil(len(combos) / ncols))
    rowsPerFig = self.rowsPerFig if self.rowsPerFig is not None else totalRows
    rowsPerFig = max(1, int(rowsPerFig))
    perFig = rowsPerFig * ncols
    nFigs = int(np.ceil(len(combos) / perFig))
    colorSeries = None
    cmap = None
    if self.colorVar and self.colorVar in subset.columns:
      colorSeries = subset.loc[numeric.index, self.colorVar]
      if np.issubdtype(colorSeries.dtype, np.number):
        cmap = 'viridis'

    for figIdx in range(nFigs):
      start = figIdx * perFig
      end = min(len(combos), (figIdx + 1) * perFig)
      pageCombos = combos[start:end]
      pageRows = int(np.ceil(len(pageCombos) / ncols))

      fig, axes = plt.subplots(pageRows, ncols, figsize=(4.5 * ncols, 3.8 * pageRows), squeeze=False)

      for ax, (xVar, yVar) in zip(axes.flat, pageCombos):
        others = [var for var in self.variables if var not in (xVar, yVar)]
        sliceIdx = self._selectSlice(numeric, xVar, yVar, others)
        if sliceIdx.empty:
          ax.text(0.5, 0.5, 'No slice\nselected', ha='center', va='center', transform=ax.transAxes)
          ax.set_xlabel(xVar)
          ax.set_ylabel(yVar)
          continue
        x = numeric.loc[sliceIdx, xVar]
        y = numeric.loc[sliceIdx, yVar]
        colorValues = None
        if colorSeries is not None:
          colorValues = colorSeries.loc[sliceIdx]
          if not np.issubdtype(colorValues.dtype, np.number):
            colorValues = colorValues.astype(str)

        scatterKwargs = dict(alpha=0.8, edgecolors='k', linewidths=0.2, s=30)
        if colorValues is not None:
          scatterKwargs['c'] = colorValues
          if cmap and np.issubdtype(colorSeries.dtype, np.number):
            scatterKwargs['cmap'] = cmap
        ax.scatter(x, y, **scatterKwargs)
        ax.set_xlabel(xVar)
        ax.set_ylabel(yVar)
        ax.grid(alpha=0.2)

      # Hide unused axes if page_combos < grid size
      totalAxes = pageRows * ncols
      for idx in range(len(pageCombos), totalAxes):
        axes.flat[idx].set_visible(False)

      title = 'Prosection matrix'
      if nFigs > 1:
        title += f' ({fig_idx + 1}/{n_figs})'
      fig.suptitle(title)
      fig.tight_layout()
      # Respect <filename> if provided, but suffix per page to avoid overwriting.
      # NOTE: PlotInterface._createFilename always prefers self.filename over defaultName,
      # so we manually apply the same rules here using the per-page name.
      baseName = self.filename if self.filename is not None else f'{self.name}.png'
      root, ext = os.path.splitext(baseName)
      ext = ext if ext else '.png'
      pageName = f'{root}{ext}' if nFigs == 1 else f'{root}_{fig_idx + 1}{ext}'

      prefix = '' if self.overwrite else f'{self.counter}-'
      filename = f'{prefix}{page_name}'
      if self.subDirectory is not None:
        filename = os.path.join(self.subDirectory, filename)
      outDir = os.path.dirname(filename)
      if outDir:
        os.makedirs(outDir, exist_ok=True)
      fig.savefig(filename, dpi=150)
      plt.close(fig)
