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
Overlay a smooth response surface (contours) with sampled optimizer points.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ResponseSurfaceOverlayPlot(PlotInterface):
  """
  Construct a contour map of an objective (or metric) across two variables and overlay sampled points.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    axes = InputData.parameterInputFactory('axes', contentType=InputTypes.StringListType,
        descr=r"""Two variable names providing the surface coordinates.""")
    spec.addSub(axes)
    spec.addSub(InputData.parameterInputFactory('response', contentType=InputTypes.StringType,
        descr=r"""Objective or metric column to contour."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> provided, optionally select a specific generation."""))
    spec.addSub(InputData.parameterInputFactory('levels', contentType=InputTypes.IntegerType,
        descr=r"""Number of contour levels (default 12)."""))
    spec.addSub(InputData.parameterInputFactory('cmap', contentType=InputTypes.StringType,
        descr=r"""Matplotlib colormap name (default viridis)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ResponseSurfaceOverlayPlot'
    self.source = None
    self.sourceName = None
    self.axes = []
    self.response = None
    self.index = None
    self.generation = None
    self.levels = 12
    self.cmap = 'viridis'

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for ResponseSurfaceOverlayPlot "{self.name}".')
    self.sourceName = src.value

    axesNode = spec.findFirst('axes')
    if axesNode is None or len(axesNode.value) != 2:
      self.raiseAnError(IOError, f'ResponseSurfaceOverlayPlot "{self.name}" requires exactly two <axes>.')
    self.axes = [entry for entry in axesNode.value if entry]

    respNode = spec.findFirst('response')
    if respNode is None or not respNode.value:
      self.raiseAnError(IOError, f'Missing <response> node for ResponseSurfaceOverlayPlot "{self.name}".')
    self.response = respNode.value

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value
    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    levelNode = spec.findFirst('levels')
    if levelNode is not None and levelNode.value is not None:
      self.levels = max(3, int(levelNode.value))

    cmapNode = spec.findFirst('cmap')
    if cmapNode is not None and cmapNode.value:
      self.cmap = cmapNode.value.strip()

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for ResponseSurfaceOverlayPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.axes) + [self.response]
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by ResponseSurfaceOverlayPlot "{self.name}".')
    self.source = src

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'Source DataObject "{self.source.name}" is empty; ResponseSurfaceOverlayPlot "{self.name}" skipped.')
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
      self.raiseAWarning(f'ResponseSurfaceOverlayPlot "{self.name}" had no rows after filtering.')
      return

    columns = [self.axes[0], self.axes[1], self.response]
    points = subset[columns].apply(pd.to_numeric, errors='coerce').dropna()
    if points.shape[0] < 3:
      self.raiseAWarning(f'Not enough finite samples to build surface for ResponseSurfaceOverlayPlot "{self.name}".')
      return
    points = points.groupby([self.axes[0], self.axes[1]], as_index=False).mean()
    if points.shape[0] < 3:
      self.raiseAWarning(f'ResponseSurfaceOverlayPlot "{self.name}" collapsed to fewer than three unique points after deduplication.')
      return
    coords = points[[self.axes[0], self.axes[1]]].to_numpy()
    centered = coords - coords.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
      self.raiseAWarning(f'ResponseSurfaceOverlayPlot "{self.name}" has nearly collinear samples; falling back to scatter-only visualization.')
      triang = None
    else:
      try:
        triang = mtri.Triangulation(coords[:, 0], coords[:, 1])
      except RuntimeError as err:
        self.raiseAWarning(f'Unable to build Delaunay triangulation for ResponseSurfaceOverlayPlot "{self.name}": {err}')
        triang = None

    x = points[self.axes[0]].to_numpy()
    y = points[self.axes[1]].to_numpy()
    z = points[self.response].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    contourf = None
    if triang is not None and triang.triangles.size > 0:
      contourf = ax.tricontourf(triang, z, levels=self.levels, cmap=self.cmap, alpha=0.8)
      ax.tricontour(triang, z, levels=self.levels, colors='k', linewidths=0.4, alpha=0.6)
    else:
      self.raiseAWarning(f'ResponseSurfaceOverlayPlot "{self.name}" could not build contours; rendering scatter overlay only.')
    ax.scatter(x, y, c=z, cmap=self.cmap, edgecolors='w', linewidths=0.3, s=30, alpha=0.9)
    ax.set_xlabel(self.axes[0])
    ax.set_ylabel(self.axes[1])
    ax.set_title(f'Response surface overlay ({self.response})')
    if contourf is not None:
      cbar = fig.colorbar(contourf, ax=ax)
      cbar.set_label(self.response)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
