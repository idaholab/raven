"""
Project multi-dimensional samples onto a star-coordinates embedding.

The axes radiate from the origin at equal angles and each point is placed by
accumulating contributions from every variable along its associated ray.
Useful for what-if analysis such as: "If I restrict the plot to the final
generation, do my decision variables still pull the nondominated designs
toward distinct quadrants, or do they blend together after reweighting?"
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class StarCoordinatesPlot(PlotInterface):
  """
  Render multi-dimensional samples in a star coordinates layout. Adjust the
  `<index>`/`<generation>` filters or colour labels to explore what-if
  scenarios where objective priorities or constraint handling shift, and the
  population migrates along different rays.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing the samples."""))
    variables = InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""List of variables to project (at least three).""")
    spec.addSub(variables)
    spec.addSub(InputData.parameterInputFactory('label', contentType=InputTypes.StringType,
        descr=r"""Optional column used to colour the samples."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> present, restrict to a specific generation."""))
    spec.addSub(InputData.parameterInputFactory('normalize', contentType=InputTypes.BoolType,
        descr=r"""If true (default), min-max normalise each variable before projection."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'StarCoordinatesPlot'
    self.source = None
    self.sourceName = None
    self.variables = []
    self.labelVar = None
    self.index = None
    self.generation = None
    self.normalize = True

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for StarCoordinatesPlot "{self.name}".')
    self.sourceName = src.value

    varNode = spec.findFirst('variables')
    if varNode is None or len(varNode.value) < 3:
      self.raiseAnError(IOError, f'StarCoordinatesPlot "{self.name}" requires at least three <variables>.')
    self.variables = [entry for entry in varNode.value if entry]

    labelNode = spec.findFirst('label')
    if labelNode is not None and labelNode.value:
      self.labelVar = labelNode.value

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    normNode = spec.findFirst('normalize')
    if normNode is not None and normNode.value is not None:
      self.normalize = bool(normNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for StarCoordinatesPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.variables)
    if self.labelVar:
      needed.append(self.labelVar)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" missing variable(s) {missing} required by StarCoordinatesPlot "{self.name}".')
    self.source = src

  @staticmethod
  def _minmax_scale(df):
    mins = df.min(axis=0)
    maxs = df.max(axis=0)
    ranges = maxs - mins
    # Avoid division by zero by falling back to zeros
    scaled = df.copy()
    for col in df.columns:
      if ranges[col] == 0:
        scaled[col] = 0.0
      else:
        scaled[col] = (df[col] - mins[col]) / ranges[col]
    return scaled

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'StarCoordinatesPlot "{self.name}" skipped because source "{self.source.name}" is empty.')
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
      self.raiseAWarning(f'StarCoordinatesPlot "{self.name}" had no samples after filtering.')
      return

    numeric = subset[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      self.raiseAWarning(f'StarCoordinatesPlot "{self.name}" found no finite samples for variables {self.variables}.')
      return

    if self.normalize:
      data = self._minmax_scale(numeric)
    else:
      data = numeric

    n_vars = len(self.variables)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_vars, endpoint=False)
    unit_vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    coords = data.to_numpy(dtype=float) @ unit_vectors

    labels = None
    cmap = None
    if self.labelVar and self.labelVar in subset.columns:
      labels = subset.loc[data.index, self.labelVar]
      if np.issubdtype(labels.dtype, np.number):
        cmap = 'viridis'
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    scatter_kwargs = dict(alpha=0.75, edgecolors='k', linewidths=0.3, s=36)
    if labels is not None:
      scatter_kwargs['c'] = labels if cmap else labels.astype(str)
      if cmap:
        scatter_kwargs['cmap'] = cmap
    ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

    # Draw axes
    max_radius = np.linalg.norm(coords, axis=1).max()
    radius = max(max_radius * 1.1, 1.0)
    for angle, var in zip(angles, self.variables):
      ax.plot([0.0, radius * np.cos(angle)], [0.0, radius * np.sin(angle)],
              color='gray', linewidth=1.0, alpha=0.5)
      ax.text(1.05 * radius * np.cos(angle), 1.05 * radius * np.sin(angle),
              var, ha='center', va='center', fontsize=9)

    ax.set_xlabel('Star coord X')
    ax.set_ylabel('Star coord Y')
    ax.set_title('Star coordinates embedding')
    ax.set_aspect('equal', 'box')
    ax.grid(alpha=0.2)
    fig.tight_layout()

    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
