"""
Train a simple self-organising map (SOM) on optimisation samples and visualise
the resulting lattice.

The SOM clusters high-dimensional points onto a 2-D grid where nearby nodes
represent similar solutions. Great for what-if questions like: "If I train on
only the latest generation, which regions of the map dominate the Pareto set,
and how would including earlier generations reshape the neighbourhoods?"
"""

import math
import random

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class SelfOrganizingMapPlot(PlotInterface):
  """
  Project samples onto a 2-D self-organising map lattice. Toggle generation
  filters or colour variables to investigate what-if scenarios: how the map
  occupancy changes if you emphasise a different objective, or how constraint
  handling shifts the clusters that survive to the final iteration.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing the samples to train on."""))
    variables = InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""List of variables to feed the SOM (at least two).""")
    spec.addSub(variables)
    spec.addSub(InputData.parameterInputFactory('grid', contentType=InputTypes.StringType,
        descr=r"""Grid shape "rows,cols" for the SOM lattice (default 10,10)."""))
    spec.addSub(InputData.parameterInputFactory('iterations', contentType=InputTypes.IntegerType,
        descr=r"""Number of training iterations (default 500)."""))
    spec.addSub(InputData.parameterInputFactory('color', contentType=InputTypes.StringType,
        descr=r"""Optional variable whose mean per node controls colour."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> present, restrict to a specific generation."""))
    spec.addSub(InputData.parameterInputFactory('seed', contentType=InputTypes.IntegerType,
        descr=r"""Random seed for reproducible map initialisation."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'SelfOrganizingMapPlot'
    self.source = None
    self.sourceName = None
    self.variables = []
    self.grid = (10, 10)
    self.iterations = 500
    self.colorVar = None
    self.index = None
    self.generation = None
    self.seed = None

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for SelfOrganizingMapPlot "{self.name}".')
    self.sourceName = src.value

    varNode = spec.findFirst('variables')
    if varNode is None or len(varNode.value) < 2:
      self.raiseAnError(IOError, f'SelfOrganizingMapPlot "{self.name}" requires at least two <variables>.')
    self.variables = [entry for entry in varNode.value if entry]

    gridNode = spec.findFirst('grid')
    if gridNode is not None and gridNode.value:
      parts = [frag.strip() for frag in str(gridNode.value).split(',') if frag.strip()]
      if len(parts) != 2:
        self.raiseAnError(IOError, f'Invalid <grid> "{gridNode.value}" for SelfOrganizingMapPlot "{self.name}". Expected "rows,cols".')
      rows, cols = int(parts[0]), int(parts[1])
      if rows <= 0 or cols <= 0:
        self.raiseAnError(IOError, f'Grid dimensions must be positive for SelfOrganizingMapPlot "{self.name}".')
      self.grid = (rows, cols)

    iterNode = spec.findFirst('iterations')
    if iterNode is not None and iterNode.value is not None:
      value = int(iterNode.value)
      if value <= 0:
        self.raiseAnError(IOError, f'Invalid <iterations> {value} for SelfOrganizingMapPlot "{self.name}".')
      self.iterations = value

    colorNode = spec.findFirst('color')
    if colorNode is not None and colorNode.value:
      self.colorVar = colorNode.value

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    seedNode = spec.findFirst('seed')
    if seedNode is not None and seedNode.value is not None:
      self.seed = int(seedNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for SelfOrganizingMapPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.variables)
    if self.colorVar:
      needed.append(self.colorVar)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" missing variable(s) {missing} required by SelfOrganizingMapPlot "{self.name}".')
    self.source = src

  @staticmethod
  def _scale(data):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    scaled = (data - mins) / np.where(ranges == 0.0, 1.0, ranges)
    return scaled, mins, ranges

  def _train_som(self, data):
    rows, cols = self.grid
    n_features = data.shape[1]
    rng = np.random.default_rng(self.seed)
    weights = rng.random((rows, cols, n_features))
    positions = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)

    time_constant = max(rows, cols) / math.log(max(rows, cols))
    learning_rate0 = 0.5
    sigma0 = max(rows, cols) / 2.0
    for t in range(self.iterations):
      sample = data[rng.integers(0, data.shape[0])]
      diff = weights - sample
      dist = np.sum(diff * diff, axis=2)
      bmu_index = np.unravel_index(np.argmin(dist), (rows, cols))
      bmu_pos = np.array(bmu_index, dtype=float)

      lr = learning_rate0 * math.exp(-t / self.iterations)
      sigma = sigma0 * math.exp(-t / time_constant) if time_constant > 0 else sigma0
      if sigma < 1e-6:
        sigma = 1e-6

      # Update neighbourhood
      grid_positions = positions.reshape(rows, cols, 2)
      sq_dist = np.sum((grid_positions - bmu_pos) ** 2, axis=2)
      influence = np.exp(-sq_dist / (2.0 * sigma * sigma))
      weights += lr * influence[..., np.newaxis] * (sample - weights)
    return weights

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'SelfOrganizingMapPlot "{self.name}" skipped because source "{self.source.name}" is empty.')
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
      self.raiseAWarning(f'SelfOrganizingMapPlot "{self.name}" had no samples after filtering.')
      return

    numeric = subset[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      self.raiseAWarning(f'SelfOrganizingMapPlot "{self.name}" found no finite samples for {self.variables}.')
      return

    data = numeric.to_numpy(dtype=float)
    scaled_data, mins, ranges = self._scale(data)
    weights = self._train_som(scaled_data)

    rows, cols = self.grid
    grid_positions = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)
    # Assign samples to BMUs
    diff = weights.reshape(rows * cols, -1)[np.newaxis, :, :] - scaled_data[:, np.newaxis, :]
    dists = np.sum(diff * diff, axis=2)
    assignments = np.argmin(dists, axis=1)

    counts = np.bincount(assignments, minlength=rows * cols).reshape(rows, cols)
    color_values = None
    if self.colorVar and self.colorVar in subset.columns:
      color_series = subset.loc[numeric.index, self.colorVar]
      if np.issubdtype(color_series.dtype, np.number):
        sums = np.zeros((rows * cols,), dtype=float)
        sums += np.bincount(assignments, weights=color_series.to_numpy(dtype=float),
                            minlength=rows * cols)
        with np.errstate(invalid='ignore'):
          averages = np.divide(sums, counts.reshape(-1), where=counts.reshape(-1) > 0)
        color_values = averages.reshape(rows, cols)
      else:
        # For categorical, store dominant label index
        labels = color_series.astype(str).to_numpy()
        unique_labels = sorted(set(labels))
        label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
        label_sums = np.zeros((rows * cols, len(unique_labels)), dtype=float)
        for idx, node in enumerate(assignments):
          label_sums[node, label_to_idx[labels[idx]]] += 1.0
        dominant = np.argmax(label_sums, axis=1)
        color_values = dominant.reshape(rows, cols)
        color_values = (color_values, unique_labels)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    im0 = axes[0].imshow(counts, cmap='Blues', origin='lower')
    axes[0].set_title('SOM occupancy')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if color_values is None:
      im1 = axes[1].imshow(np.sqrt(counts), cmap='viridis', origin='lower')
      axes[1].set_title('Node intensity (sqrt counts)')
      fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    elif isinstance(color_values, tuple):
      values, labels = color_values
      im1 = axes[1].imshow(values, cmap='tab20', origin='lower', vmin=0, vmax=len(labels) - 1)
      axes[1].set_title(f'Dominant {self.colorVar}')
      cbar = fig.colorbar(im1, ax=axes[1], ticks=range(len(labels)), fraction=0.046, pad=0.04)
      cbar.ax.set_yticklabels(labels)
    else:
      im1 = axes[1].imshow(color_values, cmap='coolwarm', origin='lower')
      axes[1].set_title(f'Mean {self.colorVar}')
      fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle('Self-organising map projection')
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
