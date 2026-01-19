"""
Render glyph-based (radar/spider) plots for representative samples.

Each selected sample is displayed as a radial glyph whose spokes correspond to
variables. Ideal for what-if comparisons such as: "Which variables differentiate
my top five Pareto solutions, and how does that profile change if I include a
different generation or constraint scenario?"
"""

import math

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class GlyphRadarPlot(PlotInterface):
  """
  Plot selected samples as radar glyphs. Adjust `<select>` filters or choose
  different ranking metrics to explore what-if scenarios—for example, how the
  glyph profiles evolve when the optimiser prioritises another objective or
  when you compare early vs late-generation candidates.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing samples for glyphs."""))
    variables = InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""Variables to include on each glyph (at least three).""")
    spec.addSub(variables)
    spec.addSub(InputData.parameterInputFactory('select', contentType=InputTypes.StringType,
        descr=r"""Selection strategy: "top", "random", or "leaders" (default top)."""))
    spec.addSub(InputData.parameterInputFactory('count', contentType=InputTypes.IntegerType,
        descr=r"""Number of samples to display (default 6)."""))
    spec.addSub(InputData.parameterInputFactory('label', contentType=InputTypes.StringType,
        descr=r"""Optional label column for the glyph titles."""))
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringType,
        descr=r"""Optional numeric column used to order samples when select="top"."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> present, restrict to a specific generation."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'GlyphRadarPlot'
    self.source = None
    self.sourceName = None
    self.variables = []
    self.select = 'top'
    self.count = 6
    self.labelVar = None
    self.metric = None
    self.index = None
    self.generation = None

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for GlyphRadarPlot "{self.name}".')
    self.sourceName = src.value

    varNode = spec.findFirst('variables')
    if varNode is None or len(varNode.value) < 3:
      self.raiseAnError(IOError, f'GlyphRadarPlot "{self.name}" requires at least three <variables>.')
    self.variables = [entry for entry in varNode.value if entry]

    selectNode = spec.findFirst('select')
    if selectNode is not None and selectNode.value:
      value = selectNode.value.strip().lower()
      if value not in ('top', 'random', 'leaders'):
        self.raiseAnError(IOError, f'Invalid <select> "{selectNode.value}" for GlyphRadarPlot "{self.name}".')
      self.select = value

    countNode = spec.findFirst('count')
    if countNode is not None and countNode.value is not None:
      value = int(countNode.value)
      if value <= 0:
        self.raiseAnError(IOError, f'Count must be positive for GlyphRadarPlot "{self.name}".')
      self.count = value

    labelNode = spec.findFirst('label')
    if labelNode is not None and labelNode.value:
      self.labelVar = labelNode.value

    metricNode = spec.findFirst('metric')
    if metricNode is not None and metricNode.value:
      self.metric = metricNode.value

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for GlyphRadarPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.variables)
    if self.labelVar:
      needed.append(self.labelVar)
    if self.metric:
      needed.append(self.metric)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" missing variable(s) {missing} required by GlyphRadarPlot "{self.name}".')
    self.source = src

  def _select_samples(self, df):
    df = df.copy()
    if self.select == 'random':
      return df.sample(n=min(self.count, len(df)), random_state=0)
    if self.metric and self.metric in df.columns:
      ordered = df.sort_values(by=self.metric, ascending=True)
    else:
      ordered = df
    return ordered.head(self.count)

  @staticmethod
  def _minmax_scale(df):
    mins = df.min(axis=0)
    maxs = df.max(axis=0)
    ranges = maxs - mins
    scaled = (df - mins) / np.where(ranges == 0.0, 1.0, ranges)
    return scaled

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'GlyphRadarPlot "{self.name}" skipped because source "{self.source.name}" is empty.')
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
      self.raiseAWarning(f'GlyphRadarPlot "{self.name}" had no samples after filtering.')
      return

    numeric = subset[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      self.raiseAWarning(f'GlyphRadarPlot "{self.name}" found no finite samples for variables {self.variables}.')
      return

    selected_idx = self._select_samples(subset.loc[numeric.index])
    if selected_idx.empty:
      self.raiseAWarning(f'GlyphRadarPlot "{self.name}" could not select samples with strategy "{self.select}".')
      return

    selected_numeric = numeric.loc[selected_idx.index]
    scaled = self._minmax_scale(selected_numeric)

    n_vars = len(self.variables)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_vars, endpoint=False).tolist()
    angles += angles[:1]

    n_samples = len(selected_idx)
    ncols = min(3, n_samples)
    nrows = int(math.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
      ax.set_axis_off()

    for ax, (idx, row) in zip(axes.flat, scaled.iterrows()):
      values = row.to_numpy(dtype=float).tolist()
      values += values[:1]
      ax.set_axis_on()
      ax.set_xticks(angles[:-1])
      ax.set_xticklabels(self.variables, fontsize=8)
      ax.set_ylim(0, 1)
      ax.plot(angles, values, color='tab:blue', linewidth=2.0)
      ax.fill(angles, values, color='tab:blue', alpha=0.2)
      title = ''
      if self.labelVar and self.labelVar in selected_idx.columns:
        title = str(selected_idx.loc[idx, self.labelVar])
      elif self.metric and self.metric in selected_idx.columns:
        title = f'{self.metric}={selected_idx.loc[idx, self.metric]:.4g}'
      ax.set_title(title, fontsize=9)

    fig.suptitle('Glyph-based radar profiles')
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
