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
Pareto chart (sorted bars + cumulative percentage line).

Two supported input modes:
  1) Category column mode:
       - <category> points to a column containing category labels (string-like)
       - optional <value> points to a numeric column to sum (defaults to counts)
  2) Constraint mode:
       - <constraints> lists constraint evaluation columns or "all"
       - bars show the count of violations per constraint (ConstraintEvaluation_* <= 0)

This is useful for quickly identifying the "vital few" drivers (e.g., which
constraints are most frequently violated, or which failure reasons dominate).
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ParetoChartPlot(PlotInterface):
  """Pareto chart for categorical frequency/weighting."""

  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing the samples."""))
    spec.addSub(InputData.parameterInputFactory('category', contentType=InputTypes.StringType,
        descr=r"""Category column name (category-column mode)."""))
    spec.addSub(InputData.parameterInputFactory('value', contentType=InputTypes.StringType,
        descr=r"""Optional numeric value column to sum per category (category-column mode). Defaults to counts."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Constraint evaluation columns (constraint mode). Values > 0 are feasible; values <= 0 are violations.
                   Use "all" to include every column named ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> provided, limits the plot to the specified generation. If omitted, uses the last generation."""))
    spec.addSub(InputData.parameterInputFactory('use_all_generations', contentType=InputTypes.BoolType,
        descr=r"""When <index> is provided, use all generations instead of only the last generation (or <generation>). Default false."""))
    spec.addSub(InputData.parameterInputFactory('top_n', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on categories shown (after sorting). Remaining categories are grouped into "Other" when <include_other> is true."""))
    spec.addSub(InputData.parameterInputFactory('include_other', contentType=InputTypes.BoolType,
        descr=r"""If true, collapses categories beyond <top_n> into an "Other" bucket. Default true."""))
    spec.addSub(InputData.parameterInputFactory('title', contentType=InputTypes.StringType,
        descr=r"""Optional plot title."""))
    spec.addSub(InputData.parameterInputFactory('bar_label', contentType=InputTypes.StringType,
        descr=r"""Optional y-axis label for bars (default 'Frequency')."""))
    spec.addSub(InputData.parameterInputFactory('line_label', contentType=InputTypes.StringType,
        descr=r"""Optional y-axis label for cumulative line (default 'Cumulative %')."""))
    spec.addSub(InputData.parameterInputFactory('bar_color', contentType=InputTypes.StringType,
        descr=r"""Bar color (default '#1f77b4')."""))
    spec.addSub(InputData.parameterInputFactory('line_color', contentType=InputTypes.StringType,
        descr=r"""Cumulative line color (default '#ff7f0e')."""))
    spec.addSub(InputData.parameterInputFactory('rotate', contentType=InputTypes.FloatType,
        descr=r"""X tick rotation in degrees (default 20)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ParetoChartPlot'
    self.source = None
    self.sourceName = None
    self.category = None
    self.value = None
    self.constraints = []
    self.useAllConstraints = False
    self.index = None
    self.generation = None
    self.useAllGenerations = False
    self.topN = None
    self.includeOther = True
    self.title = None
    self.barLabel = 'Frequency'
    self.lineLabel = 'Cumulative %'
    self.barColor = '#1f77b4'
    self.lineColor = '#ff7f0e'
    self.rotate = 20.0

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'Missing <source> node for ParetoChartPlot "{self.name}".')
    self.sourceName = src.value

    catNode = spec.findFirst('category')
    if catNode is not None and catNode.value:
      self.category = str(catNode.value)
    valNode = spec.findFirst('value')
    if valNode is not None and valNode.value:
      self.value = str(valNode.value)

    consNode = spec.findFirst('constraints')
    if consNode is not None and consNode.value:
      entries = [str(entry).strip() for entry in consNode.value if str(entry).strip()]
      if any(entry.lower() == 'all' for entry in entries):
        self.useAllConstraints = True
      else:
        self.constraints = entries

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = str(idxNode.value)
    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)
    uagNode = spec.findFirst('use_all_generations')
    if uagNode is not None and uagNode.value is not None:
      self.useAllGenerations = bool(uagNode.value)

    tnNode = spec.findFirst('top_n')
    if tnNode is not None and tnNode.value is not None:
      self.topN = int(tnNode.value)
      if self.topN <= 0:
        self.raiseAnError(IOError, f'ParetoChartPlot "{self.name}" received non-positive <top_n>.')
    ioNode = spec.findFirst('include_other')
    if ioNode is not None and ioNode.value is not None:
      self.includeOther = bool(ioNode.value)

    titleNode = spec.findFirst('title')
    if titleNode is not None and titleNode.value:
      self.title = str(titleNode.value)
    blNode = spec.findFirst('bar_label')
    if blNode is not None and blNode.value:
      self.barLabel = str(blNode.value)
    llNode = spec.findFirst('line_label')
    if llNode is not None and llNode.value:
      self.lineLabel = str(llNode.value)
    bcNode = spec.findFirst('bar_color')
    if bcNode is not None and bcNode.value:
      self.barColor = str(bcNode.value)
    lcNode = spec.findFirst('line_color')
    if lcNode is not None and lcNode.value:
      self.lineColor = str(lcNode.value)
    rotNode = spec.findFirst('rotate')
    if rotNode is not None and rotNode.value is not None:
      self.rotate = float(rotNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for ParetoChartPlot "{self.name}".')
    available = src.getVars()
    if self.useAllConstraints:
      self.constraints = sorted(var for var in available if var.startswith('ConstraintEvaluation_'))
    needed = set()
    if self.category is not None:
      needed.add(self.category)
    if self.value is not None:
      needed.add(self.value)
    if self.constraints:
      needed.update(self.constraints)
    if self.index:
      needed.add(self.index)
    if not self.constraints and self.category is None:
      self.raiseAnError(IOError, f'ParetoChartPlot "{self.name}" requires either <constraints> or <category>.')
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" is missing variable(s) {missing} required by ParetoChartPlot "{self.name}".')
    self.source = src

  @staticmethod
  def _collapse_top(series, top_n, include_other):
    if top_n is None or top_n >= len(series):
      return series
    head = series.iloc[:top_n].copy()
    if include_other:
      other_sum = float(series.iloc[top_n:].sum())
      head.loc['Other'] = other_sum
    return head

  def _extract_series(self, df):
    if self.constraints:
      counts = {}
      for c in self.constraints:
        vals = pd.to_numeric(df[c], errors='coerce')
        counts[c] = float((vals <= 0.0).sum())
      series = pd.Series(counts, dtype=float).sort_values(ascending=False)
      return series

    # category-column mode
    cat = df[self.category].astype(str)
    if self.value is None:
      series = cat.value_counts(dropna=False).astype(float)
      return series
    vals = pd.to_numeric(df[self.value], errors='coerce').fillna(0.0)
    series = vals.groupby(cat).sum().sort_values(ascending=False)
    return series

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'ParetoChartPlot "{self.name}" received an empty dataset; skipping.')
      return

    subset = df.copy()
    if self.index and not self.useAllGenerations:
      subset[self.index] = pd.to_numeric(subset[self.index], errors='coerce')
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if subset.empty:
      self.raiseAWarning(f'ParetoChartPlot "{self.name}" had no samples after filtering.')
      return

    series = self._extract_series(subset)
    series = series[series > 0.0]
    if series.empty:
      self.raiseAWarning(f'ParetoChartPlot "{self.name}" had no positive counts to plot.')
      return

    series = self._collapse_top(series, self.topN, self.includeOther)
    total = float(series.sum())
    cumulative = series.cumsum() / total * 100.0

    labels = list(series.index.astype(str))
    values = series.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(values))
    ax.bar(x, values, color=self.barColor, alpha=0.9)
    ax.set_ylabel(self.barLabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=float(self.rotate), ha='right')
    ax.grid(axis='y', alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x, cumulative.to_numpy(dtype=float), color=self.lineColor, linewidth=2.2, marker='o', markersize=4)
    ax2.set_ylim(0.0, 105.0)
    ax2.set_ylabel(self.lineLabel)

    title = self.title
    if title is None:
      title = 'Pareto chart'
      if self.constraints:
        title += ' (constraint violations)'
    ax.set_title(title)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=160)
    plt.close(fig)

