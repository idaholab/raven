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
Render glyph-based (radar/spider) plots for representative samples.

Each selected sample is displayed as a radial glyph whose spokes correspond to
variables. Ideal for what-if comparisons such as: "Which variables differentiate
my top five Pareto solutions, and how does that profile change if I include a
different generation or constraint scenario?"
"""

import math

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
        descr=r"""Selection strategy:
              - "top": best samples by <metric> (or first rows if no metric)
              - "random": random selection (deterministic seed)
              - "leaders": prefer one representative per generation, earliest-to-latest
              - "diverse": greedy farthest-point selection to maximize diversity in variable space
              Default is "top"."""))
    spec.addSub(InputData.parameterInputFactory('count', contentType=InputTypes.IntegerType,
        descr=r"""Number of samples to display (default 6)."""))
    spec.addSub(InputData.parameterInputFactory('label', contentType=InputTypes.StringType,
        descr=r"""Optional label column for the glyph titles."""))
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringType,
        descr=r"""Optional numeric column used to order samples when select="top"."""))
    spec.addSub(InputData.parameterInputFactory('metricGoal', contentType=InputTypes.StringType,
        descr=r"""When <metric> is provided, interpret it as "min" (default) or "max" for ordering."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> present, restrict to a specific generation."""))
    spec.addSub(InputData.parameterInputFactory('scale', contentType=InputTypes.StringType,
        descr=r"""Min-max scaling reference for glyph axes.
              Options:
                - selected (default): scale each variable using only the selected samples
                - population: scale using the full filtered population (can highlight small differences)."""))
    spec.addSub(InputData.parameterInputFactory('deduplicate', contentType=InputTypes.BoolType,
        descr=r"""If true (default), do not repeat the same solution more than once. Solutions are compared using the
              plotted <variables> and <deduplicateTol>. When select="leaders", if the generation-best solution was already
              seen, the next-best in that generation is chosen."""))
    spec.addSub(InputData.parameterInputFactory('deduplicateTol', contentType=InputTypes.FloatType,
        descr=r"""Absolute tolerance used when identifying duplicate solutions in variable space (default 1e-9)."""))
    spec.addSub(InputData.parameterInputFactory('labelFirstSeen', contentType=InputTypes.BoolType,
        descr=r"""If true (default), include a "first seen" generation label when <index> is provided."""))
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
    self.metricGoal = 'min'
    self.index = None
    self.generation = None
    self.scale = 'selected'
    self.deduplicate = True
    self.deduplicateTol = 1.0e-9
    self.labelFirstSeen = True

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

    metricGoalNode = spec.findFirst('metricGoal')
    if metricGoalNode is not None and metricGoalNode.value:
      value = str(metricGoalNode.value).strip().lower()
      if value not in ('min', 'max'):
        self.raiseAnError(IOError, f'Invalid <metricGoal> "{metricGoalNode.value}" for GlyphRadarPlot "{self.name}".')
      self.metricGoal = value

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

    scaleNode = spec.findFirst('scale')
    if scaleNode is not None and scaleNode.value:
      value = str(scaleNode.value).strip().lower()
      if value not in ('selected', 'population'):
        self.raiseAnError(IOError, f'Invalid <scale> "{scaleNode.value}" for GlyphRadarPlot "{self.name}".')
      self.scale = value

    dedupeNode = spec.findFirst('deduplicate')
    if dedupeNode is not None and dedupeNode.value is not None:
      self.deduplicate = bool(dedupeNode.value)

    tolNode = spec.findFirst('deduplicateTol')
    if tolNode is not None and tolNode.value is not None:
      self.deduplicateTol = float(tolNode.value)
      if self.deduplicateTol < 0:
        self.raiseAnError(IOError, f'GlyphRadarPlot "{self.name}" received negative <deduplicateTol>.')

    firstSeenNode = spec.findFirst('labelFirstSeen')
    if firstSeenNode is not None and firstSeenNode.value is not None:
      self.labelFirstSeen = bool(firstSeenNode.value)

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

  def _signature(self, row):
    """
      Create a stable signature for a solution based on plotted variables.
      @ In, row, pd.Series or array-like
      @ Out, sig, tuple
    """
    vals = np.asarray([row[var] for var in self.variables], dtype=float)
    if self.deduplicateTol <= 0:
      return tuple(np.round(vals, 12))
    return tuple(np.round(vals / self.deduplicateTol).astype(np.int64))

  def _selectSamples(self, df):
    df = df.copy()
    if self.select == 'random':
      selection = df.sample(n=min(self.count, len(df)), random_state=0)
      if not self.deduplicate:
        return selection
      # Drop duplicates (in plotted variable space) while preserving order.
      keep = []
      seen = set()
      for idx in selection.index:
        sig = self._signature(selection.loc[idx])
        if sig in seen:
          continue
        seen.add(sig)
        keep.append(idx)
      return selection.loc[keep]
    ascending = True if self.metricGoal == 'min' else False
    if self.select == 'leaders':
      if not self.index or self.index not in df.columns:
        self.raiseAWarning(f'GlyphRadarPlot "{self.name}" select="leaders" requires <index>; falling back to select="top".')
        self.select = 'top'
      else:
        if self.metric and self.metric in df.columns:
          ordered = df.sort_values(by=[self.index, self.metric], ascending=[True, ascending])
        else:
          ordered = df.sort_values(by=[self.index], ascending=True)
        gens = sorted(ordered[self.index].astype(float).unique().tolist())
        if not gens:
          return ordered.head(0)
        picked = []
        seen = set()
        for gen in gens:
          genRows = ordered[ordered[self.index] == gen]
          if genRows.empty:
            continue
          for idx in genRows.index:
            if not self.deduplicate:
              picked.append(idx)
              break
            sig = self._signature(genRows.loc[idx])
            if sig in seen:
              continue
            seen.add(sig)
            picked.append(idx)
            break
          if len(picked) >= self.count:
            break
        leaders = ordered.loc[picked] if picked else ordered.head(0)
        if len(leaders) >= self.count:
          return leaders.head(self.count)
        remaining = df.drop(index=leaders.index, errors='ignore')
        if remaining.empty:
          return leaders
        extrasNeeded = self.count - len(leaders)
        extraIdx = self._selectDiverseIndices(remaining, seedIndices=list(leaders.index), count=extrasNeeded, excludedSigs=seen)
        extras = remaining.loc[extraIdx] if extraIdx else remaining.head(0)
        return pd.concat([leaders, extras], axis=0)
    if self.select == 'diverse':
      excludedSigs = set()
      if self.deduplicate:
        excludedSigs = set()
      idxs = self._selectDiverseIndices(df, seedIndices=[], count=self.count, excludedSigs=excludedSigs)
      if not idxs:
        return df.head(0)
      return df.loc[idxs]
    if self.metric and self.metric in df.columns:
      ordered = df.sort_values(by=self.metric, ascending=ascending)
    else:
      ordered = df
    if not self.deduplicate:
      return ordered.head(self.count)
    keep = []
    seen = set()
    for idx in ordered.index:
      sig = self._signature(ordered.loc[idx])
      if sig in seen:
        continue
      seen.add(sig)
      keep.append(idx)
      if len(keep) >= self.count:
        break
    return ordered.loc[keep]

  def _selectDiverseIndices(self, df, *, seedIndices, count, excludedSigs=None):
    """
      Select indices from df that are diverse in variable space.
      Greedy farthest-point sampling in min-max scaled coordinates.
      @ In, df, pd.DataFrame, candidates (must include self.variables)
      @ In, seed_indices, list, indices already selected (may include indices not in df)
      @ In, count, int, number of additional indices to pick
      @ In, excluded_sigs, set, solution signatures that should not be re-selected
      @ Out, indices, list, selected indices (length <= count)
    """
    if count <= 0 or df.empty:
      return []
    excludedSigs = excludedSigs or set()
    numeric = df[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      return []
    # Filter out excluded signatures early.
    if excludedSigs:
      keep = []
      for idx in numeric.index:
        sig = self._signature(df.loc[idx])
        if sig in excludedSigs:
          continue
        keep.append(idx)
      numeric = numeric.loc[keep]
      if numeric.empty:
        return []
    scaled = self._minmaxScale(numeric).to_numpy(dtype=float)
    candIndex = list(numeric.index)
    # Build an initial set of selected points from seeds that are present in candidates.
    selectedPositions = []
    seedSet = set(seedIndices or [])
    for pos, idx in enumerate(candIndex):
      if idx in seedSet:
        selectedPositions.append(pos)
    # If no seeds present among candidates, start from a deterministic point.
    if not selectedPositions:
      start = 0
      if self.metric and self.metric in df.columns:
        ascending = True if self.metricGoal == 'min' else False
        ordered = df.loc[numeric.index].sort_values(by=self.metric, ascending=ascending)
        if not ordered.empty:
          startIdx = ordered.index[0]
          try:
            start = candIndex.index(startIdx)
          except ValueError:
            start = 0
      selectedPositions = [start]
    chosen = set(selectedPositions)
    picks = []
    # Precompute per-candidate min distance to selected set, update incrementally.
    selPts = scaled[selectedPositions, :]
    minDist = np.full(len(candIndex), np.inf, dtype=float)
    for p in selectedPositions:
      d = np.linalg.norm(scaled - scaled[p], axis=1)
      minDist = np.minimum(minDist, d)
    for _ in range(min(count, len(candIndex) - len(chosen))):
      # Do not pick already-selected points.
      minDist[list(chosen)] = -1.0
      nextPos = int(np.argmax(minDist))
      if minDist[nextPos] < 0:
        break
      chosen.add(nextPos)
      picks.append(candIndex[nextPos])
      d = np.linalg.norm(scaled - scaled[nextPos], axis=1)
      minDist = np.minimum(minDist, d)
    return picks

  @staticmethod
  def _minmaxScale(df):
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
      elif self.select != 'leaders':
        maxGen = subset[self.index].max()
        subset = subset[subset[self.index] == maxGen]
    if subset.empty:
      self.raiseAWarning(f'GlyphRadarPlot "{self.name}" had no samples after filtering.')
      return

    numeric = subset[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      self.raiseAWarning(f'GlyphRadarPlot "{self.name}" found no finite samples for variables {self.variables}.')
      return

    selectedIdx = self._selectSamples(subset.loc[numeric.index])
    if selectedIdx.empty:
      self.raiseAWarning(f'GlyphRadarPlot "{self.name}" could not select samples with strategy "{self.select}".')
      return

    selectedNumeric = numeric.loc[selectedIdx.index]
    if self.scale == 'population':
      scaled = self._minmaxScale(numeric).loc[selectedIdx.index]
    else:
      scaled = self._minmaxScale(selectedNumeric)

    firstSeen = {}
    if self.index and self.labelFirstSeen and self.index in subset.columns:
      # Determine the first generation a solution appeared (based on variable signature).
      try:
        gens = subset[self.index].astype(float)
      except Exception:
        gens = None
      if gens is not None:
        for idx in numeric.index:
          sig = self._signature(subset.loc[idx])
          gen = float(subset.loc[idx, self.index])
          if sig not in firstSeen or gen < firstSeen[sig]:
            firstSeen[sig] = gen

    nVars = len(self.variables)
    angles = np.linspace(0.0, 2.0 * np.pi, num=nVars, endpoint=False).tolist()
    angles += angles[:1]

    nSamples = len(selectedIdx)
    ncols = min(3, nSamples)
    nrows = int(math.ceil(nSamples / ncols))
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
      ax.fill(angles, values, color='tab:blue', alpha=0.35)
      # Titles: make them unique and informative (trajID alone is often identical across samples).
      parts = []
      if self.index and self.index in selectedIdx.columns:
        try:
          parts.append(f'{self.index}={float(selectedIdx.loc[idx, self.index]):.0f}')
        except Exception:
          parts.append(f'{self.index}={selectedIdx.loc[idx, self.index]}')
        if self.labelFirstSeen:
          sig = self._signature(selectedIdx.loc[idx])
          if sig in firstSeen:
            parts.append(f'first={firstSeen[sig]:.0f}')
      if self.labelVar and self.labelVar in selectedIdx.columns:
        parts.append(f'{self.labelVar}={selectedIdx.loc[idx, self.labelVar]}')
      if self.metric and self.metric in selectedIdx.columns:
        parts.append(f'{self.metric}={float(selectedIdx.loc[idx, self.metric]):.4g}')
      if not parts:
        parts.append(f'{self.select} sample')
      title = ', '.join(parts)
      ax.set_title(title, fontsize=9)

    fig.suptitle('Glyph-based radar profiles')
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
