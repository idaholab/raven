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
    spec.addSub(InputData.parameterInputFactory('metric_goal', contentType=InputTypes.StringType,
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
              plotted <variables> and <deduplicate_tol>. When select="leaders", if the generation-best solution was already
              seen, the next-best in that generation is chosen."""))
    spec.addSub(InputData.parameterInputFactory('deduplicate_tol', contentType=InputTypes.FloatType,
        descr=r"""Absolute tolerance used when identifying duplicate solutions in variable space (default 1e-9)."""))
    spec.addSub(InputData.parameterInputFactory('label_first_seen', contentType=InputTypes.BoolType,
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
    self.metric_goal = 'min'
    self.index = None
    self.generation = None
    self.scale = 'selected'
    self.deduplicate = True
    self.deduplicate_tol = 1.0e-9
    self.label_first_seen = True

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

    metricGoalNode = spec.findFirst('metric_goal')
    if metricGoalNode is not None and metricGoalNode.value:
      value = str(metricGoalNode.value).strip().lower()
      if value not in ('min', 'max'):
        self.raiseAnError(IOError, f'Invalid <metric_goal> "{metricGoalNode.value}" for GlyphRadarPlot "{self.name}".')
      self.metric_goal = value

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

    tolNode = spec.findFirst('deduplicate_tol')
    if tolNode is not None and tolNode.value is not None:
      self.deduplicate_tol = float(tolNode.value)
      if self.deduplicate_tol < 0:
        self.raiseAnError(IOError, f'GlyphRadarPlot "{self.name}" received negative <deduplicate_tol>.')

    firstSeenNode = spec.findFirst('label_first_seen')
    if firstSeenNode is not None and firstSeenNode.value is not None:
      self.label_first_seen = bool(firstSeenNode.value)

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
    if self.deduplicate_tol <= 0:
      return tuple(np.round(vals, 12))
    return tuple(np.round(vals / self.deduplicate_tol).astype(np.int64))

  def _select_samples(self, df):
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
    ascending = True if self.metric_goal == 'min' else False
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
          gen_rows = ordered[ordered[self.index] == gen]
          if gen_rows.empty:
            continue
          for idx in gen_rows.index:
            if not self.deduplicate:
              picked.append(idx)
              break
            sig = self._signature(gen_rows.loc[idx])
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
        extras_needed = self.count - len(leaders)
        extra_idx = self._select_diverse_indices(remaining, seed_indices=list(leaders.index), count=extras_needed, excluded_sigs=seen)
        extras = remaining.loc[extra_idx] if extra_idx else remaining.head(0)
        return pd.concat([leaders, extras], axis=0)
    if self.select == 'diverse':
      excluded_sigs = set()
      if self.deduplicate:
        excluded_sigs = set()
      idxs = self._select_diverse_indices(df, seed_indices=[], count=self.count, excluded_sigs=excluded_sigs)
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

  def _select_diverse_indices(self, df, *, seed_indices, count, excluded_sigs=None):
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
    excluded_sigs = excluded_sigs or set()
    numeric = df[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      return []
    # Filter out excluded signatures early.
    if excluded_sigs:
      keep = []
      for idx in numeric.index:
        sig = self._signature(df.loc[idx])
        if sig in excluded_sigs:
          continue
        keep.append(idx)
      numeric = numeric.loc[keep]
      if numeric.empty:
        return []
    scaled = self._minmax_scale(numeric).to_numpy(dtype=float)
    cand_index = list(numeric.index)
    # Build an initial set of selected points from seeds that are present in candidates.
    selected_positions = []
    seed_set = set(seed_indices or [])
    for pos, idx in enumerate(cand_index):
      if idx in seed_set:
        selected_positions.append(pos)
    # If no seeds present among candidates, start from a deterministic point.
    if not selected_positions:
      start = 0
      if self.metric and self.metric in df.columns:
        ascending = True if self.metric_goal == 'min' else False
        ordered = df.loc[numeric.index].sort_values(by=self.metric, ascending=ascending)
        if not ordered.empty:
          start_idx = ordered.index[0]
          try:
            start = cand_index.index(start_idx)
          except ValueError:
            start = 0
      selected_positions = [start]
    chosen = set(selected_positions)
    picks = []
    # Precompute per-candidate min distance to selected set, update incrementally.
    sel_pts = scaled[selected_positions, :]
    min_dist = np.full(len(cand_index), np.inf, dtype=float)
    for p in selected_positions:
      d = np.linalg.norm(scaled - scaled[p], axis=1)
      min_dist = np.minimum(min_dist, d)
    for _ in range(min(count, len(cand_index) - len(chosen))):
      # Do not pick already-selected points.
      min_dist[list(chosen)] = -1.0
      next_pos = int(np.argmax(min_dist))
      if min_dist[next_pos] < 0:
        break
      chosen.add(next_pos)
      picks.append(cand_index[next_pos])
      d = np.linalg.norm(scaled - scaled[next_pos], axis=1)
      min_dist = np.minimum(min_dist, d)
    return picks

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
      elif self.select != 'leaders':
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
    if self.scale == 'population':
      scaled = self._minmax_scale(numeric).loc[selected_idx.index]
    else:
      scaled = self._minmax_scale(selected_numeric)

    first_seen = {}
    if self.index and self.label_first_seen and self.index in subset.columns:
      # Determine the first generation a solution appeared (based on variable signature).
      try:
        gens = subset[self.index].astype(float)
      except Exception:
        gens = None
      if gens is not None:
        for idx in numeric.index:
          sig = self._signature(subset.loc[idx])
          gen = float(subset.loc[idx, self.index])
          if sig not in first_seen or gen < first_seen[sig]:
            first_seen[sig] = gen

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
      ax.fill(angles, values, color='tab:blue', alpha=0.35)
      # Titles: make them unique and informative (trajID alone is often identical across samples).
      parts = []
      if self.index and self.index in selected_idx.columns:
        try:
          parts.append(f'{self.index}={float(selected_idx.loc[idx, self.index]):.0f}')
        except Exception:
          parts.append(f'{self.index}={selected_idx.loc[idx, self.index]}')
        if self.label_first_seen:
          sig = self._signature(selected_idx.loc[idx])
          if sig in first_seen:
            parts.append(f'first={first_seen[sig]:.0f}')
      if self.labelVar and self.labelVar in selected_idx.columns:
        parts.append(f'{self.labelVar}={selected_idx.loc[idx, self.labelVar]}')
      if self.metric and self.metric in selected_idx.columns:
        parts.append(f'{self.metric}={float(selected_idx.loc[idx, self.metric]):.4g}')
      if not parts:
        parts.append(f'{self.select} sample')
      title = ', '.join(parts)
      ax.set_title(title, fontsize=9)

    fig.suptitle('Glyph-based radar profiles')
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
