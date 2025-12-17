#!/usr/bin/env python
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
Plot the SIMULATE core layout colored by locXX values from a PRLO optimizer export.

Intended to mirror the milestone report core plots (A.. columns, 1.. rows) and
supports symmetry views (full, quarter, eighth).
"""

import re

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class CoreLayoutPlot(PlotInterface):
  """
  Render a core map using locXX variables from a PRLO optimizer export and a SIMULATE template.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject containing locXX variables (e.g., opt_export)."""))
    spec.addSub(InputData.parameterInputFactory('template', contentType=InputTypes.StringType,
        descr=r"""Path to the SIMULATE template (EQinput.inp/firstInput.inp) containing locXX placeholders."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation/index column used to pick a row."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""Optional explicit generation value; otherwise the last available is used."""))
    spec.addSub(InputData.parameterInputFactory('accept_column', contentType=InputTypes.StringType,
        descr=r"""Optional column used to filter accepted rows (default 'accepted')."""))
    spec.addSub(InputData.parameterInputFactory('accept_value', contentType=InputTypes.StringType,
        descr=r"""Value in <accept_column> to keep (default 'final')."""))
    spec.addSub(InputData.parameterInputFactory('symmetry', contentType=InputTypes.StringType,
        descr=r"""One of: full (default), quarter, eighth."""))
    spec.addSub(InputData.parameterInputFactory('quadrant', contentType=InputTypes.StringType,
        descr=r"""For quarter symmetry, one of NE, NW, SE, SW (default NE)."""))
    spec.addSub(InputData.parameterInputFactory('label_mode', contentType=InputTypes.StringType,
        descr=r"""How to label cells: 'value' (default), 'loc', or 'both'."""))
    spec.addSub(InputData.parameterInputFactory('assembly_table', contentType=InputTypes.StringType,
        descr=r"""Optional CSV path with assembly metadata (columns like Assembly Type, FA ID, Fresh Fuel Label, Enrichment (wt%), IFBA Rods, Pyrex Rods)."""))
    spec.addSub(InputData.parameterInputFactory('color_prefix', contentType=InputTypes.StringType,
        descr=r"""Optional prefix for columns that should drive the color scale (e.g., burnup_locXX)."""))
    spec.addSub(InputData.parameterInputFactory('colorbar_label', contentType=InputTypes.StringType,
        descr=r"""Override the colorbar label (defaults to 'Fuel ID / value' or 'Burnup / color metric' when color_prefix is used)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'CoreLayoutPlot'
    self.sourceName = None
    self.templatePath = None
    self.index = None
    self.generation = None
    self.acceptColumn = 'accepted'
    self.acceptValue = 'final'
    self.symmetry = 'full'
    self.quadrant = 'NE'
    self.labelMode = 'value'
    self.assemblyTable = None
    self.colorPrefix = None
    self.colorbarLabel = None
    self.source = None
    self.locToCoords = {}
    self.locLabels = {}
    self.size = 0

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or not src.value:
      self.raiseAnError(IOError, f'CoreLayoutPlot "{self.name}" missing <source>.')
    self.sourceName = src.value

    tmpl = spec.findFirst('template')
    if tmpl is None or not tmpl.value:
      self.raiseAnError(IOError, f'CoreLayoutPlot "{self.name}" missing <template>.')
    self.templatePath = tmpl.value

    idx = spec.findFirst('index')
    if idx is not None and idx.value:
      self.index = idx.value
    gen = spec.findFirst('generation')
    if gen is not None and gen.value is not None:
      self.generation = float(gen.value)

    accCol = spec.findFirst('accept_column')
    if accCol is not None and accCol.value:
      self.acceptColumn = accCol.value
    accVal = spec.findFirst('accept_value')
    if accVal is not None and accVal.value is not None:
      self.acceptValue = accVal.value

    sym = spec.findFirst('symmetry')
    if sym is not None and sym.value:
      self.symmetry = sym.value.lower()
    quad = spec.findFirst('quadrant')
    if quad is not None and quad.value:
      self.quadrant = quad.value.upper()
    lbl = spec.findFirst('label_mode')
    if lbl is not None and lbl.value:
      mode = lbl.value.lower()
      if mode not in {'value', 'loc', 'both'}:
        self.raiseAnError(IOError, f'Invalid <label_mode> "{lbl.value}" for CoreLayoutPlot "{self.name}".')
      self.labelMode = mode
    asm = spec.findFirst('assembly_table')
    if asm is not None and asm.value:
      self.assemblyTable = asm.value
    colorPrefix = spec.findFirst('color_prefix')
    if colorPrefix is not None and colorPrefix.value:
      self.colorPrefix = colorPrefix.value
    colorbarLabel = spec.findFirst('colorbar_label')
    if colorbarLabel is not None and colorbarLabel.value:
      self.colorbarLabel = colorbarLabel.value

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'Source "{self.sourceName}" not found for CoreLayoutPlot "{self.name}".')
    self.source = src
    # parse template once
    self._parseTemplate(self.templatePath)
    if not self.locToCoords:
      self.raiseAnError(IOError, f'No locXX placeholders parsed from template "{self.templatePath}".')

  def _parseTemplate(self, path):
    self.locToCoords = {}
    self.locLabels = {}
    self.size = 0
    try:
      with open(path, 'r') as fh:
        lines = fh.readlines()
    except IOError as err:
      self.raiseAnError(IOError, f'Failed to read template "{path}": {err}')

    for line in lines:
      if self.size == 0:
        m = re.search(r"'DIM\.PWR'\s+(\d+)", line, re.IGNORECASE)
        if m:
          self.size = int(m.group(1))
      mtyp = re.match(r"\s*'FUE\.TYP'\s*[, ]\s*(\d+)\s*,?(.*)$", line, re.IGNORECASE)
      if mtyp:
        row = int(mtyp.group(1)) - 1
        tokens = re.split(r"[,\s]+", mtyp.group(2).strip().rstrip('/'))
        col = 0
        for tok in tokens:
          if not tok:
            continue
          ml = re.search(r"loc[^0-9]*([0-9]+)", tok, re.IGNORECASE)
          if ml:
            norm_loc = self._normalizeLoc(f"loc{ml.group(1)}")
            self.locToCoords.setdefault(norm_loc, []).append((row, col))
            self.locLabels.setdefault(norm_loc, f"loc{ml.group(1)}")
          col += 1
        self.size = max(self.size, row + 1, col, self.size)
        continue
      mfull = re.search(r"^\s*(\d+)\s+(\d+).*loc[^0-9]*([0-9]+)", line, re.IGNORECASE)
      if mfull:
        r = int(mfull.group(1)) - 1
        c = int(mfull.group(2)) - 1
        raw_loc = f'loc{mfull.group(3)}'.lower()
        norm_loc = self._normalizeLoc(raw_loc)
        self.locToCoords.setdefault(norm_loc, []).append((r, c))
        self.locLabels.setdefault(norm_loc, raw_loc)
    if self.size == 0 and self.locToCoords:
      self.size = max(max(r, c) for coords in self.locToCoords.values() for r, c in coords) + 1
    if self.size == 0:
      self.size = 15  # fall back

  def _normalizeLoc(self, name):
    m = re.search(r'loc0*(\d+)', name.lower())
    return f'loc{int(m.group(1))}' if m else name.lower()

  def _extractLocValues(self, row, prefix=''):
    values = {}
    prefix_lower = prefix.lower() if prefix else ''
    for col in row.index:
      col_lower = col.lower()
      if prefix_lower:
        if not col_lower.startswith(prefix_lower):
          continue
        base = col_lower[len(prefix_lower):]
      else:
        base = col_lower
      if not base.startswith('loc'):
        continue
      norm = self._normalizeLoc(base)
      try:
        values[norm] = float(row[col])
      except Exception:
        values[norm] = float(hash(row[col]) % 500)
    return values

  def _filterRow(self, df):
    subset = df.copy()
    if self.acceptColumn in subset.columns and self.acceptValue is not None:
      subset = subset[subset[self.acceptColumn] == self.acceptValue]
    if subset.empty:
      subset = df.copy()
    if self.index and self.index in subset.columns:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        subset = subset[np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if subset.empty:
      return None
    return subset.iloc[0]

  def _applySymmetry(self, grid):
    if self.symmetry == 'full':
      return grid, self.locToCoords, self.size
    half = self.size // 2 + self.size % 2
    if self.symmetry == 'quarter':
      if self.quadrant == 'NE':
        rs = slice(0, half)
        cs = slice(self.size - half, self.size)
      elif self.quadrant == 'NW':
        rs = slice(0, half)
        cs = slice(0, half)
      elif self.quadrant == 'SE':
        rs = slice(self.size - half, self.size)
        cs = slice(self.size - half, self.size)
      else:
        rs = slice(self.size - half, self.size)
        cs = slice(0, half)
      sub = grid[rs, cs]
      locs = {}
      for loc, coords in self.locToCoords.items():
        filtered = []
        for r, c in coords:
          if rs.start <= r < rs.stop and cs.start <= c < cs.stop:
            filtered.append((r - rs.start, c - cs.start))
        if filtered:
          locs[loc] = filtered
      return sub, locs, sub.shape[0]
    if self.symmetry == 'eighth':
      rs = slice(0, half)
      cs = slice(self.size - half, self.size)
      sub = grid[rs, cs].copy()
      n = sub.shape[0]
      for r in range(n):
        for c in range(n):
          if c < n - r - 1:
            sub[r, c] = np.nan
      locs = {}
      for loc, coords in self.locToCoords.items():
        filtered = []
        for r, c in coords:
          if rs.start <= r < rs.stop and cs.start <= c < cs.stop:
            rr = r - rs.start
            cc = c - cs.start
            if cc < sub.shape[1] - rr - 1:
              continue
            filtered.append((rr, cc))
        if filtered:
          locs[loc] = filtered
      return sub, locs, sub.shape[0]
    return grid, self.locToCoords, self.size

  def _buildGrid(self, values):
    grid = np.full((self.size, self.size), np.nan)
    for loc, coords in self.locToCoords.items():
      if loc not in values:
        continue
      for r, c in coords:
        grid[r, c] = values[loc]
    return grid

  def run(self):
    df = self.source.asDataset().to_dataframe()
    row = self._filterRow(df)
    if row is None:
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" found no data to plot.')
      return
    base_vals = self._extractLocValues(row)
    color_vals = self._extractLocValues(row, prefix=self.colorPrefix) if self.colorPrefix else {}
    if self.colorPrefix and not color_vals:
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" did not find columns with prefix "{self.colorPrefix}"; falling back to locXX values.')
    active_vals = color_vals or base_vals
    if not active_vals:
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" found no locXX-compatible columns to plot.')
      return

    grid = self._buildGrid(active_vals)
    grid_view, locs_view, size_view = self._applySymmetry(grid)
    if np.all(np.isnan(grid_view)):
      self.raiseAWarning(f'CoreLayoutPlot "{self.name}" grid is empty after mapping values.')
      return

    masked = np.ma.masked_invalid(grid_view)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    fig.patch.set_facecolor('#0a0f1c')
    ax.set_facecolor('#0a0f1c')
    cmap = plt.cm.viridis
    img = ax.imshow(masked, cmap=cmap, origin='upper', interpolation='none')
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = self.colorbarLabel or ('Burnup / color metric' if color_vals else 'Fuel ID / value')
    cbar.ax.set_ylabel(cbar_label, color='#e8f0ff')
    cbar.ax.yaxis.set_tick_params(color='#e8f0ff')
    for lab in cbar.ax.get_yticklabels():
      lab.set_color('#e8f0ff')

    for loc, coords in locs_view.items():
      val = active_vals.get(loc, np.nan)
      display_loc = self.locLabels.get(loc, loc)
      for r, c in coords:
        txt = ''
        if self.labelMode == 'value':
          txt = f'{val:.0f}' if np.isfinite(val) else ''
        elif self.labelMode == 'loc':
          txt = display_loc
        else:
          txt_val = f'{val:.0f}' if np.isfinite(val) else ''
          txt = f'{display_loc}\n{txt_val}' if txt_val else display_loc
        ax.text(c, r, txt, ha='center', va='center', fontsize=7, color='#e8f0ff')

    ax.set_xticks(range(size_view))
    ax.set_yticks(range(size_view))
    ax.set_xticklabels([chr(ord('A') + i) for i in range(size_view)])
    ax.set_yticklabels(range(1, size_view + 1))
    ax.set_xlabel('Column (A..)')
    ax.set_ylabel('Row (1..)')
    ax.set_title(f'Core layout: {self.name}')
    ax.grid(color='#1e2a42', linewidth=0.6)

    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=220)
    plt.close(fig)

    # Optional assembly metadata table
    if self.assemblyTable:
      try:
        import pandas as pd
        meta_df = pd.read_csv(self.assemblyTable)
        fig2, ax2 = plt.subplots(figsize=(8, 1 + 0.3 * len(meta_df)))
        ax2.axis('off')
        table = ax2.table(cellText=meta_df.values, colLabels=meta_df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        fig2.tight_layout()
        table_name = self._createFilename(defaultName=f'{self.name}_assembly_table.png')
        fig2.savefig(table_name, dpi=220, bbox_inches='tight')
        plt.close(fig2)
      except Exception as err:
        self.raiseAWarning(f'CoreLayoutPlot "{self.name}" failed to render assembly table: {err}')
