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
Plot assembly shuffle moves on a symmetric core map (full/quarter/eighth) using arrows.

Inputs:
  - template: SIMULATE template with locXX placeholders (required).
  - mapping_file: CSV with columns from_loc,to_loc[,label,color] (required).
  - symmetry: full|quarter|eighth (default eighth).
  - title: optional plot title.
"""

import csv
import io
import pathlib
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ShufflingSchemePlot(PlotInterface):
  """
  Render a shuffle pattern by drawing arrows from from_loc -> to_loc on a symmetric core view.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('template', contentType=InputTypes.StringType,
        descr=r"""Path to the SIMULATE template (EQinput.inp/firstInput.inp) containing locXX placeholders."""))
    spec.addSub(InputData.parameterInputFactory('mapping_file', contentType=InputTypes.StringType,
        descr=r"""CSV file with columns from_loc,to_loc[,label,color] describing shuffle moves."""))
    spec.addSub(InputData.parameterInputFactory('symmetry', contentType=InputTypes.StringType,
        descr=r"""One of: full, quarter, eighth (default eighth)."""))
    spec.addSub(InputData.parameterInputFactory('title', contentType=InputTypes.StringType,
        descr=r"""Optional plot title (defaults to the plot name)."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ShufflingSchemePlot'
    self.templatePath = None
    self.mappingPath = None
    self.symmetry = 'eighth'
    self.title = None

  def handleInput(self, spec):
    super().handleInput(spec)
    tmpl = spec.findFirst('template')
    if tmpl is None or not tmpl.value:
      self.raiseAnError(IOError, f'ShufflingSchemePlot "{self.name}" missing <template>.')
    self.templatePath = tmpl.value

    mapping = spec.findFirst('mapping_file')
    if mapping is None or not mapping.value:
      self.raiseAnError(IOError, f'ShufflingSchemePlot "{self.name}" missing <mapping_file>.')
    self.mappingPath = mapping.value

    sym = spec.findFirst('symmetry')
    if sym is not None and sym.value:
      cand = sym.value.lower()
      if cand not in {'full', 'quarter', 'eighth'}:
        self.raiseAnError(IOError, f'Invalid <symmetry> "{sym.value}" for ShufflingSchemePlot "{self.name}".')
      self.symmetry = cand
    title = spec.findFirst('title')
    if title is not None and title.value:
      self.title = title.value

  def _parseTemplate(self, path: str) -> Tuple[Dict[str, List[Tuple[int, int]]], int]:
    loc_to_coords: Dict[str, List[Tuple[int, int]]] = {}
    size = 0
    try:
      with open(path, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()
    except IOError as err:
      self.raiseAnError(IOError, f'ShufflingSchemePlot "{self.name}" failed to read template "{path}": {err}')
    for line in lines:
      if size == 0 and "'DIM.PWR'" in line.upper():
        tokens = line.replace(',', ' ').split()
        for idx, tok in enumerate(tokens):
          if tok.upper() == "'DIM.PWR'" and idx + 1 < len(tokens):
            try:
              size = int(tokens[idx + 1])
            except Exception:
              pass
      if line.strip().upper().startswith("'FUE.TYP'"):
        parts = line.split(',')
        if len(parts) >= 2:
          try:
            row = int(parts[1].split()[0]) - 1
          except Exception:
            row = None
          if row is not None:
            col = 0
            for tok in parts[1:]:
              tok = tok.strip().rstrip('/')
              if not tok:
                continue
              if 'LOC' in tok.upper():
                digits = ''.join(ch for ch in tok if ch.isdigit())
                if digits:
                  loc = f'loc{int(digits)}'
                  loc_to_coords.setdefault(loc.lower(), []).append((row, col))
              col += 1
            size = max(size, row + 1, col, size)
        continue
      tokens = line.strip().split()
      if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isdigit() and 'LOC' in line.upper():
        try:
          r = int(tokens[0]) - 1
          c = int(tokens[1]) - 1
          digits = ''.join(ch for ch in tokens[2] if ch.isdigit())
          if digits:
            loc = f'loc{int(digits)}'
            loc_to_coords.setdefault(loc.lower(), []).append((r, c))
        except Exception:
          pass
    if size == 0 and loc_to_coords:
      size = max(max(r, c) for coords in loc_to_coords.values() for r, c in coords) + 1
    if size == 0:
      size = 15
    return loc_to_coords, size

  def _applySymmetry(self, grid: np.ndarray, locs: Dict[str, List[Tuple[int, int]]], symmetry: str):
    size = grid.shape[0]
    if symmetry == 'full':
      return grid, locs, (0, 0)
    half = size // 2 + size % 2
    if symmetry == 'quarter':
      rs = slice(0, half)
      cs = slice(size - half, size)
    else:  # eighth
      rs = slice(0, half)
      cs = slice(size - half, size)
    sub = grid[rs, cs]
    locs_sub = {}
    for loc, coords in locs.items():
      filtered = []
      for r, c in coords:
        if rs.start <= r < rs.stop and cs.start <= c < cs.stop:
          rr = r - rs.start
          cc = c - cs.start
          if symmetry == 'eighth' and cc < sub.shape[0] - rr - 1:
            continue
          filtered.append((rr, cc))
      if filtered:
        locs_sub[loc] = filtered
    return sub, locs_sub, (rs.start, cs.start)

  def _loadMapping(self, path: str) -> List[Dict[str, str]]:
    moves: List[Dict[str, str]] = []
    try:
      with open(path, 'r', encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
          from_loc = row.get('from_loc') or row.get('from') or row.get('source')
          to_loc = row.get('to_loc') or row.get('to') or row.get('dest')
          if not from_loc or not to_loc:
            continue
          moves.append(dict(
            from_loc=from_loc.strip().lower(),
            to_loc=to_loc.strip().lower(),
            label=(row.get('label') or '').strip(),
            color=(row.get('color') or '').strip(),
          ))
    except Exception as err:
      self.raiseAnError(IOError, f'ShufflingSchemePlot "{self.name}" failed to read mapping "{path}": {err}')
    return moves

  def run(self):
    loc_coords, size = self._parseTemplate(self.templatePath)
    grid = np.full((size, size), np.nan)
    grid_view, locs_view, (r0, c0) = self._applySymmetry(grid, loc_coords, self.symmetry)
    moves = self._loadMapping(self.mappingPath)
    if not moves:
      self.raiseAWarning(f'ShufflingSchemePlot "{self.name}" found no moves to plot.')
      return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('#060914')
    ax.set_xlim(-0.5, grid_view.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid_view.shape[0] - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(grid_view.shape[1]))
    ax.set_yticks(range(grid_view.shape[0]))
    ax.grid(True, color='#1a2338', linewidth=0.7)
    ax.set_title(self.title or self.name, color='#e7edff')

    palette = ['#3cf5ff', '#ff5fd7', '#ffa64d', '#9cfb6b', '#d0a6ff', '#72d2ff']
    color_idx = 0
    def pick_color(custom: str):
      nonlocal color_idx
      if custom:
        return custom
      col = palette[color_idx % len(palette)]
      color_idx += 1
      return col

    for mv in moves:
      f = mv['from_loc']
      t = mv['to_loc']
      if f not in locs_view or t not in locs_view:
        continue
      fr = locs_view[f][0]
      tr = locs_view[t][0]
      color = pick_color(mv['color'])
      ax.annotate(
        '',
        xy=(tr[1], tr[0]),
        xytext=(fr[1], fr[0]),
        arrowprops=dict(arrowstyle='->', color=color, linewidth=2, shrinkA=6, shrinkB=6),
      )
      if mv['label']:
        midx = (fr[1] + tr[1]) / 2.0
        midy = (fr[0] + tr[0]) / 2.0
        ax.text(midx, midy, mv['label'], color=color, fontsize=9, ha='center', va='center')

    ax.set_xticklabels([chr(ord('A') + c0 + i) for i in range(grid_view.shape[1])], color='#8ea5c7')
    ax.set_yticklabels([r0 + i + 1 for i in range(grid_view.shape[0])], color='#8ea5c7')
    for spine in ax.spines.values():
      spine.set_edgecolor('#1f2b44')
    fig.tight_layout()

    fname = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(fname, dpi=150, facecolor='#04060f')
    plt.close(fig)
    self.raiseAMessage(f'ShufflingSchemePlot "{self.name}" wrote {fname}')
