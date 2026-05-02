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
  This file contains the methods designed for ad-hoc plotting methods
  created on 01/04/2022
  @author: mandd
"""

# External Imports
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from pandas.plotting import parallel_coordinates
import pandas as pd

from .. import MessageHandler
mh = getMessageHandler()

def errorFill(x, y, yerr, color=None, alphaFill=0.3, ax=None, logScale=False):
  """
    Method designed to draw a line x vs y including a shade between the min and max of y
    @ In, None
    @ Out, None
  """
  ax = ax if ax is not None else plt.gca()
  if np.isscalar(yerr) or (len(yerr) == len(y) and np.ndim(yerr) == 1):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr
  else:
    mh.message("plotUtils", f"Unhandled {yerr=} with {y=}", "warning", "quiet")
  ax.plot(x, y, color=color)
  ax.fill_between(x, ymax, ymin, color=color, alpha=alphaFill)
  if logScale:
    ax.set_yscale('symlog')


def generateParallelPlot(zs, batchID, ymins, ymaxs, ynames, fileID, line_alphas=None, line_colors=None, line_widths=None, legend_entries=None):
  """
    Main run method to generate parallel coordinate plot
    @ In, zs, pandas dataset, batch containing the set of points to be plotted
    @ In, batchID, string, ID of the batch
    @ In, ymins, np.array, minimum value for each variable
    @ In, ymaxs, np.array, maximum value for each variable
    @ In, ynames, list, list of string containing the ID of each variable
    @ In, fileID, string, name of the file containing the plot
    @ In, line_alphas, array-like, optional, alpha values for each polyline
    @ In, line_colors, array-like, optional, colors for each polyline
    @ In, line_widths, array-like, optional, linewidths for each polyline
    @ In, legend_entries, list, optional, list of dicts describing legend line samples; expected keys:
         label (str), color (str), linewidth (float), linestyle (str, optional)
    @ Out, None
  """
  if zs.size == 0:
    return
  N = zs.shape[0]
  zs = zs.astype(np.float64)
  # enforce a shared scale across all axes so relative slopes reflect actual magnitudes
  global_min = np.min(ymins)
  global_max = np.max(ymaxs)
  span = global_max - global_min
  if span == 0.0:
    span = 1.0
  zs = (zs - global_min) / span
  zs = np.clip(zs, 0.0, 1.0)

  fig, host = plt.subplots(figsize=(15, 8))

  axes = [host] + [host.twinx() for i in range(zs.shape[1] - 1)]
  for i, ax in enumerate(axes):
    ax.set_aspect('auto')
    ax.set_ylim((0.0, 1.0))
    # highlight the span that contains data for this variable
    var_min = np.clip((ymins[i] - global_min) / span, 0.0, 1.0)
    var_max = np.clip((ymaxs[i] - global_min) / span, 0.0, 1.0)
    if np.isclose(var_min, var_max):
      var_min = max(0.0, var_min - 0.01)
      var_max = min(1.0, var_max + 0.01)
    ax.axhspan(var_min, var_max, color='#d9d9d9', alpha=0.35, zorder=0)
    # map evenly spaced raw ticks back to the normalized coordinate space
    if np.isclose(ymaxs[i], ymins[i]):
      raw_ticks = np.asarray([ymins[i]])
    else:
      raw_ticks = np.linspace(ymins[i], ymaxs[i], 5)
    norm_ticks = (raw_ticks - global_min) / span
    ax.set_yticks(norm_ticks)
    ax.set_yticklabels([f'{val:g}' for val in raw_ticks])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
      ax.spines['left'].set_visible(False)
      ax.yaxis.set_ticks_position('right')
      ax.spines["right"].set_position(("axes", i / (zs.shape[1] - 1)))
      ax.tick_params(axis='y', which='major', pad=7)

  host.set_xlim(0, zs.shape[1] - 1)
  host.set_xticks(range(zs.shape[1]))
  host.set_xticklabels(ynames, fontsize=14)
  host.tick_params(axis='x', which='major', pad=7)
  host.spines['right'].set_visible(False)
  host.xaxis.tick_top()
  plot_title = 'Batch ' + str(batchID)
  host.set_title(plot_title, fontsize=14)

  if line_alphas is None:
    line_alphas = np.ones(N, dtype=float)
  else:
    line_alphas = np.asarray(line_alphas, dtype=float)
    if line_alphas.size != N:
      raise ValueError(f'line_alphas length {line_alphas.size} does not match number of lines {N}.')

  if line_colors is None:
    line_colors = np.asarray(['tab:blue'] * N, dtype=object)
  else:
    raw_colors = np.asarray(line_colors, dtype=object)
    if raw_colors.ndim > 1:
      if raw_colors.shape[0] != N:
        raise ValueError(f'line_colors length {raw_colors.size} does not match number of lines {N}.')
      packed_colors = np.empty(N, dtype=object)
      packed_colors[:] = [tuple(np.asarray(row).tolist()) for row in raw_colors]
      line_colors = packed_colors
    else:
      if raw_colors.size != N:
        raise ValueError(f'line_colors length {raw_colors.size} does not match number of lines {N}.')
      line_colors = raw_colors

  if line_widths is None:
    line_widths = np.ones(N, dtype=float)
  else:
    line_widths = np.asarray(line_widths, dtype=float)
    if line_widths.size != N:
      raise ValueError(f'line_widths length {line_widths.size} does not match number of lines {N}.')

  for j in range(N):
    host.plot(range(zs.shape[1]), zs[j,:],
              color=line_colors[j],
              linewidth=float(max(0.1, line_widths[j])),
              alpha=float(np.clip(line_alphas[j], 0.05, 1.0)))
    '''verts = list(zip([x for x in np.linspace(0, len(zs) - 1, len(zs) * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1)
    host.add_patch(patch)'''

  if legend_entries:
    from matplotlib.lines import Line2D
    handles = []
    for entry in legend_entries:
      if not entry or 'label' not in entry:
        continue
      handles.append(Line2D([0], [0],
                            color=entry.get('color', 'tab:blue'),
                            linewidth=float(entry.get('linewidth', 1.5)),
                            linestyle=entry.get('linestyle', '-'),
                            label=entry['label']))
    if handles:
      host.legend(handles=handles, loc='upper right', frameon=True, fontsize=10)

  plt.tight_layout()
  plt.savefig(fileID)
  plt.close()
