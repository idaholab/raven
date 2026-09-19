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


def generateParallelPlot(zs, batchID, ymins, ymaxs, ynames, fileID):
  """
    Main run method to generate parallel coordinate plot
    @ In, zs, pandas dataset, batch containing the set of points to be plotted
    @ In, batchID, string, ID of the batch
    @ In, ymins, np.array, minimum value for each variable
    @ In, ymaxs, np.array, maximum value for each variable
    @ In, ynames, list, list of string containing the ID of each variable
    @ In, fileID, string, name of the file containing the plot
    @ Out, None
  """
  N = zs.shape[0]
  zs = zs.astype(np.float64)
  dys = ymaxs - ymins
  zs[:, 0] = zs[:, 0]
  zs[:, 1:] = (zs[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[1:]

  fig, host = plt.subplots(figsize=(15, 8))

  axes = [host] + [host.twinx() for i in range(zs.shape[1] - 1)]
  for i, ax in enumerate(axes):
    ax.set_aspect('auto')
    ax.set_ylim((int(ymins[i]), int(ymaxs[i])))
    ax.set_yticks(np.arange(ymins[i], ymaxs[i]+1, 1))
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

  for j in range(N):
    host.plot(range(zs.shape[1]), zs[j,:])
    '''verts = list(zip([x for x in np.linspace(0, len(zs) - 1, len(zs) * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1)
    host.add_patch(patch)'''

  plt.tight_layout()
  plt.savefig(fileID)
  plt.close()


def generateConstraintParallelPlot(zs, batchID, ymins, ymaxs, ynames, fileID, lineAlphas=None, lineColors=None, lineWidths=None, legendEntries=None):
  """
    Generate a constraint-aware parallel coordinate plot. Unlike generateParallelPlot, this
    variant renders every polyline on a shared normalized scale and accepts optional per-line
    styling (alpha/color/width) plus an optional legend, so callers can encode feasibility or
    constraint-violation information. It is used by the enhanced OptParallelCoordinatePlot path and
    is kept separate so the original generateParallelPlot behavior (and its gold images) is unchanged.
    @ In, zs, np.array, batch containing the set of points to be plotted
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
  globalMin = np.min(ymins)
  globalMax = np.max(ymaxs)
  span = globalMax - globalMin
  if span == 0.0:
    span = 1.0
  zs = (zs - globalMin) / span
  zs = np.clip(zs, 0.0, 1.0)

  fig, host = plt.subplots(figsize=(15, 8))

  axes = [host] + [host.twinx() for i in range(zs.shape[1] - 1)]
  for i, ax in enumerate(axes):
    ax.set_aspect('auto')
    ax.set_ylim((0.0, 1.0))
    # highlight the span that contains data for this variable
    varMin = np.clip((ymins[i] - globalMin) / span, 0.0, 1.0)
    varMax = np.clip((ymaxs[i] - globalMin) / span, 0.0, 1.0)
    if np.isclose(varMin, varMax):
      varMin = max(0.0, varMin - 0.01)
      varMax = min(1.0, varMax + 0.01)
    ax.axhspan(varMin, varMax, color='#d9d9d9', alpha=0.35, zorder=0)
    # map evenly spaced raw ticks back to the normalized coordinate space
    if np.isclose(ymaxs[i], ymins[i]):
      rawTicks = np.asarray([ymins[i]])
    else:
      rawTicks = np.linspace(ymins[i], ymaxs[i], 5)
    normTicks = (rawTicks - globalMin) / span
    ax.set_yticks(normTicks)
    ax.set_yticklabels([f'{val:g}' for val in rawTicks])
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

  if lineAlphas is None:
    lineAlphas = np.ones(N, dtype=float)
  else:
    lineAlphas = np.asarray(lineAlphas, dtype=float)
    if lineAlphas.size != N:
      raise ValueError(f'line_alphas length {lineAlphas.size} does not match number of lines {N}.')

  if lineColors is None:
    lineColors = np.asarray(['tab:blue'] * N, dtype=object)
  else:
    rawColors = np.asarray(lineColors, dtype=object)
    if rawColors.ndim > 1:
      if rawColors.shape[0] != N:
        raise ValueError(f'line_colors length {rawColors.size} does not match number of lines {N}.')
      packedColors = np.empty(N, dtype=object)
      packedColors[:] = [tuple(np.asarray(row).tolist()) for row in rawColors]
      lineColors = packedColors
    else:
      if rawColors.size != N:
        raise ValueError(f'line_colors length {rawColors.size} does not match number of lines {N}.')
      lineColors = rawColors

  if lineWidths is None:
    lineWidths = np.ones(N, dtype=float)
  else:
    lineWidths = np.asarray(lineWidths, dtype=float)
    if lineWidths.size != N:
      raise ValueError(f'line_widths length {lineWidths.size} does not match number of lines {N}.')

  for j in range(N):
    host.plot(range(zs.shape[1]), zs[j,:],
              color=lineColors[j],
              linewidth=float(max(0.1, lineWidths[j])),
              alpha=float(np.clip(lineAlphas[j], 0.05, 1.0)))

  if legendEntries:
    from matplotlib.lines import Line2D
    handles = []
    for entry in legendEntries:
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
