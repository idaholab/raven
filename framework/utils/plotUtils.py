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


def errorFill(x, y, yerr, color=None, alphaFill=0.3, ax=None, logScale=False):
  """
    Method designed to draw a line x vs y including a shade between the min and max of y
    @ In, None
    @ Out, None
  """
  ax = ax if ax is not None else plt.gca()
  if np.isscalar(yerr) or len(yerr) == len(y):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr
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

  fig, host = plt.subplots()

  axes = [host] + [host.twinx() for i in range(zs.shape[1] - 1)]
  for i, ax in enumerate(axes):
    ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
      ax.spines["right"].set_position(("axes", i / (zs.shape[1] - 1)))

  host.set_xlim(0, zs.shape[1] - 1)
  host.set_xticks(range(zs.shape[1]))
  host.set_xticklabels(ynames, fontsize=14)
  host.tick_params(axis='x', which='major', pad=7)
  host.spines['right'].set_visible(False)
  host.xaxis.tick_top()
  plot_title = 'Batch ' + str(batchID)
  host.set_title(plot_title, fontsize=14)

  for j in range(N):
    verts = list(zip([x for x in np.linspace(0, len(zs) - 1, len(zs) * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1)
    host.add_patch(patch)
  plt.tight_layout()
  plt.savefig(fileID)
