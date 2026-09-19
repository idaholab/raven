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
Animation showing all Pareto fronts (by rank) evolving through generations.
"""

import io
import hashlib
import re

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.cm import get_cmap

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class NSGAFrontRankAnimation(PlotInterface):
  """
  Visualizes the evolution of all Pareto fronts coloured by rank, with per-frame annotations.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Two objective names to use for the scatter plot axes."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for the animation. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif" (default) or "html"."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA Rank Animation'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.index = None
    self.fps = 2.0
    self.format = 'gif'

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    objectives = spec.findFirst('objectives')
    if objectives is None:
      self.raiseAnError(IOError, f'Missing <objectives> node for NSGAFrontRankAnimation "{self.name}".')
    self.objectives = objectives.value
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, f'Missing <index> node for NSGAFrontRankAnimation "{self.name}".')
    self.index = idxNode.value
    fpsNode = spec.findFirst('fps')
    if fpsNode is not None:
      self.fps = max(fpsNode.value, 0.1)
    fmtNode = spec.findFirst('format')
    if fmtNode is not None:
      fmt = fmtNode.value.strip().lower()
      if fmt not in {'gif', 'html'}:
        self.raiseAnError(IOError, f'Unsupported format "{fmt}" for NSGAFrontRankAnimation "{self.name}".')
      self.format = fmt

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for NSGAFrontRankAnimation "{self.name}".')
    dataVars = self.source.getVars()
    required = self.objectives + [self.index, 'rank']
    missing = [var for var in required if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing variable(s) {} required by NSGAFrontRankAnimation "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)
    if len(self.objectives) != 2:
      self.raiseAnError(IOError, f'NSGAFrontRankAnimation "{self.name}" currently supports exactly two objectives.')

  def run(self):
    df = self.source.asDataset().to_dataframe()
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'No generations found for NSGAFrontRankAnimation "{self.name}".')
      return
    xVar, yVar = self.objectives
    xMin, xMax = self._scaledBounds(df[xVar].min(), df[xVar].max())
    yMin, yMax = self._scaledBounds(df[yVar].min(), df[yVar].max())
    uniqueRanks = sorted(df['rank'].unique())
    cmap = get_cmap('tab10', len(uniqueRanks))
    colorLookup = {rank: cmap(idx) for idx, rank in enumerate(uniqueRanks)}

    if self.format == 'html':
      self._writeHtml(df, generations, xVar, yVar, xMin, xMax, yMin, yMax, uniqueRanks, colorLookup,
                       filenameDefault=f'{self.name}_rank_animation.html')
    else:
      self._writeGif(df, generations, xVar, yVar, xMin, xMax, yMin, yMax, uniqueRanks, colorLookup,
                      filenameDefault=f'{self.name}_rank_animation.gif')

  def _writeGif(self, df, generations, xVar, yVar, xMin, xMax, yMin, yMax, ranks, colorLookup, filenameDefault):
    filename = self._createFilename(defaultName=filenameDefault)
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=1) as writer:
      for gen in generations:
        subset = df[df[self.index] == gen]
        fig, ax = plt.subplots()
        for rank in ranks:
          front = subset[subset['rank'] == rank]
          if front.empty:
            continue
          ax.scatter(front[xVar], front[yVar], color=colorLookup[rank], label=f'Rank {rank}', edgecolors='k', linewidths=0.3)
          if len(front) > 1:
            ordered = front.sort_values(by=xVar)
            ax.plot(ordered[xVar], ordered[yVar], color=colorLookup[rank], linewidth=0.8, alpha=0.6)
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        ax.set_xlabel(xVar)
        ax.set_ylabel(yVar)
        ax.set_title(f'Generation {gen}')
        self._addRankAnnotation(ax, subset, ranks)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
          ax.legend(loc='best')
        fig.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _writeHtml(self, df, generations, xVar, yVar, xMin, xMax, yMin, yMax, ranks, colorLookup, filenameDefault):
    filename = self._createFilename(defaultName=filenameDefault)
    fig, ax = plt.subplots()
    scatters = {}
    lines = {}
    for rank in ranks:
      scatters[rank] = ax.scatter([], [], color=colorLookup[rank], label=f'Rank {rank}', edgecolors='k', linewidths=0.3)
      lines[rank], = ax.plot([], [], color=colorLookup[rank], linewidth=0.8, alpha=0.6)
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_xlabel(xVar)
    ax.set_ylabel(yVar)
    title = ax.set_title('')
    annotationTexts = [ax.text(0.02, 0.95 - idx * 0.05, '', transform=ax.transAxes, fontsize=9, va='top')
                        for idx in range(len(ranks))]
    if ranks:
      ax.legend(loc='best')

    def init():
      for sc in scatters.values():
        sc.set_offsets(np.empty((0, 2)))
      for line in lines.values():
        line.set_data([], [])
      for txt in annotationTexts:
        txt.set_text('')
      title.set_text('')
      return list(scatters.values()) + list(lines.values()) + annotationTexts

    def update(gen):
      subset = df[df[self.index] == gen]
      for idx, rank in enumerate(ranks):
        front = subset[subset['rank'] == rank]
        offsets = np.column_stack((front[xVar].to_numpy(), front[yVar].to_numpy()))
        scatters[rank].set_offsets(offsets)
        if len(front) > 1:
          ordered = front.sort_values(by=xVar)
          lines[rank].set_data(ordered[xVar].to_numpy(), ordered[yVar].to_numpy())
        else:
          lines[rank].set_data([], [])
        annotationTexts[idx].set_text(f'Rank {rank}: {len(front)} pts')
      title.set_text(f'Generation {gen}')
      return list(scatters.values()) + list(lines.values()) + annotationTexts

    anim = animation.FuncAnimation(fig, update, frames=generations, init_func=init,
                                   interval=1000.0 / self.fps, blit=False)
    htmlStr = anim.to_jshtml()
    htmlStr = self._normalizeAnimationIds(htmlStr)
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(htmlStr)
    plt.close(fig)

  @staticmethod
  def _addRankAnnotation(ax, subset, ranks):
    y = 0.95
    for rank in ranks:
      count = (subset['rank'] == rank).sum()
      ax.text(0.02, y, f'Rank {rank}: {count} pts', transform=ax.transAxes, fontsize=9, va='top')
      y -= 0.05

  @staticmethod
  def _scaledBounds(minVal, maxVal):
    if np.isclose(minVal, maxVal):
      delta = abs(minVal) if minVal != 0 else 1.0
      return minVal - 0.1 * delta, maxVal + 0.1 * delta
    low = minVal * 0.9 if minVal >= 0 else minVal * 1.1
    high = maxVal * 1.1 if maxVal >= 0 else maxVal * 0.9
    if np.isclose(low, high):
      delta = abs(low) if low != 0 else 1.0
      low -= 0.1 * delta
      high += 0.1 * delta
    return low, high

  def _normalizeAnimationIds(self, htmlStr):
    """
    Replace randomly generated Matplotlib animation element ids with deterministic ones.
    """
    match = re.search(r'_anim_img([0-9a-f]+)', htmlStr)
    if not match:
      return htmlStr
    randomSuffix = match.group(1)
    baseName = self.name if getattr(self, 'name', None) else 'animation'
    safeName = ''.join(ch if ch.isalnum() else '_' for ch in baseName)
    seed = f'{self.__class__.__name__}:{safe_name}'
    deterministic = hashlib.md5(seed.encode('utf-8')).hexdigest()
    return htmlStr.replace(randomSuffix, deterministic)
