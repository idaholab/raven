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
Animate the cumulative hypervolume progression of a multi-objective optimizer.

Each frame extends the hypervolume time-series up to the current generation and
annotates the latest improvement. This makes sudden drops, plateaus, or surges easy
to pinpoint when reviewing optimization runs.

What-if Scenarios

Hypervolume plateaus early -> the search likely converged; consider tighter termination or
  restarting with new seeds if exploration should continue.
Hypervolume spikes followed by crash -> good points are found then discarded; audit survivor
  selection or constraint enforcement.
Stair-step growth with long flat segments -> exploitation dominates; boost mutation or sample
  injection to regain diversity.
Slow but steady climb -> healthy exploration/exploitation balance; no intervention needed.
"""

import io
import itertools
import math
import os

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import imageio.v2 as imageio

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes
from . import plotGenerationUtils


class HypervolumeMoviePlot(PlotInterface):
  """
  Animated hypervolume progression (gif/html) for multi-objective optimizers.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    objectives = InputData.parameterInputFactory('objectives', contentType=InputTypes.StringListType,
        descr=r"""Ordered list of objective columns (minimized). Two objectives render a single curve; three objectives compute a true 3D hypervolume curve; more than three objectives fall back to animating all pairwise combinations.""")
    spec.addSub(objectives)
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('referencePoint', contentType=InputTypes.StringListType,
        descr=r"""Optional reference point for hypervolume computation. If omitted, the plot uses max objective values (+5%%)."""))
    spec.addSub(InputData.parameterInputFactory('maxFrames', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on the number of generations rendered. Defaults to min(total generations, 20)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combinations."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for the generated animations. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('saveFrames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves sampled generations as standalone PNG frames alongside the animation outputs."""))
    spec.addSub(InputData.parameterInputFactory('framesMax', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <saveFrames> is true. Defaults to 10; generations are sampled evenly."""))
    plotGenerationUtils.addGenerationSelectorSpec(spec)
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'HypervolumeMoviePlot'
    self.source = None
    self.sourceName = None
    self.objectives = []
    self.objectivePairs = []
    self._useThreeD = False
    self.index = None
    self.referencePoint = None
    self._referencePoints = {}
    self.maxFrames = None
    self.explicitGenerations = None
    self.formats = {'gif', 'html'}
    self.fps = 2.0
    self.saveFrames = False
    self.frameMax = 10
    self._globalHvMax = 0.0

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for HypervolumeMoviePlot "{self.name}".')
    self.sourceName = sourceNode.value
    objNode = spec.findFirst('objectives')
    if objNode is None or not objNode.value:
      self.raiseAnError(IOError, f'Missing <objectives> node for HypervolumeMoviePlot "{self.name}".')
    self.objectives = [entry for entry in objNode.value if entry]
    if len(self.objectives) < 2:
      self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" requires at least two objectives.')
    self._useThreeD = len(self.objectives) == 3
    if len(self.objectives) == 2 or self._useThreeD:
      self.objectivePairs = [tuple(self.objectives)]
    else:
      self.objectivePairs = [tuple(pair) for pair in itertools.combinations(self.objectives, 2)]
    idxNode = spec.findFirst('index')
    if idxNode is None or idxNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for HypervolumeMoviePlot "{self.name}".')
    self.index = idxNode.value

    refNode = spec.findFirst('referencePoint')
    if refNode is not None and refNode.value:
      try:
        refVals = [float(val) for val in refNode.value]
      except ValueError as err:
        self.raiseAnError(IOError, f'Invalid <referencePoint> values for HypervolumeMoviePlot "{self.name}": {err}')
      expected = 3 if self._useThreeD else 2
      if len(refVals) != expected:
        self.raiseAnError(IOError, f'<referencePoint> must contain exactly {expected} values for HypervolumeMoviePlot "{self.name}".')
      self.referencePoint = np.asarray(refVals, dtype=float)

    maxNode = spec.findFirst('maxFrames')
    if maxNode is not None and maxNode.value is not None:
      self.maxFrames = int(maxNode.value)
      if self.maxFrames <= 0:
        self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" received non-positive <maxFrames>.')

    formatNode = spec.findFirst('format')
    if formatNode is not None and formatNode.value is not None:
      raw = formatNode.value.strip().lower()
      if raw == 'both' or not raw:
        self.formats = {'gif', 'html'}
      else:
        parts = [frag.strip() for frag in raw.split(',') if frag.strip()]
        mapped = set()
        for item in parts:
          if item in ('gif', 'html'):
            mapped.add(item)
          elif item == 'both':
            mapped.update({'gif', 'html'})
          else:
            self.raiseAnError(IOError, f'Unsupported <format> "{item}" for HypervolumeMoviePlot "{self.name}".')
        if not mapped:
          mapped = {'gif'}
        self.formats = mapped

    fpsNode = spec.findFirst('fps')
    if fpsNode is not None and fpsNode.value is not None:
      self.fps = float(fpsNode.value)
      if self.fps <= 0:
        self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" received non-positive <fps>.')

    saveFramesNode = spec.findFirst('saveFrames')
    if saveFramesNode is not None and saveFramesNode.value is not None:
      self.saveFrames = bool(saveFramesNode.value)

    framesMaxNode = spec.findFirst('framesMax')
    if framesMaxNode is not None and framesMaxNode.value is not None:
      self.frameMax = int(framesMaxNode.value)
      if self.frameMax <= 0:
        self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}" received non-positive <framesMax>.')

    self.explicitGenerations = plotGenerationUtils.parseGenerationSelectorNode(spec)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for HypervolumeMoviePlot "{self.name}".')
    self.source = src
    available = self.source.getVars()
    needed = set(self.objectives + [self.index])
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by HypervolumeMoviePlot "{self.name}".')

  def run(self):
    df = self.source.asDataset().to_dataframe().copy()
    if df.empty:
      self.raiseAWarning(f'HypervolumeMoviePlot "{self.name}" received an empty dataset; nothing to animate.')
      return
    df[self.index] = df[self.index].astype(float)
    for obj in self.objectives:
      df[obj] = df[obj].astype(float)
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'HypervolumeMoviePlot "{self.name}" found no generations in column "{self.index}".')
      return

    hvSeries = self._computeHypervolumeSeries(df, generations)
    if hvSeries is None or not hvSeries:
      self.raiseAWarning(f'HypervolumeMoviePlot "{self.name}" could not compute hypervolume; aborting.')
      return
    self._globalHvMax = 0.0
    for series in hvSeries.values():
      if series.size:
        self._globalHvMax = max(self._globalHvMax, float(np.max(series)))

    try:
      _, indices = plotGenerationUtils.resolveGenerations(
          generations, self.explicitGenerations, defaultCap=20, maxFrames=self.maxFrames)
    except ValueError as err:
      self.raiseAnError(IOError, f'HypervolumeMoviePlot "{self.name}": {err}')

    if 'gif' in self.formats:
      self._writeGif(generations, hvSeries, indices)
    if 'html' in self.formats:
      self._writeHtml(generations, hvSeries, indices)
    if self.saveFrames:
      self._writeFrames(generations, hvSeries, indices)

  def _computeHypervolumeSeries(self, df, generations):
    hvByPair = {}
    self._referencePoints = {}
    if self._useThreeD:
      if self.referencePoint is None:
        maxima = df[self.objectives].max().to_numpy(dtype=float)
        delta = np.abs(maxima) * 0.05
        delta[delta == 0.0] = 0.05
        refPoint = maxima + delta
      else:
        refPoint = np.asarray(self.referencePoint, dtype=float)
      key = tuple(self.objectives)
      self._referencePoints[key] = refPoint
      hvValues = []
      for gen in generations:
        subset = df[df[self.index] == gen]
        hvValues.append(self._computeHypervolume(subset[self.objectives].to_numpy(dtype=float), refPoint))
      hvByPair[key] = np.asarray(hvValues, dtype=float)
      return hvByPair

    for pair in self.objectivePairs:
      if self.referencePoint is None:
        maxima = df[list(pair)].max().to_numpy(dtype=float)
        delta = np.abs(maxima) * 0.05
        delta[delta == 0.0] = 0.05
        refPoint = maxima + delta
      else:
        refPoint = np.asarray(self.referencePoint, dtype=float)
      self._referencePoints[pair] = refPoint
      hvValues = []
      for gen in generations:
        subset = df[df[self.index] == gen]
        hv = self._computeHypervolume(subset[list(pair)].to_numpy(dtype=float), refPoint)
        hvValues.append(hv)
      hvByPair[pair] = np.asarray(hvValues, dtype=float)
    return hvByPair

  @staticmethod
  def _computeHypervolume(points, ref):
    if points.size == 0:
      return 0.0
    if points.shape[1] != len(ref):
      raise ValueError('Points dimensionality does not match reference point.')
    if points.shape[1] == 2:
      return HypervolumeMoviePlot._computeHypervolume2d(points, ref)
    if points.shape[1] == 3:
      return HypervolumeMoviePlot._computeHypervolume3d(points, ref)
    raise ValueError('HypervolumeMoviePlot supports hypervolume up to 3 objectives.')

  @staticmethod
  def _computeHypervolume2d(points, ref):
    order = np.argsort(points[:, 0])
    sortedPts = points[order]
    hv = 0.0
    prevX = ref[0]
    for x, y in sortedPts[::-1]:
      width = prevX - x
      if width < 0:
        width = 0.0
      height = max(0.0, ref[1] - y)
      hv += width * height
      prevX = x
    return hv

  @staticmethod
  def _computeHypervolume3d(points, ref):
    # Slice the 3D volume along the first objective and accumulate 2D slices.
    sortedIdx = np.argsort(points[:, 0])
    sortedPts = points[sortedIdx]
    hv = 0.0
    prevX = ref[0]
    for i in range(len(sortedPts) - 1, -1, -1):
      x = sortedPts[i, 0]
      width = max(0.0, prevX - x)
      yzSlice = sortedPts[:i + 1, 1:]
      area = HypervolumeMoviePlot._computeHypervolume2d(yzSlice, ref[1:])
      hv += width * area
      prevX = x
    return hv

  def _writeGif(self, generations, hvSeries, indices):
    filename = self._createFilename(defaultName=f'{self.name}.gif')
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=0) as writer:
      for idx in indices:
        fig = self._renderFrame(generations, hvSeries, idx)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _writeHtml(self, generations, hvSeries, indices):
    filename = self._createFilename(defaultName=f'{self.name}.html')
    fig, axes = self._createFigure()

    def init():
      for axis, pair in zip(axes, self.objectivePairs):
        self._drawSeries(axis, generations, hvSeries[pair], indices[0], pair)
      fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
      return fig.axes

    def update(idx):
      for axis, pair in zip(axes, self.objectivePairs):
        self._drawSeries(axis, generations, hvSeries[pair], idx, pair)
      fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
      return fig.axes

    anim = animation.FuncAnimation(fig, update, frames=indices,
                                   init_func=init, interval=1000.0 / self.fps,
                                   blit=False)
    htmlStr = anim.to_jshtml()
    centeredHtml = f'<div style="display:flex;justify-content:center;">{htmlStr}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centeredHtml)
    plt.close(fig)

  def _writeFrames(self, generations, hvSeries, indices):
    frameIdx = plotGenerationUtils.frameIndicesToSave(
        len(indices), bool(self.explicitGenerations), self.saveFrames, self.frameMax)
    if not frameIdx:
      return
    base = self._createFilename(defaultName=f'{self.name}_frames')
    template = os.path.splitext(base)[0] + '_{index:04d}.png'
    directory = os.path.dirname(template)
    if directory:
      os.makedirs(directory, exist_ok=True)
    for framePos in frameIdx:
      idx = indices[framePos]
      fig = self._renderFrame(generations, hvSeries, idx)
      fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
      fig.savefig(template.format(index=idx), dpi=150)
      plt.close(fig)

  def _renderFrame(self, generations, hvSeries, idx):
    fig, axes = self._createFigure()
    for axis, pair in zip(axes, self.objectivePairs):
      self._drawSeries(axis, generations, hvSeries[pair], idx, pair)
    fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
    return fig

  def _drawSeries(self, ax, generations, hvSeries, idx, pair):
    ax.clear()
    uptoGens = generations[:idx + 1]
    uptoHv = hvSeries[:idx + 1]
    ax.plot(uptoGens, uptoHv, color='tab:blue', linewidth=2.0)
    ax.scatter([uptoGens[-1]], [uptoHv[-1]], color='tab:orange', edgecolor='black', s=60, zorder=3)
    ax.set_xlabel(self.index)
    ax.set_ylabel('Hypervolume')
    label = ' vs '.join(pair) if len(pair) == 2 else ', '.join(pair)
    ax.set_title(f'{label} (Generation {self._formatGeneration(uptoGens[-1])})')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(min(generations), max(generations))
    ymax = self._globalHvMax if self._globalHvMax > 0.0 else (np.max(hvSeries) if hvSeries.size else 1.0)
    ax.set_ylim(0.0, ymax * 1.05 if ymax > 0.0 else 1.0)
    ax.text(0.02, 0.92,
            f'Latest: {uptoHv[-1]:.4g}\nBest: {np.max(uptoHv):.4g}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.8, edgecolor='gray'))

  @staticmethod
  def _formatGeneration(genID):
    if float(genID).is_integer():
      return int(genID)
    return genID

  def _createFigure(self):
    nPairs = len(self.objectivePairs) if self.objectivePairs else 1
    cols = max(1, nPairs)
    fig, axes = plt.subplots(1, cols, figsize=(4.8 * cols, 4.2))
    if not isinstance(axes, np.ndarray):
      axes = [axes]
    else:
      axes = axes.flatten().tolist()
    return fig, axes

  @staticmethod
  def _sampleGenerations(generations, limit):
    if limit >= len(generations):
      return list(generations), list(range(len(generations)))
    positions = np.linspace(0, len(generations) - 1, limit, dtype=int)
    selected = []
    for idx in positions:
      if idx not in selected:
        selected.append(idx)
    cursor = 0
    while len(selected) < limit and cursor < len(generations):
      if cursor not in selected:
        selected.append(cursor)
      cursor += 1
    selected = sorted(set(selected))
    if len(selected) > limit:
      selected = selected[:limit - 1] + [len(generations) - 1]
    elif selected[-1] != len(generations) - 1:
      selected[-1] = len(generations) - 1
    return [generations[i] for i in selected], selected

  def _selectFrameIndices(self, total):
    if not self.saveFrames or total <= 0 or self.frameMax <= 0:
      return []
    if total <= self.frameMax:
      return list(range(total))
    stride = int(math.ceil(total / float(self.frameMax)))
    indices = list(range(0, total, stride))
    if indices and indices[-1] != total - 1:
      if len(indices) >= self.frameMax:
        indices[-1] = total - 1
      else:
        indices.append(total - 1)
    return sorted(set(indices))
