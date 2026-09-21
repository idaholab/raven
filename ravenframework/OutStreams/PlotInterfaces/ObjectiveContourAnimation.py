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
Animation highlighting optimizer populations against concentric contour levels in objective space.
"""

import io
import math
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes
from . import plotGenerationUtils


SHADE_CMAP = LinearSegmentedColormap.from_list(
    'objectiveContourShade',
    ['#fdfdfd', '#d7e8f6', '#7fb6d8', '#1f78b4']
)

BASE_POINT_COLOR = 'tab:blue'
TOP_POINT_COLOR = 'tab:green'
BEST_POINT_COLOR = 'red'
DEFAULT_INFEASIBLE_POINT_COLOR = '#7f7f7f'
ENHANCED_INFEASIBLE_POINT_COLOR = '#f57c00'
DEFAULT_HISTORY_POINT_COLOR = '#6f6f6f'
CROSSHAIR_COLOR = '#4f4f4f'
CONSTRAINT_FILL_COLOR = '#d6d6d6'
CONSTRAINT_LINE_COLORS = (
    '#c62828', '#ef6c00', '#2e7d32', '#1565c0', '#6a1b9a'
)
HISTORY_MARKER_SIZE = 46

# Retain legacy constant name for intra-module usage until all references migrate.
INFEASIBLE_POINT_COLOR = DEFAULT_INFEASIBLE_POINT_COLOR


def _metricSeriesHelper(df, metricName, metricKind):
  series = df[metricName].astype(float)
  if metricKind == 'fitness':
    return -series, series
  return series, series


def _metricToOriginalHelper(metricValue, metricKind):
  if metricKind == 'fitness':
    return -metricValue
  return metricValue


def _isFeasibleHelper(df, constraintVars):
  if not constraintVars or df.empty:
    return np.ones(len(df), dtype=bool)
  feasible = np.ones(len(df), dtype=bool)
  for var in constraintVars:
    if var not in df.columns:
      continue
    vals = df[var].astype(float).to_numpy()
    feasible &= vals > 0.0
  return feasible


def _topCountHelper(totalPoints, topFraction, topOverride):
  if totalPoints <= 0:
    return 0
  if topOverride is not None:
    return min(totalPoints, max(1, topOverride))
  fraction = topFraction if topFraction is not None else 0.2
  fraction = min(max(fraction, 0.0), 1.0)
  return min(totalPoints, max(1, int(math.ceil(totalPoints * fraction))))


def _computePopulationColors(subset, axes, metricName, metricKind, constraintVars,
                               topFraction, topOverride, last=False,
                               infeasibleColor=DEFAULT_INFEASIBLE_POINT_COLOR,
                               topCap=None):
  totalPoints = len(subset)
  colors = np.empty((totalPoints, 4), dtype=float)
  baseRgba = mcolors.to_rgba(BASE_POINT_COLOR)
  colors[:] = baseRgba
  topGreen = mcolors.to_rgba(TOP_POINT_COLOR)
  bestRed = mcolors.to_rgba(BEST_POINT_COLOR)
  infeasibleRgba = mcolors.to_rgba(infeasibleColor)
  metricSeries, originalSeries = _metricSeriesHelper(subset, metricName, metricKind)
  if totalPoints == 0:
    summary = {'count': 0, 'pareto': 0, 'top': 0, 'best': None,
               'metric_kind': metricKind, 'metric': metricName,
               'top_fraction': 0.0, 'feasible': 0, 'feasible_fraction': 0.0,
               'has_constraints': bool(constraintVars)}
    meta = {'best_pos': None, 'best_coords': None, 'best_value': None}
    return colors, summary, meta

  feasibleMask = _isFeasibleHelper(subset, constraintVars)
  feasibleCount = int(feasibleMask.sum())
  feasibleFraction = feasibleCount / totalPoints if totalPoints else 0.0

  topCount = _topCountHelper(totalPoints, topFraction, topOverride)
  if topCap is not None:
    topCount = min(topCount, topCap)
  topMaskSeries = metricSeries[feasibleMask] if feasibleCount > 0 else metricSeries
  topIndices = topMaskSeries.nsmallest(topCount).index
  for idx in topIndices:
    colors[subset.index.get_loc(idx)] = topGreen

  bestSeries = metricSeries[feasibleMask] if feasibleCount > 0 else metricSeries
  bestMask = np.zeros(totalPoints, dtype=bool)
  bestOriginal = None
  bestIdx = None
  paretoMask = None
  paretoCount = 0
  multiObjective = False
  if 'rank' in subset.columns:
    try:
      ranks = subset['rank'].astype(int)
      paretoMask = (ranks == 1).to_numpy()
      paretoCount = int(paretoMask.sum())
      multiObjective = ranks.max() > 1
    except (ValueError, TypeError):
      paretoMask = None
  if multiObjective and paretoMask is not None and paretoCount > 0:
    colors[paretoMask] = bestRed
    bestMask = paretoMask
    bestPositions = np.where(paretoMask)[0]
    if bestPositions.size > 0:
      bestPosIdx = int(bestPositions[0])
      bestIdx = subset.index[bestPosIdx]
      if bestIdx in originalSeries.index:
        bestOriginal = float(originalSeries.loc[bestIdx])
  else:
    if not bestSeries.empty:
      bestIdx = bestSeries.idxmin()
      loc = subset.index.get_loc(bestIdx)
      bestMask[loc] = True
      bestOriginal = float(originalSeries.loc[bestIdx])
    colors[bestMask] = bestRed

  finalMask = None
  if 'accepted' in subset.columns:
    acceptedVals = subset['accepted'].astype(str).str.lower()
    finalMask = acceptedVals == 'final'
    if finalMask.any():
      finalNumpy = finalMask.to_numpy()
      colors[finalNumpy] = bestRed
      bestMask = np.logical_or(bestMask, finalNumpy)
      if bestIdx is None:
        finalIndices = finalMask[finalMask].index
        if len(finalIndices):
          bestIdx = finalIndices[0]
      if bestOriginal is None and bestIdx is not None and bestIdx in originalSeries.index:
        bestOriginal = float(originalSeries.loc[bestIdx])

  if paretoMask is not None and not (multiObjective and paretoMask.any()):
    topPositions = subset.index.get_indexer(topIndices)
    for pos in topPositions:
      if pos >= 0 and not bestMask[pos]:
        colors[pos] = topGreen

  if constraintVars:
    for pos, feasible in enumerate(feasibleMask):
      if not feasible:
        colors[pos] = infeasibleRgba

  fraction = topCount / totalPoints if totalPoints else 0.0
  summary = {'count': totalPoints,
             'pareto': paretoCount,
             'top': topCount,
             'best': bestOriginal,
             'metric_kind': metricKind,
             'metric': metricName,
             'top_fraction': fraction,
             'feasible': feasibleCount,
             'feasible_fraction': feasibleFraction,
             'has_constraints': bool(constraintVars)}
  if last:
    for pos in range(totalPoints):
      if not bestMask[pos]:
        colors[pos][3] = min(colors[pos][3], 0.45)
  bestCoords = None
  bestValue = None
  if bestMask.any():
    bestPos = int(np.where(bestMask)[0][0])
    colors[bestPos][3] = 1.0
    bestCoords = subset.iloc[bestPos][axes].to_numpy(dtype=float)
    bestValue = float(subset.iloc[bestPos][metricName])
  else:
    bestPos = None
  meta = {'best_pos': bestPos, 'best_coords': bestCoords, 'best_value': bestValue}
  return colors, summary, meta


class ObjectiveContourAnimationPlot(PlotInterface):
  """
  Draws optimizer populations over concentric contour lines of equal combined objective value.
  Points animate across generations with colors indicating relative quality.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('axes', contentType=InputTypes.StringListType,
        descr=r"""Exactly two decision-variable names defining the contour plane (e.g., x1, x3)."""))
    objectiveNode = InputData.parameterInputFactory('objective', contentType=InputTypes.StringType,
        descr=r"""Name of the metric column defining contour levels (or "all" to render each objective column separately). Optional attribute type="fitness" treats it as a maximized fitness value.""")
    objectiveNode.addParam('type', InputTypes.StringType)
    spec.addSub(objectiveNode)
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional list of constraint evaluation columns (e.g., ConstraintEvaluation_constraint1). Values > 0 are considered feasible; values <= 0 indicate constraint violation."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Name of the generation identifier (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('top', contentType=InputTypes.FloatType,
        descr=r"""Highlight threshold. Values >1 indicate a count; values in (0,1] indicate a population fraction. Defaults to 0.2 (20%)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combinations."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for generated animations. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('saveFrames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves each generation as a standalone PNG frame alongside the animation outputs."""))
    spec.addSub(InputData.parameterInputFactory('view', contentType=InputTypes.StringType,
        descr=r"""Optional legacy flag; values "2d", "3d", or "both" are accepted but the plot currently renders only the 2-D contour."""))
    spec.addSub(InputData.parameterInputFactory('surface', contentType=InputTypes.BoolType,
        descr=r"""Legacy flag; retained for compatibility but currently ignored (only the 2-D contour is rendered)."""))
    spec.addSub(InputData.parameterInputFactory('framesMax', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <saveFrames> is true. Defaults to 10; frames are sampled evenly across generations."""))
    spec.addSub(InputData.parameterInputFactory('showHistory', contentType=InputTypes.BoolType,
        descr=r"""If true, retain points from earlier generations as muted grey markers to visualize exploration history."""))
    spec.addSub(InputData.parameterInputFactory('historyAlpha', contentType=InputTypes.FloatType,
        descr=r"""Alpha value in [0, 1] applied to history markers when <showHistory> is true. Defaults to 0.15."""))
    spec.addSub(InputData.parameterInputFactory('historyColor', contentType=InputTypes.StringType,
        descr=r"""Matplotlib-compatible color for history markers when <showHistory> is true. Defaults to a neutral grey."""))
    spec.addSub(InputData.parameterInputFactory('infeasibleColor', contentType=InputTypes.StringType,
        descr=r"""Optional override for the infeasible sample color. When <showHistory> is true, the default becomes orange unless overridden here."""))
    spec.addSub(InputData.parameterInputFactory('displayFraction', contentType=InputTypes.FloatType,
        descr=r"""Optional fraction (0-1] of each generation's population to plot when the population exceeds <displayThreshold>. Defaults to 1.0 (show all)."""))
    spec.addSub(InputData.parameterInputFactory('displayThreshold', contentType=InputTypes.IntegerType,
        descr=r"""Population size above which <displayFraction> filtering activates. Defaults to 20."""))
    plotGenerationUtils.addGenerationSelectorSpec(spec)
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ObjectiveContourAnimation'
    self.source = None
    self.sourceName = None
    self.axes = []
    self.index = None
    self.metricName = None
    self.metricNames = []
    self.metricKind = 'objective'
    self.constraintVars = []
    self.topFraction = 0.2
    self.topCountOverride = None
    self.fps = 2.0
    self.formats = {'gif', 'html'}
    self.saveFrames = False
    self.frameMax = 10
    self.explicitGenerations = None
    self.showHistory = False
    self.historyAlpha = 0.15
    self.historyColor = DEFAULT_HISTORY_POINT_COLOR
    self.historyMarkerSize = HISTORY_MARKER_SIZE
    self._historyFacecolor = mcolors.to_rgba(self.historyColor, self.historyAlpha)
    self._historyLookup = {}
    self._customInfeasibleColor = None
    self._infeasiblePointColor = DEFAULT_INFEASIBLE_POINT_COLOR
    self.displayFraction = 1.0
    self.displayThreshold = 20
    self.metricAll = False
    self.metricColumns = []
    self._colorMetric = None
    self.metricKinds = {}
    self._colorMetricKind = None
    self._multiObjective = False
    self._updateVisualConfig()

  def handleInput(self, spec):
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value if spec.findFirst('source') is not None else None
    axes = spec.findFirst('axes')
    if axes is None:
      self.raiseAnError(IOError, f'Missing <axes> node for ObjectiveContourAnimationPlot "{self.name}".')
    self.axes = axes.value
    if len(self.axes) != 2:
      self.raiseAnError(IOError, f'ObjectiveContourAnimationPlot "{self.name}" requires exactly two axes variables.')
    objNode = spec.findFirst('objective')
    if objNode is None:
      self.raiseAnError(IOError, f'Missing <objective> node for ObjectiveContourAnimationPlot "{self.name}".')
    rawMetric = objNode.value.strip() if objNode.value is not None else ''
    if rawMetric.lower() == 'all':
      self.metricAll = True
      self.metricName = None
      self.metricNames = []
    else:
      if not rawMetric:
        self.raiseAnError(IOError, f'<objective> must specify a metric name or "all" for ObjectiveContourAnimationPlot "{self.name}".')
      self.metricAll = False
      entries = [item.strip() for item in rawMetric.replace(';', ',').split(',') if item.strip()]
      if not entries:
        self.raiseAnError(IOError, f'<objective> must specify at least one metric name for ObjectiveContourAnimationPlot "{self.name}".')
      self.metricNames = entries
      self.metricName = self.metricNames[0]
    objType = objNode.parameterValues.get('type', 'objective').lower()
    if objType not in {'objective', 'fitness'}:
      self.raiseAnError(IOError, f'Unsupported objective type "{objType}" for ObjectiveContourAnimationPlot "{self.name}". Use "objective" or "fitness".')
    self.metricKind = objType
    viewNode = spec.findFirst('view')  # accepted for backward compatibility; ignored
    surfaceNode = spec.findFirst('surface')
    consNode = spec.findFirst('constraints')
    if consNode is not None:
      raw = consNode.value or []
      self.constraintVars = [c for c in raw if c]
    else:
      self.constraintVars = []
    idxNode = spec.findFirst('index')
    if idxNode is None:
      self.raiseAnError(IOError, f'Missing <index> node for ObjectiveContourAnimationPlot "{self.name}".')
    self.index = idxNode.value
    topNode = spec.findFirst('top')
    if topNode is not None:
      topVal = topNode.value
      if topVal <= 0.0:
        self.raiseAnError(IOError, f'<top> must be positive for ObjectiveContourAnimationPlot "{self.name}".')
      if topVal <= 1.0:
        self.topFraction = topVal
        self.topCountOverride = None
      else:
        self.topCountOverride = int(math.ceil(topVal))
    fpsNode = spec.findFirst('fps')
    if fpsNode is not None:
      self.fps = max(fpsNode.value, 0.1)
    fmtNode = spec.findFirst('format')
    if fmtNode is not None:
      self.formats = self._parseFormats(fmtNode.value)
    else:
      self.formats = {'gif', 'html'}
    framesNode = spec.findFirst('saveFrames')
    if framesNode is not None:
      self.saveFrames = bool(framesNode.value)
    frameMaxNode = spec.findFirst('framesMax')
    if frameMaxNode is not None:
      raw = int(frameMaxNode.value)
      if raw <= 0:
        self.raiseAnError(IOError, f'<framesMax> must be positive for ObjectiveContourAnimationPlot "{self.name}".')
      self.frameMax = raw
    self.explicitGenerations = plotGenerationUtils.parseGenerationSelectorNode(spec)
    historyNode = spec.findFirst('showHistory')
    if historyNode is not None:
      self.showHistory = bool(historyNode.value)
    historyAlphaNode = spec.findFirst('historyAlpha')
    if historyAlphaNode is not None:
      alphaVal = float(historyAlphaNode.value)
      if not (0.0 <= alphaVal <= 1.0):
        self.raiseAnError(IOError, f'<historyAlpha> must be within [0, 1] for ObjectiveContourAnimationPlot "{self.name}".')
      self.historyAlpha = alphaVal
    historyColorNode = spec.findFirst('historyColor')
    if historyColorNode is not None:
      colorVal = historyColorNode.value
      try:
        mcolors.to_rgba(colorVal)
      except ValueError as err:
        self.raiseAnError(IOError, f'Invalid <historyColor> value "{colorVal}" for ObjectiveContourAnimationPlot "{self.name}": {err}')
      self.historyColor = colorVal
    infeasibleColorNode = spec.findFirst('infeasibleColor')
    if infeasibleColorNode is not None:
      infeasibleVal = infeasibleColorNode.value
      try:
        mcolors.to_rgba(infeasibleVal)
      except ValueError as err:
        self.raiseAnError(IOError, f'Invalid <infeasibleColor> value "{infeasibleVal}" for ObjectiveContourAnimationPlot "{self.name}": {err}')
      self._customInfeasibleColor = infeasibleVal
    displayFractionNode = spec.findFirst('displayFraction')
    if displayFractionNode is not None:
      fracVal = float(displayFractionNode.value)
      if not (0.0 < fracVal <= 1.0):
        self.raiseAnError(IOError, f'<displayFraction> must be in (0, 1] for ObjectiveContourAnimationPlot "{self.name}".')
      self.displayFraction = fracVal
    displayThresholdNode = spec.findFirst('displayThreshold')
    if displayThresholdNode is not None:
      thresholdVal = int(displayThresholdNode.value)
      if thresholdVal < 1:
        self.raiseAnError(IOError, f'<displayThreshold> must be positive for ObjectiveContourAnimationPlot "{self.name}".')
      self.displayThreshold = thresholdVal
    self._updateVisualConfig()

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    self.source = self.findSource(self.sourceName, stepEntities)
    if self.source is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" located for ObjectiveContourAnimationPlot "{self.name}".')
    dataVars = self.source.getVars()
    required = list(self.axes) + [self.index]
    if not self.metricAll:
      required.extend(self.metricNames)
    missing = [var for var in required if var not in dataVars]
    if missing:
      msg = 'Source DataObject "{}" is missing variable(s) {} required by ObjectiveContourAnimationPlot "{}".'.format(
          self.source.name, ', '.join(f'"{m}"' for m in missing), self.name)
      self.raiseAnError(IOError, msg)
    for var in self.constraintVars:
      if var not in dataVars:
        self.raiseAnError(IOError, f'Constraint variable "{var}" not found in source DataObject "{self.source.name}" for ObjectiveContourAnimationPlot "{self.name}".')
    if self.metricAll:
      self.metricColumns = self._discoverObjectiveColumns(dataVars)
      if not self.metricColumns:
        self.raiseAnError(IOError, f'Unable to determine objective columns when <objective> is "all" for ObjectiveContourAnimationPlot "{self.name}". Expected columns beginning with "obj".')
      self.metricName = self.metricColumns[0]
    else:
      self.metricColumns = list(self.metricNames)
      self.metricName = self.metricColumns[0]
    self.metricKinds = {}
    for metric in self.metricColumns:
      if metric not in dataVars:
        self.raiseAnError(IOError, f'Objective column "{metric}" not found in source DataObject "{self.source.name}" for ObjectiveContourAnimationPlot "{self.name}".')
      fitnessCol = f'FitnessEvaluation_{metric}'
      kind = self.metricKind
      if fitnessCol in dataVars:
        kind = 'fitness'
      self.metricKinds[metric] = kind
    self._colorMetric = self.metricColumns[0]
    self._colorMetricKind = self.metricKinds.get(self._colorMetric, self.metricKind)
    self._multiObjective = len(self.metricColumns) > 1

  def run(self):
    df = self.source.asDataset().to_dataframe()
    allGenerations = sorted(df[self.index].unique())
    if not allGenerations:
      self.raiseAWarning(f'No generations found for ObjectiveContourAnimationPlot "{self.name}".')
      return
    try:
      generations, _ = plotGenerationUtils.resolveGenerations(
          allGenerations, self.explicitGenerations, defaultCap=self.frameMax, maxFrames=None)
    except ValueError as err:
      self.raiseAnError(IOError, f'ObjectiveContourAnimationPlot "{self.name}": {err}')
    if not generations:
      self.raiseAWarning(f'ObjectiveContourAnimationPlot "{self.name}" did not select any generations to render.')
      return
    plotContext = self._buildPlotContext(df)
    historyLookup = self._buildHistoryOffsets(df, allGenerations) if self.showHistory else {}
    self._historyLookup = historyLookup
    frameTemplate = None
    if self.saveFrames:
      base = self._createFilename(defaultName=f'{self.name}_frames')
      frameTemplate = os.path.splitext(base)[0] + '_{index:04d}.png'
      frameDir = os.path.dirname(frameTemplate)
      if frameDir:
        os.makedirs(frameDir, exist_ok=True)

    for fmt in self.formats:
      if fmt == 'html':
        self._writeHtml(df, generations, plotContext,
                         historyLookup, filenameDefault=f'{self.name}.html')
      elif fmt == 'gif':
        self._writeGif(df, generations, plotContext,
                        historyLookup, filenameDefault=f'{self.name}.gif')
    if self.saveFrames:
      self._writeFrames(df, generations, plotContext,
                         historyLookup, frameTemplate)

  def _writeGif(self, df, generations, plotContext, historyLookup, filenameDefault):
    filename = self._createFilename(defaultName=filenameDefault)
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=1) as writer:
      for gen in generations:
        subset = df[df[self.index] == gen]
        historyOffsets = historyLookup.get(gen) if historyLookup else None
        fig = self._renderFrame(subset, gen, plotContext,
                                 last=(gen == generations[-1]), historyOffsets=historyOffsets)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _writeHtml(self, df, generations, plotContext, historyLookup, filenameDefault):
    filename = self._createFilename(defaultName=filenameDefault)
    firstGen = generations[0]
    historyOffsets = historyLookup.get(firstGen) if historyLookup else None
    if plotContext['multi']:
      fig, axes = self._createMultiAxes(len(self.metricColumns))
      axesList = np.atleast_1d(axes).tolist() if isinstance(axes, np.ndarray) else [axes]
      self._populateMultiAxes(axesList, df[df[self.index] == firstGen], plotContext,
                                historyOffsets=historyOffsets,
                                last=(firstGen == generations[-1]))
    else:
      fig, ax = self._createAxes(plotContext['axis_limits'])
      self._populateAxes(ax, df[df[self.index] == firstGen], plotContext,
                          historyOffsets=historyOffsets,
                          last=(firstGen == generations[-1]))
      axesList = [ax]
    suptitle = fig.suptitle('')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def init():
      suptitle.set_text('')
      return fig.axes

    def update(gen):
      subset = df[df[self.index] == gen]
      historyOffsets = historyLookup.get(gen) if historyLookup else None
      if plotContext['multi']:
        self._populateMultiAxes(axesList, subset, plotContext,
                                  historyOffsets=historyOffsets,
                                  last=(gen == generations[-1]))
      else:
        self._populateAxes(axesList[0], subset, plotContext,
                            historyOffsets=historyOffsets,
                            last=(gen == generations[-1]))
      suptitle.set_text(f'Generation {gen}')
      fig.tight_layout(rect=[0, 0, 1, 0.94])
      return fig.axes

    anim = animation.FuncAnimation(fig, update, frames=generations, init_func=init,
                                   interval=1000.0 / self.fps, blit=False)
    htmlStr = anim.to_jshtml()
    centeredHtml = f'<div style=\"display:flex;justify-content:center;\">{htmlStr}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centeredHtml)
    plt.close(fig)

  def _writeFrames(self, df, generations, plotContext, historyLookup, frameTemplate):
    if frameTemplate is None:
      return
    frameIndices = plotGenerationUtils.frameIndicesToSave(
        len(generations), bool(self.explicitGenerations), self.saveFrames, self.frameMax)
    for idx in frameIndices:
      gen = generations[idx]
      subset = df[df[self.index] == gen]
      historyOffsets = historyLookup.get(gen) if historyLookup else None
      fig = self._renderFrame(subset, gen, plotContext,
                               last=(gen == generations[-1]), historyOffsets=historyOffsets)
      framePath = frameTemplate.format(index=idx)
      fig.savefig(framePath, format='png')
      plt.close(fig)

  def _renderFrame(self, subset, gen, plotContext,
                    last=False, historyOffsets=None):
    if plotContext['multi']:
      fig, axes = self._createMultiAxes(len(self.metricColumns))
      axesList = np.atleast_1d(axes).tolist() if isinstance(axes, np.ndarray) else [axes]
      self._populateMultiAxes(axesList, subset, plotContext,
                                historyOffsets=historyOffsets, last=last)
    else:
      fig, ax = self._createAxes(plotContext['axis_limits'])
      self._populateAxes(ax, subset, plotContext,
                          historyOffsets=historyOffsets, last=last)
    fig.suptitle(f'Generation {gen}')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

  @staticmethod
  def _sampleGenerations(generations, limit):
    if limit >= len(generations):
      return list(generations)
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
    return [generations[i] for i in sorted(selected)[:limit]]

  def _selectFrameIndices(self, totalGenerations):
    if not self.saveFrames or totalGenerations <= 0 or self.frameMax <= 0:
      return []
    if totalGenerations <= self.frameMax:
      return list(range(totalGenerations))
    stride = int(math.ceil(totalGenerations / float(self.frameMax)))
    indices = list(range(0, totalGenerations, stride))
    if indices and indices[-1] != totalGenerations - 1:
      if len(indices) >= self.frameMax:
        indices[-1] = totalGenerations - 1
      else:
        indices.append(totalGenerations - 1)
    return sorted(set(indices))

  def _discoverObjectiveColumns(self, dataVars):
    """
    Infer objective column names when plotting all objectives.
    """
    outputs = []
    try:
      outputs = self.source.getVars('output')
    except Exception:
      outputs = []
    searchOrder = outputs if outputs else dataVars
    objectives = []
    seen = set()
    for var in searchOrder:
      norm = var.lower()
      if norm.startswith('obj') and var not in seen:
        objectives.append(var)
        seen.add(var)
    return objectives

  def _populateAxes(self, ax, subset, plotContext, historyOffsets=None, last=False):
    payload = self._preparePlotPayload(subset, last=last)
    axisLimits = plotContext['axis_limits']
    contourData = plotContext['contour_data']
    constraintData = plotContext['constraint_data']
    self._drawSingleAxis(
        ax, payload, axisLimits, contourData, constraintData,
        historyOffsets=historyOffsets, metricLabel=self._colorMetric,
        showSummary=True, showLegend=True, showXlabel=True, last=last)
    return payload['summary'], payload['meta']

  def _createAxes(self, axisLimits):
    fig, ax = plt.subplots(figsize=(6, 6))
    return fig, ax

  def _createMultiAxes(self, count):
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 6),
                             sharex=False, sharey=False)
    return fig, axes

  def _preparePlotPayload(self, subset, last=False):
    axisX, axisY = self.axes
    displaySubset, displayInfo = self._selectDisplaySubset(subset)
    colors, summary, meta = self._computeColors(
        displaySubset, metricName=self._colorMetric, last=last,
        topCap=displayInfo.get('top_cap'))
    offsets = displaySubset[[axisX, axisY]].to_numpy(dtype=float) if not displaySubset.empty else np.empty((0, 2))
    bestPos = meta.get('best_pos')
    plotOffsets = offsets
    plotColors = colors
    if bestPos is not None and len(offsets) > 0:
      order = np.concatenate((np.delete(np.arange(len(offsets)), bestPos), [bestPos]))
      plotOffsets = offsets[order]
      plotColors = colors[order]
    totalCount = len(subset)
    displayCount = len(displaySubset)
    summary['count_total'] = totalCount
    summary['displayed'] = displayCount
    summary['display_fraction'] = displayCount / float(totalCount) if totalCount else 0.0
    summary['count'] = totalCount
    if totalCount:
      summary['top_fraction'] = summary['top'] / float(totalCount)
    payload = {
        'subset': displaySubset,
        'full_subset': subset,
        'plot_offsets': plotOffsets,
        'plot_colors': plotColors,
        'raw_offsets': offsets,
        'colors': colors,
        'summary': summary,
        'meta': meta
    }
    return payload

  def _populateMultiAxes(self, axes, subset, plotContext, historyOffsets=None, last=False):
    if not isinstance(axes, (list, tuple, np.ndarray)):
      axes = [axes]
    else:
      axes = np.atleast_1d(axes).tolist()
    payload = self._preparePlotPayload(subset, last=last)
    constraintData = plotContext['constraint_data']
    metrics = self.metricColumns
    for idx, metric in enumerate(metrics):
      axisInfo = plotContext['geometry_map'][metric]
      showSummary = (idx == 0)
      showLegend = (idx == len(metrics) - 1)
      showXlabel = True
      self._drawSingleAxis(
          axes[idx], payload, axisInfo['axis_limits'], axisInfo['contour_data'],
          constraintData, historyOffsets=historyOffsets, metricLabel=metric,
          showSummary=showSummary, showLegend=showLegend,
          showXlabel=showXlabel, last=last)
    return payload

  def _drawSingleAxis(self, ax, payload, axisLimits, contourData, constraintData,
                        historyOffsets=None, metricLabel=None, showSummary=True,
                        showLegend=True, showXlabel=True, last=False):
    axisX, axisY = self.axes
    if hasattr(ax, '_objective_contour_summary'):
      delattr(ax, '_objective_contour_summary')
    if hasattr(ax, '_objective_contour_legend'):
      try:
        ax._objective_contour_legend.remove()
      except Exception:
        pass
      delattr(ax, '_objective_contour_legend')
    ax.clear()
    self._drawContours(ax, contourData, constraintData)
    historyPoints = historyOffsets if historyOffsets is not None else np.empty((0, 2))
    if self.showHistory and historyPoints.size:
      ax.scatter(historyPoints[:, 0], historyPoints[:, 1],
                 facecolors=[self._historyFacecolor],
                 edgecolors='none',
                 linewidths=0.0,
                 s=self.historyMarkerSize,
                 zorder=2)
    plotOffsets = payload['plot_offsets']
    plotColors = payload['plot_colors']
    if len(plotOffsets):
      ax.scatter(plotOffsets[:, 0], plotOffsets[:, 1], facecolors=plotColors,
                 edgecolors='k', linewidths=0.3, s=60, zorder=3)
    xMin, xMax = axisLimits[axisX]
    yMin, yMax = axisLimits[axisY]
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    if showXlabel:
      ax.set_xlabel(axisX)
      ax.tick_params(labelbottom=True)
    else:
      ax.set_xlabel('')
      ax.tick_params(labelbottom=False)
    ax.set_ylabel(axisY)
    label = metricLabel or payload['summary'].get('metric')
    title = f'{axisX} vs {axisY}'
    if label:
      title += f' | objective: {label}'
    ax.set_title(title)
    bestCoords = payload['meta'].get('best_coords') if payload.get('meta') else None
    crossMetric = metricLabel or self._colorMetric
    if bestCoords is None:
      bestCoords = self._bestCoordsForMetric(payload.get('full_subset'), crossMetric)
    if bestCoords is None:
      bestCoords = self._bestCoordsForMetric(payload['subset'], crossMetric)
    if bestCoords is not None:
      self._drawCrosshair(ax, bestCoords, axisLimits)
    if showSummary:
      self._updateSummary(ax, payload['summary'])
    if showLegend:
      self._ensureLegend(ax)

  def _bestCoordsForMetric(self, subset, metricName):
    if subset is None or metricName is None or subset.empty or metricName not in subset.columns:
      return None
    metricKind = self.metricKinds.get(metricName, self.metricKind)
    metricSeries, _ = _metricSeriesHelper(subset, metricName, metricKind)
    if metricSeries.empty:
      return None
    bestIdx = metricSeries.idxmin()
    if bestIdx not in subset.index:
      return None
    return subset.loc[bestIdx, self.axes].to_numpy(dtype=float)


  def _computeColors(self, subset, metricName=None, last=False, topCap=None):
    metric = metricName or self._colorMetric or (self.metricColumns[0] if self.metricColumns else self.metricName)
    metricKind = self.metricKinds.get(metric, self.metricKind)
    return _computePopulationColors(subset, self.axes, metric, metricKind,
                                      self.constraintVars, self.topFraction,
                                      self.topCountOverride, last=last,
                                      infeasibleColor=self._infeasiblePointColor,
                                      topCap=topCap)

  def _updateVisualConfig(self):
    if self._customInfeasibleColor is not None:
      targetColor = self._customInfeasibleColor
    else:
      targetColor = ENHANCED_INFEASIBLE_POINT_COLOR if self.showHistory else DEFAULT_INFEASIBLE_POINT_COLOR
    self._infeasiblePointColor = targetColor
    self._historyFacecolor = mcolors.to_rgba(self.historyColor, self.historyAlpha)

  def _buildHistoryOffsets(self, df, generations):
    """
    Construct a per-generation lookup of exploration history coordinates.
    """
    historyLookup = {}
    dims = len(self.axes)
    cumulative = np.empty((0, dims), dtype=float)
    for gen in generations:
      # Store a copy that reflects all prior generations only.
      historyLookup[gen] = cumulative.copy()
      subset = df[df[self.index] == gen]
      if subset.empty:
        continue
      coords = subset[self.axes].to_numpy(dtype=float)
      if coords.size:
        cumulative = np.vstack((cumulative, coords))
    return historyLookup

  def _selectDisplaySubset(self, subset):
    """
    Limit the displayed population when generations become large.
    Returns the filtered subset and auxiliary info for plotting.
    """
    info = {'top_cap': None, 'limited': False}
    total = len(subset)
    if total == 0:
      return subset, info
    if total <= self.displayThreshold or self.displayFraction >= 0.9999:
      return subset, info
    displayCount = max(1, int(math.ceil(total * self.displayFraction)))
    metricKind = self.metricKinds.get(self._colorMetric, self.metricKind)
    metricSeries, _ = _metricSeriesHelper(subset, self._colorMetric, metricKind)
    primary = metricSeries.nsmallest(displayCount).index.tolist()
    selection = list(primary)
    if 'rank' in subset.columns:
      try:
        ranks = subset['rank'].astype(int)
        paretoIndices = subset.index[ranks == 1].tolist()
      except (ValueError, TypeError):
        paretoIndices = []
      for idx in paretoIndices:
        if idx not in selection:
          selection.append(idx)
    if not selection:
      selection = subset.index[:displayCount].tolist()
    displaySubset = subset.loc[selection]
    info['limited'] = True
    if self.topCountOverride is None:
      info['top_cap'] = 5
    return displaySubset, info

  def _preparePlotGeometry(self, df, metricName):
    metricKind = self.metricKinds.get(metricName, self.metricKind)
    metricSeries, originalSeries = _metricSeriesHelper(df, metricName, metricKind)
    metricMin = metricSeries.min()
    metricMax = metricSeries.max()

    centerIdx = metricSeries.idxmin()
    centerPoint = df.loc[centerIdx, self.axes].to_numpy(dtype=float)
    axisPoints = df[self.axes].to_numpy(dtype=float)
    distances = np.linalg.norm(axisPoints - centerPoint, axis=1)
    maxRadius = distances.max()
    if np.isclose(maxRadius, 0.0):
      maxRadius = 1.0

    limits = {}
    for i, axis in enumerate(self.axes):
      column = df[axis].astype(float)
      dataMin = column.min()
      dataMax = column.max()
      offsetMin = abs(centerPoint[i] - dataMin)
      offsetMax = abs(dataMax - centerPoint[i])
      halfSpan = max(maxRadius, offsetMin, offsetMax)
      if np.isclose(halfSpan, 0.0):
        halfSpan = 1.0
      padding = 0.05 * halfSpan
      halfSpan += padding
      axisMin = centerPoint[i] - halfSpan
      axisMax = centerPoint[i] + halfSpan
      limits[axis] = (axisMin, axisMax)

    if np.isclose(metricMax, metricMin):
      radii = []
    else:
      levels = np.linspace(metricMin, metricMax, 6)[1:]
      radii = []
      for lvl in levels:
        fraction = max((lvl - metricMin) / (metricMax - metricMin), 0.0)
        radius = math.sqrt(fraction) * maxRadius
        if radius > 0.0:
          originalLvl = _metricToOriginalHelper(lvl, metricKind)
          radii.append((radius, float(lvl), float(originalLvl)))

    shade = None
    if maxRadius > 0:
      xVals = np.linspace(limits[self.axes[0]][0], limits[self.axes[0]][1], 200)
      yVals = np.linspace(limits[self.axes[1]][0], limits[self.axes[1]][1], 200)
      X, Y = np.meshgrid(xVals, yVals)
      radiusGrid = np.sqrt((X - centerPoint[0]) ** 2 + (Y - centerPoint[1]) ** 2)
      shade = (X, Y, np.clip(radiusGrid / maxRadius, 0.0, 1.0))

    contourData = {
        'center': centerPoint,
        'radii': radii,
        'metric_min': float(metricMin),
        'metric_max': float(metricMax),
        'kind': metricKind,
        'metric_name': metricName,
        'original_min': float(originalSeries.min()),
        'original_max': float(originalSeries.max()),
        'shade': shade,
        'max_radius': maxRadius
    }
    constraintData = self._prepareConstraintGeometry(df)
    return limits, contourData, constraintData

  def _buildPlotContext(self, df):
    if self._multiObjective:
      geometryMap = {}
      constraintDataShared = None
      for metric in self.metricColumns:
        limits, contourData, constraintData = self._preparePlotGeometry(df, metric)
        geometryMap[metric] = {'axis_limits': limits, 'contour_data': contourData}
        if constraintDataShared is None:
          constraintDataShared = constraintData
      if constraintDataShared is None:
        constraintDataShared = []
      return {'multi': True,
              'geometry_map': geometryMap,
              'constraint_data': constraintDataShared}
    axisLimits, contourData, constraintData = self._preparePlotGeometry(df, self._colorMetric)
    return {'multi': False,
            'axis_limits': axisLimits,
            'contour_data': contourData,
            'constraint_data': constraintData}

  def _drawContours(self, ax, contourData, constraintData=None):
    center = contourData['center']
    radii = contourData['radii']
    shade = contourData.get('shade')
    if constraintData:
      self._drawConstraintOverlays(ax, constraintData)
    if shade is not None:
      X, Y, Shade = shade
      ax.contourf(X, Y, Shade, levels=np.linspace(0, 1, 10), cmap=SHADE_CMAP, alpha=0.35, antialiased=True, zorder=0)
    if not radii:
      return
    for radius, lvl, originalLvl in radii:
      circle = patches.Circle(center, radius, fill=False, linestyle='--', linewidth=0.8, alpha=0.6, color='lightgray', zorder=1)
      ax.add_patch(circle)
      ax.text(center[0], center[1] + radius, f'{originalLvl:.4g}', fontsize=8, ha='center', va='bottom', color='gray', alpha=0.8, zorder=2)

  def _updateSummary(self, ax, summary):
    text = self._formatSummary(summary)
    if hasattr(ax, '_objective_contour_summary'):
      handle = ax._objective_contour_summary
      if hasattr(handle, 'set_text'):
        handle.set_text(text)
        return
    kwargs = dict(fontsize=9, va='top')
    if hasattr(ax, 'text2D') and getattr(ax, 'name', '').lower() == '3d':
      handle = ax.text2D(0.02, 0.95, text, transform=ax.transAxes, **kwargs)
    else:
      handle = ax.text(0.02, 0.95, text, transform=ax.transAxes, **kwargs)
    ax._objective_contour_summary = handle

  @staticmethod
  def _formatSummary(summary):
    if summary['count'] == 0:
      return 'Population: 0'
    parts = [
        f'Population: {summary["count"]}',
        f'Pareto rank 1: {summary["pareto"]}',
        f'Top highlighted: {summary["top"]} ({summary.get("top_fraction", 0.0)*100:.1f}%)'
    ]
    if summary.get('has_constraints'):
      feasible = summary.get('feasible', 0)
      fraction = summary.get('feasible_fraction', 0.0) * 100.0
      parts.append(f'Feasible points: {feasible} ({fraction:.1f}%)')
    if summary.get('best') is not None:
      label = summary.get('metric', 'metric')
      if summary.get('metric_kind') == 'fitness':
        parts.append(f'Best fitness ({label}): {summary["best"]:.4g}')
      else:
        parts.append(f'Best {label}: {summary["best"]:.4g}')
    displayed = summary.get('displayed')
    total = summary.get('count_total')
    if displayed is not None and total is not None and displayed != total:
      fraction = summary.get('display_fraction', 0.0) * 100.0 if total else 0.0
      parts.append(f'Displayed: {displayed} ({fraction:.1f}%)')
    return '\n'.join(parts)

  def _parseFormats(self, raw):
    if raw is None:
      return {'gif', 'html'}
    text = raw.strip().lower()
    if text == 'both':
      return {'gif', 'html'}
    tokens = {token.strip() for token in text.replace(';', ',').split(',') if token.strip()}
    if not tokens:
      return {'gif', 'html'}
    allowed = {'gif', 'html'}
    invalid = tokens - allowed
    if invalid:
      bad = ', '.join(sorted(invalid))
      self.raiseAnError(IOError, f'Unsupported format(s) "{bad}" for ObjectiveContourAnimationPlot "{self.name}". Use "gif", "html", or "both".')
    return tokens

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

  def _prepareConstraintGeometry(self, df):
    if not self.constraintVars:
      return []
    axisX, axisY = self.axes
    xVals = df[axisX].astype(float).to_numpy()
    yVals = df[axisY].astype(float).to_numpy()
    constraintData = []
    for idx, var in enumerate(self.constraintVars):
      values = df[var].astype(float).to_numpy()
      mask = np.isfinite(xVals) & np.isfinite(yVals) & np.isfinite(values)
      if mask.sum() < 3:
        self.raiseAWarning(f'Constraint "{var}" has insufficient samples to render on ObjectiveContourAnimationPlot "{self.name}".')
        continue
      x = xVals[mask]
      y = yVals[mask]
      g = values[mask]
      coords = np.column_stack((x, y))
      try:
        uniqueCoords, uniqueIdx = np.unique(coords, axis=0, return_index=True)
      except TypeError:
        # numpy <1.13 compatibility: fallback by rounding
        rounded = np.round(coords, decimals=10)
        _, uniqueIdx = np.unique(rounded, axis=0, return_index=True)
        uniqueCoords = coords[uniqueIdx]
      xUnique = uniqueCoords[:, 0]
      yUnique = uniqueCoords[:, 1]
      gUnique = g[uniqueIdx]
      finiteMask = np.isfinite(gUnique)
      xUnique = xUnique[finiteMask]
      yUnique = yUnique[finiteMask]
      gUnique = gUnique[finiteMask]
      if xUnique.size < 3:
        self.raiseAWarning(f'Constraint "{var}" collapsed to insufficient points after filtering for ObjectiveContourAnimationPlot "{self.name}".')
        continue
      triang = mtri.Triangulation(xUnique, yUnique)
      if triang.triangles.size == 0:
        continue
      constraintData.append({
          'name': var,
          'x': xUnique,
          'y': yUnique,
          'values': gUnique,
          'triangulation': triang,
          'fill_color': CONSTRAINT_FILL_COLOR,
          'line_color': CONSTRAINT_LINE_COLORS[idx % len(CONSTRAINT_LINE_COLORS)]
      })
    return constraintData

  def _drawConstraintOverlays(self, ax, constraintData):
    for info in constraintData:
      values = info['values']
      triang = info['triangulation']
      tris = triang.triangles
      triVals = values[tris]
      infeasibleMask = np.max(triVals, axis=1) <= 0.0
      if infeasibleMask.any():
        polys = np.stack((info['x'][tris[infeasibleMask]], info['y'][tris[infeasibleMask]]), axis=-1)
        collection = PolyCollection(polys, facecolors=info['fill_color'], edgecolors='none', alpha=0.35, zorder=0.5)
        ax.add_collection(collection)
      try:
        ax.tricontour(triang, values, levels=[0.0], colors=info['line_color'], linewidths=1.1, zorder=1.5)
      except Exception:
        continue

  def _drawCrosshair(self, ax, bestCoords, axisLimits):
    xLine, yLine, xHline, yHline = self._crosshairSegments(bestCoords, axisLimits)
    ax.plot(xLine, yLine, linestyle='--', color=CROSSHAIR_COLOR, linewidth=1.0, zorder=2.6)
    ax.plot(xHline, yHline, linestyle='--', color=CROSSHAIR_COLOR, linewidth=1.0, zorder=2.6)

  def _crosshairSegments(self, bestCoords, axisLimits):
    xVal, yVal = bestCoords
    xMin, _ = axisLimits[self.axes[0]]
    yMin, _ = axisLimits[self.axes[1]]
    return [xVal, xVal], [yMin, yVal], [xMin, xVal], [yVal, yVal]

  def _ensureLegend(self, ax):
    if hasattr(ax, '_objective_contour_legend'):
      try:
        ax._objective_contour_legend.remove()
      except Exception:
        pass
      delattr(ax, '_objective_contour_legend')
    baseLabel = 'Current feasible population' if self.showHistory else 'Feasible population'
    baseHandle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                         markerfacecolor=BASE_POINT_COLOR, markeredgecolor='k', label=baseLabel)
    topHandle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                        markerfacecolor=TOP_POINT_COLOR, markeredgecolor='k', label='Top highlighted')
    bestHandle = Line2D([0], [0], marker='o', linestyle='', markersize=7,
                         markerfacecolor=BEST_POINT_COLOR, markeredgecolor='k', label='Best solution')
    infeasibleHandle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                               markerfacecolor=self._infeasiblePointColor, alpha=0.85,
                               markeredgecolor='k', label='Infeasible samples')
    regionHandle = Patch(facecolor=CONSTRAINT_FILL_COLOR, alpha=0.35, label='Infeasible region')
    handles = [bestHandle, topHandle, baseHandle]
    if self.showHistory:
      historyHandle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                              markerfacecolor=self._historyFacecolor, markeredgecolor='none',
                              alpha=self.historyAlpha,
                              label='History (prior generations)')
      handles.append(historyHandle)
    handles.extend([infeasibleHandle, regionHandle])
    legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.17),
                       frameon=False, ncol=3, fontsize=9)
    ax._objective_contour_legend = legend
