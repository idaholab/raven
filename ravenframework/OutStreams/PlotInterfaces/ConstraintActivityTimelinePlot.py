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
Animate constraint violation fractions across generations.

Each frame extends the timeline to the current generation. Single-constraint runs show one
bar chart, while multi-constraint runs place each constraint in its own subplot (up to three
columns per row) so feasibility trends can be compared side-by-side.

What-if scenarios

One constraint dominates every subplot -> that condition is chronically violated; consider
  relaxing tolerances, improving repair operators, or focusing mutation on variables tied to it.
All subplots remain near 100% -> the search rarely finds feasible points; revisit initialization,
  constraint scaling, or penalty handling.
Subplots flatten but later spike again -> later algorithm stages (e.g., restarts, survivors)
  are reintroducing infeasibility; audit those transitions.
Occasional spikes across every subplot -> population resets or exploration bursts are probing
  unfamiliar territory; ensure repair operators or penalties are applied consistently.
"""

import io
import math
import os

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import imageio.v2 as imageio

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes
from . import plotGenerationUtils


class ConstraintActivityTimelinePlot(PlotInterface):
  """
  Animated constraint violation timelines with one subplot per constraint.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional list of constraint evaluation columns (values <= 0 indicate violation). Use "all" or omit this node to include every column named like ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
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
    self.printTag = 'ConstraintActivityTimelinePlot'
    self.source = None
    self.sourceName = None
    self.constraints = []
    self.useAllConstraints = False
    self.index = None
    self.maxFrames = None
    self.explicitGenerations = None
    self.formats = {'gif', 'html'}
    self.fps = 2.0
    self.saveFrames = False
    self.frameMax = 10
    self._subplotLayout = (1, 1)

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for ConstraintActivityTimelinePlot "{self.name}".')
    self.sourceName = sourceNode.value

    consNode = spec.findFirst('constraints')
    if consNode is None or not consNode.value:
      self.useAllConstraints = True
    else:
      entries = [entry.strip() for entry in consNode.value if entry and entry.strip()]
      if len(entries) == 1 and entries[0].strip().lower() == 'all':
        self.useAllConstraints = True
      else:
        self.constraints = entries
        if not self.constraints:
          self.useAllConstraints = True

    indexNode = spec.findFirst('index')
    if indexNode is None or indexNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for ConstraintActivityTimelinePlot "{self.name}".')
    self.index = indexNode.value

    maxNode = spec.findFirst('maxFrames')
    if maxNode is not None and maxNode.value is not None:
      self.maxFrames = int(maxNode.value)
      if self.maxFrames <= 0:
        self.raiseAnError(IOError, f'ConstraintActivityTimelinePlot "{self.name}" received non-positive <maxFrames>.')

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
            self.raiseAnError(IOError, f'Unsupported <format> "{item}" for ConstraintActivityTimelinePlot "{self.name}".')
        if not mapped:
          mapped = {'gif'}
        self.formats = mapped

    fpsNode = spec.findFirst('fps')
    if fpsNode is not None and fpsNode.value is not None:
      self.fps = float(fpsNode.value)
      if self.fps <= 0:
        self.raiseAnError(IOError, f'ConstraintActivityTimelinePlot "{self.name}" received non-positive <fps>.')

    saveNode = spec.findFirst('saveFrames')
    if saveNode is not None and saveNode.value is not None:
      self.saveFrames = bool(saveNode.value)

    framesMaxNode = spec.findFirst('framesMax')
    if framesMaxNode is not None and framesMaxNode.value is not None:
      self.frameMax = int(framesMaxNode.value)
      if self.frameMax <= 0:
        self.raiseAnError(IOError, f'ConstraintActivityTimelinePlot "{self.name}" received non-positive <framesMax>.')

    self.explicitGenerations = plotGenerationUtils.parseGenerationSelectorNode(spec)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for ConstraintActivityTimelinePlot "{self.name}".')
    self.source = src
    available = self.source.getVars()
    if self.useAllConstraints:
      detected = sorted(var for var in available if var.startswith('ConstraintEvaluation_'))
      if not detected:
        self.raiseAWarning(f'ConstraintActivityTimelinePlot "{self.name}" could not locate any "ConstraintEvaluation_*" columns; skipping animation.')
      self.constraints = detected
    else:
      needed = set(self.constraints + [self.index])
      missing = [var for var in needed if var not in available]
      if missing:
        self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by ConstraintActivityTimelinePlot "{self.name}".')
    if not self.constraints:
      self.raiseAWarning(f'ConstraintActivityTimelinePlot "{self.name}" has no constraints to plot; skipping animation.')

  def run(self):
    df = self.source.asDataset().to_dataframe().copy()
    if df.empty:
      self.raiseAWarning(f'ConstraintActivityTimelinePlot "{self.name}" received an empty dataset; nothing to animate.')
      return
    df[self.index] = df[self.index].astype(float)
    for cons in self.constraints:
      df[cons] = df[cons].astype(float)
    if not self.constraints:
      self.raiseAWarning(f'ConstraintActivityTimelinePlot "{self.name}" has no constraints to visualize.')
      return
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'ConstraintActivityTimelinePlot "{self.name}" found no generations in column "{self.index}".')
      return

    fractions = self._computeViolationFractions(df, generations)
    try:
      gensToRender, indices = plotGenerationUtils.resolveGenerations(
          generations, self.explicitGenerations, defaultCap=20, maxFrames=self.maxFrames)
    except ValueError as err:
      self.raiseAnError(IOError, f'ConstraintActivityTimelinePlot "{self.name}": {err}')
    if not gensToRender:
      self.raiseAWarning(f'ConstraintActivityTimelinePlot "{self.name}" did not select any generations to render.')
      return

    if 'gif' in self.formats:
      self._writeGif(generations, fractions, indices)
    if 'html' in self.formats:
      self._writeHtml(generations, fractions, indices)
    if self.saveFrames:
      self._writeFrames(generations, fractions, indices)

  def _computeViolationFractions(self, df, generations):
    results = []
    for gen in generations:
      subset = df[df[self.index] == gen]
      total = float(len(subset))
      row = []
      for cons in self.constraints:
        if total <= 0.0:
          row.append(0.0)
        else:
          values = subset[cons].to_numpy(dtype=float)
          row.append(float(np.count_nonzero(values <= 0.0)) / total)
      results.append(row)
    return np.asarray(results, dtype=float)

  def _writeGif(self, generations, fractions, indices):
    filename = self._createFilename(defaultName=f'{self.name}.gif')
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=0) as writer:
      for idx in indices:
        fig = self._renderFrame(generations, fractions, idx)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _writeHtml(self, generations, fractions, indices):
    filename = self._createFilename(defaultName=f'{self.name}.html')
    fig, axes = self._createFigure()

    def init():
      self._drawTimelines(axes, generations, fractions, indices[0])
      fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.92))
      return fig.axes

    def update(idx):
      self._drawTimelines(axes, generations, fractions, idx)
      fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.92))
      return fig.axes

    anim = animation.FuncAnimation(fig, update, frames=indices,
                                   init_func=init, interval=1000.0 / self.fps,
                                   blit=False)
    htmlStr = anim.to_jshtml()
    centeredHtml = f'<div style="display:flex;justify-content:center;">{htmlStr}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centeredHtml)
    plt.close(fig)

  def _writeFrames(self, generations, fractions, indices):
    frameIdx = plotGenerationUtils.frameIndicesToSave(
        len(indices), bool(self.explicitGenerations), self.saveFrames, self.frameMax)
    if not frameIdx:
      return
    base = self._createFilename(defaultName=f'{self.name}_frames')
    template = os.path.splitext(base)[0] + '_{index:04d}.png'
    directory = os.path.dirname(template)
    if directory:
      os.makedirs(directory, exist_ok=True)
    for position in frameIdx:
      idx = indices[position]
      fig = self._renderFrame(generations, fractions, idx)
      fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.92))
      fig.savefig(template.format(index=idx), dpi=150)
      plt.close(fig)

  def _renderFrame(self, generations, fractions, idx):
    fig, axes = self._createFigure()
    self._drawTimelines(axes, generations, fractions, idx)
    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.92))
    return fig

  def _drawTimelines(self, axes, generations, fractions, idx):
    uptoIdx = idx + 1
    xPositions = np.arange(uptoIdx)
    formattedGenerations = [self._formatGeneration(gen) for gen in generations[:uptoIdx]]
    colors = plt.get_cmap('tab10').colors
    rows, cols = self._subplotLayout
    for cIdx, axis in enumerate(axes):
      axis.clear()
      heights = fractions[:uptoIdx, cIdx]
      axis.bar(xPositions, heights, width=0.8,
               color=colors[cIdx % len(colors)], alpha=0.85, linewidth=0)
      axis.set_ylim(0.0, 1.05)
      axis.set_xticks(xPositions)
      axis.set_xticklabels(formattedGenerations, rotation=40, ha='right')
      axis.grid(axis='y', alpha=0.3, linestyle='--')
      cleanName = self.constraints[cIdx].replace('ConstraintEvaluation_', '')
      axis.set_title(cleanName or self.constraints[cIdx])
      # Only annotate axes that form the outer frame to avoid clutter
      if cIdx % cols == 0:
        axis.set_ylabel('Violation fraction')
      else:
        axis.set_ylabel('')
      if cIdx // cols == rows - 1:
        axis.set_xlabel(self.index)
      else:
        axis.set_xlabel('')
    currentGen = self._formatGeneration(generations[idx])
    if axes:
      axes[0].figure.suptitle(f'Constraint activity (Generation {currentGen})')

  def _createFigure(self):
    count = max(1, len(self.constraints))
    cols = min(3, count)
    rows = int(math.ceil(float(count) / float(cols)))
    # Aim for compact subplots while keeping room for labels and titles
    width = max(6.0, 3.1 * cols)
    height = max(3.2, 2.6 * rows)
    fig, axesGrid = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, sharey=True)
    axesFlat = axesGrid.flatten()
    axes = list(axesFlat[:count])
    # Remove any unused trailing axes generated by the grid helper
    for extra in axesFlat[count:]:
      fig.delaxes(extra)
    self._subplotLayout = (rows, cols)
    return fig, axes

  @staticmethod
  def _formatGeneration(genID):
    if float(genID).is_integer():
      return int(genID)
    return genID

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
