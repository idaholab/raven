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
Animate how crowding-distance based diversity shifts over generations using a radar chart.

At each frame the current generation's spacing statistics (crowding-distance quantiles by default)
are plotted on a polar axis and normalised against the worst‐case value observed in the run. This
creates a quick visual cue for premature convergence: shrinking polygons point to collapsing
diversity, while broad, stable shapes indicate healthy spacing.

What-if scenarios

Polygon rapidly shrinks toward the centre -> later generations are losing spacing; increase mutation,
  inject new samples, or revisit constraint handling to avoid premature convergence.
Polygon grows erratically from one frame to the next -> the search alternates between exploration and
  exploitation; review survivor selection or restart logic to smooth the transition.
Polygon remains small from the start -> the sampler never established diversity; consider larger
  initial populations, alternative seeding, or relaxed feasibility filters.
Polygon remains wide but uneven on certain axes -> only portions of the population maintain spacing;
  inspect objective scaling or crowding-distance implementation for bias.
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


class DiversityRadarPlot(PlotInterface):
  """
  Animated radar visual of crowding-distance quantiles across generations.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringType,
        descr=r"""Column to analyse for diversity. Defaults to CD (crowding distance)."""))
    spec.addSub(InputData.parameterInputFactory('quantiles', contentType=InputTypes.FloatListType,
        descr=r"""Quantiles (0-1) to display on the radar chart. Defaults to 0.1,0.25,0.5,0.75,0.9."""))
    spec.addSub(InputData.parameterInputFactory('maxFrames', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on the number of generations rendered. Defaults to min(total generations, 10)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combinations."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for generated animations. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('saveFrames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves sampled generations as standalone PNG frames alongside the animation outputs."""))
    spec.addSub(InputData.parameterInputFactory('framesMax', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <saveFrames> is true. Defaults to 10; generations are sampled evenly."""))
    plotGenerationUtils.addGenerationSelectorSpec(spec)
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'DiversityRadarPlot'
    self.source = None
    self.sourceName = None
    self.index = None
    self.metric = 'CD'
    self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    self.maxFrames = None
    self.explicitGenerations = None
    self.formats = {'gif', 'html'}
    self.fps = 2.0
    self.saveFrames = False
    self.frameMax = 10

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for DiversityRadarPlot "{self.name}".')
    self.sourceName = sourceNode.value

    indexNode = spec.findFirst('index')
    if indexNode is None or indexNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for DiversityRadarPlot "{self.name}".')
    self.index = indexNode.value

    metricNode = spec.findFirst('metric')
    if metricNode is not None and metricNode.value:
      self.metric = metricNode.value.strip()

    quantNode = spec.findFirst('quantiles')
    if quantNode is not None and quantNode.value:
      values = [float(q) for q in quantNode.value if q is not None]
      validated = []
      for q in values:
        if q <= 0.0 or q >= 1.0:
          self.raiseAnError(IOError, f'Quantile "{q}" out of range (0,1) in DiversityRadarPlot "{self.name}".')
        validated.append(q)
      if validated:
        self.quantiles = validated

    maxNode = spec.findFirst('maxFrames')
    if maxNode is not None and maxNode.value is not None:
      self.maxFrames = int(maxNode.value)
      if self.maxFrames <= 0:
        self.raiseAnError(IOError, f'DiversityRadarPlot "{self.name}" received non-positive <maxFrames>.')

    formatNode = spec.findFirst('format')
    if formatNode is not None and formatNode.value is not None:
      raw = formatNode.value.strip().lower()
      if raw == 'both' or not raw:
        self.formats = {'gif', 'html'}
      else:
        formats = set()
        for frag in raw.replace(';', ',').split(','):
          token = frag.strip()
          if not token:
            continue
          if token in ('gif', 'html'):
            formats.add(token)
          elif token == 'both':
            formats.update({'gif', 'html'})
          else:
            self.raiseAnError(IOError, f'Unsupported <format> "{token}" for DiversityRadarPlot "{self.name}".')
        if not formats:
          formats = {'gif'}
        self.formats = formats

    fpsNode = spec.findFirst('fps')
    if fpsNode is not None and fpsNode.value is not None:
      self.fps = float(fpsNode.value)
      if self.fps <= 0:
        self.raiseAnError(IOError, f'DiversityRadarPlot "{self.name}" received non-positive <fps>.')

    saveNode = spec.findFirst('saveFrames')
    if saveNode is not None and saveNode.value is not None:
      self.saveFrames = bool(saveNode.value)

    framesNode = spec.findFirst('framesMax')
    if framesNode is not None and framesNode.value is not None:
      self.frameMax = int(framesNode.value)
      if self.frameMax <= 0:
        self.raiseAnError(IOError, f'DiversityRadarPlot "{self.name}" received non-positive <framesMax>.')

    self.explicitGenerations = plotGenerationUtils.parseGenerationSelectorNode(spec)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for DiversityRadarPlot "{self.name}".')
    self.source = src
    available = self.source.getVars()
    needed = set([self.index, self.metric])
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by DiversityRadarPlot "{self.name}".')

  def run(self):
    data = self.source.asDataset().to_dataframe().copy()
    if data.empty:
      self.raiseAWarning(f'DiversityRadarPlot "{self.name}" received an empty dataset; nothing to animate.')
      return
    data[self.index] = data[self.index].astype(float)
    data[self.metric] = data[self.metric].astype(float)
    generations = sorted(data[self.index].unique())
    if not generations:
      self.raiseAWarning(f'DiversityRadarPlot "{self.name}" found no generations in column "{self.index}".')
      return

    quantMatrix = self._computeQuantiles(data, generations)
    if quantMatrix.size == 0:
      self.raiseAWarning(f'DiversityRadarPlot "{self.name}" could not compute quantiles for metric "{self.metric}".')
      return
    try:
      selectedGenerations, indices = plotGenerationUtils.resolveGenerations(
          generations, self.explicitGenerations, defaultCap=10, maxFrames=self.maxFrames)
    except ValueError as err:
      self.raiseAnError(IOError, f'DiversityRadarPlot "{self.name}": {err}')

    angles = self._computeAngles(len(self.quantiles))
    labels = [f'Q{int(q*100)}' for q in self.quantiles]
    normMatrix = self._normalize(quantMatrix)

    if 'gif' in self.formats:
      self._writeGif(selectedGenerations, indices, angles, labels, normMatrix)
    if 'html' in self.formats:
      self._writeHtml(selectedGenerations, indices, angles, labels, normMatrix)
    if self.saveFrames:
      self._writeFrames(selectedGenerations, indices, angles, labels, normMatrix)

  def _computeQuantiles(self, df, generations):
    quantiles = []
    for gen in generations:
      subset = df[df[self.index] == gen]
      values = subset[self.metric].to_numpy(dtype=float)
      values = values[np.isfinite(values)]
      if values.size == 0:
        quantiles.append([0.0 for _ in self.quantiles])
      else:
        quantiles.append(np.quantile(values, self.quantiles))
    return np.asarray(quantiles, dtype=float)

  @staticmethod
  def _computeAngles(count):
    if count <= 0:
      return np.asarray([], dtype=float)
    base = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return base

  @staticmethod
  def _normalize(matrix):
    maxVal = np.nanmax(matrix)
    if not np.isfinite(maxVal) or np.isclose(maxVal, 0.0):
      maxVal = 1.0
    return np.clip(matrix / maxVal, 0.0, 1.0)

  def _writeGif(self, gens, indices, angles, labels, normMatrix):
    filename = self._createFilename(defaultName=f'{self.name}.gif')
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=0) as writer:
      for i, idx in enumerate(indices):
        fig = self._renderFrame(gens[i], angles, labels, normMatrix[idx])
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _writeHtml(self, gens, indices, angles, labels, normMatrix):
    filename = self._createFilename(defaultName=f'{self.name}.html')
    fig, ax, line, fill = self._setupRadarAxes(angles, labels)
    fillContainer = {'patch': fill}

    def init():
      line.set_data([], [])
      patch = fillContainer['patch']
      patch.remove()
      fillContainer['patch'] = ax.fill([], [], color='#1f77b4', alpha=0.35)[0]
      ax.set_title('')
      return line, fillContainer['patch']

    def update(i):
      idx = indices[i]
      generation = gens[i]
      values = normMatrix[idx]
      anglesClosed, valuesClosed = self._closePolygon(angles, values)
      line.set_data(anglesClosed, valuesClosed)
      patch = fillContainer['patch']
      patch.remove()
      fillContainer['patch'] = ax.fill(anglesClosed, valuesClosed, color='#1f77b4', alpha=0.35)[0]
      ax.set_title(f'Generation {self._format_generation(generation)}')
      return line, fillContainer['patch']

    anim = animation.FuncAnimation(fig, update, frames=len(indices),
                                   init_func=init, interval=1000.0 / self.fps,
                                   blit=False)
    htmlStr = anim.to_jshtml()
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(f'<div style="display:flex;justify-content:center;">{html_str}</div>')
    plt.close(fig)

  def _writeFrames(self, gens, indices, angles, labels, normMatrix):
    framePositions = plotGenerationUtils.frameIndicesToSave(
        len(indices), bool(self.explicitGenerations), self.saveFrames, self.frameMax)
    if not framePositions:
      return
    base = self._createFilename(defaultName=f'{self.name}_frames')
    template = os.path.splitext(base)[0] + '_{index:04d}.png'
    directory = os.path.dirname(template)
    if directory:
      os.makedirs(directory, exist_ok=True)
    for pos in framePositions:
      idx = indices[pos]
      fig = self._renderFrame(gens[pos], angles, labels, normMatrix[idx])
      fig.savefig(template.format(index=int(gens[pos])), dpi=150)
      plt.close(fig)

  def _renderFrame(self, generation, angles, labels, values):
    fig, ax, line, fill = self._setupRadarAxes(angles, labels)
    anglesClosed, valuesClosed = self._closePolygon(angles, values)
    line.set_data(anglesClosed, valuesClosed)
    fill.remove()
    fill = ax.fill(anglesClosed, valuesClosed, color='#1f77b4', alpha=0.35)[0]
    ax.set_title(f'Generation {self._format_generation(generation)}')
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    return fig

  def _setupRadarAxes(self, angles, labels):
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True, linewidth=0.6, alpha=0.3)
    ax.spines['polar'].set_visible(False)
    anglesClosed, valuesClosed = self._closePolygon(angles, np.zeros(len(angles)))
    line, = ax.plot(anglesClosed, valuesClosed, color='#1f77b4', linewidth=2.0)
    fill = ax.fill(anglesClosed, valuesClosed, color='#1f77b4', alpha=0.35)[0]
    return fig, ax, line, fill

  @staticmethod
  def _closePolygon(angles, values):
    if values.size == 0:
      return np.asarray([]), np.asarray([])
    anglesClosed = np.concatenate([angles, [angles[0]]])
    valuesClosed = np.concatenate([values, [values[0]]])
    return anglesClosed, valuesClosed

  @staticmethod
  def _formatGeneration(genID):
    if float(genID).is_integer():
      return int(genID)
    return genID

  @staticmethod
  def _sampleGenerations(generations, limit):
    if limit >= len(generations):
      indices = list(range(len(generations)))
      return generations, indices
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
    if selected[-1] != len(generations) - 1:
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
