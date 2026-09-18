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
Visualise generational convergence in single-objective runs via best/mean fitness bands.

Each generation is represented by three elements:
  * A line joining the best (min or max) fitness, highlighting improvement plateaus.
  * A line for the mean fitness.
  * A shaded band spanning mean ± one standard deviation to emphasise narrowing variability.
  * Individual samples are plotted as semi-transparent scatters to expose outliers.

What-if scenarios

Best line flat while mean drops slowly -> convergence has stalled; consider increasing mutation or
  restarts to escape local minima.
Mean remains far from best with wide variance band -> population is diverse but not converging; tighten
  selection pressure or reduce exploration.
Variance collapses early yet best keeps improving -> exploitation dominates successfully; optionally
  reduce early elitism to avoid missing alternate basins.
Best oscillates while variance spikes -> potential instability (e.g., repair operators or penalty swings);
  inspect constraint handling or fitness scaling.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class FitnessFunnelPlot(PlotInterface):
  """
  Static line/scatter plot summarising best and mean fitness per generation.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the optimizer SolutionExport DataObject."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringListType,
        descr=r"""One or more columns containing the objective/fitness values to evaluate.
              When more than one is given, the plot renders one funnel subplot per column
              (e.g., two objectives produce two stacked subplots); a single column reproduces
              the classic single-objective funnel."""))
    spec.addSub(InputData.parameterInputFactory('goal', contentType=InputTypes.StringListType,
        descr=r"""Optimisation goal per <metric>: "min" (default) or "max". Provide either one value
              (applied to every metric) or one value per metric, in the same order as <metric>."""))
    spec.addSub(InputData.parameterInputFactory('maxPerFigure', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of per-metric funnel panels drawn in a single figure (default 4). When the
              number of metrics exceeds this cap, the panels are split evenly across multiple figures
              (e.g., 6 metrics -> two figures of 3, 9 -> three of 3, 12 -> three of 4) so panels stay
              legible for high-objective problems. Within a figure, up to three panels stack in a single
              column (sharing the generation axis) and four use a 2x2 grid. A single figure is written as
              "<name>.png"; multiple figures are written as "<name>_p01.png", "<name>_p02.png", ..."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'FitnessFunnelPlot'
    self.source = None
    self.sourceName = None
    self.index = None
    self.metrics = []
    self.goals = []
    self.maxPerFigure = 4

  def handleInput(self, spec):
    super().handleInput(spec)
    sourceNode = spec.findFirst('source')
    if sourceNode is None or sourceNode.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for FitnessFunnelPlot "{self.name}".')
    self.sourceName = sourceNode.value

    indexNode = spec.findFirst('index')
    if indexNode is None or indexNode.value is None:
      self.raiseAnError(IOError, f'Missing <index> node for FitnessFunnelPlot "{self.name}".')
    self.index = indexNode.value

    metricNode = spec.findFirst('metric')
    if metricNode is None or metricNode.value is None:
      self.raiseAnError(IOError, f'Missing <metric> node for FitnessFunnelPlot "{self.name}".')
    self.metrics = [m.strip() for m in metricNode.value if m is not None and m.strip()]
    if not self.metrics:
      self.raiseAnError(IOError, f'<metric> node is empty for FitnessFunnelPlot "{self.name}".')

    goalNode = spec.findFirst('goal')
    if goalNode is not None and goalNode.value:
      goals = [g.strip().lower() for g in goalNode.value if g is not None and g.strip()]
      for goal in goals:
        if goal not in ('min', 'max'):
          self.raiseAnError(IOError, f'Unsupported <goal> "{goal}" for FitnessFunnelPlot "{self.name}". Use "min" or "max".')
      if len(goals) == 1:
        self.goals = [goals[0]] * len(self.metrics)
      elif len(goals) == len(self.metrics):
        self.goals = goals
      else:
        self.raiseAnError(IOError, f'FitnessFunnelPlot "{self.name}" received {len(goals)} <goal> values for '
                          f'{len(self.metrics)} <metric> values; provide one goal, or one per metric.')
    else:
      self.goals = ['min'] * len(self.metrics)

    maxNode = spec.findFirst('maxPerFigure')
    if maxNode is not None and maxNode.value is not None:
      self.maxPerFigure = int(maxNode.value)
      if self.maxPerFigure <= 0:
        self.raiseAnError(IOError, f'FitnessFunnelPlot "{self.name}" received non-positive <maxPerFigure>.')

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for FitnessFunnelPlot "{self.name}".')
    self.source = src
    variables = self.source.getVars()
    missing = [var for var in ([self.index] + self.metrics) if var not in variables]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{self.source.name}" is missing variables {missing} required by FitnessFunnelPlot "{self.name}".')

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'FitnessFunnelPlot "{self.name}" received an empty dataset; skipping.')
      return

    df[self.index] = df[self.index].astype(float)
    for metric in self.metrics:
      df[metric] = df[metric].astype(float)
    generations = sorted(df[self.index].unique())
    if not generations:
      self.raiseAWarning(f'FitnessFunnelPlot "{self.name}" found no generations in column "{self.index}".')
      return

    baseName = self._createFilename(defaultName=f'{self.name}.png')
    directory = os.path.dirname(baseName)
    if directory:
      os.makedirs(directory, exist_ok=True)

    pages = self._paginate(list(zip(self.metrics, self.goals)))
    multiPage = len(pages) > 1
    stem, ext = os.path.splitext(baseName)
    anyPlotted = False
    for pageNum, page in enumerate(pages, start=1):
      filename = f'{stem}_p{pageNum:02d}{ext}' if multiPage else baseName
      plotted = self._render_figure(df, generations, page, filename)
      anyPlotted = anyPlotted or plotted

    if not anyPlotted:
      self.raiseAWarning(f'FitnessFunnelPlot "{self.name}" could not compute statistics for any metric; nothing written.')

  def _render_figure(self, df, generations, page, filename):
    """
      Draw and save a single figure holding the funnels for one page of metrics.
      @ In, df, pandas.DataFrame, the solution export data
      @ In, generations, list, sorted unique generation identifiers
      @ In, page, list, [(metric, goal), ...] to draw in this figure
      @ In, filename, str, output PNG path
      @ Out, plotted, bool, True if at least one panel was drawn
    """
    nPanels = len(page)
    nrows, ncols = self._grid_shape(nPanels)
    if ncols == 1:
      figsize = (7.0, 4.2 * nrows)
    else:
      figsize = (5.2 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                             figsize=figsize, squeeze=False)
    flat = axes.ravel()
    plotted = False
    for idx, (metric, goal) in enumerate(page):
      ax = flat[idx]
      stats = self._compute_statistics(df, generations, metric, goal)
      if stats is None:
        self.raiseAWarning(f'FitnessFunnelPlot "{self.name}" could not compute statistics for metric "{metric}".')
        ax.set_axis_off()
        continue
      plotted = True
      best, mean, std = stats
      ax.scatter(df[self.index], df[metric], s=18, alpha=0.35, c='#1f77b4', linewidths=0)
      ax.plot(generations, mean, color='#ff7f0e', linewidth=2.0, label='Mean')
      ax.fill_between(generations, mean - std, mean + std, color='#ffbb78', alpha=0.35, label='±1σ')
      ax.plot(generations, best, color='#2ca02c', linewidth=2.4, label=f'Best ({goal})')
      ax.set_ylabel(metric)
      ax.set_title(f'{metric} ({goal})')
      ax.grid(alpha=0.3, linestyle='--')
      ax.legend(frameon=False)

    # blank out unused grid cells (ragged last row)
    for j in range(nPanels, len(flat)):
      flat[j].set_axis_off()

    if not plotted:
      plt.close(fig)
      return False

    # label the generation axis on the bottom-most populated panel of each column, and
    # re-enable its tick labels (sharex hides them for any panel that is not on the last grid row)
    for col in range(ncols):
      bottom = None
      for row in range(nrows):
        cell = row * ncols + col
        if cell < nPanels:
          bottom = cell
      if bottom is not None:
        flat[bottom].set_xlabel(self.index)
        flat[bottom].tick_params(labelbottom=True)

    fig.suptitle('Fitness Funnel')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return True

  def _paginate(self, items):
    """
      Split the (metric, goal) pairs into evenly-sized pages of at most maxPerFigure panels each,
      so no page is left with a single orphan panel (e.g., 5 with cap 3 -> [3, 2], not [3, 1, 1]).
      @ In, items, list, ordered [(metric, goal), ...]
      @ Out, pages, list, list of pages, each a sublist of @ items
    """
    n = len(items)
    if n <= self.maxPerFigure:
      return [items]
    nPages = int(math.ceil(n / float(self.maxPerFigure)))
    base, extra = divmod(n, nPages)
    pages = []
    cursor = 0
    for p in range(nPages):
      size = base + (1 if p < extra else 0)
      pages.append(items[cursor:cursor + size])
      cursor += size
    return pages

  def _grid_shape(self, n):
    """
      Choose a subplot grid (rows, cols) for @ n per-metric funnels within one figure.
      Up to three panels stack in a single column (preserving funnel width and a shared generation
      axis); more use a near-square grid.
      @ In, n, int, number of panels to lay out
      @ Out, (nrows, ncols), tuple(int, int), grid dimensions covering all n panels
    """
    if n <= 3:
      ncols = 1
    else:
      ncols = int(math.ceil(math.sqrt(n)))
    ncols = max(1, ncols)
    nrows = int(math.ceil(n / float(ncols)))
    return nrows, ncols

  def _compute_statistics(self, df, generations, metric, goal):
    best = []
    mean = []
    std = []
    for gen in generations:
      values = df[df[self.index] == gen][metric].to_numpy(dtype=float)
      values = values[np.isfinite(values)]
      if values.size == 0:
        best.append(np.nan)
        mean.append(np.nan)
        std.append(np.nan)
        continue
      mean.append(float(np.mean(values)))
      std.append(float(np.std(values)))
      if goal == 'max':
        best.append(float(np.max(values)))
      else:
        best.append(float(np.min(values)))
    best = np.asarray(best, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if not np.isfinite(best).any():
      return None
    return best, mean, std
