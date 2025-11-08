"""
Visualise pairwise relationships between variables via a circular chord diagram.

The plot maps each variable to an arc on the unit circle, then draws chords
whose width/opacity reflect a chosen association metric (correlation by
default). Handy for what-if exploration like: "If I focus on the latest
generation, which decision variables remain tightly coupled, and which links
disappear when I relax a constraint?"
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib import patches, path
import numpy as np
import pandas as pd

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes


class ChordDiagramPlot(PlotInterface):
  """
  Draw a chord diagram to highlight variable associations. Adjust filters or
  thresholds to perform what-if studies—e.g., raising `<threshold>` to show
  only the strongest correlations after a design change surfaces different
  couplings in the population.
  """
  @classmethod
  def getInputSpecification(cls):
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""Name of the DataObject providing the samples."""))
    variables = InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""Variables to include in the chord diagram (at least three).""")
    spec.addSub(variables)
    spec.addSub(InputData.parameterInputFactory('metric', contentType=InputTypes.StringType,
        descr=r"""Association metric: "pearson" (default) or "spearman"."""))
    spec.addSub(InputData.parameterInputFactory('threshold', contentType=InputTypes.FloatType,
        descr=r"""Minimum absolute association strength to draw a chord (default 0.3)."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Optional generation identifier column (e.g., batchId)."""))
    spec.addSub(InputData.parameterInputFactory('generation', contentType=InputTypes.FloatType,
        descr=r"""When <index> present, restrict to a specific generation."""))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'ChordDiagramPlot'
    self.source = None
    self.sourceName = None
    self.variables = []
    self.metric = 'pearson'
    self.threshold = 0.3
    self.index = None
    self.generation = None

  def handleInput(self, spec):
    super().handleInput(spec)
    src = spec.findFirst('source')
    if src is None or src.value is None:
      self.raiseAnError(IOError, f'Missing <source> node for ChordDiagramPlot "{self.name}".')
    self.sourceName = src.value

    varNode = spec.findFirst('variables')
    if varNode is None or len(varNode.value) < 3:
      self.raiseAnError(IOError, f'ChordDiagramPlot "{self.name}" requires at least three <variables>.')
    self.variables = [entry for entry in varNode.value if entry]

    metricNode = spec.findFirst('metric')
    if metricNode is not None and metricNode.value:
      value = metricNode.value.strip().lower()
      if value not in ('pearson', 'spearman'):
        self.raiseAnError(IOError, f'Unsupported <metric> "{metricNode.value}" for ChordDiagramPlot "{self.name}".')
      self.metric = value

    thresholdNode = spec.findFirst('threshold')
    if thresholdNode is not None and thresholdNode.value is not None:
      self.threshold = float(thresholdNode.value)

    idxNode = spec.findFirst('index')
    if idxNode is not None and idxNode.value:
      self.index = idxNode.value

    genNode = spec.findFirst('generation')
    if genNode is not None and genNode.value is not None:
      self.generation = float(genNode.value)

  def initialize(self, stepEntities):
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" found for ChordDiagramPlot "{self.name}".')
    available = src.getVars()
    needed = list(self.variables)
    if self.index:
      needed.append(self.index)
    missing = [var for var in needed if var not in available]
    if missing:
      self.raiseAnError(IOError, f'Source DataObject "{src.name}" missing variable(s) {missing} required by ChordDiagramPlot "{self.name}".')
    self.source = src

  @staticmethod
  def _bezier_arc(start_angle, end_angle, radius=1.0):
    """
    Construct a cubic Bezier path approximating an arc from start_angle to end_angle.
    """
    if end_angle < start_angle:
      end_angle += 2.0 * np.pi
    angle = end_angle - start_angle
    n_segments = int(np.ceil(abs(angle) / (np.pi / 2.0)))
    angle_step = angle / n_segments
    path_segments = []
    current_angle = start_angle
    for _ in range(n_segments):
      next_angle = current_angle + angle_step
      alpha = np.tan(angle_step / 4.0)
      p0 = np.array([radius * np.cos(current_angle), radius * np.sin(current_angle)])
      p3 = np.array([radius * np.cos(next_angle), radius * np.sin(next_angle)])
      p1 = p0 + alpha * np.array([-radius * np.sin(current_angle), radius * np.cos(current_angle)])
      p2 = p3 - alpha * np.array([-radius * np.sin(next_angle), radius * np.cos(next_angle)])
      path_segments.append((p0, p1, p2, p3))
      current_angle = next_angle
    vertices = [path_segments[0][0]]
    codes = [path.Path.MOVETO]
    for seg in path_segments:
      vertices.extend(seg[1:])
      codes.extend([path.Path.CURVE4, path.Path.CURVE4, path.Path.CURVE4])
    return path.Path(vertices, codes)

  def run(self):
    df = self.source.asDataset().to_dataframe()
    if df.empty:
      self.raiseAWarning(f'ChordDiagramPlot "{self.name}" skipped because source "{self.source.name}" is empty.')
      return
    subset = df.copy()
    if self.index:
      subset[self.index] = subset[self.index].astype(float)
      if self.generation is not None:
        mask = np.isclose(subset[self.index].to_numpy(dtype=float), self.generation)
        subset = subset[mask]
      else:
        max_gen = subset[self.index].max()
        subset = subset[subset[self.index] == max_gen]
    if subset.empty:
      self.raiseAWarning(f'ChordDiagramPlot "{self.name}" had no samples after filtering.')
      return

    numeric = subset[self.variables].astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
      self.raiseAWarning(f'ChordDiagramPlot "{self.name}" found no finite samples for variables {self.variables}.')
      return

    if self.metric == 'spearman':
      corr = numeric.rank().corr(method='pearson')
    else:
      corr = numeric.corr(method='pearson')
    corr = corr.fillna(0.0)

    n_vars = len(self.variables)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_vars + 1)
    var_to_angle = {var: (angles[i], angles[i + 1]) for i, var in enumerate(self.variables)}

    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw={'projection': 'polar'})
    ax.set_axis_off()

    for idx, var in enumerate(self.variables):
      start_angle, end_angle = var_to_angle[var]
      mid_angle = 0.5 * (start_angle + end_angle)
      ax.bar(x=[mid_angle], height=[1.0], width=end_angle - start_angle,
             bottom=0.95, color='#e0e0e0', edgecolor='gray', linewidth=1.0, alpha=0.8)
      ax.text(mid_angle, 1.1, var, ha='center', va='center', rotation=np.degrees(mid_angle),
              rotation_mode='anchor', fontsize=9)

    for i in range(n_vars):
      for j in range(i + 1, n_vars):
        strength = corr.iloc[i, j]
        if abs(strength) < self.threshold:
          continue
        start_a = 0.5 * sum(var_to_angle[self.variables[i]])
        end_a = 0.5 * sum(var_to_angle[self.variables[j]])
        chords_path = self._bezier_arc(start_a, end_a, radius=0.9)
        color = plt.cm.coolwarm((strength + 1.0) / 2.0)
        width = 2.0 + 4.0 * abs(strength)
        patch = patches.PathPatch(chords_path, facecolor='none', edgecolor=color,
                                  linewidth=width, alpha=0.7)
        ax.add_patch(patch)

    ax.set_ylim(0, 1.2)
    fig.suptitle('Variable association chord diagram')
    fig.tight_layout()
    filename = self._createFilename(defaultName=f'{self.name}.png')
    fig.savefig(filename, dpi=150)
    plt.close(fig)
