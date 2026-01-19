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
import matplotlib
matplotlib.use('Agg', force=True)
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


def _metric_series_helper(df, metric_name, metric_kind):
  series = df[metric_name].astype(float)
  if metric_kind == 'fitness':
    return -series, series
  return series, series


def _metric_to_original_helper(metric_value, metric_kind):
  if metric_kind == 'fitness':
    return -metric_value
  return metric_value


def _is_feasible_helper(df, constraint_vars):
  if not constraint_vars or df.empty:
    return np.ones(len(df), dtype=bool)
  feasible = np.ones(len(df), dtype=bool)
  for var in constraint_vars:
    if var not in df.columns:
      continue
    vals = df[var].astype(float).to_numpy()
    feasible &= vals > 0.0
  return feasible


def _top_count_helper(total_points, top_fraction, top_override):
  if total_points <= 0:
    return 0
  if top_override is not None:
    return min(total_points, max(1, top_override))
  fraction = top_fraction if top_fraction is not None else 0.2
  fraction = min(max(fraction, 0.0), 1.0)
  return min(total_points, max(1, int(math.ceil(total_points * fraction))))


def _compute_population_colors(subset, axes, metric_name, metric_kind, constraint_vars,
                               top_fraction, top_override, last=False,
                               infeasible_color=DEFAULT_INFEASIBLE_POINT_COLOR,
                               top_cap=None):
  total_points = len(subset)
  colors = np.empty((total_points, 4), dtype=float)
  base_rgba = mcolors.to_rgba(BASE_POINT_COLOR)
  colors[:] = base_rgba
  top_green = mcolors.to_rgba(TOP_POINT_COLOR)
  best_red = mcolors.to_rgba(BEST_POINT_COLOR)
  infeasible_rgba = mcolors.to_rgba(infeasible_color)
  metric_series, original_series = _metric_series_helper(subset, metric_name, metric_kind)
  if total_points == 0:
    summary = {'count': 0, 'pareto': 0, 'top': 0, 'best': None,
               'metric_kind': metric_kind, 'metric': metric_name,
               'top_fraction': 0.0, 'feasible': 0, 'feasible_fraction': 0.0,
               'has_constraints': bool(constraint_vars)}
    meta = {'best_pos': None, 'best_coords': None, 'best_value': None}
    return colors, summary, meta

  feasible_mask = _is_feasible_helper(subset, constraint_vars)
  feasible_count = int(feasible_mask.sum())
  feasible_fraction = feasible_count / total_points if total_points else 0.0

  top_count = _top_count_helper(total_points, top_fraction, top_override)
  if top_cap is not None:
    top_count = min(top_count, top_cap)
  top_mask_series = metric_series[feasible_mask] if feasible_count > 0 else metric_series
  top_indices = top_mask_series.nsmallest(top_count).index
  for idx in top_indices:
    colors[subset.index.get_loc(idx)] = top_green

  best_series = metric_series[feasible_mask] if feasible_count > 0 else metric_series
  best_mask = np.zeros(total_points, dtype=bool)
  best_original = None
  best_idx = None
  pareto_mask = None
  pareto_count = 0
  multi_objective = False
  if 'rank' in subset.columns:
    try:
      ranks = subset['rank'].astype(int)
      pareto_mask = (ranks == 1).to_numpy()
      pareto_count = int(pareto_mask.sum())
      multi_objective = ranks.max() > 1
    except (ValueError, TypeError):
      pareto_mask = None
  if multi_objective and pareto_mask is not None and pareto_count > 0:
    colors[pareto_mask] = best_red
    best_mask = pareto_mask
    best_positions = np.where(pareto_mask)[0]
    if best_positions.size > 0:
      best_pos_idx = int(best_positions[0])
      best_idx = subset.index[best_pos_idx]
      if best_idx in original_series.index:
        best_original = float(original_series.loc[best_idx])
  else:
    if not best_series.empty:
      best_idx = best_series.idxmin()
      loc = subset.index.get_loc(best_idx)
      best_mask[loc] = True
      best_original = float(original_series.loc[best_idx])
    colors[best_mask] = best_red

  final_mask = None
  if 'accepted' in subset.columns:
    accepted_vals = subset['accepted'].astype(str).str.lower()
    final_mask = accepted_vals == 'final'
    if final_mask.any():
      final_numpy = final_mask.to_numpy()
      colors[final_numpy] = best_red
      best_mask = np.logical_or(best_mask, final_numpy)
      if best_idx is None:
        final_indices = final_mask[final_mask].index
        if len(final_indices):
          best_idx = final_indices[0]
      if best_original is None and best_idx is not None and best_idx in original_series.index:
        best_original = float(original_series.loc[best_idx])

  if pareto_mask is not None and not (multi_objective and pareto_mask.any()):
    top_positions = subset.index.get_indexer(top_indices)
    for pos in top_positions:
      if pos >= 0 and not best_mask[pos]:
        colors[pos] = top_green

  if constraint_vars:
    for pos, feasible in enumerate(feasible_mask):
      if not feasible:
        colors[pos] = infeasible_rgba

  fraction = top_count / total_points if total_points else 0.0
  summary = {'count': total_points,
             'pareto': pareto_count,
             'top': top_count,
             'best': best_original,
             'metric_kind': metric_kind,
             'metric': metric_name,
             'top_fraction': fraction,
             'feasible': feasible_count,
             'feasible_fraction': feasible_fraction,
             'has_constraints': bool(constraint_vars)}
  if last:
    for pos in range(total_points):
      if not best_mask[pos]:
        colors[pos][3] = min(colors[pos][3], 0.45)
  best_coords = None
  best_value = None
  if best_mask.any():
    best_pos = int(np.where(best_mask)[0][0])
    colors[best_pos][3] = 1.0
    best_coords = subset.iloc[best_pos][axes].to_numpy(dtype=float)
    best_value = float(subset.iloc[best_pos][metric_name])
  else:
    best_pos = None
  meta = {'best_pos': best_pos, 'best_coords': best_coords, 'best_value': best_value}
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
        descr=r"""Highlight threshold. Values >1 indicate a count; values in (0,1] indicate a population fraction. Defaults to 0.2 (20%%)."""))
    spec.addSub(InputData.parameterInputFactory('format', contentType=InputTypes.StringType,
        descr=r"""Output format. Options: "gif", "html", "both", or comma-separated combinations."""))
    spec.addSub(InputData.parameterInputFactory('fps', contentType=InputTypes.FloatType,
        descr=r"""Frames per second for generated animations. Defaults to 2."""))
    spec.addSub(InputData.parameterInputFactory('save_frames', contentType=InputTypes.BoolType,
        descr=r"""If true, saves each generation as a standalone PNG frame alongside the animation outputs."""))
    spec.addSub(InputData.parameterInputFactory('view', contentType=InputTypes.StringType,
        descr=r"""Optional legacy flag; values "2d", "3d", or "both" are accepted but the plot currently renders only the 2-D contour."""))
    spec.addSub(InputData.parameterInputFactory('surface', contentType=InputTypes.BoolType,
        descr=r"""Legacy flag; retained for compatibility but currently ignored (only the 2-D contour is rendered)."""))
    spec.addSub(InputData.parameterInputFactory('frames_max', contentType=InputTypes.IntegerType,
        descr=r"""Maximum number of PNG frames to save when <save_frames> is true. Defaults to 10; frames are sampled evenly across generations."""))
    spec.addSub(InputData.parameterInputFactory('show_history', contentType=InputTypes.BoolType,
        descr=r"""If true, retain points from earlier generations as muted grey markers to visualize exploration history."""))
    spec.addSub(InputData.parameterInputFactory('history_alpha', contentType=InputTypes.FloatType,
        descr=r"""Alpha value in [0, 1] applied to history markers when <show_history> is true. Defaults to 0.15."""))
    spec.addSub(InputData.parameterInputFactory('history_color', contentType=InputTypes.StringType,
        descr=r"""Matplotlib-compatible color for history markers when <show_history> is true. Defaults to a neutral grey."""))
    spec.addSub(InputData.parameterInputFactory('infeasible_color', contentType=InputTypes.StringType,
        descr=r"""Optional override for the infeasible sample color. When <show_history> is true, the default becomes orange unless overridden here."""))
    spec.addSub(InputData.parameterInputFactory('display_fraction', contentType=InputTypes.FloatType,
        descr=r"""Optional fraction (0-1] of each generation's population to plot when the population exceeds <display_threshold>. Defaults to 1.0 (show all)."""))
    spec.addSub(InputData.parameterInputFactory('display_threshold', contentType=InputTypes.IntegerType,
        descr=r"""Population size above which <display_fraction> filtering activates. Defaults to 20."""))
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
    self.top_fraction = 0.2
    self.top_count_override = None
    self.fps = 2.0
    self.formats = {'gif', 'html'}
    self.save_frames = False
    self.frame_max = 10
    self.show_history = False
    self.history_alpha = 0.15
    self.history_color = DEFAULT_HISTORY_POINT_COLOR
    self.history_marker_size = HISTORY_MARKER_SIZE
    self._history_facecolor = mcolors.to_rgba(self.history_color, self.history_alpha)
    self._history_lookup = {}
    self._custom_infeasible_color = None
    self._infeasible_point_color = DEFAULT_INFEASIBLE_POINT_COLOR
    self.display_fraction = 1.0
    self.display_threshold = 20
    self.metricAll = False
    self.metricColumns = []
    self._color_metric = None
    self.metricKinds = {}
    self._color_metric_kind = None
    self._multi_objective = False
    self._update_visual_config()

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
    raw_metric = objNode.value.strip() if objNode.value is not None else ''
    if raw_metric.lower() == 'all':
      self.metricAll = True
      self.metricName = None
      self.metricNames = []
    else:
      if not raw_metric:
        self.raiseAnError(IOError, f'<objective> must specify a metric name or "all" for ObjectiveContourAnimationPlot "{self.name}".')
      self.metricAll = False
      entries = [item.strip() for item in raw_metric.replace(';', ',').split(',') if item.strip()]
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
      top_val = topNode.value
      if top_val <= 0.0:
        self.raiseAnError(IOError, f'<top> must be positive for ObjectiveContourAnimationPlot "{self.name}".')
      if top_val <= 1.0:
        self.top_fraction = top_val
        self.top_count_override = None
      else:
        self.top_count_override = int(math.ceil(top_val))
    fpsNode = spec.findFirst('fps')
    if fpsNode is not None:
      self.fps = max(fpsNode.value, 0.1)
    fmtNode = spec.findFirst('format')
    if fmtNode is not None:
      self.formats = self._parse_formats(fmtNode.value)
    else:
      self.formats = {'gif', 'html'}
    framesNode = spec.findFirst('save_frames')
    if framesNode is not None:
      self.save_frames = bool(framesNode.value)
    frameMaxNode = spec.findFirst('frames_max')
    if frameMaxNode is not None:
      raw = int(frameMaxNode.value)
      if raw <= 0:
        self.raiseAnError(IOError, f'<frames_max> must be positive for ObjectiveContourAnimationPlot "{self.name}".')
      self.frame_max = raw
    historyNode = spec.findFirst('show_history')
    if historyNode is not None:
      self.show_history = bool(historyNode.value)
    historyAlphaNode = spec.findFirst('history_alpha')
    if historyAlphaNode is not None:
      alpha_val = float(historyAlphaNode.value)
      if not (0.0 <= alpha_val <= 1.0):
        self.raiseAnError(IOError, f'<history_alpha> must be within [0, 1] for ObjectiveContourAnimationPlot "{self.name}".')
      self.history_alpha = alpha_val
    historyColorNode = spec.findFirst('history_color')
    if historyColorNode is not None:
      color_val = historyColorNode.value
      try:
        mcolors.to_rgba(color_val)
      except ValueError as err:
        self.raiseAnError(IOError, f'Invalid <history_color> value "{color_val}" for ObjectiveContourAnimationPlot "{self.name}": {err}')
      self.history_color = color_val
    infeasibleColorNode = spec.findFirst('infeasible_color')
    if infeasibleColorNode is not None:
      infeasible_val = infeasibleColorNode.value
      try:
        mcolors.to_rgba(infeasible_val)
      except ValueError as err:
        self.raiseAnError(IOError, f'Invalid <infeasible_color> value "{infeasible_val}" for ObjectiveContourAnimationPlot "{self.name}": {err}')
      self._custom_infeasible_color = infeasible_val
    displayFractionNode = spec.findFirst('display_fraction')
    if displayFractionNode is not None:
      frac_val = float(displayFractionNode.value)
      if not (0.0 < frac_val <= 1.0):
        self.raiseAnError(IOError, f'<display_fraction> must be in (0, 1] for ObjectiveContourAnimationPlot "{self.name}".')
      self.display_fraction = frac_val
    displayThresholdNode = spec.findFirst('display_threshold')
    if displayThresholdNode is not None:
      threshold_val = int(displayThresholdNode.value)
      if threshold_val < 1:
        self.raiseAnError(IOError, f'<display_threshold> must be positive for ObjectiveContourAnimationPlot "{self.name}".')
      self.display_threshold = threshold_val
    self._update_visual_config()

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
      self.metricColumns = self._discover_objective_columns(dataVars)
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
      fitness_col = f'FitnessEvaluation_{metric}'
      kind = self.metricKind
      if fitness_col in dataVars:
        kind = 'fitness'
      self.metricKinds[metric] = kind
    self._color_metric = self.metricColumns[0]
    self._color_metric_kind = self.metricKinds.get(self._color_metric, self.metricKind)
    self._multi_objective = len(self.metricColumns) > 1

  def run(self):
    df = self.source.asDataset().to_dataframe()
    all_generations = sorted(df[self.index].unique())
    if not all_generations:
      self.raiseAWarning(f'No generations found for ObjectiveContourAnimationPlot "{self.name}".')
      return
    cap = max(1, min(self.frame_max, len(all_generations)))
    generations = self._sample_generations(all_generations, cap)
    plot_context = self._build_plot_context(df)
    history_lookup = self._build_history_offsets(df, all_generations) if self.show_history else {}
    self._history_lookup = history_lookup
    frame_template = None
    if self.save_frames:
      base = self._createFilename(defaultName=f'{self.name}_frames')
      frame_template = os.path.splitext(base)[0] + '_{index:04d}.png'
      frame_dir = os.path.dirname(frame_template)
      if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)

    for fmt in self.formats:
      if fmt == 'html':
        self._write_html(df, generations, plot_context,
                         history_lookup, filename_default=f'{self.name}.html')
      elif fmt == 'gif':
        self._write_gif(df, generations, plot_context,
                        history_lookup, filename_default=f'{self.name}.gif')
    if self.save_frames:
      self._write_frames(df, generations, plot_context,
                         history_lookup, frame_template)

  def _write_gif(self, df, generations, plot_context, history_lookup, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    duration = 1.0 / self.fps
    with imageio.get_writer(filename, mode='I', duration=duration, loop=1) as writer:
      for gen in generations:
        subset = df[df[self.index] == gen]
        history_offsets = history_lookup.get(gen) if history_lookup else None
        fig = self._render_frame(subset, gen, plot_context,
                                 last=(gen == generations[-1]), history_offsets=history_offsets)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer))

  def _write_html(self, df, generations, plot_context, history_lookup, filename_default):
    filename = self._createFilename(defaultName=filename_default)
    first_gen = generations[0]
    history_offsets = history_lookup.get(first_gen) if history_lookup else None
    if plot_context['multi']:
      fig, axes = self._create_multi_axes(len(self.metricColumns))
      axes_list = np.atleast_1d(axes).tolist() if isinstance(axes, np.ndarray) else [axes]
      self._populate_multi_axes(axes_list, df[df[self.index] == first_gen], plot_context,
                                history_offsets=history_offsets,
                                last=(first_gen == generations[-1]))
    else:
      fig, ax = self._create_axes(plot_context['axis_limits'])
      self._populate_axes(ax, df[df[self.index] == first_gen], plot_context,
                          history_offsets=history_offsets,
                          last=(first_gen == generations[-1]))
      axes_list = [ax]
    suptitle = fig.suptitle('')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def init():
      suptitle.set_text('')
      return fig.axes

    def update(gen):
      subset = df[df[self.index] == gen]
      history_offsets = history_lookup.get(gen) if history_lookup else None
      if plot_context['multi']:
        self._populate_multi_axes(axes_list, subset, plot_context,
                                  history_offsets=history_offsets,
                                  last=(gen == generations[-1]))
      else:
        self._populate_axes(axes_list[0], subset, plot_context,
                            history_offsets=history_offsets,
                            last=(gen == generations[-1]))
      suptitle.set_text(f'Generation {gen}')
      fig.tight_layout(rect=[0, 0, 1, 0.94])
      return fig.axes

    anim = animation.FuncAnimation(fig, update, frames=generations, init_func=init,
                                   interval=1000.0 / self.fps, blit=False)
    html_str = anim.to_jshtml()
    centered_html = f'<div style=\"display:flex;justify-content:center;\">{html_str}</div>'
    with open(filename, 'w', encoding='utf-8') as out:
      out.write(centered_html)
    plt.close(fig)

  def _write_frames(self, df, generations, plot_context, history_lookup, frame_template):
    if frame_template is None:
      return
    frame_indices = self._select_frame_indices(len(generations))
    for idx in frame_indices:
      gen = generations[idx]
      subset = df[df[self.index] == gen]
      history_offsets = history_lookup.get(gen) if history_lookup else None
      fig = self._render_frame(subset, gen, plot_context,
                               last=(gen == generations[-1]), history_offsets=history_offsets)
      frame_path = frame_template.format(index=idx)
      fig.savefig(frame_path, format='png')
      plt.close(fig)

  def _render_frame(self, subset, gen, plot_context,
                    last=False, history_offsets=None):
    if plot_context['multi']:
      fig, axes = self._create_multi_axes(len(self.metricColumns))
      axes_list = np.atleast_1d(axes).tolist() if isinstance(axes, np.ndarray) else [axes]
      self._populate_multi_axes(axes_list, subset, plot_context,
                                history_offsets=history_offsets, last=last)
    else:
      fig, ax = self._create_axes(plot_context['axis_limits'])
      self._populate_axes(ax, subset, plot_context,
                          history_offsets=history_offsets, last=last)
    fig.suptitle(f'Generation {gen}')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

  @staticmethod
  def _sample_generations(generations, limit):
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

  def _select_frame_indices(self, total_generations):
    if not self.save_frames or total_generations <= 0 or self.frame_max <= 0:
      return []
    if total_generations <= self.frame_max:
      return list(range(total_generations))
    stride = int(math.ceil(total_generations / float(self.frame_max)))
    indices = list(range(0, total_generations, stride))
    if indices and indices[-1] != total_generations - 1:
      if len(indices) >= self.frame_max:
        indices[-1] = total_generations - 1
      else:
        indices.append(total_generations - 1)
    return sorted(set(indices))

  def _discover_objective_columns(self, data_vars):
    """
    Infer objective column names when plotting all objectives.
    """
    outputs = []
    try:
      outputs = self.source.getVars('output')
    except Exception:
      outputs = []
    search_order = outputs if outputs else data_vars
    objectives = []
    seen = set()
    for var in search_order:
      norm = var.lower()
      if norm.startswith('obj') and var not in seen:
        objectives.append(var)
        seen.add(var)
    return objectives

  def _populate_axes(self, ax, subset, plot_context, history_offsets=None, last=False):
    payload = self._prepare_plot_payload(subset, last=last)
    axis_limits = plot_context['axis_limits']
    contour_data = plot_context['contour_data']
    constraint_data = plot_context['constraint_data']
    self._draw_single_axis(
        ax, payload, axis_limits, contour_data, constraint_data,
        history_offsets=history_offsets, metric_label=self._color_metric,
        show_summary=True, show_legend=True, show_xlabel=True, last=last)
    return payload['summary'], payload['meta']

  def _create_axes(self, axis_limits):
    fig, ax = plt.subplots(figsize=(6, 6))
    return fig, ax

  def _create_multi_axes(self, count):
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 6),
                             sharex=False, sharey=False)
    return fig, axes

  def _prepare_plot_payload(self, subset, last=False):
    axis_x, axis_y = self.axes
    display_subset, display_info = self._select_display_subset(subset)
    colors, summary, meta = self._compute_colors(
        display_subset, metric_name=self._color_metric, last=last,
        top_cap=display_info.get('top_cap'))
    offsets = display_subset[[axis_x, axis_y]].to_numpy(dtype=float) if not display_subset.empty else np.empty((0, 2))
    best_pos = meta.get('best_pos')
    plot_offsets = offsets
    plot_colors = colors
    if best_pos is not None and len(offsets) > 0:
      order = np.concatenate((np.delete(np.arange(len(offsets)), best_pos), [best_pos]))
      plot_offsets = offsets[order]
      plot_colors = colors[order]
    total_count = len(subset)
    display_count = len(display_subset)
    summary['count_total'] = total_count
    summary['displayed'] = display_count
    summary['display_fraction'] = display_count / float(total_count) if total_count else 0.0
    summary['count'] = total_count
    if total_count:
      summary['top_fraction'] = summary['top'] / float(total_count)
    payload = {
        'subset': display_subset,
        'full_subset': subset,
        'plot_offsets': plot_offsets,
        'plot_colors': plot_colors,
        'raw_offsets': offsets,
        'colors': colors,
        'summary': summary,
        'meta': meta
    }
    return payload

  def _populate_multi_axes(self, axes, subset, plot_context, history_offsets=None, last=False):
    if not isinstance(axes, (list, tuple, np.ndarray)):
      axes = [axes]
    else:
      axes = np.atleast_1d(axes).tolist()
    payload = self._prepare_plot_payload(subset, last=last)
    constraint_data = plot_context['constraint_data']
    metrics = self.metricColumns
    for idx, metric in enumerate(metrics):
      axis_info = plot_context['geometry_map'][metric]
      show_summary = (idx == 0)
      show_legend = (idx == len(metrics) - 1)
      show_xlabel = True
      self._draw_single_axis(
          axes[idx], payload, axis_info['axis_limits'], axis_info['contour_data'],
          constraint_data, history_offsets=history_offsets, metric_label=metric,
          show_summary=show_summary, show_legend=show_legend,
          show_xlabel=show_xlabel, last=last)
    return payload

  def _draw_single_axis(self, ax, payload, axis_limits, contour_data, constraint_data,
                        history_offsets=None, metric_label=None, show_summary=True,
                        show_legend=True, show_xlabel=True, last=False):
    axis_x, axis_y = self.axes
    if hasattr(ax, '_objective_contour_summary'):
      delattr(ax, '_objective_contour_summary')
    if hasattr(ax, '_objective_contour_legend'):
      try:
        ax._objective_contour_legend.remove()
      except Exception:
        pass
      delattr(ax, '_objective_contour_legend')
    ax.clear()
    self._draw_contours(ax, contour_data, constraint_data)
    history_points = history_offsets if history_offsets is not None else np.empty((0, 2))
    if self.show_history and history_points.size:
      ax.scatter(history_points[:, 0], history_points[:, 1],
                 facecolors=[self._history_facecolor],
                 edgecolors='none',
                 linewidths=0.0,
                 s=self.history_marker_size,
                 zorder=2)
    plot_offsets = payload['plot_offsets']
    plot_colors = payload['plot_colors']
    if len(plot_offsets):
      ax.scatter(plot_offsets[:, 0], plot_offsets[:, 1], facecolors=plot_colors,
                 edgecolors='k', linewidths=0.3, s=60, zorder=3)
    xMin, xMax = axis_limits[axis_x]
    yMin, yMax = axis_limits[axis_y]
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    if show_xlabel:
      ax.set_xlabel(axis_x)
      ax.tick_params(labelbottom=True)
    else:
      ax.set_xlabel('')
      ax.tick_params(labelbottom=False)
    ax.set_ylabel(axis_y)
    label = metric_label or payload['summary'].get('metric')
    title = f'{axis_x} vs {axis_y}'
    if label:
      title += f' | objective: {label}'
    ax.set_title(title)
    best_coords = payload['meta'].get('best_coords') if payload.get('meta') else None
    cross_metric = metric_label or self._color_metric
    if best_coords is None:
      best_coords = self._best_coords_for_metric(payload.get('full_subset'), cross_metric)
    if best_coords is None:
      best_coords = self._best_coords_for_metric(payload['subset'], cross_metric)
    if best_coords is not None:
      self._draw_crosshair(ax, best_coords, axis_limits)
    if show_summary:
      self._update_summary(ax, payload['summary'])
    if show_legend:
      self._ensure_legend(ax)

  def _best_coords_for_metric(self, subset, metric_name):
    if subset is None or metric_name is None or subset.empty or metric_name not in subset.columns:
      return None
    metric_kind = self.metricKinds.get(metric_name, self.metricKind)
    metric_series, _ = _metric_series_helper(subset, metric_name, metric_kind)
    if metric_series.empty:
      return None
    best_idx = metric_series.idxmin()
    if best_idx not in subset.index:
      return None
    return subset.loc[best_idx, self.axes].to_numpy(dtype=float)


  def _compute_colors(self, subset, metric_name=None, last=False, top_cap=None):
    metric = metric_name or self._color_metric or (self.metricColumns[0] if self.metricColumns else self.metricName)
    metric_kind = self.metricKinds.get(metric, self.metricKind)
    return _compute_population_colors(subset, self.axes, metric, metric_kind,
                                      self.constraintVars, self.top_fraction,
                                      self.top_count_override, last=last,
                                      infeasible_color=self._infeasible_point_color,
                                      top_cap=top_cap)

  def _update_visual_config(self):
    if self._custom_infeasible_color is not None:
      target_color = self._custom_infeasible_color
    else:
      target_color = ENHANCED_INFEASIBLE_POINT_COLOR if self.show_history else DEFAULT_INFEASIBLE_POINT_COLOR
    self._infeasible_point_color = target_color
    self._history_facecolor = mcolors.to_rgba(self.history_color, self.history_alpha)

  def _build_history_offsets(self, df, generations):
    """
    Construct a per-generation lookup of exploration history coordinates.
    """
    history_lookup = {}
    dims = len(self.axes)
    cumulative = np.empty((0, dims), dtype=float)
    for gen in generations:
      # Store a copy that reflects all prior generations only.
      history_lookup[gen] = cumulative.copy()
      subset = df[df[self.index] == gen]
      if subset.empty:
        continue
      coords = subset[self.axes].to_numpy(dtype=float)
      if coords.size:
        cumulative = np.vstack((cumulative, coords))
    return history_lookup

  def _select_display_subset(self, subset):
    """
    Limit the displayed population when generations become large.
    Returns the filtered subset and auxiliary info for plotting.
    """
    info = {'top_cap': None, 'limited': False}
    total = len(subset)
    if total == 0:
      return subset, info
    if total <= self.display_threshold or self.display_fraction >= 0.9999:
      return subset, info
    display_count = max(1, int(math.ceil(total * self.display_fraction)))
    metric_kind = self.metricKinds.get(self._color_metric, self.metricKind)
    metric_series, _ = _metric_series_helper(subset, self._color_metric, metric_kind)
    primary = metric_series.nsmallest(display_count).index.tolist()
    selection = list(primary)
    if 'rank' in subset.columns:
      try:
        ranks = subset['rank'].astype(int)
        pareto_indices = subset.index[ranks == 1].tolist()
      except (ValueError, TypeError):
        pareto_indices = []
      for idx in pareto_indices:
        if idx not in selection:
          selection.append(idx)
    if not selection:
      selection = subset.index[:display_count].tolist()
    display_subset = subset.loc[selection]
    info['limited'] = True
    if self.top_count_override is None:
      info['top_cap'] = 5
    return display_subset, info

  def _prepare_plot_geometry(self, df, metric_name):
    metric_kind = self.metricKinds.get(metric_name, self.metricKind)
    metric_series, original_series = _metric_series_helper(df, metric_name, metric_kind)
    metric_min = metric_series.min()
    metric_max = metric_series.max()

    center_idx = metric_series.idxmin()
    center_point = df.loc[center_idx, self.axes].to_numpy(dtype=float)
    axis_points = df[self.axes].to_numpy(dtype=float)
    distances = np.linalg.norm(axis_points - center_point, axis=1)
    max_radius = distances.max()
    if np.isclose(max_radius, 0.0):
      max_radius = 1.0

    limits = {}
    for i, axis in enumerate(self.axes):
      column = df[axis].astype(float)
      data_min = column.min()
      data_max = column.max()
      offset_min = abs(center_point[i] - data_min)
      offset_max = abs(data_max - center_point[i])
      half_span = max(max_radius, offset_min, offset_max)
      if np.isclose(half_span, 0.0):
        half_span = 1.0
      padding = 0.05 * half_span
      half_span += padding
      axis_min = center_point[i] - half_span
      axis_max = center_point[i] + half_span
      limits[axis] = (axis_min, axis_max)

    if np.isclose(metric_max, metric_min):
      radii = []
    else:
      levels = np.linspace(metric_min, metric_max, 6)[1:]
      radii = []
      for lvl in levels:
        fraction = max((lvl - metric_min) / (metric_max - metric_min), 0.0)
        radius = math.sqrt(fraction) * max_radius
        if radius > 0.0:
          original_lvl = _metric_to_original_helper(lvl, metric_kind)
          radii.append((radius, float(lvl), float(original_lvl)))

    shade = None
    if max_radius > 0:
      x_vals = np.linspace(limits[self.axes[0]][0], limits[self.axes[0]][1], 200)
      y_vals = np.linspace(limits[self.axes[1]][0], limits[self.axes[1]][1], 200)
      X, Y = np.meshgrid(x_vals, y_vals)
      radius_grid = np.sqrt((X - center_point[0]) ** 2 + (Y - center_point[1]) ** 2)
      shade = (X, Y, np.clip(radius_grid / max_radius, 0.0, 1.0))

    contour_data = {
        'center': center_point,
        'radii': radii,
        'metric_min': float(metric_min),
        'metric_max': float(metric_max),
        'kind': metric_kind,
        'metric_name': metric_name,
        'original_min': float(original_series.min()),
        'original_max': float(original_series.max()),
        'shade': shade,
        'max_radius': max_radius
    }
    constraint_data = self._prepare_constraint_geometry(df)
    return limits, contour_data, constraint_data

  def _build_plot_context(self, df):
    if self._multi_objective:
      geometry_map = {}
      constraint_data_shared = None
      for metric in self.metricColumns:
        limits, contour_data, constraint_data = self._prepare_plot_geometry(df, metric)
        geometry_map[metric] = {'axis_limits': limits, 'contour_data': contour_data}
        if constraint_data_shared is None:
          constraint_data_shared = constraint_data
      if constraint_data_shared is None:
        constraint_data_shared = []
      return {'multi': True,
              'geometry_map': geometry_map,
              'constraint_data': constraint_data_shared}
    axis_limits, contour_data, constraint_data = self._prepare_plot_geometry(df, self._color_metric)
    return {'multi': False,
            'axis_limits': axis_limits,
            'contour_data': contour_data,
            'constraint_data': constraint_data}

  def _draw_contours(self, ax, contour_data, constraint_data=None):
    center = contour_data['center']
    radii = contour_data['radii']
    shade = contour_data.get('shade')
    if constraint_data:
      self._draw_constraint_overlays(ax, constraint_data)
    if shade is not None:
      X, Y, Shade = shade
      ax.contourf(X, Y, Shade, levels=np.linspace(0, 1, 10), cmap=SHADE_CMAP, alpha=0.35, antialiased=True, zorder=0)
    if not radii:
      return
    for radius, lvl, original_lvl in radii:
      circle = patches.Circle(center, radius, fill=False, linestyle='--', linewidth=0.8, alpha=0.6, color='lightgray', zorder=1)
      ax.add_patch(circle)
      ax.text(center[0], center[1] + radius, f'{original_lvl:.4g}', fontsize=8, ha='center', va='bottom', color='gray', alpha=0.8, zorder=2)

  def _update_summary(self, ax, summary):
    text = self._format_summary(summary)
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
  def _format_summary(summary):
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

  def _parse_formats(self, raw):
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
  def _scaled_bounds(min_val, max_val):
    if np.isclose(min_val, max_val):
      delta = abs(min_val) if min_val != 0 else 1.0
      return min_val - 0.1 * delta, max_val + 0.1 * delta
    low = min_val * 0.9 if min_val >= 0 else min_val * 1.1
    high = max_val * 1.1 if max_val >= 0 else max_val * 0.9
    if np.isclose(low, high):
      delta = abs(low) if low != 0 else 1.0
      low -= 0.1 * delta
      high += 0.1 * delta
    return low, high

  def _prepare_constraint_geometry(self, df):
    if not self.constraintVars:
      return []
    axis_x, axis_y = self.axes
    x_vals = df[axis_x].astype(float).to_numpy()
    y_vals = df[axis_y].astype(float).to_numpy()
    constraint_data = []
    for idx, var in enumerate(self.constraintVars):
      values = df[var].astype(float).to_numpy()
      mask = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(values)
      if mask.sum() < 3:
        self.raiseAWarning(f'Constraint "{var}" has insufficient samples to render on ObjectiveContourAnimationPlot "{self.name}".')
        continue
      x = x_vals[mask]
      y = y_vals[mask]
      g = values[mask]
      coords = np.column_stack((x, y))
      try:
        unique_coords, unique_idx = np.unique(coords, axis=0, return_index=True)
      except TypeError:
        # numpy <1.13 compatibility: fallback by rounding
        rounded = np.round(coords, decimals=10)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        unique_coords = coords[unique_idx]
      x_unique = unique_coords[:, 0]
      y_unique = unique_coords[:, 1]
      g_unique = g[unique_idx]
      finite_mask = np.isfinite(g_unique)
      x_unique = x_unique[finite_mask]
      y_unique = y_unique[finite_mask]
      g_unique = g_unique[finite_mask]
      if x_unique.size < 3:
        self.raiseAWarning(f'Constraint "{var}" collapsed to insufficient points after filtering for ObjectiveContourAnimationPlot "{self.name}".')
        continue
      triang = mtri.Triangulation(x_unique, y_unique)
      if triang.triangles.size == 0:
        continue
      constraint_data.append({
          'name': var,
          'x': x_unique,
          'y': y_unique,
          'values': g_unique,
          'triangulation': triang,
          'fill_color': CONSTRAINT_FILL_COLOR,
          'line_color': CONSTRAINT_LINE_COLORS[idx % len(CONSTRAINT_LINE_COLORS)]
      })
    return constraint_data

  def _draw_constraint_overlays(self, ax, constraint_data):
    for info in constraint_data:
      values = info['values']
      triang = info['triangulation']
      tris = triang.triangles
      tri_vals = values[tris]
      infeasible_mask = np.max(tri_vals, axis=1) <= 0.0
      if infeasible_mask.any():
        polys = np.stack((info['x'][tris[infeasible_mask]], info['y'][tris[infeasible_mask]]), axis=-1)
        collection = PolyCollection(polys, facecolors=info['fill_color'], edgecolors='none', alpha=0.35, zorder=0.5)
        ax.add_collection(collection)
      try:
        ax.tricontour(triang, values, levels=[0.0], colors=info['line_color'], linewidths=1.1, zorder=1.5)
      except Exception:
        continue

  def _draw_crosshair(self, ax, best_coords, axis_limits):
    x_line, y_line, x_hline, y_hline = self._crosshair_segments(best_coords, axis_limits)
    ax.plot(x_line, y_line, linestyle='--', color=CROSSHAIR_COLOR, linewidth=1.0, zorder=2.6)
    ax.plot(x_hline, y_hline, linestyle='--', color=CROSSHAIR_COLOR, linewidth=1.0, zorder=2.6)

  def _crosshair_segments(self, best_coords, axis_limits):
    x_val, y_val = best_coords
    x_min, _ = axis_limits[self.axes[0]]
    y_min, _ = axis_limits[self.axes[1]]
    return [x_val, x_val], [y_min, y_val], [x_min, x_val], [y_val, y_val]

  def _ensure_legend(self, ax):
    if hasattr(ax, '_objective_contour_legend'):
      try:
        ax._objective_contour_legend.remove()
      except Exception:
        pass
      delattr(ax, '_objective_contour_legend')
    base_label = 'Current feasible population' if self.show_history else 'Feasible population'
    base_handle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                         markerfacecolor=BASE_POINT_COLOR, markeredgecolor='k', label=base_label)
    top_handle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                        markerfacecolor=TOP_POINT_COLOR, markeredgecolor='k', label='Top highlighted')
    best_handle = Line2D([0], [0], marker='o', linestyle='', markersize=7,
                         markerfacecolor=BEST_POINT_COLOR, markeredgecolor='k', label='Best solution')
    infeasible_handle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                               markerfacecolor=self._infeasible_point_color, alpha=0.85,
                               markeredgecolor='k', label='Infeasible samples')
    region_handle = Patch(facecolor=CONSTRAINT_FILL_COLOR, alpha=0.35, label='Infeasible region')
    handles = [best_handle, top_handle, base_handle]
    if self.show_history:
      history_handle = Line2D([0], [0], marker='o', linestyle='', markersize=6,
                              markerfacecolor=self._history_facecolor, markeredgecolor='none',
                              alpha=self.history_alpha,
                              label='History (prior generations)')
      handles.append(history_handle)
    handles.extend([infeasible_handle, region_handle])
    legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.17),
                       frameon=False, ncol=3, fontsize=9)
    ax._objective_contour_legend = legend
