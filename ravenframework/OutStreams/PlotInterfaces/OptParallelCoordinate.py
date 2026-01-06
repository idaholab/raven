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
Created on November 20th, 2021

@author: mandd
"""

# External Imports
import numpy as np
import imageio

# Internal Imports
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from ...utils import plotUtils
from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class OptParallelCoordinatePlot(PlotInterface):
  """
    Plots input coordinate in a parallel coordinate plot.

    Optional constraint encoding can be enabled via <constraints>:
    - feasible polylines are colored with <feasible_color>
    - infeasible polylines are colored either with <infeasible_color> or (if <color_mode>='violation') a colormap
    - linewidth can optionally scale with total constraint violation (<thickness_mode>='violation')
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""The name of the RAVEN DataObject from which the data should be taken for this plotter.
              This should be the SolutionExport for a MultiRun with an Optimizer."""))
    spec.addSub(InputData.parameterInputFactory('vars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject whose optimization paths should be plotted."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Names of the variable that refers to the batch index"""))
    spec.addSub(InputData.parameterInputFactory('max_frames', contentType=InputTypes.IntegerType,
        descr=r"""Optional cap on the number of generations rendered. If omitted, the plotter shows at most ten evenly spaced generations."""))
    spec.addSub(InputData.parameterInputFactory('trail_generations', contentType=InputTypes.IntegerType,
        descr=r"""Optional count of historical generations to overlay (with fading) in each frame. Older trails are drawn with lower opacity."""))
    spec.addSub(InputData.parameterInputFactory('constraints', contentType=InputTypes.StringListType,
        descr=r"""Optional list of constraint evaluation columns. Values > 0 are treated as feasible;
                   values <= 0 indicate violation. Use "all" to include every column named
                   ConstraintEvaluation_*."""))
    spec.addSub(InputData.parameterInputFactory('color_mode', contentType=InputTypes.StringType,
        descr=r"""How to color infeasible paths when <constraints> are provided.
                   Options: "feasibility" (default; infeasible uses <infeasible_color>),
                   "violation" (infeasible color encodes violation magnitude), or "none"."""))
    spec.addSub(InputData.parameterInputFactory('violation_metric', contentType=InputTypes.StringType,
        descr=r"""When <color_mode> or <thickness_mode> uses violations, reduce multiple constraints into a scalar using:
                   "sum" (default), "max", or "l2"."""))
    spec.addSub(InputData.parameterInputFactory('thickness_mode', contentType=InputTypes.StringType,
        descr=r"""How to set linewidths when <constraints> are provided. Options: "none" (default) or "violation"."""))
    spec.addSub(InputData.parameterInputFactory('linewidth_bounds', contentType=InputTypes.FloatListType,
        descr=r"""Two floats giving min,max linewidth for infeasible samples when using <thickness_mode>='violation' (default 0.6,2.6)."""))
    spec.addSub(InputData.parameterInputFactory('feasible_color', contentType=InputTypes.StringType,
        descr=r"""Color for feasible paths when <constraints> are provided (default '#2e7d32')."""))
    spec.addSub(InputData.parameterInputFactory('infeasible_color', contentType=InputTypes.StringType,
        descr=r"""Color for infeasible paths when <constraints> are provided and <color_mode>='feasibility' (default '#d32f2f')."""))
    spec.addSub(InputData.parameterInputFactory('infeasible_cmap', contentType=InputTypes.StringType,
        descr=r"""Matplotlib colormap for infeasible paths when <color_mode>='violation' (default 'Reds')."""))
    spec.addSub(InputData.parameterInputFactory('show_infeasible', contentType=InputTypes.BoolType,
        descr=r"""If false, hide infeasible paths when <constraints> are provided (default true)."""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OptParallelCoordinatePlot'
    self.source = None      # reference to DataObject source
    self.sourceName = None  # name of DataObject source
    self.vars = None        # variables to plot
    self.index = None       # index ID for each batch
    self.maxFrames = None   # user-specified generation cap
    self.trailGenerations = None  # number of trailing generations to overlay
    self.constraints = []
    self.useAllConstraints = False
    self.colorMode = 'feasibility'
    self.violationMetric = 'sum'
    self.thicknessMode = 'none'
    self.linewidthBounds = (0.6, 2.6)
    self.feasibleColor = '#2e7d32'
    self.infeasibleColor = '#d32f2f'
    self.infeasibleCmap = 'Reds'
    self.showInfeasible = True
    self._globalViolationMax = None

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    params, notFound = spec.findNodesAndExtractValues(['source','vars','index'])

    for node in notFound:
      self.raiseAnError(IOError, "Missing " +str(node) +" node in the OptParallelCoordinatePlot " + str(self.name))
    else:
      self.sourceName = params['source']
      self.vars       = params['vars']
      self.index      = params['index']
    maxNode = spec.findFirst('max_frames')
    if maxNode is not None and maxNode.value is not None:
      self.maxFrames = int(maxNode.value)
      if self.maxFrames <= 0:
        self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" received non-positive <max_frames>.')
    trailNode = spec.findFirst('trail_generations')
    if trailNode is not None and trailNode.value is not None:
      self.trailGenerations = int(trailNode.value)
      if self.trailGenerations <= 0:
        self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" received non-positive <trail_generations>.')

    consNode = spec.findFirst('constraints')
    if consNode is not None and consNode.value:
      entries = [str(entry).strip() for entry in consNode.value if str(entry).strip()]
      if any(entry.lower() == 'all' for entry in entries):
        self.useAllConstraints = True
      else:
        self.constraints = entries

    modeNode = spec.findFirst('color_mode')
    if modeNode is not None and modeNode.value:
      self.colorMode = str(modeNode.value).strip()

    vmNode = spec.findFirst('violation_metric')
    if vmNode is not None and vmNode.value:
      self.violationMetric = str(vmNode.value).strip()

    tmNode = spec.findFirst('thickness_mode')
    if tmNode is not None and tmNode.value:
      self.thicknessMode = str(tmNode.value).strip()

    lwNode = spec.findFirst('linewidth_bounds')
    if lwNode is not None and lwNode.value:
      vals = [float(v) for v in lwNode.value]
      if len(vals) != 2:
        self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" requires two values in <linewidth_bounds>.')
      self.linewidthBounds = (min(vals[0], vals[1]), max(vals[0], vals[1]))

    fcNode = spec.findFirst('feasible_color')
    if fcNode is not None and fcNode.value:
      self.feasibleColor = str(fcNode.value).strip()

    icNode = spec.findFirst('infeasible_color')
    if icNode is not None and icNode.value:
      self.infeasibleColor = str(icNode.value).strip()

    cmapNode = spec.findFirst('infeasible_cmap')
    if cmapNode is not None and cmapNode.value:
      self.infeasibleCmap = str(cmapNode.value).strip()

    siNode = spec.findFirst('show_infeasible')
    if siNode is not None and siNode.value is not None:
      self.showInfeasible = bool(siNode.value)

    if self.colorMode not in {'feasibility', 'violation', 'none'}:
      self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" received unsupported <color_mode> "{self.colorMode}".')
    if self.violationMetric not in {'sum', 'max', 'l2'}:
      self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" received unsupported <violation_metric> "{self.violationMetric}".')
    if self.thicknessMode not in {'none', 'violation'}:
      self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" received unsupported <thickness_mode> "{self.thicknessMode}".')
    if not mcolors.is_color_like(self.feasibleColor):
      self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" received invalid <feasible_color> "{self.feasibleColor}".')
    if not mcolors.is_color_like(self.infeasibleColor):
      self.raiseAnError(IOError, f'OptParallelCoordinatePlot "{self.name}" received invalid <infeasible_color> "{self.infeasibleColor}".')


  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, stepEntities, dict, contains all the Objects are going to be used in the
                                current step. The sources are searched into this.
      @ Out, None
    """
    super().initialize(stepEntities)
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" was found in the Step for SamplePlot "{self.name}"!')
    self.source = src
    dataVars = self.source.getVars()
    if self.useAllConstraints:
      self.constraints = sorted(var for var in dataVars if var.startswith('ConstraintEvaluation_'))
    missing = [var for var in (self.vars) if var not in dataVars]
    if missing:
      msg = f'Source DataObject "{self.source.name}" is missing the following variables ' +\
            f'expected by OptPath plotter "{self.name}": '
      msg += ', '.join(f'"{m}"' for m in missing)
      self.raiseAnError(IOError, msg)
    if self.constraints:
      missing_constraints = [var for var in self.constraints if var not in dataVars]
      if missing_constraints:
        self.raiseAWarning(f'OptParallelCoordinatePlot "{self.name}" could not find constraint column(s) {missing_constraints}; proceeding with available constraints only.')
        self.constraints = [var for var in self.constraints if var in dataVars]

    if self.constraints:
      data = self.source.asDataset().to_dataframe().copy()
      if not data.empty:
        self._globalViolationMax = float(np.nanmax(self._constraint_violation(data)))
      else:
        self._globalViolationMax = 0.0

  @staticmethod
  def _is_feasible(df, constraints):
    if df is None or df.empty or not constraints:
      return np.ones(0 if df is None else len(df), dtype=bool)
    feasible = np.ones(len(df), dtype=bool)
    for var in constraints:
      if var not in df.columns:
        continue
      vals = df[var].astype(float).to_numpy()
      feasible &= vals > 0.0
    return feasible

  def _constraint_violation(self, df):
    if df is None or df.empty or not self.constraints:
      return np.zeros(0 if df is None else len(df), dtype=float)
    values = []
    for var in self.constraints:
      if var not in df.columns:
        continue
      vals = df[var].astype(float).to_numpy()
      values.append(np.maximum(0.0, -vals))
    if not values:
      return np.zeros(len(df), dtype=float)
    stacked = np.vstack(values)
    if self.violationMetric == 'max':
      return np.max(stacked, axis=0)
    if self.violationMetric == 'l2':
      return np.sqrt(np.sum(stacked * stacked, axis=0))
    return np.sum(stacked, axis=0)

  def run(self):
    """
      Main run method
      @ In, None
      @ Out, None
    """
    data = self.source.asDataset().to_dataframe().copy()
    if data.empty:
      self.raiseAWarning(f'OptParallelCoordinatePlot "{self.name}" received an empty dataset; no plot generated.')
      return
    data[self.index] = data[self.index].astype(float)

    generations = sorted(data[self.index].unique())
    if not generations:
      self.raiseAWarning(f'OptParallelCoordinatePlot "{self.name}" found no generations in "{self.index}".')
      return

    numeric = data[self.vars].astype(float)
    yMin = numeric.min().to_numpy()
    yMax = numeric.max().to_numpy()

    def _select_generations(all_gens, limit):
      if limit >= len(all_gens):
        return list(all_gens)
      positions = np.linspace(0, len(all_gens) - 1, limit, dtype=int)
      selected_indices = []
      for idx in positions:
        if idx not in selected_indices:
          selected_indices.append(idx)
      # fill up if duplicates occurred
      cursor = 0
      while len(selected_indices) < limit and cursor < len(all_gens):
        if cursor not in selected_indices:
          selected_indices.append(cursor)
        cursor += 1
      selected_indices = sorted(selected_indices)
      if selected_indices[-1] != len(all_gens) - 1:
        selected_indices[-1] = len(all_gens) - 1
      selected_indices = sorted(selected_indices)
      return [all_gens[i] for i in selected_indices]

    default_cap = min(len(generations), 10)
    frame_cap = self.maxFrames if self.maxFrames is not None else default_cap
    frame_cap = max(1, min(frame_cap, len(generations)))
    gens_to_render = _select_generations(generations, frame_cap)
    index_lookup = {gen: idx for idx, gen in enumerate(generations)}

    trail_len = self.trailGenerations if self.trailGenerations is not None else min(5, len(generations))
    trail_len = max(1, trail_len)

    filesID = []

    for genID in gens_to_render:
      gen_position = index_lookup[genID]
      trail_start = max(0, gen_position - trail_len + 1)
      trail_gens = generations[trail_start:gen_position + 1]
      if len(trail_gens) == 1:
        alpha_values = np.array([1.0])
      else:
        alpha_values = np.linspace(0.3, 1.0, len(trail_gens))
      line_blocks = []
      alpha_blocks = []
      color_blocks = []
      width_blocks = []
      for alpha, trail_gen in zip(alpha_values, trail_gens):
        population = data[data[self.index] == trail_gen]
        if population.empty:
          continue
        if self.constraints:
          feasibleMask = self._is_feasible(population, self.constraints)
          if not self.showInfeasible:
            population = population[feasibleMask]
            feasibleMask = np.ones(len(population), dtype=bool)
          violation = self._constraint_violation(population) if not population.empty else np.zeros(0, dtype=float)
          denom = float(self._globalViolationMax) if self._globalViolationMax is not None else float(np.nanmax(violation) if violation.size else 0.0)
          denom = denom if np.isfinite(denom) and denom > 0.0 else 1.0
          vnorm = np.clip(violation / denom, 0.0, 1.0)
          if self.colorMode == 'violation':
            cmap = cm.get_cmap(self.infeasibleCmap)
            colors = []
            for is_feas, v in zip(feasibleMask, vnorm):
              if is_feas:
                colors.append(self.feasibleColor)
              else:
                colors.append(cmap(float(v)))
            colors = np.asarray(colors, dtype=object)
          elif self.colorMode == 'none':
            colors = np.asarray(['tab:blue'] * len(population), dtype=object)
          else:
            colors = np.where(feasibleMask, self.feasibleColor, self.infeasibleColor).astype(object)
          if self.thicknessMode == 'violation':
            lw_min, lw_max = self.linewidthBounds
            widths = np.where(feasibleMask, lw_min, (lw_min + (lw_max - lw_min) * vnorm)).astype(float)
          else:
            widths = np.where(feasibleMask, 0.9, 1.6).astype(float)
        else:
          colors = np.asarray(['tab:blue'] * len(population), dtype=object)
          widths = np.ones(len(population), dtype=float)
        values = population[self.vars].astype(float).to_numpy()
        line_blocks.append(values)
        alpha_blocks.extend([alpha] * len(values))
        color_blocks.extend(list(colors))
        width_blocks.extend(list(widths))
      if not line_blocks:
        continue
      stacked = np.vstack(line_blocks)
      fileID = f'{self.name}_{genID}.png'
      legend_entries = None
      if self.constraints:
        legend_entries = [
          {'label': 'Feasible (all constraints > 0)', 'color': self.feasibleColor, 'linewidth': 2.0},
          {'label': 'Infeasible (violation encoded)', 'color': self.infeasibleColor if self.colorMode != 'violation' else cm.get_cmap(self.infeasibleCmap)(0.85), 'linewidth': 2.0},
        ]
      plotUtils.generateParallelPlot(stacked, genID, yMin, yMax, self.vars, fileID,
                                     line_alphas=alpha_blocks,
                                     line_colors=color_blocks,
                                     line_widths=width_blocks,
                                     legend_entries=legend_entries)
      filesID.append(fileID)

    if not filesID:
      self.raiseAWarning(f'OptParallelCoordinatePlot "{self.name}" did not produce any frames.')
      return

    # create filename
    giffilename = self._createFilename(defaultName=f'{self.name}.gif')

    with imageio.get_writer(giffilename, mode='I') as writer:
      for filename in filesID:
        image = imageio.imread(filename)
        writer.append_data(image)
