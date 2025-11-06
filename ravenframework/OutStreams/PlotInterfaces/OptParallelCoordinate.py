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
from ...utils import plotUtils
from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class OptParallelCoordinatePlot(PlotInterface):
  """
    Plots input coordinate in a parallel coordinate plot
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
    missing = [var for var in (self.vars) if var not in dataVars]
    if missing:
      msg = f'Source DataObject "{self.source.name}" is missing the following variables ' +\
            f'expected by OptPath plotter "{self.name}": '
      msg += ', '.join(f'"{m}"' for m in missing)
      self.raiseAnError(IOError, msg)

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
      for alpha, trail_gen in zip(alpha_values, trail_gens):
        population = data[data[self.index] == trail_gen]
        if population.empty:
          continue
        values = population[self.vars].astype(float).to_numpy()
        line_blocks.append(values)
        alpha_blocks.extend([alpha] * len(values))
      if not line_blocks:
        continue
      stacked = np.vstack(line_blocks)
      fileID = f'{self.name}_{genID}.png'
      plotUtils.generateParallelPlot(stacked, genID, yMin, yMax, self.vars, fileID, line_alphas=alpha_blocks)
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

