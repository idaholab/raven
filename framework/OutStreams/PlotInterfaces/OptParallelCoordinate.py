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
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import imageio

from .PlotInterface import PlotInterface
from utils import InputData, InputTypes

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
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OptPath Plot'
    self.source = None      # reference to DataObject source
    self.sourceName = None  # name of DataObject source
    self.vars = None        # variables to plot

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    self.sourceName = spec.findFirst('source').value
    self.vars = spec.findFirst('vars').value
    # checker; this should be superceded by "required" in input params
    if self.sourceName is None:
      self.raiseAnError(IOError, "Missing <source> node!")
    if self.vars is None:
      self.raiseAnError(IOError, "Missing <vars> node!")

  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, stepEntities, dict, contains all the Objects are going to be used in the
                                current step. The sources are searched into this.
      @ Out, None
    """
    src = self.findSource(self.sourceName, stepEntities)
    if src is None:
      self.raiseAnError(IOError, f'No source named "{self.sourceName}" was found in the Step for SamplePlot "{self.name}"!')
    self.source = src
    # sanity check
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
    data = self.source.asDataset().to_dataframe()
    ynames  = self.source._inputs

    min_Gen = int(min(data['batchId']))
    max_Gen = int(max(data['batchId']))
    
    yMin = np.zeros(4)
    yMax = np.zeros(4)
    
    for idx,inp in enumerate(ynames):
      yMin[idx] = min(data[inp])
      yMax[idx] = max(data[inp])
    
    filesID = []
    
    for idx,genID in enumerate(range(min_Gen,max_Gen+1,1)):
      population = data[data['batchId']==genID]
      ys = population[ynames].values
      fileID = f'{self.name}' + str(genID) + '.png'
      generateParallelPlot(ys,genID,yMin,yMax,ynames,fileID)
      filesID.append(fileID)
    
    fig = plt.figure()
    with imageio.get_writer(f'{self.name}.gif', mode='I') as writer:
      for filename in filesID:
        image = imageio.imread(filename)
        writer.append_data(image)


def generateParallelPlot(zs,batchID,ymins,ymaxs,ynames,fileID):
  """
    Main run method.
    @ In, zs, pandas dataset, batch containing the set of points to be plotted
    @ In, batchID, string, ID of the batch 
    @ In, ymins, np.array, minimum value for each variable
    @ In, ymaxs, np.array, maximum value for each variable
    @ In, ynames, list, list of string containing the ID of each variable
    @ In, fileID, string, name of the file containing the plot
    @ Out, None
  """
  N = zs.shape[0]

  fig, host = plt.subplots()

  axes = [host] + [host.twinx() for i in range(zs.shape[1] - 1)]
  for i, ax in enumerate(axes):
    ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
      ax.spines["right"].set_position(("axes", i / (zs.shape[1] - 1)))

  host.set_xlim(0, zs.shape[1] - 1)
  host.set_xticks(range(zs.shape[1]))
  host.set_xticklabels(ynames, fontsize=14)
  host.tick_params(axis='x', which='major', pad=7)
  host.spines['right'].set_visible(False)
  host.xaxis.tick_top()
  plot_title = 'Generation ' + str(batchID)
  host.set_title(plot_title, fontsize=14)

  for j in range(N):
    verts = list(zip([x for x in np.linspace(0, len(zs) - 1, len(zs) * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1)
    host.add_patch(patch)
  plt.tight_layout()
  plt.savefig(fileID)
    