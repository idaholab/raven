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
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import pandas as pd
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
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OptParallelCoordinate Plot'
    self.source = None      # reference to DataObject source
    self.sourceName = None  # name of DataObject source
    self.vars = None        # variables to plot
    self.index = None       # index ID for each batch


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
    minGen = int(min(data[self.index]))
    maxGen = int(max(data[self.index]))

    yMin = np.zeros(len(self.vars))
    yMax = np.zeros(len(self.vars))

    for idx,inp in enumerate(self.vars):
      yMin[idx] = min(data[inp])
      yMax[idx] = max(data[inp])

    filesID = []

    for idx,genID in enumerate(range(minGen,maxGen+1,1)):
      population = data[data[self.index]==genID]
      ys = population[self.vars].values
      fileID = f'{self.name}' + str(genID) + '.png'
      plotUtils.generateParallelPlot(ys,genID,yMin,yMax,self.vars,fileID)
      filesID.append(fileID)

    fig = plt.figure()
    with imageio.get_writer(f'{self.name}.gif', mode='I') as writer:
      for filename in filesID:
        image = imageio.imread(filename)
        writer.append_data(image)




