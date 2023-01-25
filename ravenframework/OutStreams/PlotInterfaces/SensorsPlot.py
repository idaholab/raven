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
# from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Internal Imports
from ...utils import plotUtils
from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class SensorsPlot(PlotInterface):
  """
    Plots Optimal Sensor Placement coordinate in input and output space
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.setStrictMode(False)
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringType,
        descr=r"""The name of the RAVEN DataObject from which the data should be taken for this plotter.
              This should be the SolutionExport for a MultiRun with an Optimizer."""))
    spec.addSub(InputData.parameterInputFactory('vars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject whose optimization paths should be plotted."""))
    # spec.addSub(InputData.parameterInputFactory('logVars', contentType=InputTypes.StringListType,
    #     descr=r"""Names of the variables from the DataObject to be plotted on a log scale."""))
    spec.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.StringType,
        descr=r"""Names of the variable that refers to the batch index"""))
    spec.addSub(InputData.parameterInputFactory('how', contentType=InputTypes.StringType,
        descr=r"""Digital format of the generated picture"""))
    spec.addSub(InputData.parameterInputFactory('marker', contentType=InputTypes.StringType,
        descr=r"""Marker of the scatter plot"""))
    spec.addSub(InputData.parameterInputFactory('xlabel', contentType=InputTypes.StringType,
        descr=r"""X-axis label of the scatter plot"""))
    spec.addSub(InputData.parameterInputFactory('ylabel', contentType=InputTypes.StringType,
        descr=r"""Y-axis label of the scatter plot"""))
    spec.addSub(InputData.parameterInputFactory('c', contentType=InputTypes.StringType,
        descr=r"""Colour of points on the scatter plot"""))
    spec.addSub(InputData.parameterInputFactory('s', contentType=InputTypes.FloatType,
        descr=r"""The marker size in points**2"""))
    spec.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.FloatType,
        descr=r"""The alpha blending value, between 0 (transparent) and 1 (opaque)"""))
    spec.addSub(InputData.parameterInputFactory('linewidths', contentType=InputTypes.FloatType,
        descr=r"""The linewidth of the marker edges"""))
    spec.addSub(InputData.parameterInputFactory('cmap', contentType=InputTypes.StringType,
        descr=r"""A Colormap instance or registered colormap name"""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag   = 'SPSL  Plot'
    self.source     = None      # reference to DataObject source
    self.sourceName = None      # name of DataObject source
    self.vars       = None      # variables to plot
    self.logVars    = None      # variables to plot in log scale
    self.index      = None      # index ID for each batch
    self.how        = None      # format of the generated picture
    self.marker     = None      # marker of the scatter plot
    self.xlabel     = None      # X-axis label of the scatter plot
    self.ylabel     = None      # Y-axis label of the scatter plot
    self.c          = None      # Colour of the points on the scatter plot
    self.s          = None      # The marker size in points**2
    self.alpha      = None      # The alpha blending value, between 0 (transparent) and 1 (opaque)
    self.linewidths = None      # The linewidth of the marker edges
    self.cmap       = None      # A Colormap instance or registered colormap name

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    params, notFound = spec.findNodesAndExtractValues(['source','vars','index','how','marker','xlabel','ylabel','c','s','alpha','linewidths','cmap'])

    listNotFound = notFound.copy()
    for req in listNotFound:
      if req == 's':
        params['s'] = 20
      elif req == 'cmap':
        params['cmap'] = None
      elif req == 'alpha':
        params['alpha'] = None
      elif req == 'linewidths':
        params['linewidths'] = None
      elif req == 'marker':
        params['marker'] = 'o'
      elif req == 'c':
        params['c'] = 'b'
      notFound.remove(req)

    for node in notFound:
      self.raiseAnError(IOError, "Missing " +str(node) +" node in the SensorsPlot " + str(self.name))
    else:
      self.sourceName = params['source']
      self.vars       = params['vars']
      self.index      = params['index']
      self.how        = params['how']
      self.marker     = params['marker']
      self.xlabel     = params['xlabel']
      self.ylabel     = params['ylabel']
      self.c          = params['c']
      self.s          = params['s']
      self.alpha      = params['alpha']
      self.linewidths = params['linewidths']
      self.cmap       = params['cmap']
    # params, notFound = spec.findNodesAndExtractValues(['logVars'])
    # if notFound:
    #   self.logVars = None
    # else:
    #   self.logVars = params['logVars']


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
      Main run method.
      @ In, None
      @ Out, None
    """
    data = self.source.asDataset()
    outVars = self.source.getVars(subset='output')
    inpVars = self.source._data['sensor']
    # nFigures = len(self.vars)
    fig, axs = plt.subplots(1,1)
    fig.suptitle('Sensors Plot')
    # plt.scatter(data.sel(loc = inpVars[0]).sensorLocs.values,data.sel(loc = inpVars[1]).sensorLocs.values,marker = self.marker, c = self.c, s = self.s, alpha = self.alpha, linewidths = self.linewidths, cmap=self.cmap)
    pl = plt.scatter(data['X (m)']*100,data['Y (m)']*100, s=self.s, c=data['Temperature (K)'].data[-1],cmap=plt.cm.coolwarm)
    plt.scatter(data['X (m)'].to_numpy()*100,data['Y (m)'].to_numpy()*100,marker='x',Color='red')
    cbar = plt.colorbar(pl)
    cbar.set_label('Temperature ($^{\circ}K$)')
    plt.xlabel(self.xlabel)
    plt.tick_params(axis='x', labelrotation = 90)
    plt.ylabel(self.ylabel)
    plt.grid()
    axs=plt.gca()
    axs.set_aspect(0.7)
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")

    if self.how in ['png','pdf','svg','jpeg']:
      fileName = self.name +'.%s'  % self.how
      plt.savefig(fileName, format=self.how)
    else:
      self.raiseAnError(IOError, f'Digital format of the plot "{self.name}" is not available!')