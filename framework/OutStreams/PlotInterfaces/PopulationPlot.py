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
import numpy as np
from matplotlib.lines import Line2D

from .PlotInterface import PlotInterface
from utils import InputData, InputTypes

class PopulationPlot(PlotInterface):
  """
    Plots population coordinate in input and output space
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
    spec.addSub(InputData.parameterInputFactory('logVars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject to be plotted on a log scale."""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag   = 'GAPopulation Plot'
    self.source     = None      # reference to DataObject source
    self.sourceName = None      # name of DataObject source
    self.vars       = None      # variables to plot
    self.logVars    = None      # variables to plot in log scale

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    params, notFound = spec.findNodesAndExtractValues(['source','vars'])

    for node in notFound:
      self.raiseAnError(IOError, "Missing " +str(node) +" node in the PopulationPlot " + str(self.name))
    else:
      self.sourceName = params['source']
      self.vars       = params['vars']

    params, notFound = spec.findNodesAndExtractValues(['logVars'])
    if notFound:
      self.logVars = None
    else:
      self.logVars = params['logVars']


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
      Main run method.
      @ In, None
      @ Out, None
    """
    data = self.source.asDataset().to_dataframe()
    inVars  = self.source._inputs
    outVars = self.source._outputs

    nFigures = len(self.vars)
    fig, axs = plt.subplots(nFigures,1)
    fig.suptitle('Population Plot')

    min_Gen = int(min(data['batchId']))
    max_Gen = int(max(data['batchId']))

    for indexVar,var in enumerate(self.vars):
        min_fit = np.zeros(max_Gen-min_Gen+1)
        max_fit = np.zeros(max_Gen-min_Gen+1)
        avg_fit = np.zeros(max_Gen-min_Gen+1)

        for idx,genID in enumerate(range(min_Gen,max_Gen+1,1)):
            population = data[data['batchId']==genID]
            min_fit[idx] = min(population[var])
            max_fit[idx] = max(population[var])
            avg_fit[idx] = population[var].mean()

        if var in inVars:
          if var in self.logVars:
            errorfill(range(min_Gen,max_Gen+1,1), avg_fit, [min_fit,max_fit], color='g', ax=axs[indexVar],logscale=True)
          else:
            errorfill(range(min_Gen,max_Gen+1,1), avg_fit, [min_fit,max_fit], color='g', ax=axs[indexVar])
        else:
          if var in self.logVars:
            errorfill(range(min_Gen,max_Gen+1,1), avg_fit, [min_fit,max_fit], color='b', ax=axs[indexVar],logscale=True)
          else:
            errorfill(range(min_Gen,max_Gen+1,1), avg_fit, [min_fit,max_fit], color='b', ax=axs[indexVar])
        axs[indexVar].set_ylabel(var)
        if var == self.vars[-1]:
          axs[indexVar].set_xlabel('Generation #')

    plt.savefig(f'{self.name}.png')

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, logscale=False):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    if logscale:
        ax.set_yscale('symlog')

