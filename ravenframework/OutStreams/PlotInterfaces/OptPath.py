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
Created on April 6, 2021

@author: talbpaul
"""
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class OptPath(PlotInterface):
  """
    Plots the path that variables took during an optimization, including accepted and rejected runs.
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
    self.markerMap = {'first': 'yo',
                      'accepted': 'go',
                      'rejected': 'rx',
                      'rerun': 'c.',
                      'final': 'mo'}
    self.markers = defaultdict(lambda: 'k.', self.markerMap)

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
    missing = [var for var in (self.vars+['accepted']) if var not in dataVars]
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
    fig, axes = plt.subplots(len(self.vars), 1, sharex=True)
    fig.suptitle('Optimization Path')
    for r in range(len(self.source)): # realizations
      rlz = self.source.realization(index=r, asDataSet=True, unpackXArray=False)
      accepted = rlz['accepted']
      for v, var in enumerate(self.vars):
        ax = axes[v]
        value = rlz[var]
        self.addPoint(ax, r, value, accepted)
        if v == len(self.vars) - 1:
          ax.set_xlabel('Optimizer Iteration')
        ax.set_ylabel(var)
    # common legend
    fig.subplots_adjust(right=0.80)
    lns = []
    for cond in self.markerMap.keys():
      lns.append(Line2D([0], [0], color=self.markerMap[cond][0], marker=self.markerMap[cond][1]))
    fig.legend(lns, list(self.markerMap.keys()),
               loc='center right',
               borderaxespad=0.1,
               title='Legend')
    plt.savefig(f'{self.name}.png')

  def addPoint(self, ax, i, value, accepted):
    """
      Plots a point in the optimization path.
      @ In, ax, pyplot axis, axis to plot on
      @ In, i, int, iteration number
      @ In, value, float, variable value
      @ In, accepted, str, acceptance condition
      @ Out, lines, list, lines created
    """
    lines = ax.plot(i, value, f'{self.markers[accepted]}')
    return lines
