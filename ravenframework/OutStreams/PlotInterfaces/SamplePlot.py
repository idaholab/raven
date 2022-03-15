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
Created on April 1, 2021

@author: talbpaul
"""
import matplotlib.pyplot as plt

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class SamplePlot(PlotInterface):
  """
    Plots variables as a function of realization; a simple demonstration plotter.
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
        descr=r"""The name of the RAVEN DataObject from which the data should be taken for this plotter."""))
    spec.addSub(InputData.parameterInputFactory('vars', contentType=InputTypes.StringListType,
        descr=r"""Names of the variables from the DataObject whose realizations should be plotted."""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'PlotInterface'
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
    # sanity check
    dataVars = self.source.getVars()
    missing = [var for var in self.vars if var not in dataVars]
    if missing:
      msg = f'Source DataObject "{self.source.name}" is missing the following variables expected by SamplePlot "{self.name}": '
      msg += ', '.join(f'"{m}"' for m in missing)
      self.raiseAnError(IOError, msg)

  def run(self):
    """
      Main run method.
      @ In, None
      @ Out, None
    """
    fig, axes = plt.subplots(len(self.vars), 1, sharex=True)
    allDims = self.source.getDimensions()
    data, meta = self.source.getData()
    sampleIDs = data['RAVEN_sample_ID'].values
    for v, var in enumerate(self.vars):
      ax = axes[v]
      dims = allDims[var]
      if len(dims) > 0:
        self.raiseAnError(RuntimeError, f'Variable "{var}" has too high dimensionality ({len(dims)}) for the SamplePlotter!')
      values = data[var].values
      self.plotScalar(ax, sampleIDs, values)
      ax.set_ylabel(var)
    axes[-1].set_xlabel('RAVEN Sample Number')
    fig.align_ylabels(axes[:])
    plt.savefig(f'{self.name}.png')

  def plotScalar(self, ax, ids, vals):
    """
      Plots a scalar by sample.
      @ In, ax, pyplot axis, axis to plot on
      @ In, ids, np.array, RAVEN sample IDs to plot against
      @ In, vals, np.array, values to plot
      @ Out, None
    """
    ax.plot(ids, vals, '.-')
