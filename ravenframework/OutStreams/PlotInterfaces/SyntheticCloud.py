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
Created on September 2, 2021

@author: talbpaul
Definition for a common plotting need with synthetic histories versus training data.
"""

import matplotlib.pyplot as plt

from .PlotInterface import PlotInterface
from ...utils import InputData, InputTypes

class SyntheticCloud(PlotInterface):
  """
    Plots the training data along with a cloud of sampled data for synthetic histories.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('training', contentType=InputTypes.StringType,
        descr=r"""The name of the RAVEN DataObject from which the training (or original) data should
        be taken for this plotter.
        This should be the data used to train the surrogate."""))
    spec.addSub(InputData.parameterInputFactory('samples', contentType=InputTypes.StringType,
        descr=r"""The name of the RAVEN DataObject from which the sampled synthetic histories should
        be taken for this plotter."""))
    spec.addSub(InputData.parameterInputFactory('macroParam', contentType=InputTypes.StringType,
        descr=r"""Name of the macro variable (e.g. Year)."""))
    spec.addSub(InputData.parameterInputFactory('microParam', contentType=InputTypes.StringType,
        descr=r"""Name of the micro variable or pivot parameter (e.g. Time)."""))
    spec.addSub(InputData.parameterInputFactory('variables', contentType=InputTypes.StringListType,
        descr=r"""Name of the signal variables to plot."""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OptPath Plot'
    self.training = None     # DataObject with training data
    self.trainingName = None # name of training D.O.
    self.samples = None      # DataObject with sample data
    self.samplesName = None  # name of samples D.O.
    self.macroName = None    # name of macro parameter (e.g. Year)
    self.microName = None    # name of micro parameter (e.g. Time)
    self.variables = None    # variable names to plot
    self.clusterName = '_ROM_Cluster' # TODO magic name

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    self.trainingName = spec.findFirst('training').value
    self.samplesName = spec.findFirst('samples').value
    self.macroName = spec.findFirst('macroParam').value
    self.microName = spec.findFirst('microParam').value
    self.variables = spec.findFirst('variables').value
    # checker; this should be superceded by "required" in input params
    if self.trainingName is None:
      self.raiseAnError(IOError, "Missing <training> node!")
    if self.samplesName is None:
      self.raiseAnError(IOError, "Missing <samples> node!")


  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, stepEntities, dict, contains all the Objects are going to be used in the
                                current step. The sources are searched into this.
      @ Out, None
    """
    train = self.findSource(self.trainingName, stepEntities)
    if train is None:
      self.raiseAnError(IOError, f'No input named "{self.trainingName}" was found in the Step for Plotter "{self.name}"!')
    if train.isEmpty:
      self.raiseAnError(IOError, f'Data object "{self.trainingName}" is empty!')
    self.training = train
    sample = self.findSource(self.samplesName, stepEntities)
    if sample is None:
      self.raiseAnError(IOError, f'No input named "{self.samplesName}" was found in the Step for Plotter "{self.name}"!')
    if sample.isEmpty:
      self.raiseAnError(IOError, f'Data object "{self.samplesName}" is empty!')
    if self.clusterName in sample.getVars():
      self.raiseAnError(IOError, f'Data object "{self.samplesName}" uses clusters! For this plotting, please take full samples.')
    self.samples = sample

  def run(self):
    """
      Main run method.
      @ In, None
      @ Out, None
    """
    tTag = self.training.sampleTag
    sTag = self.samples.sampleTag
    training = self.training.asDataset()
    samples = self.samples.asDataset()
    alpha = max(0.05, .5/len(samples))
    varNames = self.variables
    numVars = len(varNames)
    # use the len of macro, cluster from samples to build plots
    macro = samples[self.macroName]
    figCounter = 0
    for m, mac in enumerate(macro):
      figCounter += 1
      fig, axes = plt.subplots(numVars, 1, sharex=True)
      if numVars == 1:
        axes = [axes]
      axes[-1].set_xlabel(self.microName)

      mSamples = samples.sel({self.macroName: mac}, drop=True)
      mTraining = None
      if self.macroName in training:
        if int(mac) in training[self.macroName]:
          if self.macroName in training.dims:
            mTraining = training.drop_sel({self.macroName: mac})
          else:
            mTraining = training.where(training[self.macroName]==mac, drop=True).squeeze()
      for v, var in enumerate(varNames):
        ax = axes[v]
        # plot cloud of sample data
        for s in mSamples[sTag].values:
          samp = mSamples[{sTag: s}]
          ax.plot(samp[self.microName].values, samp[var].values, 'b-.', alpha=alpha)
        ax.set_title(f'{var}, {self.macroName} {int(mac)}')
        ax.set_ylabel(var)
        if mTraining is not None:
          ax.plot(mTraining[self.microName].values, mTraining[var].values, 'k-.')

      filename =  f'{self.name}_{m}.png'
      plt.savefig(filename)
      self.raiseAMessage(f'Wrote "{filename}".')

