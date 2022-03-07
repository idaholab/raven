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
Created on August 3, 2021

@author: talbpaul
"""
import numpy as np

from .PostProcessorInterface import PostProcessorInterface
from ...utils import InputData, InputTypes
from ...TSA import TSAUser

class TSACharacterizer(PostProcessorInterface, TSAUser):
  """
    Maps realizations from contiguous time histories to characterization space using TSA algorithms.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, input specification
    """
    specs = super().getInputSpecification()
    cls.addTSASpecs(specs, subset='characterize')
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    PostProcessorInterface.__init__(self)
    TSAUser.__init__(self)

  def _handleInput(self, inp):
    """
      Function to handle the parsed paramInput for this class.
      @ In, inp, InputData.parameterInput, the already-parsed input.
      @ Out, None
    """
    super()._handleInput(inp)
    self.readTSAInput(inp)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the PostProcessor
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    # register metadata variable names
    names = list(sorted(self.getCharacterizingVariableNames()))
    self.addMetaKeys(names)

  def checkInput(self, currentInp):
    """
      Check provided input.
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) > 1:
      self.raiseAnError(IOError, 'Expected 1 input DataObject, but received {} inputs!'.format(len(currentInp)))
    currentInp = currentInp[0]
    if currentInp.type not in ['HistorySet']:
      self.raiseAnError(IOError, f'TSACharacterizer postprocessor "{self.name}" requires a HistorySet input! ' +
                                 f'Got "{currentInp.type}".')
    return currentInp

  def run(self, inp):
    """
      This method executes the postprocessor action.
      @ In, inp, object, object contained the data to process.
      @ Out, rlzs, list, list of realizations obtained
    """
    self.checkInput(inp)
    inp = inp[0]
    rlzs = []
    for r, rlz in enumerate(inp.sliceByIndex(inp.sampleTag)):
      self.raiseADebug(f'Characterizing realization {r} ...')
      self._tsaReset()
      pivots = rlz[self.pivotParameterID]
      targetVals = np.zeros((1, len(pivots), len(self.target))) # shape: (rlzs, time, targets)
      for t, target in enumerate(self.target):
        targetVals[0, :, t] = rlz[target]
      self.trainTSASequential(targetVals)
      rlz = self.getParamsAsVars()
      rlzs.append(rlz)
    return rlzs

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    rlzs = evaluation[1]
    for r, rlz in enumerate(rlzs):
      for key, value in rlz.items():
        rlz[key] = np.atleast_1d(value)
      output.addRealization(rlz)
