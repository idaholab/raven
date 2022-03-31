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
Created on August 28, 2018

@author: talbpaul
"""
#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import utils
from ...utils import InputData, InputTypes
#Internal Modules End-----------------------------------------------------------

class ValueDuration(PostProcessorInterface):
  """
    Constructs a load duration curve.
    x-axis is time spent above a particular variable's value,
    y-axis is the value of the variable.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inSpec= super(ValueDuration, cls).getInputSpecification()
    inSpec.addSub(InputData.parameterInputFactory('target',
                                                  contentType=InputTypes.StringListType,
                                                  strictMode=True))
    inSpec.addSub(InputData.parameterInputFactory('bins',
                                                  contentType=InputTypes.IntegerType))
    return inSpec

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dynamic = True # from base class, indicates time-dependence is handled internally
    self.numBins = None # integer number of bins to use in creating the duration curve. TODO default?
    self.targets = None # list of strings, variables to apply postprocessor to

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      tag = child.getName()
      if tag == 'target':
        self.targets = set(child.value)
      elif tag == 'bins':
        self.numBins = child.value

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      In this case, we only want data objects!
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) > 1:
      self.raiseAnError(IOError, 'Expected 1 input HistorySet, but received {} inputs!'.format(len(currentInp)))
    currentInp = currentInp[0]
    if currentInp.type not in ['HistorySet']:
      self.raiseAnError(IOError, 'ValueDuration postprocessor "{}" requires a HistorySet input! Got "{}".'
                                 .format(self.name, currentInput.type))
    return currentInp

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, realizations, list, list of realizations obtained
    """
    inData = self.inputToInternal(inputIn)
    realizations = []
    # new load duration curve for each sample
    for s in range(len(inData)):
      # obtain relevant sample
      sample = inData.realization(index=s)
      # solutions for each realization are stored in this dict
      rlz = {}
      # separate curve for each target
      for target in self.targets:
        data = sample[target]
        # bin values
        counts, edges = np.histogram(data,self.numBins)
        ## reverse order of histogram, edges to get load duration axes
        counts = counts[::-1]
        edges = edges[::-1]
        ## cumulatively stack counts, starting with highest value bin
        cumulative = np.cumsum(counts)

        # store results
        rlz['counts_'+target] = cumulative
        ## only keep upper value of edges so lengths match
        rlz['bins_'+target] = edges[1:]

        realizations.append(rlz)

    return realizations

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    realizations = evaluation[1]
    for rlz in realizations:
      output.addRealization(rlz)
