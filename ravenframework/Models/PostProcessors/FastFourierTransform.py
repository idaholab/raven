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
Created on September 11, 2018

@author: talbpaul
"""
#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from ...utils import InputData, InputTypes
from .PostProcessorInterface import PostProcessorInterface
#Internal Modules End-----------------------------------------------------------

class FastFourierTransform(PostProcessorInterface):
  """
    Constructs fast-fourier transform data for a history
    Outputs are "frequency" for each index and "amplitude" for each target
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
    inSpec = super(FastFourierTransform, cls).getInputSpecification()
    inSpec.addSub(InputData.parameterInputFactory('target',
                                                  contentType=InputTypes.StringListType,
                                                  strictMode=True))
    return inSpec

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dynamic = True # from base class, indicates time-dependence is handled internally
    self.targets = None # list of strings, variables to apply postprocessor to
    self.indices = None # dict of {string:string}, key is target and value is index (independent monotonic var)

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

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      In this case, we only want data objects!
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) > 1:
      self.raiseAnError(IOError, 'Expected 1 input HistorySet or DataSet, but received {} inputs!'.format(len(currentInp)))
    currentInp = currentInp[0]
    if currentInp.type not in ['HistorySet','DataSet']:
      self.raiseAnError(IOError, 'FastFourierTransform postprocessor "{}" requires a HistorySet or DataSet input! Got "{}".'
                                 .format(self.name, currentInput.type))
    # check targets requested depend only on one index (no more or less)
    okay = True
    for target in self.targets:
      indices = currentInp.getDimensions(var=target)
      if len(indices) != 1:
        okay = False
        if len(indices) == 0:
          self.raiseAWarning('Target "{}" is a scalar! FFT cannot be performed.'.format(target))
        else:
          self.raiseAWarning('Target "{}" has {} index dimensions! FFT can only currently handle one.'.format(target,len(indices)))
    if not okay:
      self.raiseAnError(IndexError,'Some targets were not dimensioned correctly. See warnings for more details. Exiting.')
    # return data object
    return currentInp

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, realizations, list, list of realizations obtained
    """
    # do checking and isolate input
    inData = self.inputToInternal(inputIn)

    # storage for each realization
    realizations = []
    # loop over realizations and perform fft
    ## TODO can the loop be avoided?
    for s in range(len(inData)):
      # obtain relevant sample
      sample = inData.realization(index=s)
      # solutions for each realization are stored in this dict
      rlz = {}
      # separate curve for each target
      for target in self.targets:
        # get complex ndarray (real and imaginary parts)
        data = sample[target].values
        N = len(data)
        mixed = np.fft.fft(data)
        freq = np.fft.fftfreq(N)
        # select positive values
        freq = freq[:int(N/2)]
        mixed = mixed[:int(N/2)].real
        #data = zip(freq,1.0/freq,mixed.real)
        #data.sort(key=lambda x:abs(x[2]), reverse=True)
        #freq,period,amp = zip(*data)
        # TODO change frequencies based on delta index? Example: time
        rlz[target+'_fft_frequency'] = freq
        rlz[target+'_fft_period'] = 1.0/freq
        rlz[target+'_fft_amplitude'] = mixed
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
