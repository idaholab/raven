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
  This module contains the Umbrella sampling strategy

  Created on Feb 15, 2021
  @author: Tanaya
  supercedes Samplers.py from crisr
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
from operator import mul
from functools import reduce
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .ForwardSampler import ForwardSampler
from utils import utils,randomUtils,InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class Importance(ForwardSampler):
  """
    MONTE CARLO Sampler
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
    inputSpecification = super(Importance, cls).getInputSpecification()
    ImportanceInput = InputData.parameterInputFactory("Importance", contentType=InputTypes.StringType)
    samplerInitInput = InputData.parameterInputFactory("samplerInit")
    limit = InputData.parameterInputFactory("limit", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(limit)
    targetDistribution = InputData.parameterInputFactory("distribution")
    samplerInitInput.addSub(targetDistribution)
    samplingTypeInput = InputData.parameterInputFactory("samplingType", contentType=InputTypes.StringType)
    samplerInitInput.addSub(samplingTypeInput)
    inputSpecification.addSub(samplerInitInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    ForwardSampler.__init__(self)
    self.printTag = 'SAMPLER UMBRELLA'
    self.samplingType = None
    self.limit = None

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    ForwardSampler.readSamplerInit(self,xmlNode)
    # if paramInput.findFirst('samplerInit') != None:
    #   if self.limit is None:
    #     self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+' needs the limit block (number of samples) in the samplerInit block')
    #   if paramInput.findFirst('samplerInit').findFirst('samplingType')!= None:
    #     self.samplingType = paramInput.findFirst('samplerInit').findFirst('samplingType').value
    #     if self.samplingType not in ['uniform']:
    #       self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+': specified type of samplingType is not recognized. Allowed type is: uniform')
    #   else:
    #     self.samplingType = None
    # else:
    #   self.raiseAnError(IOError,self,'Umbrella sampler '+self.name+' needs the samplerInit block')

  def stratified_uniform_sample(self, m_per_bin, n_bins):
    """
    function to generate a uniform stratified sample
    :param m_per_bin:
    :param n_bins:
    :return:
    """
    print("In importance sampling")
    n = n_bins
    m = m_per_bin
    samples = []
    bounds = [num / n for num in range(0, n + 1)]

    for i in range(0, n):
      LB = bounds[i]
      UB = bounds[i + 1]
      print("IN IMPORTANCE SAMPLING!!!!!!")
      print(LB, UB)
      bin_sample = np.random.uniform(LB, UB, m)

      samples[(i * m + 1):((i + 1) * m)] = bin_sample
      print(samples)
    return bounds[1:], samples


  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    n_bins = 1
    m_per_bin = 1

    uSample = np.random.uniform(0, 1, 1)
    importanceSample = self.distDict['importance'].ppf(uSample[0])

    importanceWeight = self.distDict['target'].pdf(
      importanceSample) / self.distDict['importance'].pdf(importanceSample)

    self.inputInfo['SampledVars']['sample'] = importanceSample
    self.inputInfo['ProbabilityWeight'] = 1.0
    self.inputInfo['ProbabilityWeight-target'] = self.distDict['target'].pdf(importanceSample)
    self.inputInfo['ProbabilityWeight-importance'] = importanceWeight
    self.inputInfo['PointProbability'] = importanceWeight
    self.inputInfo['SamplerType'] = 'Importance'
    # print(self.inputInfo)
