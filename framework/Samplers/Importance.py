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
    MONTE CARLO Importance Sampler. This is currently a uni-variate sampler. Support for this sampling
    method currently includes one target distribution and one "importance" distribution for weighting.
    Future iterations of this class may include functionality for multivariate distributions.
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
    samplerInitInput = InputData.parameterInputFactory("samplerInit")
    limit = InputData.parameterInputFactory("limit", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(limit)

    constantInput = InputData.parameterInputFactory("ConstantSource")
    importanceVariableName = InputData.parameterInputFactory("importanceVariable", contentType=InputTypes.IntegerType)
    constantInput.addSub(importanceVariableName)

    targetDistribution = InputData.parameterInputFactory("distribution")
    samplerInitInput.addSub(targetDistribution)
    initialSeedInput = InputData.parameterInputFactory("initialSeed", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(initialSeedInput)
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
    self.importance = None

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    ForwardSampler.readSamplerInit(self,xmlNode)

    if paramInput.findFirst('samplerInit') != None:
      if self.limit is None:
        self.raiseAnError(IOError,self,'Importance sampler '+self.name+' needs the limit block (number of samples) in the samplerInit block')
    else:
      self.raiseAnError(IOError,self,'Importance sampler '+self.name+' needs the samplerInit block')

    # input check

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    distributions = [key for key in self.distDict.keys()]
    target = distributions[0]
    importance=distributions[1]

    importanceSample = self.distDict[importance].rvs()

    importanceWeight = self.distDict[target].pdf(
      importanceSample) / self.distDict[importance].pdf(importanceSample)

    sample = importanceSample*importanceWeight
    key = target
    for kkey in key.split(','):
      self.values[kkey] = np.atleast_1d(sample)[0]

    self.inputInfo['ProbabilityWeight-'+importance] = importanceWeight
    self.inputInfo['ProbabilityWeight-'+target] = self.distDict['target'].pdf(importanceSample)
    self.inputInfo['PointProbability'] = self.distDict[target].pdf(importanceSample)
    self.inputInfo['ProbabilityWeight'] = 0
    self.inputInfo['SamplerType'] = 'Importance'