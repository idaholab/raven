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
  This module contains the Monte Carlo sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from crisr
"""
# External Modules----------------------------------------------------------------------------------
from operator import mul
from functools import reduce
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .Sampler import Sampler
from ..utils import utils,randomUtils,InputData, InputTypes
# Internal Modules End------------------------------------------------------------------------------

class MonteCarlo(Sampler):
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
    inputSpecification = super(MonteCarlo, cls).getInputSpecification()

    samplerInitInput = InputData.parameterInputFactory("samplerInit")
    limitInput = InputData.parameterInputFactory("limit", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(limitInput)
    initialSeedInput = InputData.parameterInputFactory("initialSeed", contentType=InputTypes.IntegerType)
    samplerInitInput.addSub(initialSeedInput)
    distInitInput = InputData.parameterInputFactory("distInit", contentType=InputTypes.StringType)
    distSubInput = InputData.parameterInputFactory("distribution")
    distSubInput.addParam("name", InputTypes.StringType)
    distSubInput.addSub(InputData.parameterInputFactory("initialGridDisc", contentType=InputTypes.IntegerType))
    distSubInput.addSub(InputData.parameterInputFactory("tolerance", contentType=InputTypes.FloatType))

    distInitInput.addSub(distSubInput)
    samplerInitInput.addSub(distInitInput)
    samplingTypeInput = InputData.parameterInputFactory("samplingType", contentType=InputTypes.StringType)
    samplerInitInput.addSub(samplingTypeInput)
    reseedEachIterationInput = InputData.parameterInputFactory("reseedEachIteration", contentType=InputTypes.StringType)
    samplerInitInput.addSub(reseedEachIterationInput)

    inputSpecification.addSub(samplerInitInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'SAMPLER MONTECARLO'
    self.samplingType = None
    self.limit = None

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    # TODO remove using xmlNode
    Sampler.readSamplerInit(self, xmlNode)
    if paramInput.findFirst('samplerInit') is not None:
      if self.limit is None:
        self.raiseAnError(IOError,self, f'Monte Carlo sampler {self.name} needs the limit block (number of samples) in the samplerInit block')
      if paramInput.findFirst('samplerInit').findFirst('samplingType') is not None:
        self.samplingType = paramInput.findFirst('samplerInit').findFirst('samplingType').value
        if self.samplingType not in ['uniform']:
          self.raiseAnError(IOError, self, f'Monte Carlo sampler {self.name}: specified type of samplingType is not recognized. Allowed type is: uniform')
      else:
        self.samplingType = None
    else:
      self.raiseAnError(IOError, self, f'Monte Carlo sampler {self.name} needs the samplerInit block')

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    # create values dictionary
    weight = 1.0
    for key in sorted(self.distDict):
      # check if the key is a comma separated list of strings
      # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
      totDim = self.variables2distributionsMapping[key]['totDim']
      dist   = self.variables2distributionsMapping[key]['name']
      reducedDim = self.variables2distributionsMapping[key]['reducedDim']
      weight = 1.0
      if totDim == 1:
        if self.samplingType == 'uniform':
          distData = self.distDict[key].getCrowDistDict()
          if ('xMin' not in distData.keys()) or ('xMax' not in distData.keys()):
            self.raiseAnError(IOError,"In the Monte-Carlo sampler a uniform sampling type has been chosen;"
                   + " however, one or more distributions have not specified either the lowerBound or the upperBound")
          lower = distData['xMin']
          upper = distData['xMax']
          rvsnum = lower + (upper - lower) * randomUtils.random()
          # TODO (wangc): I think the calculation for epsilon need to be updated as following
          # epsilon = (upper-lower)/(self.limit+1) * 0.5
          epsilon = (upper-lower)/self.limit
          midPlusCDF  = self.distDict[key].cdf(rvsnum + epsilon)
          midMinusCDF = self.distDict[key].cdf(rvsnum - epsilon)
          weight *= midPlusCDF - midMinusCDF
        else:
          rvsnum = self.distDict[key].rvs()
        for kkey in key.split(','):
          self.values[kkey] = np.atleast_1d(rvsnum)[0]
        self.inputInfo['SampledVarsPb'][key] = self.distDict[key].pdf(rvsnum)
        self.inputInfo['ProbabilityWeight-' + key] = 1.
      elif totDim > 1:
        if reducedDim == 1:
          if self.samplingType is None:
            rvsnum = self.distDict[key].rvs()
            coordinate = np.atleast_1d(rvsnum).tolist()
          else:
            coordinate = np.zeros(totDim)
            for i in range(totDim):
              lower = self.distDict[key].returnLowerBound(i)
              upper = self.distDict[key].returnUpperBound(i)
              coordinate[i] = lower + (upper - lower) * randomUtils.random()
          if reducedDim > len(coordinate):
            self.raiseAnError(IOError, "The dimension defined for variables drew from the multivariate normal distribution is exceeded by the dimension used in Distribution (MultivariateNormal) ")
          probabilityValue = self.distDict[key].pdf(coordinate)
          self.inputInfo['SampledVarsPb'][key] = probabilityValue
          for var in self.distributions2variablesMapping[dist]:
            varID  = utils.first(var.keys())
            varDim = var[varID]
            for kkey in varID.strip().split(','):
              self.values[kkey] = np.atleast_1d(rvsnum)[varDim-1]
          self.inputInfo[f'ProbabilityWeight-{dist}'] = 1.
      else:
        self.raiseAnError(IOError, "Total dimension for given distribution should be >= 1")

    if len(self.inputInfo['SampledVarsPb'].keys()) > 0:
      self.inputInfo['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    else:
      self.inputInfo['PointProbability'] = 1.0
    if self.samplingType == 'uniform':
      self.inputInfo['ProbabilityWeight'  ] = weight
    else:
      self.inputInfo['ProbabilityWeight' ] = 1.0 # MC weight is 1/N => weight is one
    self.inputInfo['SamplerType'] = 'MonteCarlo'

  def _localHandleFailedRuns(self, failedRuns):
    """
      Specialized method for samplers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns) > 0:
      self.raiseADebug('  Continuing with reduced-size Monte-Carlo sampling.')
