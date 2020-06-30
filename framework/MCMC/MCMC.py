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
  Markov Chain Monte Carlo
  This base class defines the principle methods required for MCMC

  Created on June 26, 2020
  @author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import Distributions
from Samplers import AdaptiveSampler
from utils import utils,randomUtils,InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class MCMC(AdaptiveSampler):
  """
    Markov Chain Monte Carlo Sampler
  """
  @classmethod
  def userManualDescription(cls):
    """
      Provides a user manual description for this actor. Should only be needed for base classes.
      @ In, None
      @ Out, descr, string, description
    """
    descr = r"""
    \section{Markov Chain Monte Carlo} \label{sec:MCMC}
    The Markov chain Monte Carlo (MCMC) is another important entity in the RAVEN framework.
    It provides enormous scope for realistic statistical modeling. MCMC is essentially
    Monte Carlo integration using Markov chain. Bayesians, and sometimes also frequentists,
    need to integrate over possibly high-dimensional probability distributions to make inference
    about model parameters or to make predictions. Bayesians need to integrate over the posterior
    distributions of model parameters given the data, and frequentists may need to integrate
    over the distribution of observables given parameter values. Monte Carlo integration draws
    samples from the required distribution, and then forms samples averages to approximate expectations.
    MCMC draws these samples by running a cleverly constructed Markov chain for a long time.
    There are a large number of MCMC algorithms, and popular families include Gibbs sampling,
    Metropolis-Hastings, slice sampling, Hamiltonian Monte Carlo, and many others. Regardless
    of the algorithm, the goal in Bayesian inference is to maximize the unnormalized joint
    posterior distribution and collect samples of the target distributions, which are marginal
    posterior distributions, later to be used for inference.
    """
    return descr

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(MCMC, cls).getInputSpecification()

    samplerInitInput = InputData.parameterInputFactory("samplerInit", strictMode=True,
        printPriority=10,
        descr=r"""collection of nodes that describe the initialization of the MCMC algorithm.""")
    limitInput = InputData.parameterInputFactory("limit", contentType=InputTypes.IntegerType,
        descr=r"""the limit for the total samples""")
    samplerInitInput.addSub(limitInput)
    initialSeedInput = InputData.parameterInputFactory("initialSeed", contentType=InputTypes.IntegerType,
        descr='')
    samplerInitInput.addSub(initialSeedInput)
    tuneInput = InputData.parameterInputFactory("tune", contentType=InputTypes.IntegerType,
        descr='')
    samplerInitInput.addSub(tuneInput)
    inputSpecification.addSub(samplerInitInput)

    inputSpecification.addSub(InputData.parameterInputFactory("likelihood",contentType=InputTypes.StringType,
        printPriority=5,
        descr=r"""Output of likelihood function"""))
    # modify Sampler variable nodes
    variable = inputSpecification.getSub('variable')
    variable.addSub(InputData.parameterInputFactory('initial', contentType=InputTypes.FloatType,
        descr=r"""inital value for given variable"""))
    variable.addSub(InputData.assemblyInputFactory('proposal', contentType=InputTypes.StringType, strictMode=True,
        printPriority=30,
        descr=r"""name of the Distribution that is used as proposal distribution"""))

    inputSpecification.addSub(variable)
    return inputSpecification

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, vars, dict, {varName: manual description} for each solution export option
    """
    vars = super(AdaptiveSampler, cls).getSolutionExportVariableNames()
    # TODO: multiple chains for MCMC
    new = {'traceID': 'integer identifying which iteration a Markov chain is on',
           '{VAR}': r'any variable from the \xmlNode{TargetEvaluation} input or output at current iteration'
           }
    vars.update(new)
    return vars

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    AdaptiveSampler.__init__(self)
    self._initialValues = {}
    self._updateValues = {}
    self._currentValues = {}
    self._proposal = {}
    self._tune = 0
    self._likelihood = None
    self._availProposal = {'normal': Distributions.Normal(0.0, 1.0),
                           'uniform': Distributions.Uniform(-1.0, 1.0)}
    self._acceptDist = Distributions.Uniform(0.0, 1.0)
    self._currentRlz = None

    self.addAssemblerObject('proposal', '-n', True)

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      unfortunately-named method that serves as a pass-through for input reading.
      comes from inheriting from Sampler and _readMoreXML chain.
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node (don't use!)
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    # this is just a passthrough until sampler gets reworked or renamed
    self.handleInput(paramInput)

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    likelihood = paramInput.findFirst('likelihood')
    if likelihood is not None:
      self._likelihood = likelihood.value
    else:
      self.raiseAnError(IOError, "likelihood is required, but not provided!")
    init = paramInput.findFirst('samplerInit')
    if init is not None:
      # limit
      limit = init.findFirst('limit')
      if limit is not None:
        self.limit = limit.value
      else:
        self.raiseAnError(IOError, 'MCMC', self.name, 'needs the limit block (number of samples) in the samplerInit block')
      # initialSeed
      seed = init.findFirst('initialSeed')
      if seed is not None:
        self.initSeed = seed.value
      else:
        self.initSeed = randomUtils.randomIntegers(0,2**31,self)
      tune = init.findFirst('tune')
      if tune is not None:
        self._tune = tune.value
    else:
      self.raiseAnError(IOError, 'MCMC', self.name, 'needs the samplerInit block')
    # variables additional reading
    for varNode in paramInput.findAll('variable'):
      var = varNode.parameterValues['name']
      initNode = varNode.findFirst('initial')
      if initNode:
        self._initialValues[var] = initNode.value
      else:
        self._initialValues[var] = None
      proposal = varNode.findFirst('proposal')
      if proposal:
        self._proposal[var] = proposal.value
      else:
        self._proposal[var] = None
    # TargetEvaluation Node (Required)
    targetEval = paramInput.findFirst('TargetEvaluation')
    self._targetEvaluation = targetEval.value

    self._updateValues = copy.copy(self._initialValues)

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean MCMC is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    # TODO: currently, we only consider uncorrelated case
    # initialize distributions
    for _, dist in self._availProposal.items():
      dist.initializeDistribution()
    self._acceptDist.initializeDistribution()
    for var, dist in self._proposal.items():
      if dist:
        self._proposal[var] = self.retrieveObjectFromAssemblerDict('proposal', dist)
      else:
        self._proposal[var] = self._availProposal['normal']
    AdaptiveSampler.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    # retrieve target evaluation
    # self._targetEvaluation = self.retrieveObjectFromAssemblerDict('TargetEvaluation', self._targetEvaluation)

    # initialize the input variable values
    for key, value in self._updateValues.items():
      totDim = self.variables2distributionsMapping[key]['totDim']
      dist = self.distDict[key]
      if totDim != 1:
        self.raiseAnError(IOError,"Total dimension for given distribution {} should be 1".format(dist.type))
      if value is None:
        value = dist.rvs()
        self._updateValues[key] = value

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    for key, value in self._updateValues.items():
      self.values[key] = value
      self.inputInfo['SampledVarsPb'][key] = self.distDict[key].pdf(value)
      self.inputInfo['ProbabilityWeight-' + key] = 1.
    self.inputInfo['PointProbability'] = 1.0
    self.inputInfo['ProbabilityWeight' ] = 1.0

  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      General function (available to all samplers) that finalize the sampling
      calculation just ended. In this case, The function is aimed to check if
      all the batch calculations have been performed
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ Out, None
    """
    AdaptiveSampler.localFinalizeActualSampling(self, jobObject, model, myInput)

  def _addToSolutionExport(self, rlz):
    """
      add realizations to solution export
      @ In, rlz, dict, sampled realization
      @ Out, None
    """
    rlz = dict((var, np.atleast_1d(val)) for var, val in rlz.items())
    self._solutionExport.addRealization(rlz)

  def localStillReady(self, ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    ready = AdaptiveSampler.localStillReady(self, ready)
    return ready

  def _localHandleFailedRuns(self, failedRuns):
    """
      Specialized method for samplers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns)>0:
      self.raiseADebug('  Continuing with reduced-size MCMC sampling.')

  def _formatSolutionExportVariableNames(self, acceptable):
    """
      Does magic formatting for variables, based on this class's needs.
      Extend in inheritors as needed.
      @ In, acceptable, set, set of acceptable entries for solution export for this entity
      @ Out, acceptable, set, modified set of acceptable variables with all formatting complete
    """
    acceptable = AdaptiveSampler._formatSolutionExportVariableNames(self, acceptable)
    return acceptable
