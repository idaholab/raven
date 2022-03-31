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
from ... import Distributions
from ...Samplers import AdaptiveSampler
from ...utils import utils,randomUtils,InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class MCMC(AdaptiveSampler):
  """
    Markov Chain Monte Carlo Sampler.
  """

  @classmethod
  def userManualDescription(cls):
    """
      Provides a user manual description for this actor. Should only be needed for base classes.
      @ In, None
      @ Out, descr, string, description
    """
    descr = r"""
      \subsection{Markov Chain Monte Carlo}
      \label{subsec:MCMC}
      The Markov chain Monte Carlo (MCMC) is a Sampler entity in the RAVEN framework.
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
        descr=r"""The initial seed for random number generator""")
    samplerInitInput.addSub(initialSeedInput)
    burnInInput = InputData.parameterInputFactory("burnIn", contentType=InputTypes.IntegerType,
        descr=r"""The number of samples that will be discarded""")
    samplerInitInput.addSub(burnInInput)
    tune = InputData.parameterInputFactory("tune", contentType=InputTypes.BoolType,
        descr=r"""The option to tune the scaling parameter""")
    samplerInitInput.addSub(tune)
    tuneInterval = InputData.parameterInputFactory("tuneInterval", contentType=InputTypes.IntegerType,
        descr=r"""The number of sample steps for each tuning of scaling parameter""")
    samplerInitInput.addSub(tuneInterval)
    inputSpecification.addSub(samplerInitInput)
    likelihoodInp = InputData.parameterInputFactory("likelihood",contentType=InputTypes.StringType,
        printPriority=5,
        descr=r"""Output of likelihood function""")
    likelihoodInp.addParam('log', InputTypes.BoolType, required=False,
        descr=r"""True if the user provided is the log likelihood, otherwise, treat it as
        the standard likelihood""")
    inputSpecification.addSub(likelihoodInp)
    # modify Sampler variable nodes
    variable = inputSpecification.getSub('variable')
    variable.addSub(InputData.parameterInputFactory('initial', contentType=InputTypes.FloatType,
        descr=r"""inital value for given variable"""))
    proposal = InputData.assemblyInputFactory('proposal', contentType=InputTypes.StringType, strictMode=True,
        printPriority=30,
        descr=r"""name of the Distribution that is used as proposal distribution""")
    proposal.addParam('dim', InputTypes.IntegerType, required=False,
        descr=r"""for an ND proposal distribution, indicates the dimension within the ND Distribution that corresponds
              to this variable""")
    variable.addSub(proposal)
    variable.addSub(InputData.assemblyInputFactory('probabilityFunction', contentType=InputTypes.StringType, strictMode=True,
        printPriority=30,
        descr=r"""name of the function that is used as prior distribution (doesn't need to be normalized)"""))
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
           '{VAR}': r'any variable from the \xmlNode{TargetEvaluation} input or output at current iteration',
           'LogPosterior': 'log-posterior distribution value',
           'AcceptRate': 'the accept rate of MCMC algorithm'
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
    self.onlySampleAfterCollecting = True
    self._initialValues = {} # dict stores the user provided initial values, i.e. {var: val}
    self._updateValues = {} # dict stores input variables values for the current MCMC iteration, i.e. {var:val}
    self._proposal = {} # dict stores the proposal distributions for input variables, i.e. {var:dist}
    self._proposalDist = {} # dist stores the input variables for each proposal distribution, i.e. {distName:[(var,dim)]}
    self._priorFuns = {} # dict stores the prior functions for input variables, i.e. {var:fun}
    self._burnIn = 0      # integers indicate how many samples will be discarded
    self._likelihood = None # stores the output from the likelihood
    self._logLikelihood = False # True if the user provided likelihood is in log format
    self._availProposal = {'normal': Distributions.Normal,
                           'multivariateNormal': Distributions.MultivariateNormal} # available proposal distributions
    self._acceptDist = Distributions.Uniform(0.0, 1.0) # uniform distribution for accept/rejection purpose
    self.toBeCalibrated = {} # parameters that will be calibrated
    self._correlated = False # True if input variables are correlated else False
    self.netLogPosterior = 0.0 # log-posterior vs iteration
    self._localReady = True # True if the submitted job finished
    self._currentRlz = None # dict stores the current realizations, i.e. {var: val}
    self._acceptRate = 1. # The accept rate for MCMC
    self._acceptCount = 1 # The total number of accepted samples
    self._tune = True # Tune the scaling parameter if True
    self._tuneInterval = 100 # the number of sample steps for each tuning of scaling parameter
    self._scaling = 1.0 # The initial scaling parameter
    self._countsUntilTune = self._tuneInterval # The remain number of sample steps until the next tuning
    self._acceptInTune = 0 # The accepted number of samples for given tune interval
    self._accepted = False # The indication of current samples, True if accepted otherwise False
    self._stdProposalDefault = 0.2 # the initial scaling of the std of proposal distribution (only apply to default)
    # assembler objects
    self.addAssemblerObject('proposal', InputData.Quantity.zero_to_infinity)
    self.addAssemblerObject('probabilityFunction', InputData.Quantity.zero_to_infinity)

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Method that serves as a pass-through for input reading.
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
      self._logLikelihood = likelihood.parameterValues.get('log', False)
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
      burnIn = init.findFirst('burnIn')
      if burnIn is not None:
        self._burnIn = burnIn.value
      tune = init.findFirst('tune')
      if tune is not None:
        self._tune = tune.value
      tuneInterval = init.findFirst('tuneInterval')
      if tuneInterval is not None:
        self._tuneInterval = tuneInterval.value
    else:
      self.raiseAnError(IOError, 'MCMC', self.name, 'needs the samplerInit block')
    if self._burnIn >= self.limit:
      self.raiseAnError(IOError, 'Provided "burnIn" value must be less than "limit" value!')
    # TargetEvaluation Node (Required)
    targetEval = paramInput.findFirst('TargetEvaluation')
    self._targetEvaluation = targetEval.value
    self._updateValues = copy.copy(self._initialValues)

  def _readInVariable(self, child, prefix):
    """
      Reads in a "variable" input parameter node.
      This method is called by "_readMoreXMLbase" method within base class "Sampler".
      @ In, child, utils.InputData.ParameterInput, input parameter node to read from
      @ In, prefix, str, pass through parameter (not used), i.e. empty string. It is used
        by Sampler base class to indicate "Distribution"
      @ Out, None
    """
    assert prefix == "", '"prefix" should be empty, but got {}!'.format(prefix)
    varName = child.parameterValues['name']
    foundDist = child.findFirst('distribution')
    foundFunc = child.findFirst('function')
    foundPrior = child.findFirst('probabilityFunction')
    if (foundDist and foundFunc) or (foundDist and foundPrior) or (foundFunc and foundPrior):
      self.raiseAnError(IOError, 'Sampled variable "{}" can only have one node among "distribution, function, \
        probabilityFunction", more than one of them are provided. Please check your input!'.format(varName))
    elif not (foundDist or foundFunc or foundPrior):
      self.raiseAnError(IOError, 'Sampled variable "{}" requires only one node among "distribution, function, \
        probabilityFunction", but none of them is provided. Please check your input!'.format(varName))
    # set shape if present
    if 'shape' in child.parameterValues:
      self.variableShapes[varName] = child.parameterValues['shape']
    # read subnodes
    for childChild in child.subparts:
      if childChild.getName() == 'distribution':
        # name of the distribution to sample
        toBeSampled = childChild.value
        varData = {}
        varData['name'] = childChild.value
        # variable dimensionality
        if 'dim' not in childChild.parameterValues:
          dim = 1
          if self._correlated:
            self.raiseAnError(IOError, 'Found both correlated and uncorrelated variables in "{}" Samplers!'.format(self.type),
                             'Please check your input, all variables should either be uncorrelated or correlated!')
        else:
          dim = childChild.parameterValues['dim']
          if not self._correlated:
            self._correlated = True
        varData['dim'] = dim
        # set up mapping for variable to distribution
        self.variables2distributionsMapping[varName] = varData
        # flag distribution as needing to be sampled
        self.toBeSampled[varName] = toBeSampled
        self.toBeCalibrated[varName] = toBeSampled
        if varName not in self._initialValues:
          self._initialValues[varName] = None
      elif childChild.getName() == 'function':
        # function name
        toBeSampled = childChild.value
        # track variable as a functional sample
        self.dependentSample[varName] = toBeSampled
      elif childChild.getName() == 'initial':
        self._initialValues[varName] = childChild.value
      elif childChild.getName() == 'proposal':
        distName = childChild.value
        dim = childChild.parameterValues.get('dim', None)
        self._proposal[varName] = distName
        if distName not in self._proposalDist:
          self._proposalDist[distName] = [(varName, dim)]
        else:
          self._proposalDist[distName].append((varName, dim))
      elif childChild.getName() == 'probabilityFunction':
        toBeSampled = childChild.value
        self.toBeCalibrated[varName] = toBeSampled
        self._priorFuns[varName] = toBeSampled
        if varName not in self._initialValues:
          self._initialValues[varName] = None

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean MCMC is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    self._acceptDist.initializeDistribution()
    AdaptiveSampler.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    ## TODO: currently AdaptiveSampler is still using self.assemblerDict to retrieve the target evaluation.
    # We should change it to using the following method.
    # retrieve target evaluation
    # self._targetEvaluation = self.retrieveObjectFromAssemblerDict('TargetEvaluation', self._targetEvaluation)
    for var, priorFun in self._priorFuns.items():
      self._priorFuns[var] = self.retrieveObjectFromAssemblerDict('probabilityFunction', priorFun)
      if "pdf" not in self._priorFuns[var].availableMethods():
        self.raiseAnError(IOError,'Function', self._priorFuns[var], 'does not contain a method named "pdf". \
          It must be present if this needs to be used in a MCMC Sampler!')
        if not self._initialValues[var]:
          self.raiseAnError(IOError, '"initial" is required when using "probabilityFunction", but not found \
            for variable "{}"'.format(var))
      # initialize the input variable values
    for var, dist in self.distDict.items():
      distType = dist.getDistType()
      if distType != 'Continuous':
        self.raiseAnError(IOError, 'variable "{}" requires continuous distribution, but "{}" is provided!'.format(var, distType))

    meta = ['LogPosterior', 'AcceptRate']
    self.addMetaKeys(meta)

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
      if key in self.distDict:
        self.inputInfo['SampledVarsPb'][key] = self.distDict[key].pdf(value)
      else:
        self.inputInfo['SampledVarsPb'][key] = self._priorFuns[key].evaluate("pdf", self._updateValues)
      self.inputInfo['ProbabilityWeight-' + key] = 1.
    self.inputInfo['PointProbability'] = 1.0
    self.inputInfo['ProbabilityWeight'] = 1.0
    self.inputInfo['LogPosterior'] = self.netLogPosterior
    self.inputInfo['AcceptRate'] = self._acceptRate

  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      General function (available to all samplers) that finalizes the sampling
      calculation just ended. In this case, The function is aimed to check if
      all the batch calculations have been performed
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ Out, None
    """
    self._localReady = True
    AdaptiveSampler.localFinalizeActualSampling(self, jobObject, model, myInput)
    prefix = jobObject.getMetadata()['prefix']
    full = self._targetEvaluation.realization(index=self.counter-1)
    rlz = dict((var, full[var]) for var in (list(self.toBeCalibrated.keys()) + [self._likelihood] + list(self.dependentSample.keys())))
    rlz['traceID'] = self.counter
    rlz['LogPosterior'] = self.inputInfo['LogPosterior']
    rlz['AcceptRate'] = self.inputInfo['AcceptRate']
    if self.counter == 1:
      self._addToSolutionExport(rlz)
      self._currentRlz = rlz
    if self.counter > 1:
      alpha = self._useRealization(rlz, self._currentRlz)
      self.netLogPosterior = alpha
      self._accepted = self._checkAcceptance(alpha)
      if self._accepted:
        self._currentRlz = rlz
        self._addToSolutionExport(rlz)
        self._updateValues = dict((var, rlz[var]) for var in self._updateValues)
      else:
        self._currentRlz.update({'traceID':self.counter, 'LogPosterior': self.inputInfo['LogPosterior'], 'AcceptRate':self.inputInfo['AcceptRate']})
        self._addToSolutionExport(self._currentRlz)
        self._updateValues = dict((var, self._currentRlz[var]) for var in self._updateValues)
    if self._tune:
      self._acceptInTune = self._acceptInTune + 1 if self._accepted else self._acceptInTune
      self._countsUntilTune -= 1
    ## tune scaling parameter
    if not self._countsUntilTune and self._tune:
      ### tune
      self._scaling = self.tuneScalingParam(self._scaling, self._acceptInTune/float(self._tuneInterval))
      ### reset counter
      self._countsUntilTune = self._tuneInterval
      self._acceptInTune = 0

  @abc.abstractmethod
  def _useRealization(self, newRlz, currentRlz):
    """
      Used to feedback the collected runs within the sampler
      @ In, newRlz, dict, new generated realization
      @ In, currentRlz, dict, the current existing realization
      @ Out, netLogPosterior, float, the accepted probabilty
    """

  def _checkAcceptance(self, alpha):
    """
      Method to check the acceptance
      @ In, alpha, float, the accepted probabilty
      @ Out, acceptable, bool, True if we accept the new sampled point
    """
    acceptValue = np.log(self._acceptDist.rvs())
    acceptable = alpha > acceptValue
    if acceptable:
      self._acceptCount += 1
    self._acceptRate = self._acceptCount/self.counter
    return acceptable

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

  def _addToSolutionExport(self, rlz):
    """
      add realizations to solution export
      @ In, rlz, dict, sampled realization
      @ Out, None
    """
    if self._burnIn < self.counter:
      rlz = dict((var, np.atleast_1d(val)) for var, val in rlz.items())
      self._solutionExport.addRealization(rlz)

  @staticmethod
  def tuneScalingParam(scale, acceptRate):
    """
      Tune the scaling parameter for the proposal distribution
      Borrowed from PyMC3
      @ In, scale, float, the scaling parameter for the proposal distribution
      @ In, acceptRate, float, the accept rate
      @ Out, scale, float, the adjusted scaling parameter
    """
    if acceptRate < 0.001:
      scale *= 0.1
    elif acceptRate < 0.05:
      scale *= 0.5
    elif acceptRate < 0.2:
      scale *= 0.9
    elif acceptRate > 0.95:
      scale *= 10.0
    elif acceptRate > 0.75:
      scale *= 2.0
    elif acceptRate > 0.5:
      scale *= 1.1
    return scale
