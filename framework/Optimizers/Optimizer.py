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
  Module where the base class of optimizer is. Adapted from Sampler.py.

  Created on June 16, 2016
  @author: chenj
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import abc
import numpy as np
from collections import deque
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils,randomUtils,InputData
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
from Samplers import Sampler
#Internal Modules End--------------------------------------------------------------------------------

class Optimizer(Sampler):
  """
    This is the base class for optimizers
    Optimizer is a special type of "samplers" that own the optimization strategy (Type) and they generate the input values to optimize a loss function.
    The most significant deviation from the Samplers is that they do not use distributions.
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    #### FULL INPUT LIST SPSA #### -> * means part of FiniteDiff as well
    # so far, there's no distinction between optimizers, so all the input from SPSA is here.  Fix this when we sort out optimizers.

    inputSpecification = super(Optimizer,cls).getInputSpecification()
    # assembled objects
    targEval = InputData.parameterInputFactory('TargetEvaluation', contentType=InputData.StringType, strictMode=True)
    targEval.addParam('type', InputData.StringType, True)
    targEval.addParam('class', InputData.StringType, True)
    inputSpecification.addSub(targEval)
    sampler = InputData.parameterInputFactory('Sampler', contentType=InputData.StringType, strictMode=True)
    sampler.addParam('type', InputData.StringType, True)
    sampler.addParam('class', InputData.StringType, True)
    inputSpecification.addSub(sampler)
    function = InputData.parameterInputFactory('Function', contentType=InputData.StringType, strictMode=True)
    function.addParam('type', InputData.StringType, True)
    function.addParam('class', InputData.StringType, True)
    inputSpecification.addSub(function)
    precond = InputData.parameterInputFactory('Preconditioner', contentType=InputData.StringType, strictMode=True)
    precond.addParam('type', InputData.StringType, True)
    precond.addParam('class', InputData.StringType, True)
    inputSpecification.addSub(precond)

    # variable
    ## was also part of Sampler, but we need to rewrite variable, so remove it first
    inputSpecification.removeSub('variable')
    variable = InputData.parameterInputFactory('variable', strictMode=True)
    variable.addParam("name", InputData.StringType, True)
    variable.addParam("shape", InputData.IntegerListType, required=False)
    upperBound = InputData.parameterInputFactory('upperBound', contentType=InputData.FloatType, strictMode=True)
    lowerBound = InputData.parameterInputFactory('lowerBound', contentType=InputData.FloatType, strictMode=True)
    initial = InputData.parameterInputFactory('initial',contentType=InputData.StringListType)
    variable.addSub(upperBound)
    variable.addSub(lowerBound)
    variable.addSub(initial)
    inputSpecification.addSub(variable)
    # constant -> use the Sampler's specs.

    # objectVar
    objectVar = InputData.parameterInputFactory('objectVar', contentType=InputData.StringType, strictMode=True)
    inputSpecification.addSub(objectVar)

    # initialization
    init = InputData.parameterInputFactory('initialization', strictMode=True)
    whenWriteEnum = InputData.makeEnumType('whenWriteEnum','whenWriteType',['final','every'])
    limit      = InputData.parameterInputFactory('limit', contentType=InputData.IntegerType)
    seed       = InputData.parameterInputFactory('initialSeed', contentType=InputData.IntegerType)
    minmaxEnum = InputData.makeEnumType('MinMax','OptimizerTypeType',['min','max'])
    minmax     = InputData.parameterInputFactory('type', contentType=minmaxEnum)
    thresh     = InputData.parameterInputFactory('thresholdTrajRemoval', contentType=InputData.FloatType)
    write      = InputData.parameterInputFactory('writeSteps',contentType=whenWriteEnum)
    init.addSub(limit)
    init.addSub(seed)
    init.addSub(minmax)
    init.addSub(thresh)
    init.addSub(write)
    inputSpecification.addSub(init)

    # convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True)
    itLim   = InputData.parameterInputFactory('iterationLimit'   , contentType=InputData.IntegerType)
    pers    = InputData.parameterInputFactory('persistence'      , contentType=InputData.IntegerType)
    rel     = InputData.parameterInputFactory('relativeThreshold', contentType=InputData.FloatType  )
    abst    = InputData.parameterInputFactory('absoluteThreshold', contentType=InputData.FloatType  )
    grad    = InputData.parameterInputFactory('gradientThreshold', contentType=InputData.FloatType  )
    minstep = InputData.parameterInputFactory('minStepSize'      , contentType=InputData.FloatType  )
    grow    = InputData.parameterInputFactory('gainGrowthFactor' , contentType=InputData.FloatType  )
    shrink  = InputData.parameterInputFactory('gainShrinkFactor' , contentType=InputData.FloatType  )
    conv.addSub(itLim)
    conv.addSub(pers)
    conv.addSub(rel)
    conv.addSub(abst)
    conv.addSub(grad)
    conv.addSub(minstep)
    conv.addSub(grow)
    conv.addSub(shrink)
    inputSpecification.addSub(conv)

    # parameter
    param = InputData.parameterInputFactory('parameter', strictMode=True)
    stochEnum = InputData.makeEnumType('StochDistEnum','StochDistType',['Hypersphere','Bernoulli'])
    num    = InputData.parameterInputFactory('numGradAvgIterations'   , contentType=InputData.IntegerType)
    stoch  = InputData.parameterInputFactory('stochasticDistribution' , contentType=stochEnum            )
    bisect = InputData.parameterInputFactory('innerBisectionThreshold', contentType=InputData.FloatType  )
    loop   = InputData.parameterInputFactory('innerLoopLimit'         , contentType=InputData.IntegerType)
    param.addSub(num)
    param.addSub(stoch)
    param.addSub(bisect)
    param.addSub(loop)
    inputSpecification.addSub(param)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Sampler.__init__(self)
    #counters
    ## while "counter" is scalar in Sampler, it's more complicated in Optimizer
    self.counter                        = {}                        # Dict containing counters used for based and derived class
    self.counter['mdlEval']             = 0                         # Counter of the model evaluation performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.counter['varsUpdate']          = 0                         # Counter of the optimization iteration.
    self.counter['recentOptHist']       = {}                        # as {traj: [pt0, pt1]} where each pt is {'inputs':{var:val}, 'output':val}, the two most recently-accepted points by value
    self.counter['persistence'  ]       = {}                        # as {traj: n} where n is the number of consecutive converges
    #limits
    ## while "limit" is scalar in Sampler, it's more complicated in Optimizer
    self.limit                          = {}                        # Dict containing limits for each counter
    self.limit['mdlEval']               = 2000                      # Maximum number of the loss function evaluation
    self.limit['varsUpdate']            = 650                       # Maximum number of the optimization iteration.
    #variable lists
    self.objVar                         = None                      # Objective variable to be optimized
    self.optVars                        = {}                        # By trajectory, current decision variables for optimization
    self.fullOptVars                    = None                      # Decision variables for optimization, full space
    self.optTraj                        = None                      # Identifiers of parallel optimization trajectories
    #initialization parameters
    self.optVarsInit                    = {}                        # Dict containing upper/lower bounds and initial of each decision variables
    self.optVarsInit['upperBound']      = {}                        # Dict containing upper bounds of each decision variables
    self.optVarsInit['lowerBound']      = {}                        # Dict containing lower bounds of each decision variables
    self.optVarsInit['initial']         = {}                        # Dict containing initial values of each decision variables
    self.optVarsInit['ranges']          = {}                        # Dict of the ranges (min and max) of each variable's domain
    self.optType                        = None                      # Either max or min
    self.writeSolnExportOn              = None                      # Determines when we write to solution export (every step or final solution)
    self.paramDict                      = {}                        # Dict containing additional parameters for derived class
    self.initializationSampler          = None                      # Sampler that can be used to initialize the optimizer trajectories
    self.optVarsInitialized             = {}                        # Dict {var1:<initial> present?,var2:<initial> present?}
    #convergence tools
    self.optVarsHist                    = {}                        # History of normalized decision variables for each iteration
    self.thresholdTrajRemoval           = None                      # Threshold used to determine the convergence of parallel optimization trajectories
    self.absConvergenceTol              = 0.0                       # Convergence threshold (absolute value)
    self.relConvergenceTol              = 1.e-3                     # Convergence threshold (relative value)
    self.convergencePersistence         = 1                         # number of retries to attempt before accepting convergence
    # TODO REWORK minStepSize is for gradient-based specifically
    self.minStepSize                    = 1e-9                      # minimum allowable step size (in abs. distance, in input space)
    #sampler-step communication
    self.submissionQueue                = {}                        # by traj, a place (deque) to store points that should be submitted some time after they are discovered
    #functions and dataojbects
    self.constraintFunction             = None                      # External constraint function, could be not present
    self.preconditioners                = {}                        # by name, Models that might be used as preconditioners
    self.solutionExport                 = None                      # This is the data used to export the solution
    self.mdlEvalHist                    = None                      # Containing information of all model evaluation
    self.objSearchingROM                = None                      # ROM used internally for fast loss function evaluation
    #stateful tracking
    self.recommendedOptPoint            = {}                        # by traj, the next recommended point (as a dict) in the input space to move to
    self.nextActionNeeded               = (None,None)               # tool for localStillReady to inform localGenerateInput on the next action needed
    self.status                         = {}                        # by trajectory, ("string-based status", arbitrary, other, entries)
    ### EXPLANATION OF STATUS SYSTEM
    #
    # Due to the complicated nature of adaptive sampling in a forward-sampling approach, we keep track
    # of the current "process" and "reason" for each trajectory.  These processes and reasons are set by the
    # individual optimizers for their own use in checking readiness, convergence, etc.
    # They are stored as self.status[traj] = {'process':'','reason':''}
    # Common processes to all optimizers:
    # TODO these are the ones for SPSA, this should get moved or something when we rework this module
    # Processes:
    #   "submitting grad eval points" - submitting new points so later we can evaluate a gradient and take an opt step
    #   "collecting grad eval points" - all the required gradient evaluation points are submitted, so we're just waiting to collect them
    #   "submitting new opt points"   - a new optimal point has been postulated, and is being submitted for evaluationa (not actually used)
    #   "collecting new opt points"   - the new  hypothetical optimal point has been submitted, and we're waiting on it to finish
    #   "evaluate gradient"           - localStillReady notes we have all the new grad eval points, and has flagged for gradient to be evaluated in localGenerateInput
    #   "following traj <#>"          - trajectory is following another one, given by the last word
    # Reasons:
    #   "just started"                - the optimizer has only just begun operation, and doesn't know what it's doing yet
    #   "found new opt point"         - the last hypothetical optimal point has been accepted, so we need to move forward
    #   "rejecting bad opt point"     - the last hypothetical optimal point was rejected, so we need to reconsider
    #   "seeking new opt point"       - the process of looking for a new opt point has started
    #   "converged"                   - the trajectory is in convergence
    #   "removed as redundant"        - the trajectory has ended because it follows another one
    #   "received recommended point"  - something other than the normal algorithm (such as a preconditioner) suggested a point
    # example usage:
    #   self.status[traj]['process'] == 'submitting grad eval points' and self.status[traj]['reason'] == 'rejecting bad opt point'
    #
    ### END explanation
    self.addAssemblerObject('TargetEvaluation','1')
    self.addAssemblerObject('Function','-1')
    self.addAssemblerObject('Preconditioner','-n')
    self.addAssemblerObject('Sampler','-1')   #This Sampler can be used to initialize the optimization initial points (e.g. partially replace the <initial> blocks for some variables)

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      Overloads the base Sampler class since optimizer has different requirements
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    self.assemblerDict['Functions'    ] = []
    self.assemblerDict['Distributions'] = []
    self.assemblerDict['DataObjects'  ] = []
    for mainClass in ['Functions','Distributions','DataObjects']:
      for funct in initDict[mainClass]:
        self.assemblerDict[mainClass].append([mainClass,initDict[mainClass][funct].type,funct,initDict[mainClass][funct]])

  def _localWhatDoINeed(self):
    """
      Identifies needed distributions and functions.
      Overloads Sampler base implementation because of unique needs.
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {}
    needDict['Distributions'] = [(None,'all')] # We get ALL Distributions in case a Sampler is used for the initialization of the initial points
    needDict['Functions'    ] = [(None,'all')] # We get ALL Functions in case a Sampler is used for the initialization of the initial points
    needDict['DataObjects'  ] = [(None,'all')] # We get ALL DataObjects in case a CustomSampler is used for the initialization of the initial points
    return needDict

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    # TODO can be combined with Sampler's _readMoreXML, but needs to implement paramInput passing to localInputAndChecks (new input checker)
    Assembler._readMoreXML(self,xmlNode)
    self._readMoreXMLbase(xmlNode)
    self.localInputAndChecks(xmlNode)

  def _readMoreXMLbase(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to the base optimizer only
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node1
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    # TODO some merging with base sampler XML reading might be possible, but in general requires different entries
    # first read all XML nodes
    for child in paramInput.subparts:
      #FIXME: the common variable reading should be wrapped up in a method to reduce the code redundancy
      if child.getName() == "variable":
        if self.fullOptVars is None:
          self.fullOptVars = []
        # store variable name
        varName = child.parameterValues['name']
        self.optVarsInitialized[varName] = False
        # store varible requested shape, if any
        if 'shape' in child.parameterValues:
          self.variableShapes[varName] = child.parameterValues['shape']
        self.fullOptVars.append(varName)
        self.optVarsInit['initial'][varName] = {}
        for childChild in child.subparts:
          if childChild.getName() == "upperBound":
            self.optVarsInit['upperBound'][varName] = childChild.value
          elif childChild.getName() == "lowerBound":
            self.optVarsInit['lowerBound'][varName] = childChild.value
          elif childChild.getName() == "initial":
            self.optVarsInit['initial'][varName] = {}
            self.optVarsInitialized[varName] = True
            initPoints = childChild.value
            for trajInd, initVal in enumerate(initPoints):
              try:
                self.optVarsInit['initial'][varName][trajInd] = float(initVal)
              except ValueError:
                self.raiseAnError(ValueError,
                    'Unable to convert to float the intial value for variable "{}" in trajectory "{}": {}'
                    .format(varName,trajInd,initVal))
            if self.optTraj == None:
              self.optTraj = list(range(len(self.optVarsInit['initial'][varName].keys())))

      elif child.getName() == "constant":
        name,value = self._readInConstant(child)
        self.constants[child.parameterValues['name']] = value

      elif child.getName() == "objectVar":
        self.objVar = child.value.strip()

      elif child.getName() == "initialization":
        self.initSeed = randomUtils.randomIntegers(0,2**31,self)
        for childChild in child.subparts:
          if childChild.getName() == "limit":
            self.limit['mdlEval'] = childChild.value
            #the manual once claimed that "A" defaults to iterationLimit/10, but it's actually this number/10.
          elif childChild.getName() == "type":
            self.optType = childChild.value
            if self.optType not in ['min', 'max']:
              self.raiseAnError(IOError, 'Unknown optimization type "{}". Available: "min" or "max"'.format(childChild.value))
          elif childChild.getName() == "initialSeed":
            self.initSeed = childChild.value
          elif childChild.getName() == 'thresholdTrajRemoval':
            self.thresholdTrajRemoval = childChild.value
          elif childChild.getName() == 'writeSteps':
            whenToWrite = childChild.value.strip().lower()
            if whenToWrite == 'every':
              self.writeSolnExportOn = 'every'
            elif whenToWrite == 'final':
              self.writeSolnExportOn = 'final'
            else:
              self.raiseAnError(IOError,'Unexpected frequency for <writeSteps>: "{}". Expected "every" or "final".'.format(whenToWrite))
          else:
            self.raiseAnError(IOError,'Unknown tag: '+childChild.getName())

      elif child.getName() == "convergence":
        for childChild in child.subparts:
          if childChild.getName() == "iterationLimit":
            self.limit['varsUpdate'] = childChild.value
          elif childChild.getName() == "absoluteThreshold":
            self.absConvergenceTol = childChild.value
          elif childChild.getName() == "relativeThreshold":
            self.relConvergenceTol = childChild.value
          elif childChild.getName() == "minStepSize":
            self.minStepSize = childChild.value
          elif childChild.getName() == 'persistence':
            self.convergencePersistence = childChild.value

      elif child.getName() == "restartTolerance":
        self.restartTolerance = child.value

      elif child.getName() == 'parameter':
        for childChild in child.subparts:
          self.paramDict[childChild.getName()] = childChild.value

    # now that XML is read, do some checks and defaults
    # set defaults
    if self.writeSolnExportOn is None:
      self.writeSolnExportOn = 'every'
    self.raiseAMessage('Writing to solution export on "{}" optimizer iteration.'.format(self.writeSolnExportOn))
    if self.optType is None:
      self.optType = 'min'
    if self.thresholdTrajRemoval is None:
      self.thresholdTrajRemoval = 0.05
    if self.initSeed is None:
      self.initSeed = randomUtils.randomIntegers(0,2**31,self)
    # NOTE: optTraj can be changed in "initialize" if the user provides a sampler for seeding
    if self.optTraj is None:
      self.optTraj = [0]

    # check required settings TODO this can probably be removed thanks to the input checking!
    if self.objVar is None:
      self.raiseAnError(IOError, 'Object variable is not specified for optimizer!')
    if self.fullOptVars is None:
      self.raiseAnError(IOError, 'Decision variable(s) not specified for optimizer!')

    for var in self.getOptVars():
      if var not in self.variableShapes:
        self.variableShapes[var] = (1,)
      else:
        if len(self.variableShapes[var]) > 1:
          self.raiseAnError(NotImplementedError,'Matrices as inputs are not yet supported in the Optimizer. For variable "{}" received shape "{}"!'.format(var,self.variableShapes[var]))

    for varName in self.fullOptVars:
      if varName not in self.optVarsInit['upperBound'].keys():
        self.raiseAnError(IOError, 'Upper bound for '+varName+' is not provided' )
      if varName not in self.optVarsInit['lowerBound'].keys():
        self.raiseAnError(IOError, 'Lower bound for '+varName+' is not provided' )
      #store ranges of variables
      self.optVarsInit['ranges'][varName] = self.optVarsInit['upperBound'][varName] - self.optVarsInit['lowerBound'][varName]
      if len(self.optVarsInit['initial'][varName]) == 0:
        for traj in self.optTraj:
          self.optVarsInit['initial'][varName][traj] = None

  def initialize(self,externalSeeding=None,solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    for entry in self.assemblerDict.get('Preconditioner',[]):
      cls,typ,name,model = entry
      if cls != 'Models' or typ != 'ExternalModel':
        self.raiseAnError(IOError,'Currently only "ExternalModel" models can be used as preconditioners! Got "{}.{}" for "{}".'.format(cls,typ,name))
      self.preconditioners[name] = model
      model.initialize({},[])

    for entry in self.assemblerDict.get('Sampler',[]):
      cls,typ,name,sampler = entry
      forwardSampler = False
      for baseClass in sampler.__class__.__bases__:
        if "ForwardSampler" in baseClass.__name__:
          forwardSampler = True
          break
      if not forwardSampler:
        self.raiseAnError(IOError,'Only "ForwardSampler"s (e.g. MonteCarlo, Grid, etc.) can be used for initializing the trajectories in the Optimizer! Got "{}.{}" for "{}".'.format(cls,typ,name))
      self.initializationSampler = sampler
      initDict = {}
      for entity in ['Distributions','Functions','DataObjects']:
        initDict[entity] = dict((entry[2],entry[3]) for entry in self.assemblerDict.get(entity,[]))
      self.initializationSampler._localGenerateAssembler(initDict)
      for key in self.initializationSampler.getInitParams().keys():
        if key.startswith("sampled variable:"):
          var = key.replace("sampled variable:","").strip()
          # check if the sampled variables are among the optimization parameters
          if var not in self.getOptVars():
            self.raiseAnError(IOError,'The variable "'+var+'" sampled by the initialization Sampler "'+self.initializationSampler.name+'" is not among the optimization parameters!')
          # check if the sampled variables have been already initialized in the optimizer (i.e. <initial>)
          if self.optVarsInitialized[var]:
            self.raiseAnError(IOError,'The variable "'+var+'" sampled by the initialization Sampler "'+self.initializationSampler.name+
                                      '" has been already initialized in the Optimizer block. Remove <initial> XML node in Optimizer or the <variable> XML node in the Sampler!')
      # generate the initial coordinates by the sampler and check if they are inside the boundaries
      self.initializationSampler.initialize(externalSeeding)
      # check the number of trajectories (i.e. self.initializationSample.limit in the Sampler)
      currentNumberTrajectories = len(self.optTraj)
      if currentNumberTrajectories > 1:
        if currentNumberTrajectories != self.initializationSampler.limit:
          self.raiseAnError(IOError,"The number of samples generated by the initialization Sampler are different "+
                                    "than the one inputted in the Optimizer (from the variables where the <initial> XML block has been inputted)")
      else:
        self.optTraj = list(range(self.initializationSampler.limit))
        for varName in self.optVarsInit['initial'].keys():
          self.optVarsInit['initial'][varName] = dict.fromkeys(self.optTraj, self.optVarsInit['initial'][varName][0])
      while self.initializationSampler.amIreadyToProvideAnInput():
        self.initializationSampler.localGenerateInput(None,None)
        self.initializationSampler.inputInfo['prefix'] = self.initializationSampler.counter
        sampledVars = self.initializationSampler.inputInfo['SampledVars']
        for varName, value in sampledVars.items():
          self.optVarsInit['initial'][varName][self.initializationSampler.counter] = np.atleast_1d(value)
        self.initializationSampler.counter +=1

    # NOTE: counter['varsUpdate'] needs to be set AFTER self.optTraj length is set by the sampler (if used exclusively)
    self.counter['mdlEval'] = 0
    self.counter['varsUpdate'] = [0]*len(self.optTraj)
    self.optTrajLive = copy.deepcopy(self.optTraj)

    self.mdlEvalHist = self.assemblerDict['TargetEvaluation'][0][3]
    # check if the TargetEvaluation feature and target spaces are consistent
    ins  = self.mdlEvalHist.getVars("input")
    outs = self.mdlEvalHist.getVars("output")
    for varName in self.fullOptVars:
      if varName not in ins:
        self.raiseAnError(RuntimeError,"the optimization variable "+varName+" is not contained in the TargetEvaluation object "+self.mdlEvalHist.name)
    if self.objVar not in outs:
      self.raiseAnError(RuntimeError,"the optimization objective variable "+self.objVar+" is not contained in the TargetEvaluation object "+self.mdlEvalHist.name)
    self.objSearchingROM = SupervisedLearning.returnInstance('SciKitLearn', self, **{'SKLtype':'neighbors|KNeighborsRegressor', 'Features':','.join(list(self.fullOptVars)), 'Target':self.objVar, 'n_neighbors':1,'weights':'distance'})
    self.solutionExport = solutionExport
    if self.solutionExport is None:
      self.raiseAnError(IOError,'The results of optimization cannot be obtained without a SolutionExport defined in the Step!')

    if type(solutionExport).__name__ not in ["PointSet","DataSet"]:
      self.raiseAnError(IOError,'solutionExport type must be a PointSet or DataSet. Got '+\
                                 type(solutionExport).__name__+ '!')

    if 'Function' in self.assemblerDict.keys():
      self.constraintFunction = self.assemblerDict['Function'][0][3]
      if 'constrain' not in self.constraintFunction.availableMethods():
        self.raiseAnError(IOError,'the function provided to define the constraints must have an implemented method called "constrain"')

    # initialize dictionary entries
    # TODO a bunch of the gradient-level trajectory initializations should be moved here.
    for traj in self.optTraj:
      self.optVars[traj]            = self.getOptVars()
      self.submissionQueue[traj]    = deque()

    #check initial point array consistency
    rightLen = len(self.optTraj) #the hypothetical correct length
    for var in self.getOptVars():
      haveLen = len(self.optVarsInit['initial'][var])
      if haveLen != rightLen:
        self.raiseAnError(RuntimeError,'The number of trajectories for variable "{}" is incorrect!  Got {} but expected {}!  Check the <initial> block.'.format(var,haveLen,rightLen))

    # check the constraint here to check if the initial values violate it
    varK = {}
    for trajInd in self.optTraj:
      for varName in self.getOptVars():
        varK[varName] = self.optVarsInit['initial'][varName][trajInd]
      satisfied, _ = self.checkConstraint(varK)
      if not satisfied:
        # get a random value between the the lower and upper bounds
        self.raiseAWarning("the initial values specified for trajectory "+str(trajInd)+" do not satify the contraints. Picking random ones!")
        randomGuessesCnt = 0
        while not satisfied and randomGuessesCnt < self.constraintHandlingPara['innerLoopLimit']:
          for varName in self.getOptVars():
            varK[varName] = self.optVarsInit['lowerBound'][varName]+randomUtils.random()*self.optVarsInit['ranges'][varName]
            self.optVarsInit['initial'][varName][trajInd] = varK[varName]
          satisfied, _ = self.checkConstraint(varK)
        if not satisfied:
          self.raiseAnError(Exception,"It was not possible to find any initial values that could satisfy the constraints for trajectory "+str(trajInd))

    # extend multivalue variables (aka vector variables, or variables with "shape")
    ## TODO someday take array of initial values from a DataSet
    for var,shape in self.variableShapes.items():
      if np.prod(shape) > 1:
        for traj in self.optTraj:
          baseVal = self.optVarsInit['initial'][var][traj]
          newVal = np.ones(shape)*baseVal
          self.optVarsInit['initial'][var][traj] = newVal

    if self.initSeed is not None:
      randomUtils.randomSeed(self.initSeed)

    self.localInitialize(solutionExport=solutionExport)

  ###############
  # Run Methods #
  ###############
  def amIreadyToProvideAnInput(self):
    """
      This is a method that should be called from any user of the optimizer before requiring the generation of a new input.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of model evaluation, convergence criteria met, etc.
      @ In, None
      @ Out, ready, bool, indicating the readiness of the optimizer to generate a new input.
    """
    ready = True if self.counter['mdlEval'] < self.limit['mdlEval'] else False
    if not ready:
      self.raiseAMessage('Reached limit for number of model evaluations!')
    convergence = self.checkConvergence()
    ready = self.localStillReady(ready)
    return ready

  ###################
  # Utility Methods #
  ###################
  def cancelJobs(self, ids):
    """
      Flags jobs with the ids provided to be cancelled.
      @ In, ids, list(str), prefixes/job IDs that need to be cancelled
      @ Out, None
    """
    # first knock them out of the submission queues
    for traj in self.optTraj:
      toRemove = []
      for job in self.submissionQueue[traj]:
        prefix = job['prefix']
        if prefix in ids:
          toRemove.append(job)
          ids.remove(prefix)
          self.raiseADebug('Removing {} from run list by request'.format(prefix))
      for r in toRemove:
        self.submissionQueue[traj].remove(r)
    # then put them in the termination list
    self._jobsToEnd.extend(ids)

  def checkConstraint(self, optVars):
    """
      Method to check whether a set of decision variables satisfy the constraint or not in UNNORMALIZED input space
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ Out, satisfaction, tuple, (bool,list) => (variable indicating the satisfaction of constraints at the point optVars, masks for the under/over violations)
    """
    violatedConstrains = {'internal':[],'external':[]}
    if self.constraintFunction == None:
      satisfied = True
    else:
      satisfied = True if self.constraintFunction.evaluate("constrain",optVars) == 1 else False
      if not satisfied:
        violatedConstrains['external'].append(self.constraintFunction.name)
    for var in optVars:
      varSatisfy=True
      # this should work whether optVars is an array or a single value
      check = np.atleast_1d(optVars[var])
      overMask = check > self.optVarsInit['upperBound'][var]
      underMask = check < self.optVarsInit['lowerBound'][var]
      if np.sum(overMask)+np.sum(underMask) > 0:
        self.raiseAWarning('A variable violated boundary constraints! Details below (enable DEBUG printing)')
        self.raiseADebug('Violating values: "{}"={}'.format(var,optVars[var]))
        satisfied = False
        violatedConstrains['internal'].append( (var,underMask,overMask) )

    satisfied = self.localCheckConstraint(optVars, satisfied)
    satisfaction = satisfied,violatedConstrains
    return satisfaction

  @abc.abstractmethod
  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, none,
      @ Out, convergence, bool, variable indicating whether the convergence criteria has been met.
    """

  def checkInputs(self,inp):
    """
      Checks that all the values of the optimization variables have been set for the point.
      @ In, inp, dict, {var:val} input space point
      @ Out, okay, bool, True if all inputs there, False if not
      @ Out, missing, list, list of missing variables
    """
    missing = []
    for var in self.getOptVars():
      if inp.get(var,None) is None:
        missing.append(var)
        okay = False
    return len(missing)==0,missing

  def checkIfBetter(self,a,b):
    """
      Checks if a is preferable to b for this optimization problem.  Helps mitigate needing to keep
      track of whether a minimization or maximation problem is being run.
      @ In, a, float, value to be compared
      @ In, b, float, value to be compared against
      @ Out, checkIfBetter, bool, True if a is preferable to b for this optimization
    """
    if self.optType == 'min':
      return a <= b
    elif self.optType == 'max':
      return a >= b

  @abc.abstractmethod
  def clearCurrentOptimizationEffort(self):
    """
      Used to inform the subclass optimization effor that it needs to forget whatever opt data it is using
      for the current point (for example, gradient determination points) so that we can start new.
      @ In, None
      @ Out, None
    """
    # README: this method is necessary because the base class optimizer doesn't know what needs to be reset in the
    #         subclass, but the subclass doesn't know when it needs to call this method.
    pass

  @abc.abstractmethod
  def _createEvaluationIdentifier(self, *args, **kwargs):
    """
      Creates a unique ID to identifiy particular realizations as they return from the JobHandler.
      Inputs should be specific to the needs of individual optimizers.
      @ In, args, list, list of arguments
      @ In, kwargs, dict, dictionary of keyword arguments
      @ Out, identifier, str, the evaluation identifier
    """
    pass

  def denormalizeData(self, optVars):
    """
      Method to normalize the data
      @ In, optVars, dict, dictionary containing the value of decision variables to be deormalized, in form of {varName: varValue}
      @ Out, optVarsDenorm, dict, dictionary containing the value of denormalized decision variables, in form of {varName: varValue}
    """
    optVarsDenorm = {}
    for var in optVars.keys():
      try:
        optVarsDenorm[var] = optVars[var]*(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])+self.optVarsInit['lowerBound'][var]
      except KeyError:
        optVarsDenorm[var] = optVars[var]
    return optVarsDenorm

  def endJobRunnable(self):
    """
      Returns the maximum number of inputs allowed to be created by the optimizer right after a job ends
      @ In, None
      @ Out, endJobRunnable, int, number of runnable jobs at the end of each job
    """
    return self._endJobRunnable

  def _expandVectorVariables(self):
    """
      Normally used to extend variables; in the Optimizer, we do that in localGenerateInput
      @ In, None
      @ Out, None
    """
    pass

  def getCurrentSetting(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
                              and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['counter_mdlEval'       ] = self.counter['mdlEval']
    paramDict['counter_varsUpdate'    ] = self.counter['varsUpdate']
    paramDict['initial seed'  ] = self.initSeed
    for key in self.inputInfo:
      if key!='SampledVars':
        paramDict[key] = self.inputInfo[key]
      else:
        for var in self.inputInfo['SampledVars'].keys():
          paramDict['Variable: '+var+' has value'] = paramDict[key][var]
    paramDict.update(self.localGetCurrentSetting())
    return paramDict

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
                              and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for variable in self.getOptVars():
      paramDict[variable] = 'is sampled as a decision variable'
    paramDict['limit_mdlEval' ]        = self.limit['mdlEval']
    paramDict['limit_optIter']         = self.limit['varsUpdate']
    paramDict['initial seed' ]         = self.initSeed
    paramDict.update(self.localGetInitParams())
    return paramDict

  @abc.abstractmethod
  def _getJobsByID(self):
    """
      Overwritten by the base class; obtains new solution export values
      @ In, None
      @ Out, None
    """
    pass

  def getLossFunctionGivenId(self, evaluationID):
    """
      Method to get the Loss Function value given an evaluation ID
      @ In, evaluationID, string, the evaluation identifier (prefix)
      @ Out, objeciveValue, float, the loss function value
    """
    # get matching realization by matching "prefix"
    # TODO the EnsembleModel prefix breaks this pattern!
    _,rlz  = self.mdlEvalHist.realization(matchDict={'prefix':evaluationID})
    # if no match found, return None
    if rlz is None:
      return None
    # otherwise, return value (float assures single value)
    return float(rlz[self.objVar])

  def getOptVars(self):
    """
      Returns the variables in the active optimization space
      @ In, None
      @ Out, optVars, list(string), variables in the current optimization space
    """
    return self.fullOptVars

  def _incrementCounter(self):
    """
      Increments counter and sets up prefix.
      @ In, None
      @ Out, None
    """
    self.counter['mdlEval'] +=1 #since we are creating the input for the next run we increase the counter and global counter
    self.inputInfo['prefix'] = str(self.counter['mdlEval'])

  @abc.abstractmethod
  def localCheckConstraint(self, optVars, satisfaction = True):
    """
      Local method to check whether a set of decision variables satisfy the constraint or not
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ In, satisfaction, bool, optional, variable indicating how the caller determines the constraint satisfaction at the point optVars
      @ Out, satisfaction, bool, variable indicating the satisfaction of constraints at the point optVars
    """
    return satisfaction

  def normalizeData(self, optVars):
    """
      Method to normalize the data
      @ In, optVars, dict, dictionary containing the value of decision variables to be normalized, in form of {varName: varValue}
      @ Out, optVarsNorm, dict, dictionary containing the value of normalized decision variables, in form of {varName: varValue}
    """
    optVarsNorm = {}
    for var in optVars.keys():
      try:
        optVarsNorm[var] = (optVars[var]-self.optVarsInit['lowerBound'][var])/(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])
      except KeyError:
        optVarsNorm[var] = optVars[var]
    return optVarsNorm

  def _numberOfSamples(self,traj=None):
    """
      Calculates the number of independent variables (one for each scalar plus each scalar in each vector).
      @ In, traj, int, optional, if provided then only count variables in current trajectory
      @ Out, _numberOfSamples, int, total number of independent values that need sampling
    """
    return sum(np.prod(self.variableShapes[var]) for var in self.getOptVars())

  def removeConvergedTrajectory(self,convergedTraj):
    """
      Appropriate process for clearing out converged histories.
      @ In, convergedTraj, int, trajectory that has converged and might need to be removed
      @ Out, None
    """
    for t,traj in enumerate(self.optTrajLive):
      if traj == convergedTraj:
        self.optTrajLive.pop(t)
        break

  def proposeNewPoint(self,traj,point):
    """
      Sets a proposed point for the next in the opt chain.  Recommended to be overwritten in subclasses.
      @ In, traj, int, trajectory who is getting proposed point
      @ In, point, dict, new input space point as {var:val}
      @ Out, None
    """
    point = copy.deepcopy(point)
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = point
    self.recommendedOptPoint[traj] = point

  def updateVariableHistory(self,data,traj):
    """
      Stores new historical points into the optimization history.
      @ In, data, dict, UNNORMALIZED new input space entries as {var:#, var:#}
      @ In, traj, int, integer label of the current trajectory of interest
      @ Out, None
    """
    # collect static vars, values
    allData = {}
    allData.update(self.normalizeData(data)) # data point not normalized a priori
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = copy.deepcopy(allData)
