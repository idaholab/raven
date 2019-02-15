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

  Created on Jan. 15, 2019
  @author: wangc, mandd
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
from utils import utils, randomUtils, InputData
from Assembler import Assembler
from Samplers import AdaptiveSampler
#Internal Modules End--------------------------------------------------------------------------------

class OptimizerBase(AdaptiveSampler):
  """
    This is the base class for optimizers.
    Optimizer is a special type of "samplers" that own the optimization strategy (Type) and they generate the input
    values to optimize a loss function. The most significant deviation from the Samplers is that they do not use
    distributions.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    inputSpecification = super(OptimizerBase, cls).getInputSpecification()
    # assembled objects
    # TargetEvaluation represents the container where the model evaluations are stored
    targEval = InputData.parameterInputFactory('TargetEvaluation', contentType=InputData.StringType, strictMode=True)
    targEval.addParam('type', InputData.StringType, True)
    targEval.addParam('class', InputData.StringType, True)
    inputSpecification.addSub(targEval)
    # Sampler can be used to initialize the starting points for some of the variables
    sampler = InputData.parameterInputFactory('Sampler', contentType=InputData.StringType, strictMode=True)
    sampler.addParam('type', InputData.StringType, True)
    sampler.addParam('class', InputData.StringType, True)
    inputSpecification.addSub(sampler)
    # Function indicateds the external function where the constraints are stored
    function = InputData.parameterInputFactory('Function', contentType=InputData.StringType, strictMode=True)
    function.addParam('type', InputData.StringType, True)
    function.addParam('class', InputData.StringType, True)
    inputSpecification.addSub(function)
    # variable
    ## was also part of Sampler, but we need to rewrite variable, so remove it first
    inputSpecification.removeSub('variable')
    variable = InputData.parameterInputFactory('variable', strictMode=True)
    variable.addParam("name", InputData.StringType, True)
    variable.addParam("shape", InputData.IntegerListType, required=False)
    upperBound = InputData.parameterInputFactory('upperBound', contentType=InputData.FloatType, strictMode=True)
    lowerBound = InputData.parameterInputFactory('lowerBound', contentType=InputData.FloatType, strictMode=True)
    initial = InputData.parameterInputFactory('initial',contentType=InputData.FloatListType)
    variable.addSub(upperBound)
    variable.addSub(lowerBound)
    variable.addSub(initial)
    inputSpecification.addSub(variable)
    # objectVar
    objectVar = InputData.parameterInputFactory('objectVar', contentType=InputData.StringType, strictMode=True)
    inputSpecification.addSub(objectVar)
    # initialization
    init = InputData.parameterInputFactory('initialization', strictMode=True)
    limit      = InputData.parameterInputFactory('limit', contentType=InputData.IntegerType)
    whenWriteEnum = InputData.makeEnumType('whenWriteEnum','whenWriteType',['final','every'])
    minmaxEnum = InputData.makeEnumType('MinMax','OptimizerTypeType',['min','max'])
    minmax     = InputData.parameterInputFactory('type', contentType=minmaxEnum)
    write      = InputData.parameterInputFactory('writeSteps',contentType=whenWriteEnum)
    init.addSub(limit)
    init.addSub(minmax)
    init.addSub(write)
    inputSpecification.addSub(init)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    AdaptiveSampler.__init__(self)
    self.varsUpdate                     = 0                         # Counter of the optimization iteration.
    self.recentOptHist                  = {}                        # as {traj: [pt0, pt1]} where each pt is {'inputs':{var:val}, 'output':val}, the two most recently-accepted points by value
    self.prefixHistory                  = {}                        # as {traj: [prefix1, prefix2]} where each prefix is the job identifier for each trajectory
    self.objVar                         = None                      # Objective variable to be optimized
    self.optVars                        = {}                        # By trajectory, current decision variables for optimization
    self.optType                        = None                      # Either max or min
    self.optTraj                        = None                      # Identifiers of parallel optimization trajectories
    self.optVarsInitialized             = {}                        # Dict {var1:<initial> present?,var2:<initial> present?}
    #initialization parameters
    self.optVarsInit                    = {}                        # Dict containing upper/lower bounds and initial of each decision variables
    self.optVarsInit['upperBound']      = {}                        # Dict containing upper bounds of each decision variables
    self.optVarsInit['lowerBound']      = {}                        # Dict containing lower bounds of each decision variables
    self.optVarsInit['initial']         = {}                        # Dict containing initial values of each decision variables
    self.optVarsInit['ranges']          = {}                        # Dict of the ranges (min and max) of each variable's domain
    self.optVarsHist                    = {}                        # History of decision variables for each iteration
    self.writeSolnExportOn              = None                      # Determines when we write to solution export (every step or final solution)
    self.paramDict                      = {}                        # Dict containing additional parameters for derived class
    #sampler-step communication
    self.submissionQueue                = {}                        # by traj, a place (deque) to store points that should be submitted some time after they are discovered
    self.constraintFunction             = None                      # External constraint function, could be not present
    self.solutionExport                 = None                      # This is the data used to export the solution
    self.nextActionNeeded               = (None,None)               # tool for localStillReady to inform localGenerateInput on the next action needed
    self.modelEvaluationsHist           = None                      # Containing information of all model evaluation

    self.addAssemblerObject('TargetEvaluation','1')
    self.addAssemblerObject('Function','-1')
    self.addAssemblerObject('Preconditioner','-n')
    self.addAssemblerObject('Sampler','-1')   #This Sampler can be used to initialize the optimization initial points (e.g. partially replace the <initial> blocks for some variables)

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
        varName = child.parameterValues['name']
        self.optVarsInitialized[varName] = False
        # store variable requested shape, if any
        if 'shape' in child.parameterValues:
          self.variableShapes[varName] = child.parameterValues['shape']
        self.toBeOptimized.append(varName)
        self.optVarsInit['initial'][varName] = {}
        for childChild in child.subparts:
          if childChild.getName() == "upperBound":
            self.optVarsInit['upperBound'][varName] = childChild.value
          elif childChild.getName() == "lowerBound":
            self.optVarsInit['lowerBound'][varName] = childChild.value
          elif childChild.getName() == "initial":
            # for consistent with multi trajectory, we initialize with only one trajectory with index '0'
            self.optVarsInit['initial'][varName][0] = childChild.value
            self.optVarsInitialized[varName] = True
            initPoints = childChild.value
      elif child.getName() == "constant":
        name,value = self._readInConstant(child)
        self.constants[child.parameterValues['name']] = value
      elif child.getName() == "objectVar":
        self.objVar = child.value.strip()
      elif child.getName() == "restartTolerance":
        self.restartTolerance = child.value
      elif child.getName() == "initialization":
        for childChild in child.subparts:
          if childChild.getName() == "limit":
            self.limit = childChild.value
            #the manual once claimed that "A" defaults to iterationLimit/10, but it's actually this number/10.
          elif childChild.getName() == "type":
            self.optType = childChild.value
            if self.optType not in ['min', 'max']:
              self.raiseAnError(IOError, 'Unknown optimization type "{}". Available: "min" or "max"'.format(childChild.value))
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
    # now that XML is read, do some checks and defaults
    # set defaults
    if self.writeSolnExportOn is None:
      self.writeSolnExportOn = 'final'
    self.raiseAMessage('Writing to solution export on "{}" optimizer iteration.'.format(self.writeSolnExportOn))
    if self.optType is None:
      self.optType = 'min'
    # NOTE: optTraj can be changed in "initialize" if the user provides a sampler for seeding
    if self.optTraj is None:
      self.optTraj = [0]
    # check required settings TODO this can probably be removed thanks to the input checking!
    if self.objVar is None:
      self.raiseAnError(IOError, 'Object variable is not specified for optimizer!')
    if len(self.toBeOptimized) == 0:
      self.raiseAnError(IOError, 'Decision variable(s) not specified for optimizer!')

    for var in self.toBeOptimized:
      if var not in self.variableShapes:
        self.variableShapes[var] = (1,)
      else:
        if len(self.variableShapes[var]) > 1:
          self.raiseAnError(NotImplementedError,'Matrices as inputs are not yet supported in the Optimizer. For variable "{}" received shape "{}"!'.format(var,self.variableShapes[var]))

    for varName in self.toBeOptimized:
      if varName not in self.optVarsInit['upperBound'].keys():
        self.raiseAnError(IOError, 'Upper bound for '+varName+' is not provided' )
      if varName not in self.optVarsInit['lowerBound'].keys():
        self.raiseAnError(IOError, 'Lower bound for '+varName+' is not provided' )
      #store ranges of variables
      self.optVarsInit['ranges'][varName] = self.optVarsInit['upperBound'][varName] - self.optVarsInit['lowerBound'][varName]
      if len(self.optVarsInit['initial'][varName]) == 0:
        for traj in self.optTraj:
          self.optVarsInit['initial'][varName][traj] = None

  def checkConstraint(self, optVars):
    """
      Method to check whether a set of decision variables satisfy the constraint
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of
        {varName: varValue}
      @ Out, satisfied, bool, variable indicating the satisfaction of constraints at the point optVars
    """
    if self.constraintFunction == None:
      satisfied = True
    else:
      satisfied = True if self.constraintFunction.evaluate("constrain",optVars) == 1 else False
    return satisfied

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

  def denormalizeData(self, optVars):
    """
      Method to normalize the data
      @ In, optVars, dict, dictionary containing the value of decision variables to be deormalized,
        in form of {varName: varValue}
      @ Out, optVarsDenorm, dict, dictionary containing the value of denormalized decision variables,
        in form of {varName: varValue}
    """
    pass

  def normalizeData(self, optVars):
    """
      Method to normalize the data
      @ In, optVars, dict, dictionary containing the value of decision variables to be normalized,
        in form of {varName: varValue}
      @ Out, optVarsNorm, dict, dictionary containing the value of normalized decision variables,
        in form of {varName: varValue}
    """
    pass

  def getQueuedPoint(self, traj=0, denorm=False):
    """
      Pops the first point off the submission queue (or errors if empty).
      @ In, traj, int, the trajectory from whose queue we should obtain an entry
      @ In, denorm, bool, optional, if True the input data will be denormalized before returning
      @ Out, prefix, str, #_#_#
      @ Out, point, dict, {var:val}
    """
    try:
      entry = self.submissionQueue[traj].popleft()
    except IndexError:
      self.raiseAnError(RuntimeError,'Tried to get a point from submission queue of trajectory "{}" but it is empty!'.format(traj))
    prefix = entry['prefix']
    point = entry['inputs']
    if denorm:
      point = self.denormalizeData(point)
    return prefix,point

  def getCurrentSetting(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
                              and each parameter's initial value as the dictionary values
    """
    paramDict = AdaptiveSampler.getCurrentSetting(self)
    paramDict['counter_varsUpdate'    ] = self.varsUpdate
    return paramDict

  def initialize(self,externalSeeding=None,solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    # NOTE: 'varsUpdate' needs to be set AFTER self.optTraj length is set by the sampler (if used exclusively)
    self.counter = 0
    self.varsUpdate = [0]*len(self.optTraj)
    self.optTrajLive = copy.deepcopy(self.optTraj)
    # TODO: We should use retrieveObjectFromAssemblerDict to get the Instance
    self.modelEvaluationsHist = self.assemblerDict['TargetEvaluation'][0][3]
    # check if the TargetEvaluation feature and target spaces are consistent
    ins  = self.modelEvaluationsHist.getVars("input")
    outs = self.modelEvaluationsHist.getVars("output")
    for varName in self.toBeOptimized:
      if varName not in ins:
        self.raiseAnError(RuntimeError,"the optimization variable "+varName+" is not contained in the TargetEvaluation object "+self.modelEvaluationsHist.name)
    if self.objVar not in outs:
      self.raiseAnError(RuntimeError,"the optimization objective variable "+self.objVar+" is not contained in the TargetEvaluation object "+self.modelEvaluationsHist.name)
    self.solutionExport = solutionExport
    if self.solutionExport is None:
      self.raiseAnError(IOError,'The results of optimization cannot be obtained without a SolutionExport defined in the Step!')

    if type(solutionExport).__name__ not in ["PointSet","DataSet"]:
      self.raiseAnError(IOError,'solutionExport type must be a PointSet or DataSet. Got '+\
                                 type(solutionExport).__name__+ '!')
    # TODO: We should use retrieveObjectFromAssemblerDict to get the Instance
    if 'Function' in self.assemblerDict.keys():
      self.constraintFunction = self.assemblerDict['Function'][0][3]
      if 'constrain' not in self.constraintFunction.availableMethods():
        self.raiseAnError(IOError,'the function provided to define the constraints must have an implemented method called "constrain"')

    # TODO a bunch of the gradient-level trajectory initializations should be moved here.
    for traj in self.optTraj:
      self.optVars[traj]            = self.toBeOptimized #initial as full space
      self.submissionQueue[traj]    = deque()

    #check initial point array consistency
    rightLen = len(self.optTraj) #the hypothetical correct length
    for var in self.toBeOptimized:
      haveLen = len(self.optVarsInit['initial'][var])
      if haveLen != rightLen:
        self.raiseAnError(RuntimeError,'The number of trajectories for variable "{}" is incorrect!  Got {} but expected {}!  Check the <initial> block.'.format(var,haveLen,rightLen))

    # extend multivalue variables (aka vector variables, or variables with "shape")
    ## TODO someday take array of initial values from a DataSet
    for var,shape in self.variableShapes.items():
      if np.prod(shape) > 1:
        for traj in self.optTraj:
          baseVal = self.optVarsInit['initial'][var][traj]
          if len(baseVal) == 1:
            newVal = np.ones(shape)*baseVal
            self.optVarsInit['initial'][var][traj] = newVal
          elif len(baseVal) != np.prod(shape):
            self.raiseAnError(IOError, "The number of initial values provided for vector variable", var," is not equal the vector total size!" )

    # check the constraint here to check if the initial values violate it
    varK = {}
    for trajInd in self.optTraj:
      for varName in self.toBeOptimized:
        varK[varName] = self.optVarsInit['initial'][varName][trajInd]
        self.checkConstraint(varK)

    self.localInitialize(solutionExport=solutionExport)

  def updateVariableHistory(self,data,traj=0):
    """
      Stores new historical points into the optimization history.
      @ In, data, dict, new input space entries as {var:#, var:#}
      @ In, traj, int, integer label of the current trajectory of interest
      @ Out, None
    """
    self.optVarsHist[traj][self.varsUpdate][traj] = copy.deepcopy(data)

  @abc.abstractmethod
  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, none,
      @ Out, convergence, bool, variable indicating whether the convergence criteria has been met.
    """

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

  def writeToSolutionExport(self, traj=0):
    """
      Standardizes how the solution export is written to.
      Uses data from "recentOptHist" and other counters to fill in values.
      @ In, traj, int, the trajectory for which an entry is being written
      @ Out, None
    """
    pass

  def _checkModelFinish(self, traj=0, updateKey=0, evalID='v'):
    """
      Determines if the Model has finished running an input and returned the output
      @ In, traj, int, traj on which the input is being checked
      @ In, updateKey, int, the id of variable update on which the input is being checked
      @ In, evalID, int or string, indicating the id of the perturbation (int) or its a variable update (string 'v')
      @ Out, _checkModelFinish, tuple(bool, int), indicating whether the Model has finished the evaluation over input
             identified by traj+updateKey+evalID, the index of the location of the input in dataobject
    """
    if len(self.modelEvaluationsHist) == 0:
      return (False,-1)
    lookFor = '{}_{}_{}'.format(traj,updateKey,evalID)
    index,match = self.modelEvaluationsHist.realization(matchDict = {'prefix':lookFor})
    # if no match, return False
    if match is None:
      return False,-1
    # otherwise, return index of match
    return True, index

  def _createEvaluationIdentifier(self, trajID=0, iterID=0, evalType='v'):
    """
      Create evaluation identifier
      @ In, trajID, integer, trajectory identifier
      @ In, iterID, integer, iteration number (identifier)
      @ In, evalType, integer or string, evaluation type (v for variable update; otherwise id for gradient evaluation)
      @ Out, identifier, string, the evaluation identifier
    """
    identifier = str(trajID) + '_' + str(iterID) + '_' + str(evalType)
    return identifier

  @abc.abstractmethod
  def _getJobsByID(self):
    """
      Overwritten by the base class; obtains new solution export values
      @ In, None
      @ Out, None
    """
    pass
