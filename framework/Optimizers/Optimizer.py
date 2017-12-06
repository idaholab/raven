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
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import abc
import numpy as np
from collections import deque
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils,randomUtils
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class Optimizer(utils.metaclass_insert(abc.ABCMeta,BaseType),Assembler):
  """
    This is the base class for optimizers
    Optimizer is a special type of "samplers" that own the optimization strategy (Type) and they generate the input values to optimize a loss function.
    The most significant deviation from the Samplers is that they do not use distributions.
  """
  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    #FIXME: Since the similarity of this class with the base sampler, we should merge this
    BaseType.__init__(self)
    Assembler.__init__(self)
    self.ableToHandelFailedRuns         = False                     # is this optimizer able to handle failed runs?
    #counters
    self.counter                        = {}                        # Dict containing counters used for based and derived class
    self.counter['mdlEval']             = 0                         # Counter of the model evaluation performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.counter['varsUpdate']          = 0                         # Counter of the optimization iteration.
    self.counter['recentOptHist']       = {}                        # as {traj: [pt0, pt1]} where each pt is {'inputs':{var:val}, 'output':val}, the two most recently-accepted points by value
    self.counter['prefixHistory']       = {}                        # as {traj: [prefix1, prefix2]} where each prefix is the job identifier for each trajectory
    self.counter['persistence'  ]       = {}                        # as {traj: n} where n is the number of consecutive converges
    #limits
    self.limit                          = {}                        # Dict containing limits for each counter
    self.limit['mdlEval']               = 2000                      # Maximum number of the loss function evaluation
    self.limit['varsUpdate']            = 650                       # Maximum number of the optimization iteration.
    self._endJobRunnable                = sys.maxsize               # max number of inputs creatable by the optimizer right after a job ends
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
    self.initSeed                       = None                      # Seed for random number generators
    self.optType                        = None                      # Either max or min
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
    self.values                         = {}                        # for each variable the current value {'var name':value}
    self.inputInfo                      = {}                        # depending on the optimizer several different type of keywarded information could be present only one is mandatory, see below
    self.inputInfo['SampledVars']       = self.values               # this is the location where to get the values of the sampled variables
    self.constants                      = {}                        # dictionary of constants variables
    self.printTag                       = self.type                 # prefix for all prints (optimizer type)
    self.submissionQueue                = {}                        # by traj, a place (deque) to store points that should be submitted some time after they are discovered
    #functions and dataojbects
    self.constraintFunction             = None                      # External constraint function, could be not present
    self.preconditioners                = {}                        # by name, Models that might be used as preconditioners
    self.solutionExport                 = None                      # This is the data used to export the solution (it could also not be present)
    self.mdlEvalHist                    = None                      # Containing information of all model evaluation
    self.objSearchingROM                = None                      # ROM used internally for fast loss function evaluation
    #multilevel
    self.multilevel                     = False                     # indicates if operating in multilevel mode
    self.mlBatches                      = {}                        # dict of {batchName:[list,of,vars]} that defines input subspaces
    self.mlHoldBatches                  = {}                        # dict of {batchName:[list,of,vars]} that defines the optional output subspaces that need to be kept constant till convergence of this space
    self.mlSequence                     = []                        # list of batch names that determines the order of convergence.  Last entry is converged most often and fastest (innermost loop).
    self.mlDepth                        = {}                        # {traj: #} index of current recursion depth within self.mlSequence, must be initialized to None
    self.mlStaticValues                 = {}                        # by traj, dictionary of static values for variables in fullOptVars but not in optVars due to multilevel
    self.mlOutputStaticVariables        = {}                        # by traj, dictionary of list of output that must be kept constant due to multilevel
    self.mlActiveSpaceSteps             = {}                        # by traj, integer to track iterations performed in optimizing the current, active subspace
    self.mlBatchInfo                    = {}                        # by batch, by traj, info includes 'lastStepSize','gradientHistory','recommendToGain'
    self.mlPreconditioners              = {}                        # by batch, the preconditioner models to use when transitioning subspaces
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
    self.addAssemblerObject('Restart' ,'-n',True)
    self.addAssemblerObject('TargetEvaluation','1')
    self.addAssemblerObject('Function','-1')
    self.addAssemblerObject('Preconditioner','-n')
    self.addAssemblerObject('Sampler','-1')   #This Sampler can be used to initialize the optimization initial points (e.g. partially replace the <initial> blocks for some variables)

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    self.assemblerDict['Functions'    ] = []
    self.assemblerDict['Distributions'] = []
    for mainClass in ['Functions','Distributions']:
      for funct in initDict[mainClass]:
        self.assemblerDict[mainClass].append([mainClass,initDict[mainClass][funct].type,funct,initDict[mainClass][funct]])

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the optimizers that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {}
    needDict['Distributions'] = [(None,'all')] # We get ALL Distributions in case a Sampler is used for the initialization of the initial points
    needDict['Functions']     = [(None,'all')] # We get ALL Functions in case a Sampler is used for the initialization of the initial points
    return needDict

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
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
    for child in xmlNode:
      #FIXME: the common variable reading should be wrapped up in a method to reduce the code redondancy
      if child.tag == "variable":
        if self.fullOptVars is None:
          self.fullOptVars = []
        try:
          varname = child.attrib['name']
          self.optVarsInitialized[varname] = False
        except KeyError:
          self.raiseAnError(IOError, child.tag+' node does not have the "name" attribute')
        self.fullOptVars.append(varname)
        self.optVarsInit['initial'][varname] = {}
        for childChild in child:
          if   childChild.tag == "upperBound":
            self.optVarsInit['upperBound'][varname] = float(childChild.text)
          elif childChild.tag == "lowerBound":
            self.optVarsInit['lowerBound'][varname] = float(childChild.text)
          elif childChild.tag == "initial":
            self.optVarsInit['initial'][varname] = {}
            self.optVarsInitialized[varname] = True
            temp = childChild.text.split(',')
            for trajInd, initVal in enumerate(temp):
              try:
                self.optVarsInit['initial'][varname][trajInd] = float(initVal)
              except ValueError:
                self.raiseAnError(ValueError, 'Unable to convert to float the intial value for variable "{}" in trajectory "{}": {}'.format(varname,trajInd,initVal))
            if self.optTraj == None:
              self.optTraj = range(len(self.optVarsInit['initial'][varname].keys()))
      elif child.tag == "constant":
        value = utils.partialEval(child.text)
        if value is None:
          self.raiseAnError(IOError,'The body of "constant" XML block should be a number. Got: ' +child.text)
        try:
          self.constants[child.attrib['name']] = value
        except KeyError:
          self.raiseAnError(KeyError,child.tag+' must have the attribute "name"!!!')
      elif child.tag == "objectVar":
        self.objVar = child.text

      elif child.tag == "initialization":
        self.initSeed = randomUtils.randomIntegers(0,2**31,self)
        for childChild in child:
          if childChild.tag == "limit":
            self.limit['mdlEval'] = int(childChild.text)
            #the manual once claimed that "A" defaults to iterationLimit/10, but it's actually this number/10.
          elif childChild.tag == "type":
            self.optType = childChild.text
            if self.optType not in ['min', 'max']:
              self.raiseAnError(IOError, 'Unknown optimization type '+childChild.text+'. Available: "min" or "max"')
          elif childChild.tag == "initialSeed":
            self.initSeed = int(childChild.text)
          elif childChild.tag == 'thresholdTrajRemoval':
            self.thresholdTrajRemoval = float(childChild.text)
          else:
            self.raiseAnError(IOError,'Unknown tag '+childChild.tag+' .Available: limit, type, initialSeed!')

      elif child.tag == "convergence":
        for childChild in child:
          if childChild.tag == "iterationLimit":
            self.limit['varsUpdate'] = int(childChild.text)
          elif childChild.tag == "absoluteThreshold":
            self.absConvergenceTol = float(childChild.text)
          elif childChild.tag == "relativeThreshold":
            self.relConvergenceTol = float(childChild.text)
          elif childChild.tag == "minStepSize":
            self.minStepSize = float(childChild.text)
          elif childChild.tag == 'persistence':
            self.convergencePersistence = int(childChild.text)
      elif child.tag == "restartTolerance":
        self.restartTolerance = float(child.text)

      elif child.tag == 'parameter':
        for childChild in child:
          self.paramDict[childChild.tag] = childChild.text

      elif child.tag == 'multilevel':
        self.multilevel = True
        for subnode in child:
          if subnode.tag == 'subspace':
            #subspace name
            try:
              name = subnode.attrib['name']
            except KeyError:
              self.raiseAnError(IOError, 'A multilevel subspace is missing the "name" attribute!')
            if name in self.mlBatches.keys():
              self.raiseAnError(IOError,'Multilevel subspace "{}" has a duplicate name!'.format(name))
            if "holdOutputSpace" in subnode.attrib:
              self.mlHoldBatches[name] =  [var.strip() for var in subnode.attrib['holdOutputSpace'].split(",")]
              self.raiseAMessage('For subspace "'+name+'" the following output space is asked to be kept on hold: '+','.join(self.mlHoldBatches[name]))
            #subspace text
            subspaceVars = list(x.strip() for x in subnode.text.split(','))
            if len(subspaceVars) < 1:
              self.raiseAnError(IOError,'Multilevel subspace "{}" has no variables specified!'.format(name))
            self.mlBatches[name] = subspaceVars
            #subspace preconditioner
            precond = subnode.attrib.get('precond')
            if precond is not None:
              self.mlPreconditioners[name] = precond
          elif subnode.tag == 'sequence':
            self.mlSequence = list(x.strip() for x in subnode.text.split(','))

    if self.optType is None:
      self.optType = 'min'
    if self.thresholdTrajRemoval is None:
      self.thresholdTrajRemoval = 0.05
    if self.initSeed is None:
      self.initSeed = randomUtils.randomIntegers(0,2**31,self)
    if self.objVar is None:
      self.raiseAnError(IOError, 'Object variable is not specified for optimizer!')
    if self.fullOptVars is None:
      self.raiseAnError(IOError, 'Decision variable is not specified for optimizer!')
    self.fullOptVars.sort()
    if self.optTraj is None:
      self.optTraj = [0]
    for varname in self.fullOptVars:
      if varname not in self.optVarsInit['upperBound'].keys():
        self.raiseAnError(IOError, 'Upper bound for '+varname+' is not provided' )
      if varname not in self.optVarsInit['lowerBound'].keys():
        self.raiseAnError(IOError, 'Lower bound for '+varname+' is not provided' )
      #store ranges of variables
      self.optVarsInit['ranges'][varname] = self.optVarsInit['upperBound'][varname] - self.optVarsInit['lowerBound'][varname]
      if len(self.optVarsInit['initial'][varname]) == 0:
        for traj in self.optTraj:
          self.optVarsInit['initial'][varname][traj] = None
    # NOTE: optTraj can be changed in "initialize" if the user provides a sampler for seeding
    if self.multilevel:
      if len(self.mlSequence) < 1:
        self.raiseAnError(IOError,'No "sequence" was specified for multilevel optimization!')
      if set(self.mlSequence) != set(self.mlBatches.keys()):
        self.raiseAWarning('There is a mismatch between the multilevel batches defined and batches used in the sequence!  Some variables may not be optimized correctly ...')

  def getOptVars(self,traj=None,full=False):
    """
      Returns the variables in the active optimization space
      @ In, full, bool, optional, if True will always give ALL the opt variables
      @ Out, optVars, list(string), variables in the current optimization space
    """
    if full or not self.multilevel or traj is None:
      return self.fullOptVars
    else:
      return self.optVars[traj]

  def localInputAndChecks(self,xmlNode):
    """
      Local method. Place here the additional reading, remember to add initial parameters in the method localGetInitParams
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    pass # To be overwritten by subclass

  def endJobRunnable(self):
    """
      Returns the maximum number of inputs allowed to be created by the optimizer right after a job ends
      @ In, None
      @ Out, endJobRunnable, int, number of runnable jobs at the end of each job
    """
    return self._endJobRunnable

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

  def localGetInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
                              and each parameter's initial value as the dictionary values
    """
    return {}

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

  def localGetCurrentSetting(self):
    """
      Returns a dictionary with class specific information regarding the
      current status of the object.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
                              and each parameter's initial value as the dictionary values
    """
    return {}

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
      availableDist, availableFunc = {}, {} # {'dist name: object}
      for entry in self.assemblerDict.get('Distributions',[]):
        availableDist[entry[2]] = entry[3]
      for entry in self.assemblerDict.get('Functions',[]):
        availableFunc[entry[2]] = entry[3]
      self.initializationSampler._generateDistributions(availableDist,availableFunc)
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
        self.optTraj = range(self.initializationSampler.limit)
        for varName in self.optVarsInit['initial'].keys():
          self.optVarsInit['initial'][varName] = dict.fromkeys(self.optTraj, self.optVarsInit['initial'][varName][0])
      while self.initializationSampler.amIreadyToProvideAnInput():
        self.initializationSampler.localGenerateInput(None,None)
        self.initializationSampler.inputInfo['prefix'] = self.initializationSampler.counter
        sampledVars = self.initializationSampler.inputInfo['SampledVars']
        for varName, value in sampledVars.items():
          self.optVarsInit['initial'][varName][self.initializationSampler.counter] = value
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

    if type(solutionExport).__name__ != "HistorySet":
      self.raiseAnError(IOError,'solutionExport type must be a HistorySet. Got '+ type(solutionExport).__name__+ '!')

    if 'Function' in self.assemblerDict.keys():
      self.constraintFunction = self.assemblerDict['Function'][0][3]
      if 'constrain' not in self.constraintFunction.availableMethods():
        self.raiseAnError(IOError,'the function provided to define the constraints must have an implemented method called "constrain"')

    # initialize multilevel trajectory-based structures
    # TODO a bunch of the gradient-level trajectory initializations should be moved here.
    for traj in self.optTraj:
      self.optVars[traj]            = self.getOptVars() #initial as full space
      self.mlDepth[traj]            = None
      self.mlStaticValues[traj]     = {}
      self.mlActiveSpaceSteps[traj] = 0
      self.submissionQueue[traj]    = deque()
    for batch in self.mlBatches.keys():
      self.mlBatchInfo[batch]       = {}
    # line up preconditioners with their batches
    for batch,precondName in self.mlPreconditioners.items():
      try:
        self.mlPreconditioners[batch] = self.preconditioners[precondName]
      except IndexError:
        self.raiseAnError(IOError,'Could not find preconditioner "{}" in <Preconditioner> nodes!'.format(precondName))

    # apply multilevel preconditioners, in order
    for traj in self.optTraj:
      # initial point(s) are in self.optVarsInit['initial']
      initPoint = dict((var,self.optVarsInit['initial'][var][traj]) for var in self.optVarsInit['initial'].keys())
      # run all preconditioners on that point
      for depth in range(len(self.mlSequence)):
        batch = self.mlSequence[depth]
        initPoint = self.applyPreconditioner(batch,initPoint,denormalize=False)
      #check initial point consistency
      okay,missing = self.checkInputs(initPoint)
      if not okay:
        self.raiseAnError(IOError,'While initializing model inputs, some were not set! Set them through preconditioners or using the <initial> block or a linked Sampler.\n  Missing:', ', '.join(missing))
      # set the initial values that come from preconditioning
      for var in self.getOptVars(full=True):
        self.optVarsInit['initial'][var][traj] = initPoint[var]

    #check initial point array consistency
    rightLen = len(self.optTraj) #the hypothetical correct length
    for var in self.getOptVars(full=True):
      haveLen = len(self.optVarsInit['initial'][var])
      if haveLen != rightLen:
        self.raiseAnError(RuntimeError,'The number of trajectories for variable "{}" is incorrect!  Got {} but expected {}!  Check the <initial> block.'.format(var,haveLen,rightLen))

    # check the constraint here to check if the initial values violate it
    varK = {}
    for trajInd in self.optTraj:
      for varname in self.getOptVars():
        varK[varname] = self.optVarsInit['initial'][varname][trajInd]
      satisfied, _ = self.checkConstraint(varK)
      if not satisfied:
        # get a random value between the the lower and upper bounds
        self.raiseAWarning("the initial values specified for trajectory "+str(trajInd)+" do not satify the contraints. Picking random ones!")
        randomGuessesCnt = 0
        while not satisfied and randomGuessesCnt < self.constraintHandlingPara['innerLoopLimit']:
          for varname in self.getOptVars():
            varK[varname] = self.optVarsInit['lowerBound'][varname]+randomUtils.random()*self.optVarsInit['ranges'][varname]
            self.optVarsInit['initial'][varname][trajInd] = varK[varname]
          satisfied, _ = self.checkConstraint(varK)
        if not satisfied:
          self.raiseAnError(Exception,"It was not possible to find any initial values that could satisfy the constraints for trajectory "+str(trajInd))

    if self.initSeed != None:
      randomUtils.randomSeed(self.initSeed)

    self.localInitialize(solutionExport=solutionExport)

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

  def applyPreconditioner(self,batch,originalPoint,denormalize=True):
    """
      Applies the preconditioner model of a batch to the original point given.
      @ In, batch, string, name of the subsequence batch whose preconditioner needs to be applied
      @ In, originalPoint, dict, {var:val} the point that needs preconditioning (normalized space)
      @ In, denormalize, bool, optional, if True then the originalPoint will be denormalized before running in the preconditioner
      @ Out, results, dict, {var:val} the preconditioned point (still normalized space)
    """
    precond = self.mlPreconditioners.get(batch,None)
    if precond is not None:
      self.raiseADebug('Running preconditioner on batch "{}"'.format(batch))
      # TODO someday this might need to be extended when other models or more complex external models are used for precond
      precond.createNewInput([{}],'Optimizer')
      if denormalize:
        originalPoint = self.denormalizeData(originalPoint)
      infoDict = {'SampledVars':originalPoint}
      for key,value in self.constants.items():
        infoDict['SampledVars'][key] = value
      try:
        _,(preResults,_) = precond.evaluateSample([infoDict['SampledVars']],'Optimizer',infoDict)
      except RuntimeError:
        self.raiseAnError(RuntimeError,'There was an error running the preconditioner for batch "{}"! See messages above for details.'.format(batch))
      # flatten results #TODO breaks for multi-entry arrays
      for key,val in preResults.items():
        preResults[key] = float(val)
      #restore to normalized space if the original point was normalized space
      if denormalize:
        preResults = self.normalizeData(preResults)
      # construct new input point from results + originalPoint
      results = {}
      for key in originalPoint.keys():
        if key in preResults.keys():
          results[key] = preResults[key]
        else:
          results[key] = originalPoint[key]
      return results
    else:
      return originalPoint

  def localInitialize(self,solutionExport):
    """
      Use this function to add initialization features to the derived class
      it is call at the beginning of each step
      @ In, solutionExport, DataObject, a PointSet to hold the solution
      @ Out, None
    """
    pass # To be overwritten by subclass

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
    #if converged and not ready, the optimizer believes it is done; check multilevel
    # -> however, if we're waiting on point collection, don't do multilevel check; only when we want to submit a new point.
    # REASONS TO INTERCEDE in multilevel:
    #   1.) We're at the beginning so we need to initialize multilevel subspace distinction,
    #   2.) We're in the outermost subspace, and have perturbed and converged, so we're completely converged
    #   3.) We've converged the innermost subspace so we need to move to one subspace higher
    #   4.) We're in a non-innermost subspace, and have perturbed but not converged, so we need to move back to innermost again
    #   5.) We're in an intermediate subspace, and have perturbed and converged, so we need to move to one subspace higher
    mlIntervene = False #will be True if we changed the state of the optimizer
    #get the trajectory from the list of "next action needed"
    if self.nextActionNeeded[1] is not None:
      checkMLTrajs = [self.nextActionNeeded[1]]
    else:
      checkMLTrajs = []
      for traj in self.status.keys():
        if self.status[traj]['reason'] == 'converged':
          checkMLTrajs.append(traj)
    for traj in checkMLTrajs:
      if self.multilevel and self.status[traj]['reason'] in ['found new opt point','converged'] :
        # do we have any opt points yet?
        if len(self.counter['recentOptHist'][traj][0]) > 0:
          # get the latset optimization point (normalized)
          latestPoint = self.counter['recentOptHist'][traj][0]['inputs']
          #some flags for clarity of checking
          justStarted = self.mlDepth[traj] is None
          inInnermost = self.mlDepth[traj] is not None and self.mlDepth[traj] == len(self.mlSequence)-1
          inOutermost = self.mlDepth[traj] is not None and self.mlDepth[traj] == 0
          trajConverged = self.status[traj]['reason'] == 'converged'
          # if we only have evaluated the initial point, set the depth to innermost and start grad sampling
          if justStarted:
            self.raiseADebug('Multilevel: initializing for trajectory "{}"'.format(traj))
            self.updateMultilevelDepth(traj, len(self.mlSequence)-1, latestPoint, setAll=True)
            mlIntervene = True
          # if we haven't taken (and accepted) a new opt step, don't change anything
          # otherwise, if we're in the outermost subspace AND we're converged, we're done!
          # otherwise, if we're in the innermost subspace AND we're converged, then move to a higher subspace
          elif trajConverged:#inOutermost and trajConverged:
            if inOutermost:
              self.raiseADebug('Multilevel: outermost subspace converged for trajectory "{}"!'.format(traj))
            else:
              self.raiseADebug('Multilevel: moving from converged subspace to higher subspace for trajectory "{}"'.format(traj))
              self.updateMultilevelDepth(traj,self.mlDepth[traj]-1,latestPoint)
              mlIntervene = True
          # otherwise, if we're not in innermost and not converged, move to innermost subspace
          else: #aka not converged
            if not inInnermost and self.mlActiveSpaceSteps[traj] >= 1:
              self.raiseADebug('Multilevel: moving from perturbed higher subspace back to innermost subspace for trajectory "{}"'.format(traj))
              self.updateMultilevelDepth(traj, len(self.mlSequence)-1, latestPoint, setAll=True)
              mlIntervene = True
          #otherwise, we don't interfere with existing readiness
    #if multilevel intervened, recheck readiness (should always result in ready=True???)
    if mlIntervene:
      self.raiseADebug('Because multilevel intervened, rechecking readiness of optimizer for trajectory "{}"'.format(traj))
      ready = self.localStillReady(True)
    return ready

  @abc.abstractmethod
  def getPreviousIdentifierGivenCurrent(self,prefix):
    """
      Method to get the previous identifier given the current prefix
      @ In, prefix, str, the current identifier
      @ Out, previousPrefix, str, the previous identifier
    """
    pass

  def updateMultilevelDepth(self, traj, depth, optPoint, setAll=False):
    """
      Updates the multilevel depth with static values for inactive subspaces
      @ In, traj, the trajectory whose multilevel depth needs updating
      @ In, depth, int, recursion depth in subspace loops, which ranges between 0 and the last index of self.multilevelSequence
      @ In, optPoint, dict, dictionary point of latest optimization as {var:#, var:#} (normalized)
      @ In, setAll, bool, optional, if True then we set ALL the static variables, not just the old active space
      @ Out, None
    """
    #retain the old batch so we know which static values to set
    if self.mlDepth[traj] is not None:
      oldDepth = self.mlDepth[traj]
      oldBatch = self.mlSequence[oldDepth]
      firstTime = False
      # retain th current state of the algorithm so we can set it later when we return to this batch
      self.mlBatchInfo[oldBatch][traj] = self._getAlgorithmState(traj)
    else:
      firstTime = True
      oldBatch = 'pre-initialize'
      oldDepth = depth
    # set the new active space
    self.mlDepth[traj] = depth
    newBatch = self.mlSequence[self.mlDepth[traj]]
    self.raiseADebug('Transitioning multilevel subspace from "{}" to "{}" for trajectory "{}"...'.format(oldBatch,newBatch,traj))
    # reset the number of iterations in each subspace
    self.mlActiveSpaceSteps[traj] = 0
    # set the active space to include only the desired batch
    self.optVars[traj] = self.mlBatches[newBatch]
    # set the remainder to static variables
    if setAll:
      toMakeStatic = set(self.fullOptVars)-set(self.mlBatches[newBatch])
    else:
      toMakeStatic = self.mlBatches[oldBatch]
    if traj in self.mlOutputStaticVariables:
      self.mlOutputStaticVariables.pop(traj)
    if newBatch in self.mlHoldBatches:
      self.raiseAMessage('For subspace "'+newBatch+'" the following output space is going to be kept on hold: '+','.join(self.mlHoldBatches[newBatch]))
      self.mlOutputStaticVariables[traj] = self.mlHoldBatches[newBatch]

    for var in toMakeStatic:
      self.mlStaticValues[traj][var] = copy.deepcopy(optPoint[var])
    # remove newBatch static values
    for var in self.mlBatches[newBatch]:
      try:
        del self.mlStaticValues[traj][var]
      except KeyError:
        #it wasn't static before, so no problem
        pass
    # clear existing gradient determination data
    if not firstTime:
      self.clearCurrentOptimizationEffort(traj)
    # apply preconditioner IFF we're going towards INNER loops
    newInput = copy.deepcopy(optPoint)
    if depth > oldDepth:
      self.raiseADebug('Preconditioning subsets below',oldDepth,range(oldDepth+1,depth+1))
      #apply changes all the way down
      for d in range(oldDepth+1,depth+1):
        precondBatch = self.mlSequence[d]
        newInput = self.applyPreconditioner(precondBatch,newInput)
        # TODO I don't like that this is called every time!
        self.proposeNewPoint(traj,newInput)
        self.status[traj]['process'] = 'submitting new opt points'
        self.status[traj]['reason'] = 'received recommended point'
    # if there's batch info about the new batch, set it
    self._setAlgorithmState(traj,self.mlBatchInfo[newBatch].get(traj,None))
    #make sure trajectory is live
    if traj not in self.optTrajLive:
      self.optTrajLive.append(traj)

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

  def localStillReady(self,ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, variable indicating whether the caller is prepared for another input.
    """
    return ready # To be overwritten by subclass

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

  def checkConstraint(self, optVars):
    """
      Method to check whether a set of decision variables satisfy the constraint or not in UNNORMALIZED input space
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ Out, satisfaction, tuple, (bool,list) => (variable indicating the satisfaction of constraints at the point optVars, list of the violated constrains)
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
      if optVars[var] > self.optVarsInit['upperBound'][var]:
        violatedConstrains['internal'].append([var,self.optVarsInit['upperBound'][var]])
        varSatisfy = False
      elif optVars[var] < self.optVarsInit['lowerBound'][var]:
        violatedConstrains['internal'].append([var,self.optVarsInit['lowerBound'][var]])
        varSatisfy = False
      if not varSatisfy:
        self.raiseAWarning('A variable violated boundary constraints! "{}"={}'.format(var,optVars[var]))
        satisfied=False

    satisfied = self.localCheckConstraint(optVars, satisfied)
    satisfaction = satisfied,violatedConstrains
    return satisfaction

  @abc.abstractmethod
  def localCheckConstraint(self, optVars, satisfaction = True):
    """
      Local method to check whether a set of decision variables satisfy the constraint or not
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ In, satisfaction, bool, optional, variable indicating how the caller determines the constraint satisfaction at the point optVars
      @ Out, satisfaction, bool, variable indicating the satisfaction of constraints at the point optVars
    """
    return satisfaction

  @abc.abstractmethod
  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, none,
      @ Out, convergence, bool, variable indicating whether the convergence criteria has been met.
    """

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

  def generateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, generateInput, tuple(int,dict), (1,realization dictionary)
    """
    self.counter['mdlEval'] +=1 #since we are creating the input for the next run we increase the counter and global counter
    self.inputInfo['prefix'] = str(self.counter['mdlEval'])
    model.getAdditionalInputEdits(self.inputInfo)
    self.localGenerateInput(model,oldInput)
    ####   UPDATE STATICS   ####
    # get trajectory asking for eval from LGI variable set
    traj = self.inputInfo['trajectory']

    self.values.update(self.denormalizeData(self.mlStaticValues[traj]))
    staticOutputVars = self.mlOutputStaticVariables[traj] if traj in self.mlOutputStaticVariables else None #self.mlOutputStaticVariables.pop(traj,None)
    #if "holdOutputSpace" in self.inputInfo:
    #  self.inputInfo.pop("holdOutputSpace")
    if staticOutputVars is not None:
      # check if the model can hold a portion of the output space
      if not model.acceptHoldOutputSpace():
        self.raiseAnError(RuntimeError,'The user requested to hold a certain output space but the model "'+model.name+'" does not allow it!')
      # try to hold this output variables (multilevel)
      ID = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj]-1,"")
      self.inputInfo["holdOutputErase"] = ID
    #### CONSTANT VARIABLES ####
    if len(self.constants) > 0:
      self.values.update(self.constants)
    self.raiseADebug('Found new input to evaluate:',self.values)
    # "0" means a new sample is found, oldInput is the input that should be perturbed
    return 0,oldInput

  @abc.abstractmethod
  def localGenerateInput(self,model,oldInput):
    """
      This class need to be overwritten since it is here that the magic of the optimizer happens.
      After this method call the self.inputInfo should be ready to be sent to the model
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    pass

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
    allData.update(self.mlStaticValues[traj]) # these are normalized
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = copy.deepcopy(allData)

  def removeConvergedTrajectory(self,convergedTraj):
    """
      Appropriate process for clearing out converged histories.  This lets the multilevel process intercede
      when a trajectory is flagged for removal, in the event it is part of an inner loop.
      @ In, convergedTraj, int, trajectory that has converged and might need to be removed
      @ Out, None
    """
    for t,traj in enumerate(self.optTrajLive):
      if traj == convergedTraj:
        self.optTrajLive.pop(t)
        break

  def finalizeActualSampling(self,jobObject,model,myInput):
    """
      This function is used by optimizers that need to collect information from a finished run.
      Provides a generic interface that all optimizers will use, for specifically
      handling any sub-class, the localFinalizeActualSampling should be overridden
      instead, as finalizeActualSampling provides only generic functionality
      shared by all optimizers and will in turn call the localFinalizeActualSampling
      before returning.
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
    """
    self.localFinalizeActualSampling(jobObject,model,myInput)

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """
      Overwrite only if you need something special at the end of each run....
      This function is used by optimizers that need to collect information from the just ended run
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
    """
    pass

  @abc.abstractmethod
  def _getJobsByID(self):
    """
      Overwritten by the base class; obtains new solution export values
      @ In, None
      @ Out, None
    """
    pass

  def handleFailedRuns(self,failedRuns):
    """
      Collects the failed runs from the Step and allows optimizer to handle them individually if need be.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    self.raiseADebug('===============')
    self.raiseADebug('| RUN SUMMARY |')
    self.raiseADebug('===============')
    if len(failedRuns)>0:
      self.raiseAWarning('There were %i failed runs!  Run with verbosity = debug for more details.' %(len(failedRuns)))
      for run in failedRuns:
        metadata = run.getMetadata()
        ## FIXME: run.command no longer exists, so I am only outputting the
        ## run's identifier.
        self.raiseADebug('  Run number %s FAILED:' %run.identifier)
        self.raiseADebug('      return code :',run.getReturnCode())
        if metadata is not None:
          self.raiseADebug('      sampled vars:')
          for v,k in metadata['SampledVars'].items():
            self.raiseADebug('         ',v,':',k)
    else:
      self.raiseADebug('All runs completed without returning errors.')
    self._localHandleFailedRuns(failedRuns)
    self.raiseADebug('===============')
    self.raiseADebug('  END SUMMARY  ')
    self.raiseADebug('===============')

  def _localHandleFailedRuns(self,failedRuns):
    """
      Specialized method for optimizers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns)>0:
      self.raiseAnError(IOError,'There were failed runs; aborting RAVEN.')

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
