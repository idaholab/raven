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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
import Distributions
#Internal Modules End--------------------------------------------------------------------------------

class Optimizer(utils.metaclass_insert(abc.ABCMeta,BaseType),Assembler):
  """
    This is the base class for optimizers
    Optimizer is a special type of "samplers" that own the optimization strategy (Type) and they generate the
    input values to optimize a loss function. They do not have distributions inside!!!!

    --Instance--
    myInstance = Optimizer()
    myInstance.XMLread(xml.etree.ElementTree.Element)  This method generates all the information that will be permanent for the object during the simulation

    --usage--
    myInstance = Optimizer()
    myInstance.XMLread(xml.etree.ElementTree.Element)  This method generate all permanent information of the object from <Simulation>
    myInstance.whatDoINeed()                           -see Assembler class-
    myInstance.initialize()                            This method is called from the <Step> before the Step process start.
    myInstance.amIreadyToProvideAnInput                Requested from <Step> used to verify that the optimizer is available to generate a new input for the model
    myInstance.generateInput(self,model,oldInput)      Requested from <Step> to generate a new input. Generate the new values and request to model to modify according the input and returning it back

    --Other inherited methods--
    myInstance.whoAreYou()                            -see BaseType class-
    myInstance.myCurrentSetting()                     -see BaseType class-

    --Adding a new Optimizer subclass--
    <MyClass> should inherit at least from Optimizer or from another derived class already presents

    DO NOT OVERRIDE any of the class method that are not starting with self.local*

    ADD your class to the dictionary __InterfaceDict in the Factory submodule

    The following method overriding is MANDATORY:
    self.localGenerateInput(model,oldInput)  : this is where the step happens, after this call the output is ready
    self._localGenerateAssembler(initDict)
    self._localWhatDoINeed()

    the following methods could be overrode:
    self.localInputAndChecks(xmlNode)
    self.localGetInitParams()
    self.localGetCurrentSetting()
    self.localInitialize()
    self.localStillReady(ready)
    self.localFinalizeActualSampling(jobObject,model,myInput)
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
    self.counter                        = {}                        # Dict containing counters used for based and derived class
    self.counter['mdlEval']             = 0                         # Counter of the model evaluation performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.counter['varsUpdate']          = 0                         # Counter of the optimization iteration.
    self.counter['recentOptHist']       = {}                        # as {traj: [pt0, pt1]} where each pt is {'inputs':{var:val}, 'output':val}, the two most recently-accepted points by value
    self.limit                          = {}                        # Dict containing limits for each counter
    self.limit['mdlEval']               = 2000                      # Maximum number of the loss function evaluation
    self.limit['varsUpdate']            = 650                       # Maximum number of the optimization iteration.
    self.initSeed                       = None                      # Seed for random number generators
    self.optVars                        = None                      # Decision variables for optimization
    self.fullOptVars                    = None                      # Decision variables for optimization, full space
    self.optVarsInit                    = {}                        # Dict containing upper/lower bounds and initial of each decision variables
    self.optVarsInit['upperBound']      = {}                        # Dict containing upper bounds of each decision variables
    self.optVarsInit['lowerBound']      = {}                        # Dict containing lower bounds of each decision variables
    self.optVarsInit['initial']         = {}                        # Dict containing initial values of each decision variables
    self.optVarsInit['ranges']          = {}                        # Dict of the ranges (min and max) of each variable's domain
    self.optVarsHist                    = {}                        # History of normalized decision variables for each iteration
    #self.nVar                           = 0                         # Number of decision variables
    self.objVar                         = None                      # Objective variable to be optimized
    self.optType                        = None                      # Either maximize or minimize
    self.optTraj                        = None                      # Identifiers of parallel optimization trajectories
    self.thresholdTrajRemoval           = None                      # Threshold used to determine the convergence of parallel optimization trajectories
    self.paramDict                      = {}                        # Dict containing additional parameters for derived class
    self.absConvergenceTol              = 0.0                       # Convergence threshold (absolute value)
    self.relConvergenceTol              = 1.e-3                     # Convergence threshold (relative value)
    self.solutionExport                 = None                      #This is the data used to export the solution (it could also not be present)
    self.values                         = {}                        # for each variable the current value {'var name':value}
    self.inputInfo                      = {}                        # depending on the optimizer several different type of keywarded information could be present only one is mandatory, see below
    self.inputInfo['SampledVars'     ]  = self.values               # this is the location where to get the values of the sampled variables
    self.constants                      = {}                        # dictionary of constants variables
    self.FIXME                          = False                     # FIXME flag
    self.printTag                       = self.type                 # prefix for all prints (optimizer type)

    self._endJobRunnable                = sys.maxsize               # max number of inputs creatable by the optimizer right after a job ends

    self.constraintFunction             = None                      # External constraint function, could be not present
    self.mdlEvalHist                    = None                      # Containing information of all model evaluation
    self.objSearchingROM                = None                      # ROM used internally for fast loss function evaluation

    self.nextActionNeeded               = (None,None)               # tool for localStillReady to inform localGenerateInput on the next action needed

    self.multilevel                     = False                     # indicates if operating in multilevel mode
    self.mlBatches                      = {}                        # dict of {batchName:[list,of,vars]} that defines input subspaces
    self.mlTolerances                   = {}                        # dict of {batchName:float} that gives convergence tolerance for each subspace
    self.mlSequence                     = []                        # list of batch names that determines the order of convergence.  Last entry is converged most often and fastest (innermost loop).
    self.mlDepth                        = None                      # index of current recursion depth within self.mlSequence
    self.mlStaticValues                 = {}                        # dictionary of static values for variables in fullOptVars but not in optVars due to multilevel
    self.mlActiveSpaceSteps             = 0                         # integer to track iterations performed in optimizing the current, active subspace

    self.status                         = {}                        # by trajectory, ("string-based status", arbitrary, other, entries)
    ### EXPLANATION OF STATUS SYSTEM
    #
    # Due to the complicated nature of adaptive sampling in a forward-sampling approach, we keep track
    # of the current "process" and "reason" for each trajectory.  These processes and reasons are set by the
    # individual optimizers for their own use in checking readiness, convergence, etc.
    # Common processes to all optimizers:
    # TODO these are the ones for SPSA, this should get moved or something when we rework this module
    # Processes:
    #   "submitting grad eval points" - submitting new points so later we can evaluate a gradient and take an opt step
    #   "collecting grad eval points" - all the required gradient evaluation points are submitted, so we're just waiting to collect them
    #   "submitting new opt point"    - a new optimal point has been postulated, and is being submitted for evaluationa (not actually used)
    #   "collecting new opt point"    - the new  hypothetical optimal point has been submitted, and we're waiting on it to finish
    #   "evaluate gradient"           - localStillReady notes we have all the new grad eval points, and has flagged for gradient to be evaluated in localGenerateInput
    # Reasons:
    #   "just started"            - the optimizer has only just begun operation, and doesn't know what it's doing yet
    #   "found new opt point"     - the last hypothetical optimal point has been accepted, so we need to move forward
    #   "rejecting bad opt point" - the last hypothetical optimal point was rejected, so we need to reconsider
    #   "converged"               - the trajectory is in convergence
    self.addAssemblerObject('Restart' ,'-n',True)
    self.addAssemblerObject('TargetEvaluation','1')
    self.addAssemblerObject('Function','-1')

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    ## FIX ME -- this method is inherited from sampler and may not be needed by optimizer
    ## Currently put here as a place holder
    pass

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the optimizers that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    ## FIX ME -- this method is inherited from sampler and may not be needed by optimizer
    ## Currently put here as a place holder
    return {}

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
        if self.optVars is None:
          self.optVars = []
        try:
          varname = child.attrib['name']
        except KeyError:
          self.raiseAnError(IOError, child.tag+' node does not have the "name" attribute')
        self.optVars.append(varname)
        for childChild in child:
          if   childChild.tag == "upperBound":
            self.optVarsInit['upperBound'][varname] = float(childChild.text)
          elif childChild.tag == "lowerBound":
            self.optVarsInit['lowerBound'][varname] = float(childChild.text)
          elif childChild.tag == "initial"   :
            self.optVarsInit['initial'][varname] = {}
            temp = childChild.text.split(',')
            for trajInd, initVal in enumerate(temp):
              try:
                self.optVarsInit['initial'][varname][trajInd] = float(initVal)
              except ValueError:
                self.raiseAnError(ValueError, "Unable to convert to float the intial value for variable "+varname+ " in trajectory "+str(trajInd))
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
        self.initSeed = Distributions.randomIntegers(0,2**31,self)
        for childChild in child:
          if childChild.tag == "limit":
            self.limit['mdlEval'] = int(childChild.text) #FIXME what's the difference between this and self.limit['varsUpdate']?
            #the manual once claimed that "A" defaults to iterationLimit/10, but it's actually this number/10.
          elif childChild.tag == "type":
            self.optType = childChild.text
            if self.optType not in ['min', 'max']:
              self.raiseAnError(IOError, 'Unknown optimization type '+childChild.text+'. Available: mix or max')
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
          if childChild.tag == "absoluteThreshold":
            self.absConvergenceTol = float(childChild.text)
          if childChild.tag == "relativeThreshold":
            self.relConvergenceTol = float(childChild.text)
      elif child.tag == "restartTolerance":
        self.restartTolerance = float(child.text)

      elif child.tag == 'parameter':
        for childChild in child:
          self.paramDict[childChild.tag] = childChild.text

      elif child.tag == 'multilevel':
        self.multilevel = True
        for subnode in child:
          if subnode.tag == 'subspace':
            attribs = {}
            try:
              name = subnode.attrib['name']
            except KeyError:
              self.raiseAnError(IOError, 'A multilevel subspace is missing the "name" attribute!')
            if name in self.mlBatches.keys():
              self.raiseAnError(IOError,'Multilevel subspace "{}" has a duplicate name!'.format(name))
            subspaceVars = list(x.strip() for x in subnode.text.split(','))
            if len(subspaceVars)<1:
              self.raiseAnError(IOError,'Multilevel subspace "{}" has no variables specified!'.format(name))
            self.mlBatches[name] = subspaceVars
          elif subnode.tag == 'sequence':
            self.mlSequence = list(x.strip() for x in subnode.text.split(','))

    if self.optType is None:
      self.optType = 'min'
    if self.thresholdTrajRemoval is None:
      self.thresholdTrajRemoval = 0.05
    if self.initSeed is None:
      self.initSeed = Distributions.randomIntegers(0,2**31,self)
    if self.objVar is None:
      self.raiseAnError(IOError, 'Object variable is not specified for optimizer!')
    if self.optVars is None:
      self.raiseAnError(IOError, 'Decision variable is not specified for optimizer!')
    else:
      self.optVars.sort()
    if self.optTraj is None:
      self.optTraj = [0]
    for varname in self.optVars:
      if varname not in self.optVarsInit['upperBound'].keys():
        self.raiseAnError(IOError, 'Upper bound for '+varname+' is not provided' )
      if varname not in self.optVarsInit['lowerBound'].keys():
        self.raiseAnError(IOError, 'Lower bound for '+varname+' is not provided' )
      if varname not in self.optVarsInit['initial'].keys():
        self.optVarsInit['initial'][varname] = {}
        for trajInd in self.optTraj:
          self.optVarsInit['initial'][varname][trajInd] = (self.optVarsInit['upperBound'][varname]+self.optVarsInit['lowerBound'][varname])/2.0
      else:
        for trajInd in self.optTraj:
          initVal =  self.optVarsInit['initial'][varname][trajInd]
          if initVal < self.optVarsInit['lowerBound'][varname] or initVal > self.optVarsInit['upperBound'][varname]:
            self.raiseAnError(IOError,"The initial value for variable "+varname+" and trajectory "+str(trajInd) +" is outside the domain identified by the lower and upper bounds!")
      if len(self.optTraj) != len(self.optVarsInit['initial'][varname].keys()):
        self.raiseAnError(ValueError, 'Number of initial values does not equal to the number of parallel optimization trajectories')
      #store ranges of variables
      self.optVarsInit['ranges'][varname] = self.optVarsInit['upperBound'][varname] - self.optVarsInit['lowerBound'][varname]
    self.optTrajLive = copy.deepcopy(self.optTraj)
    self.fullOptVars = copy.deepcopy(self.optVars)
    if self.multilevel:
      if len(self.mlSequence) < 1:
        self.raiseAnError(IOError,'No "sequence" was specified for multilevel optimization!')
      if set(self.mlSequence) != set(self.mlBatches.keys()):
        self.raiseAWarning('There is a mismatch between the multilevel batches defined and batches used in the sequence!  Some variables may not be optimized correctly ...')
      if len(self.optTraj) > 1:
        self.raiseAnError(NotImplementedError,'Multilevel with multiple trajectories is not ready yet; use single trajectory.')

  def getOptVars(self,full=False):
    """
      Returns the variables in the active optimization space
      @ In, full, bool, optional, if True will always give ALL the opt variables
      @ Out, optVars, list(string), variables in the current optimization space
    """
    if full or not self.multilevel:
      return self.fullOptVars
    else:
      return self.optVars

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
    for variable in self.optVars:
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
    self.counter['mdlEval'] = 0
    self.counter['varsUpdate'] = [0]*len(self.optTraj)
    #self.nVar = len(self.optVars)

    self.mdlEvalHist = self.assemblerDict['TargetEvaluation'][0][3]
    self.objSearchingROM = SupervisedLearning.returnInstance('SciKitLearn', self, **{'SKLtype':'neighbors|KNeighborsRegressor', 'Features':','.join(list(self.fullOptVars)), 'Target':self.objVar, 'n_neighbors':1,'weights':'distance'})
    self.solutionExport = solutionExport

    if solutionExport != None and type(solutionExport).__name__ != "HistorySet":
      self.raiseAnError(IOError,'solutionExport type is not a HistorySet. Got '+ type(solutionExport).__name__+ '!')

    if 'Function' in self.assemblerDict.keys():
      self.constraintFunction = self.assemblerDict['Function'][0][3]
      if 'constrain' not in self.constraintFunction.availableMethods():
        self.raiseAnError(IOError,'the function provided to define the constraints must have an implemented method called "constrain"')
    # check the constraint here to check if the initial values violate it
    varK = {}
    for trajInd in self.optTraj:
      for varname in self.optVars:
        varK[varname] = self.optVarsInit['initial'][varname][trajInd]
      satisfied, _ = self.checkConstraint(self.normalizeData(varK))
      if not satisfied:
        # get a random value between the the lower and upper bounds
        self.raiseAWarning("the initial values specified for trajectory "+str(trajInd)+" do not satify the contraints. Picking random ones!")
        randomGuessesCnt = 0
        while not satisfied and randomGuessesCnt < self.constraintHandlingPara['innerLoopLimit']:
          for varname in self.optVars:
            varK[varname] = self.optVarsInit['lowerBound'][varname]+Distributions.random()*self.optVarsInit['ranges'][varname]
            self.optVarsInit['initial'][varname][trajInd] = varK[varname]
          satisfied, _ = self.checkConstraint(varK)
        if not satisfied:
          self.raiseAnError(Exception,"It was not possible to find any initial values that could satisfy the constraints for trajectory "+str(trajInd))


    if self.initSeed != None:
      Distributions.randomSeed(self.initSeed)

    # specializing the self.localInitialize()
    if solutionExport != None:
      self.localInitialize(solutionExport=solutionExport)
    else:
      self.localInitialize()

  def localInitialize(self,solutionExport=None):
    """
      Use this function to add initialization features to the derived class
      it is call at the beginning of each step
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
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
    print('DEBUGG AIR status:',self.status)
    ready = True if self.counter['mdlEval'] < self.limit['mdlEval'] else False
    convergence = self.checkConvergence()
    ready = self.localStillReady(ready)
    #if converged and not ready, the optimizer believes it is done; check multilevel
    # -> however, if we're waiting on point collection, don't do multilevel check; only when we want to submit a new point.
    traj = 0 #FIXME for multiple trajectories
    if self.multilevel and self.status[traj][0].startswith('collecting'):
      print('DEBUGG ------------ MULTILEVEL CHECK ---------------')
      print('DEBUGG ... ready and converged:',ready,convergence)
      # make sure we have optimization history points; otherwise, there's nothing to be done
      if len(self.counter['recentOptHist'][traj][0]) > 0: #len(self.optVarsHist[traj])>0:
        # get the latset optimization point (normalized)
        latest_point = self.counter['recentOptHist'][traj][0]['inputs'] #self.latestPoint[traj]['inputs']#self.optVarsHist[traj][self.counter['varsUpdate'][traj]]
        # is this the first optimization point we've received?
        if self.mlDepth is None:
          # if so, then initialize multilevel, and no status change needed
          print('DEBUGG ... first opt point')
          self.updateMultilevelDepth(traj, len(self.mlSequence)-1, latest_point, setAll=True)
          # now we need to optimize the innermost space, so we are ready to provide samples
          return True
        # else if not the first opt point ... get the current batch since we'll use that going forward ...
        currentBatch = self.mlSequence[self.mlDepth]
        # are we in the innermost loop?
        if self.mlDepth == len(self.mlSequence)-1:
          print('DEBUGG ... in innermost loop ...')
          # then, are we converged?
          if convergence: #FIXME could be checked through self.status?
            print('DEBUGG ... converged ...')
            # then, move out one subspace in our loop
            self.updateMultilevelDepth(traj,self.mlDepth-1,latest_point)
            ready = True #FIXME needed? --> yes, because "ready" is false, but we know that we just changed levels and need to keep the sampling going
          # else if not converged ...
          else:
            # ... then we're ready to perturb (or wait for the return of) the active (innermost) space
            print('DEBUGG ... not converged ...')
            return ready
        # else if we're not in the innermost space ...
        else:
          # has the active space been perturbed since last convergence?
          if self.mlActiveSpaceSteps >= 1: #TODO someday this could be a user setting; for now, we only take one step in the outer loops at a time
            # are we in the outermost loop?
            if self.mlDepth == 0:
              # are we converged:
              if convergence:
                # we're done!
                self.raiseADebug('Outermost subspace converged!')
                return False #ready
              # else if not converged ...
              else:
                # set the active space to be the innermost loop so it can converge for the new outer loop value
                self.raiseADebug('Having perturbed outer space "{}", we now return to the innermost space ...'.format(currentBatch))
                self.updateMultilevelDepth(traj,len(self.mlSequence)-1,latest_point)
                #return True
            # else if not in the outermost loop ...
            else:
              # set the active space to the innermost loop so it can converge for the outer loop value
              self.raiseADebug('Converging outer loop "{}" by returning to innermost loop ...'.format(currentBatch))
              self.updateMultilevelDepth(traj,len(self.mlSequence)-1,latest_point)
          # else we haven't sampled a perturbed point yet for the active subspace ...
          else:
            # allow the current subspace to be perturbed
            self.raiseADebug('Perturbing subspace "{}" ...'.format(currentBatch))
            return ready
      print('DEBUGG ################# amiready repinging lsr #################')
      ready = self.localStillReady(True)
    print('DEBUGG amiready returning',ready)
    return ready

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
    if self.mlDepth is not None:
      oldDepth = self.mlDepth
      oldBatch = self.mlSequence[oldDepth]
      firstTime = False
    else:
      firstTime = True
      oldBatch = 'pre-initialize'
    # set the new active space
    self.mlDepth = depth
    newBatch = self.mlSequence[self.mlDepth]
    self.raiseADebug('Transitioning multilevel subspace from "{}" to "{}" ...'.format(oldBatch,newBatch))
    # reset the number of iterations in each subspace
    self.mlActiveSpaceSteps = 0
    # set the active space to include only the desired batch
    self.optVars = self.mlBatches[newBatch]
    # set the remainder to static variables
    if setAll:
      toMakeStatic = set(self.fullOptVars)-set(self.mlBatches[newBatch])
    else:
      toMakeStatic = self.mlBatches[oldBatch]
    for var in toMakeStatic:
      self.mlStaticValues[var] = optPoint[var]
    # remove newBatch static values
    for var in self.mlBatches[newBatch]:
      try:
        del self.mlStaticValues[var]
      except KeyError:
        #it wasn't static before, so no problem
        pass
    # clear existing gradient determination data
    if not firstTime:
      self.clearCurrentOptimizationEffort(traj)
    #self.mlReinitialize = True #FIXME Needed?

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
      @ Out, functionValue, float, the loss function value
    """
    objective  = self.mdlEvalHist.getParametersValues('outputs', nodeId = 'RecontructEnding')[self.objVar]
    prefix = self.mdlEvalHist.getMetadata('prefix',nodeId='RecontructEnding')
    if len(prefix) > 0 and utils.returnIdSeparator() in prefix[0]:
      # ensemble model id modification
      # FIXME: Need to find a better way to handle this case
      prefix = [key.split(utils.returnIdSeparator())[-1] for key in prefix]
    search = dict(zip(prefix, objective))
    functionValue = search.get(evaluationID,None)
    return functionValue

  def lossFunctionEval(self, optVars):
    """
      Method to evaluate the loss function based on all model evaluation.
      @ In, optVars, dict, dictionary containing the values of decision variables to be evaluated
                           optVars should have the form {varName1:[value11, value12,...value1n], varName2:[value21, value22,...value2n]...}
      @ Out, lossFunctionValue, numpy array, loss function values corresponding to each point in optVars
    """
    tempDict = copy.copy(self.mdlEvalHist.getParametersValues('inputs', nodeId = 'RecontructEnding'))
    tempDict.update(self.mdlEvalHist.getParametersValues('outputs', nodeId = 'RecontructEnding'))
    for key in tempDict.keys():
      tempDict[key] = np.asarray(tempDict[key])
    self.objSearchingROM.train(tempDict)
    # extend optVars to include static values
    numGradPts = len(optVars.values()[0])
    #print('DEBUGG optVars:',optVars)
    #print('DEBUGG in static:',self.mlStaticValues)
    for key,val in self.mlStaticValues.items():
      val = self.denormalizeData({key:val})[key]
      optVars[key] = np.array([val]*numGradPts)
    # denormalize the data # TODO right? since we train ROM on unnormalized data
    #print('DEBUGG optVars:',optVars)
    #optVars = self.denormalizeData(optVars)
    #print('DEBUGG optVars:',optVars)
    # fix data type
    for key in optVars.keys():
      optVars[key] = np.atleast_1d(optVars[key])
    # use KNN ROM to evaluate the loss function
    lossFunctionValue = self.objSearchingROM.evaluate(optVars)[self.objVar]
    #flip the solution around origin if performing maximization search
    if self.optType == 'min':
      return lossFunctionValue
    else:
      return lossFunctionValue*-1.0

  def checkConstraint(self, optVars):
    """
      Method to check whether a set of decision variables satisfy the constraint or not
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
    optVars = self.denormalizeData(optVars)
    for var in optVars:
      if optVars[var] > self.optVarsInit['upperBound'][var] or optVars[var] < self.optVarsInit['lowerBound'][var]:
        satisfied = False
        if optVars[var] > self.optVarsInit['upperBound'][var]:
          violatedConstrains['internal'].append([var,self.optVarsInit['upperBound'][var]])
        if optVars[var] < self.optVarsInit['lowerBound'][var]:
          violatedConstrains['internal'].append([var,self.optVarsInit['lowerBound'][var]])

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
    self.values.update(self.denormalizeData(self.mlStaticValues))
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
    allData.update(self.mlStaticValues) # these are normalized
    allData.update(self.normalizeData(data)) # data point not normalized a priori
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = copy.deepcopy(allData)
    self.mlActiveSpaceSteps += 1


  def removeConvergedTrajectory(self,convergedTraj):
    """
      Appropriate process for clearing out converged histories.  This lets the multilevel process intercede
      when a trajectory is flagged for removal, in the event it is part of an inner loop.
      @ In, convergedTraj, int, trajectory that has converged and might need to be removed
      @ Out, None
    """
    if self.multilevel:
      #FIXME when we work on multiple traj and multileve, figure out what to do here
      pass
    else:
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

