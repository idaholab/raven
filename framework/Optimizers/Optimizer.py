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
import utils
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
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
    BaseType.__init__(self)
    Assembler.__init__(self)
    self.counter                        = {}                        # Dict containing counters used for based and derived class
    self.counter['mdlEval']             = 0                         # Counter of the model evaluation performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.counter['varsUpdate']          = 0                         # Counter of the optimization iteration.
    self.limit                          = {}                        # Dict containing limits for each counter
    self.limit['mdlEval']               = sys.maxsize               # Maximum number of the loss function evaluation
    self.limit['varsUpdate']            = sys.maxsize               # Maximum number of the optimization iteration.
    self.initSeed                       = None                      # Seed for random number generators
    self.optVars                        = None                      # Decision variables for optimization
    self.optVarsInit                    = {}                        # Dict containing upper/lower bounds and initial of each decision variables
    self.optVarsInit['upperBound']      = {}                        # Dict containing upper bounds of each decision variables
    self.optVarsInit['lowerBound']      = {}                        # Dict containing lower bounds of each decision variables
    self.optVarsInit['initial']         = {}                        # Dict containing initial values of each decision variables
    self.optVarsHist                    = {}                        # History of decision variables for each iteration
    self.nVar                           = 0                         # Number of decision variables
    self.objVar                         = None                      # Objective variable to be optimized
    self.optType                        = None                      # Either maximize or minimize
    self.paramDict                      = {}                        # Dict containing additional parameters for derived class
    self.convergenceTol                 = 1e-3                      # Convergence threshold
    self.solutionExport                 = None                      #This is the data used to export the solution (it could also not be present)
    self.values                         = {}                        # for each variable the current value {'var name':value}
    self.inputInfo                      = {}                        # depending on the optimizer several different type of keywarded information could be present only one is mandatory, see below
    self.inputInfo['SampledVars'     ]  = self.values               # this is the location where to get the values of the sampled variables
    self.FIXME                          = False                     # FIXME flag
    self.printTag                       = self.type                 # prefix for all prints (optimizer type)

    self._endJobRunnable                = sys.maxsize               # max number of inputs creatable by the optimizer right after a job ends

    self.constraintFunction             = None                      # External constraint function, could be not present
    self.mdlEvalHist                    = None                      # Containing information of all model evaluation
    self.objSearchingROM                = None                      # ROM used internally for fast loss function evaluation

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
      if child.tag == "variable":
        if self.optVars == None: self.optVars = []
        varname = str(child.attrib['name'])
        self.optVars.append(varname)
        for childChild in child:
          if   childChild.tag == "upperBound": self.optVarsInit['upperBound'][varname] = float(childChild.text)
          elif childChild.tag == "lowerBound": self.optVarsInit['lowerBound'][varname] = float(childChild.text)
          elif childChild.tag == "initial"   : self.optVarsInit['initial'][varname] = float(childChild.text)
        if varname not in self.optVarsInit['upperBound'].keys(): self.optVarsInit['upperBound'][varname] = sys.maxsize
        if varname not in self.optVarsInit['lowerBound'].keys(): self.optVarsInit['lowerBound'][varname] = -sys.maxsize
        if varname not in self.optVarsInit['initial'].keys()   : self.optVarsInit['initial'][varname] = 0.0

      elif child.tag == "objectVar":
        self.objVar = child.text

      elif child.tag == "initialization":
        self.initSeed = Distributions.randomIntegers(0,2**31,self)
        for childChild in child:
          if childChild.tag == "limit":
            self.limit['mdlEval'] = int(childChild.text)
          elif childChild.tag == "type":
            self.optType = childChild.text
            if self.optType not in ['min', 'max']:
              self.raiseAnError(IOError, 'Unknown optimization type '+childChild.text+'. Available: mix or max')
          elif childChild.tag == "initialSeed":
            self.initSeed = int(childChild.text)
          else: self.raiseAnError(IOError,'Unknown tag '+childChild.tag+' .Available: limit, type, initialSeed!')

      elif child.tag == "convergence":
        for childChild in child:
          if childChild.tag == "iterationLimit":
            self.limit['varsUpdate'] = int(childChild.text)
          if childChild.tag == "threshold":
            self.convergenceTol = float(childChild.text)

      elif child.tag == "restartTolerance":
        self.restartTolerance = float(child.text)

      elif child.tag == 'parameter':
        for childChild in child:
          self.paramDict[childChild.tag] = childChild.text

    if self.optType == None:    self.optType = 'min'
    if self.initSeed == None:   self.initSeed = Distributions.randomIntegers(0,2**31,self)
    if self.objVar == None:     self.raiseAnError(IOError, 'Object variable is not specified for optimizer!')
    if self.optVars == None:
      self.raiseAnError(IOError, 'Decision variable is not specified for optimizer!')
    else:
      self.optVars.sort()

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
      @ In, solutionExport, DataObject, optional
      @ Out, None
    """
    self.counter['mdlEval'] = 0
    self.counter['varsUpdate'] = 0
    self.nVar = len(self.optVars)

    self.mdlEvalHist = self.assemblerDict['TargetEvaluation'][0][3]
    self.objSearchingROM = SupervisedLearning.returnInstance('SciKitLearn', self, **{'SKLtype':'neighbors|KNeighborsRegressor', 'Features':','.join(list(self.optVars)), 'Target':self.objVar, 'n_neighbors':1})
    self.solutionExport = solutionExport

    if solutionExport != None and type(solutionExport).__name__ != "PointSet":
      self.raiseAnError(IOError,'solutionExport type is not a PointSet. Got '+ type(solutionExport).__name__+ '!')

    if 'Function' in self.assemblerDict.keys():
      self.constraintFunction = self.assemblerDict['Function'][0][3]
      if 'constrain' not in self.constrainFunction.availableMethods():
        self.raiseAnError(IOError,'the function provided to define the constraints must have an implemented method called "constrain"')

    if self.initSeed != None:           Distributions.randomSeed(self.initSeed)

    # specializing the self.localInitialize()
    if solutionExport != None : self.localInitialize(solutionExport=solutionExport)
    else                      : self.localInitialize()

  def localInitialize(self,solutionExport=None):
    """
      use this function to add initialization features to the derived class
      it is call at the beginning of each step
      @ In, solutionExport, DataObject, optional
      @ Out, None
    """
    pass # To be overwritten by subclass

  def amIreadyToProvideAnInput(self): #inLastOutput=None):
    """
      This is a method that should be called from any user of the optimizer before requiring the generation of a new input.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of model evaluation, convergence criteria met, etc.
      @ In, None
      @ Out, ready, bool, indicating the readiness of the optimizer to generate a new input.
    """
    ready = True if self.counter['mdlEval'] < self.limit['mdlEval'] and self.counter['varsUpdate'] < self.limit['varsUpdate'] else False
    convergence = self.checkConvergence()
    ready = self.localStillReady(ready, convergence)
    return ready

  def localStillReady(self,ready, convergence = False): #,lastOutput=None
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @In, ready, bool, boolean variable indicating whether the caller is prepared for another input.
      @In, convergence, boolean variable indicating whether the convergence criteria has been met.
      @Out, ready, bool, boolean variable indicating whether the caller is prepared for another input.
    """
    return ready # To be overwritten by subclass

  def lossFunctionEval(self, optVars):
    """
      Method to evaluate the loss function based on all model evaluation.
      @In, optVars, dict containing the values of decision variables to be evaluated
           optVars should have the form {varName1:[value11, value12,...value1n], varName2:[value21, value22,...value2n]...}
      @Out, lossFunctionValue, array, loss function values corresponding to each point in optVars
    """
    tempDict = copy.copy(self.mdlEvalHist.getParametersValues('inputs', nodeId = 'RecontructEnding'))
    tempDict.update(self.mdlEvalHist.getParametersValues('outputs', nodeId = 'RecontructEnding'))
    for key in tempDict.keys():                   tempDict[key] = np.asarray(tempDict[key])
    self.objSearchingROM.train(tempDict)

    for key in optVars.keys():                    optVars[key] = np.atleast_1d(optVars[key])
    lossFunctionValue = self.objSearchingROM.evaluate(optVars)
    if self.optType == 'min':           return lossFunctionValue
    else:                               return lossFunctionValue*-1.0

  def checkConstraint(self, optVars):
    """
      Method to check whether a set of decision variables satisfy the constraint or not
      @In, optVars, dict containing the value of decision variables to be checked, in form of {varName: varValue}
      @Out, satisfaction, boolean variable indicating the satisfaction of contraints at the point optVars
    """
    if self.constraintFunction == None:
      satisfaction = True
    else:
      satisfaction = True if self.constraintFunction.evaluate("constrain",optVars) == 1 else False
    satisfaction = self.localCheckConstraint(optVars, satisfaction)
    return satisfaction

  def localCheckConstraint(self, optVars, satisfaction = True):
    """
      Local method to check whether a set of decision variables satisfy the constraint or not
      @In, optVars, dict containing the value of decision variables to be checked, in form of {varName: varValue}
      @In, satisfaction, boolean variable indicating how the caller determines the constraint satisfaction at the point optVars
      @Out, satisfaction, boolean variable indicating the satisfaction of constraints at the point optVars
    """
    return satisfaction # To be overwritten by subclass

  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, none,
      @ Out, convergence, boolean variable indicating whether the convergence criteria has been met.
    """
    if self.counter['varsUpdate'] < 2:
      convergence = False
    elif abs(self.lossFunctionEval(self.optVarsHist[self.counter['varsUpdate']])-self.lossFunctionEval(self.optVarsHist[self.counter['varsUpdate']-1])) < self.convergenceTol:
      convergence = True
    else:
      convergence = False
    convergence = self.localCheckConvergence(convergence)
    return convergence

  @abc.abstractmethod
  def localCheckConvergence(self, convergence = False):
    """
      Local method to check convergence.
      @ In, convergence, boolean variable indicating how the caller determines the convergence.
      @ Out, convergence, boolean variable indicating whether the convergence criteria has been met.
    """
    return convergence

  def generateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    """
    self.counter['mdlEval'] +=1                              #since we are creating the input for the next run we increase the counter and global counter
    self.inputInfo['prefix'] = str(self.counter['mdlEval'])

    model.getAdditionalInputEdits(self.inputInfo)
    self.localGenerateInput(model,oldInput)

    self.raiseADebug('Found new input to evaluate:',self.values)
    return 0,model.createNewInput(oldInput,self.type,**self.inputInfo)

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
        metadata = run.returnMetadata()
        self.raiseADebug('  Run number %s FAILED:' %run.identifier,run.command)
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
