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
Module containing the different type of step allowed
Step is called by simulation
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import atexit
import time
import abc
import os
import sys
import pickle
import copy
import numpy as np
#import pickle as cloudpickle
import cloudpickle
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from EntityFactoryBase import EntityFactory
from BaseClasses import BaseEntity, InputDataUser
import Files
from utils import utils
from utils import InputData, InputTypes
import Models
from OutStreams import OutStreamEntity
from DataObjects import DataObject
from Databases import Database
#Internal Modules End--------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
class Step(utils.metaclass_insert(abc.ABCMeta, BaseEntity, InputDataUser)):
  """
    This class implement one step of the simulation pattern.
    Usage:
    myInstance = Step()                                !Generate the instance
    myInstance.XMLread(xml.etree.ElementTree.Element)  !This method read the xml and perform all the needed checks
    myInstance.takeAstep()                             !This method perform the step
    --Internal chain [in square brackets methods that can be/must be overwritten]
    self.XMLread(xml)-->self._readMoreXML(xml)     -->[self._localInputAndChecks(xmlNode)]
    self.takeAstep() -->self_initializeStep()      -->[self._localInitializeStep()]
                     -->[self._localTakeAstepRun()]
                     -->self._endStepActions()
    --Other external methods--
    myInstance.whoAreYou()                 -see BaseType class-
    myInstance.myCurrentSetting()          -see BaseType class-
    myInstance.printMe()                   -see BaseType class-
    --Adding a new step subclass--
     **<MyClass> should inherit at least from Step or from another step already presents
     **DO NOT OVERRIDE any of the class method that are not starting with self.local*
     **ADD your class to the dictionary __InterfaceDict at the end of the module
    Overriding the following methods overriding unless you inherit from one of the already existing methods:
    self._localInputAndChecks(xmlNode)      : used to specialize the xml reading and the checks
    self._localGetInitParams()              : used to retrieve the local parameters and values to be printed
    self._localInitializeStep(inDictionary) : called after this call the step should be able the accept the call self.takeAstep(inDictionary):
    self._localTakeAstepRun(inDictionary)   : this is where the step happens, after this call the output is ready
  """

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__(**kwargs)
    self.parList    = []   # List of list [[role played in the step, class type, specialization, global name (user assigned by the input)]]
    self.sleepTime  = 0.005  # Waiting time before checking if a run is finished
    #If a step possess re-seeding instruction it is going to ask to the sampler to re-seed according
    #  re-seeding = a number to be used as a new seed
    #  re-seeding = 'continue' the use the already present random environment
    #If there is no instruction (self.initSeed = None) the sampler will reinitialize
    self.initSeed = None
    self._excludeFromModelValidation = ['SolutionExport']
    # how to handle failed runs. By default, the step fails.
    # If the attribute "repeatFailureRuns" is inputted, a certain number of repetitions are going to be performed
    self.failureHandling = {"fail":True, "repetitions":0, "perturbationFactor":0.0, "jobRepetitionPerformed":{}}
    self.printTag = 'STEPS'
    self._clearRunDir = None

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.description = r"""
                       The \textbf{MultiRun} step allows the user to assemble the calculation flow of
                       an analysis that requires multiple ``runs'' of the same model.
                       This step is used, for example, when the input (space) of the model needs to be
                       perturbed by a particular sampling strategy.
                       The specifications of this type of step must be defined within a
                       \xmlNode{MultiRun} XML block."""

    inputSpecification.addParam("sleepTime", InputTypes.FloatType,
        descr='Determines the wait time between successive iterations within this step, in seconds.')
    inputSpecification.addParam("re-seeding", InputTypes.StringType, descr=r"""
              this optional
              attribute could be used to control the seeding of the random number generator (RNG).
              If inputted, the RNG can be reseeded. The value of this attribute
              can be: either 1) an integer value with the seed to be used (e.g. \xmlAttr{re-seeding} =
              ``20021986''), or 2) string value named ``continue'' where the RNG is not re-initialized""")
    inputSpecification.addParam("pauseAtEnd", InputTypes.StringType)
    inputSpecification.addParam("fromDirectory", InputTypes.StringType)
    inputSpecification.addParam("repeatFailureRuns", InputTypes.StringType)
    inputSpecification.addParam("clearRunDir", InputTypes.BoolType,
        descr=r"""indicates whether the run directory should be cleared (removed) before beginning
              the Step calculation. The run directory has the same name as the Step and is located
              within the WorkingDir. Note this directory is only used for Steps with certain Models,
              such as Code.
              \default{True}""")

    # for convenience, map subnodes to descriptions and loop through them
    subOptions = {'Input': 'Inputs to the step operation',
                  'Model': 'Entity containing the model to be executed',
                  'Sampler': 'Entity containing the sampling strategy',
                  'Output': 'Entity to store results of the step',
                  'Optimizer': 'Entity containing the optimization strategy',
                  'SolutionExport': 'Entity containing auxiliary output for the solution of this step',
                  'Function': 'Functional definition for use within this step',
                 }
    for stepPart, description in subOptions.items():
      stepPartInput = InputData.parameterInputFactory(stepPart,
                                                      contentType=InputTypes.StringType,
                                                      descr=description)
      stepPartInput.addParam("class", InputTypes.StringType, True)
      stepPartInput.addParam("type", InputTypes.StringType, True)
      inputSpecification.addSub(stepPartInput)

    return inputSpecification

  def _readMoreXML(self,xmlNode):
    """
      Handles the reading of all the XML describing the step
      Since step are not reused there will not be changes in the parameter describing the step after this reading
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    printString = 'For step of type {0:15} and name {1:15} the attribute {3:10} has been assigned to a not understandable value {2:10}'
    self.raiseADebug('move this tests to base class when it is ready for all the classes')
    if 're-seeding' in paramInput.parameterValues:
      self.initSeed=paramInput.parameterValues['re-seeding']
      if self.initSeed.lower()   == "continue":
        self.initSeed  = "continue"
      else:
        try:
          self.initSeed  = int(self.initSeed)
        except:
          self.raiseAnError(IOError,printString.format(self.type,self.name,self.initSeed,'re-seeding'))
    if 'sleepTime' in paramInput.parameterValues:
      self.sleepTime = paramInput.parameterValues['sleepTime']
    self._clearRunDir = paramInput.parameterValues.get('clearRunDir', True)
    for child in paramInput.subparts:
      classType = child.parameterValues['class']
      classSubType = child.parameterValues['type']
      self.parList.append([child.getName(),classType,classSubType,child.value])

    self.pauseEndStep = False
    if 'pauseAtEnd' in paramInput.parameterValues:
      if utils.stringIsTrue(paramInput.parameterValues['pauseAtEnd']):
        self.pauseEndStep = True
      elif utils.stringIsFalse(paramInput.parameterValues['pauseAtEnd']):
        self.pauseEndStep = False
      else:
        self.raiseAnError(IOError,printString.format(self.type,self.name,paramInput.parameterValues['pauseAtEnd'],'pauseAtEnd'),
                                  'expected one of {}'.format(utils.boolThingsFull))
    if 'repeatFailureRuns' in paramInput.parameterValues:
      failureSettings = str(paramInput.parameterValues['repeatFailureRuns']).split("|")
      self.failureHandling['fail'] = False
      #failureSettings = str(xmlNode.attrib['repeatFailureRuns']).split("|")
      #if len(failureSettings) not in [1,2]: (for future usage)
      #  self.raiseAnError(IOError,'repeatFailureRuns format error. Expecting either the repetition number only ' +
      #                            'or the repetition number and the perturbation factor separated by "|" symbol')
      if len(failureSettings) != 1:
        self.raiseAnError(IOError,'repeatFailureRuns format error. Expecting the repetition number only ')
      self.failureHandling['repetitions'] = utils.intConversion(failureSettings[0])
      #if len(failureSettings) == 2:
      #  self.failureHandling['perturbationFactor'] = utils.floatConversion(failureSettings[1])
      if self.failureHandling['repetitions'] is None:
        self.raiseAnError(IOError,'In Step named '+self.name+' it was not possible to cast "repetitions" attribute into an integer!')
      #if self.failureHandling['perturbationFactor'] is None:
      #  self.raiseAnError(IOError,'In Step named '+self.name+' it was not possible to cast "perturbationFactor" attribute into a float!')
    self._localInputAndCheckParam(paramInput)
    if None in self.parList:
      self.raiseAnError(IOError,'A problem was found in  the definition of the step '+str(self.name))

  @abc.abstractmethod
  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    pass

  def getInitParams(self):
    """
      Exports a dictionary with the information that will stay constant during the existence of the instance of this class. Overloaded from BaseType
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['Sleep time'  ] = str(self.sleepTime)
    paramDict['Initial seed'] = str(self.initSeed)
    for List in self.parList:
      paramDict[List[0]] = 'Class: '+str(List[1]) +' Type: '+str(List[2]) + '  Global name: '+str(List[3])
    paramDict.update(self._localGetInitParams())
    return paramDict

  @abc.abstractmethod
  def _localGetInitParams(self):
    """
      Place here a specialization of the exporting of what in the step is added to the initial parameters
      the printing format of paramDict is key: paramDict[key]
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}

  def _initializeStep(self,inDictionary):
    """
      Method to initialize the current step.
      the job handler is restarted and re-seeding action are performed
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    inDictionary['jobHandler'].startingNewStep()
    self.raiseADebug('jobHandler initialized')
    self._localInitializeStep(inDictionary)

  @abc.abstractmethod
  def _localInitializeStep(self,inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def _localTakeAstepRun(self,inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    pass

  def _registerMetadata(self,inDictionary):
    """
      collects expected metadata keys and deliver them to output data objects
      @ In, inDictionary, dict, initialization dictionary
      @ Out, None
    """
    ## first collect them
    metaKeys = set()
    metaParams = dict()
    for role,entities in inDictionary.items():
      if isinstance(entities,list):
        for entity in entities:
          if hasattr(entity,'provideExpectedMetaKeys'):
            keys, params = entity.provideExpectedMetaKeys()
            metaKeys = metaKeys.union(keys)
            metaParams.update(params)
      else:
        if hasattr(entities,'provideExpectedMetaKeys'):
          keys, params = entities.provideExpectedMetaKeys()
          metaKeys = metaKeys.union(keys)
          metaParams.update(params)
    ## then give them to the output data objects
    for out in inDictionary['Output']+(inDictionary['TargetEvaluation'] if 'TargetEvaluation' in inDictionary else []):
      if 'addExpectedMeta' in dir(out):
        out.addExpectedMeta(metaKeys,metaParams)

  def _endStepActions(self,inDictionary):
    """
      This method is intended for performing actions at the end of a step
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    if self.pauseEndStep:
      for i in range(len(inDictionary['Output'])):
        if inDictionary['Output'][i].type in ['Plot']:
          inDictionary['Output'][i].endInstructions('interactive')

  def takeAstep(self,inDictionary):
    """
      This should work for everybody just split the step in an initialization and the run itself
      inDictionary[role]=instance or list of instance
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    self.raiseAMessage('***  Beginning initialization ***')
    self._initializeStep(inDictionary)
    self.raiseAMessage('***    Initialization done    ***')
    self.raiseAMessage('***       Beginning run       ***')
    self._localTakeAstepRun(inDictionary)
    self.raiseAMessage('***       Run finished        ***')
    self.raiseAMessage('***     Closing the step      ***')
    self._endStepActions(inDictionary)
    self.raiseAMessage('***        Step closed        ***')
#
#
#
class SingleRun(Step):
  """
    This is the step that will perform just one evaluation
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.samplerType = 'Sampler'
    self.failedRuns = []
    self.lockedFileName = "ravenLocked.raven"
    self.printTag = 'STEP SINGLERUN'

  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    self.raiseADebug('the mapping used in the model for checking the compatibility of usage should be more similar to self.parList to avoid the double mapping below','FIXME')
    found     = 0
    rolesItem = []
    #collect model, other entries
    for index, parameter in enumerate(self.parList):
      if parameter[0]=='Model':
        found +=1
        modelIndex = index
      else:
        rolesItem.append(parameter[0])
    #test the presence of one and only one model
    if found > 1:
      self.raiseAnError(IOError,'Only one model is allowed for the step named '+str(self.name))
    elif found == 0:
      self.raiseAnError(IOError,'No model has been found for the step named '+str(self.name))
    #clarify run by roles
    roles      = set(rolesItem)
    if 'Optimizer' in roles:
      self.samplerType = 'Optimizer'
      if 'Sampler' in roles:
        self.raiseAnError(IOError, 'Only Sampler or Optimizer is alloweed for the step named '+str(self.name))
    #if single run, make sure model is an instance of Code class
    if self.type == 'SingleRun':
      if self.parList[modelIndex][2] != 'Code':
        self.raiseAnError(IOError,'<SingleRun> steps only support running "Code" model types!  Consider using a <MultiRun> step using a "Custom" sampler for other models.')
      if 'Optimizer' in roles or 'Sampler' in roles:
        self.raiseAnError(IOError,'<SingleRun> steps does not allow the usage of <Sampler> or <Optimizer>!  Consider using a <MultiRun> step.')
      if 'SolutionExport' in roles:
        self.raiseAnError(IOError,'<SingleRun> steps does not allow the usage of <SolutionExport>!  Consider using a <MultiRun> step with a <Sampler>/<Optimizer> that allows its usage.')
    #build entry list for verification of correct input types
    toBeTested = {}
    for role in roles:
      toBeTested[role]=[]
    for  myInput in self.parList:
      if myInput[0] in rolesItem:
        toBeTested[ myInput[0]].append({'class':myInput[1],'type':myInput[2]})
    #use the models static testing of roles compatibility
    for role in roles:
      if role not in self._excludeFromModelValidation:
        Models.validate(self.parList[modelIndex][2], role, toBeTested[role])
    self.raiseADebug('reactivate check on Input as soon as loadCsv gets out from the PostProcessor models!')
    if 'Output' not in roles:
      self.raiseAnError(IOError,'It is not possible a run without an Output!')

  def _localInitializeStep(self,inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    #Model initialization
    modelInitDict = {'Output':inDictionary['Output']}
    if 'SolutionExport' in inDictionary.keys():
      modelInitDict['SolutionExport'] = inDictionary['SolutionExport']
    if inDictionary['Model'].createWorkingDir:
      currentWorkingDirectory = os.path.join(inDictionary['jobHandler'].runInfoDict['WorkingDir'],
                                             inDictionary['jobHandler'].runInfoDict['stepName'])
      workingDirReady = False
      alreadyTried = False
      while not workingDirReady:
        try:
          os.mkdir(currentWorkingDirectory)
          workingDirReady = True
        except FileExistsError:
          if utils.checkIfPathAreAccessedByAnotherProgram(currentWorkingDirectory,3.0):
            self.raiseAWarning('directory '+ currentWorkingDirectory + ' is likely used by another program!!! ')
          if utils.checkIfLockedRavenFileIsPresent(currentWorkingDirectory,self.lockedFileName):
              self.raiseAnError(RuntimeError, self, "another instance of RAVEN is running in the working directory "+ currentWorkingDirectory+". Please check your input!")
          if self._clearRunDir and not alreadyTried:
            self.raiseAWarning(f'The calculation run directory {currentWorkingDirectory} already exists, ' +
                              'clearing existing files. This action can be disabled through the RAVEN Step input.')

            utils.removeDir(currentWorkingDirectory)
            alreadyTried = True
            continue
          else:
            if alreadyTried:
              self.raiseAWarning(f'The calculation run directory {currentWorkingDirectory} already exists, ' +
                                'and was not able to be removed. ' +
                                'Files present in this directory may be replaced, and error handling may not occur as expected.')
            else:
              self.raiseAWarning(f'The calculation run directory {currentWorkingDirectory} already exists. ' +
                                'Files present in this directory may be replaced, and error handling may not occur as expected.')
            workingDirReady = True
          # register function to remove the locked file at the end of execution
        atexit.register(utils.removeFile,os.path.join(currentWorkingDirectory,self.lockedFileName))
    inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'],modelInitDict)

    self.raiseADebug('for the role Model  the item of class {0:15} and name {1:15} has been initialized'.format(
      inDictionary['Model'].type,inDictionary['Model'].name))

    #Database initialization
    for i in range(len(inDictionary['Output'])):
      #if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
      # if 'Database' in inDictionary['Output'][i].type:
      if isinstance(inDictionary['Output'][i], Database):
        inDictionary['Output'][i].initialize(self.name)
      elif isinstance(inDictionary['Output'][i], OutStreamEntity):
        inDictionary['Output'][i].initialize(inDictionary)
      self.raiseADebug('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(inDictionary['Output'][i].type,inDictionary['Output'][i].name))
    self._registerMetadata(inDictionary)

  def _localTakeAstepRun(self,inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    jobHandler     = inDictionary['jobHandler']
    model          = inDictionary['Model'     ]
    sampler        = inDictionary.get(self.samplerType,None)
    inputs         = inDictionary['Input'     ]
    outputs        = inDictionary['Output'    ]

    # the input provided by a SingleRun is simply the file to be run.  model.run, however, expects stuff to perturb.
    # get an input to run -> different between SingleRun and PostProcessor runs
    # if self.type == 'SingleRun':
    #   newInput = model.createNewInput(inputs,'None',**{'SampledVars':{},'additionalEdits':{}})
    # else:
    #   newInput = inputs

    ## The single run should still collect its SampledVars for the output maybe?
    ## The problem here is when we call Code.collectOutput(), the sampledVars
    ## is empty... The question is where do we ultimately get this information
    ## the input object's input space or the desired output of the Output object?
    ## I don't think all of the outputs need to specify their domain, so I suppose
    ## this should default to all of the ones in the input? Is it possible to
    ## get an input field in the outputs variable that is not in the inputs
    ## variable defined above? - DPM 4/6/2017
    #empty dictionary corresponds to sampling data in MultiRun
    model.submit(inputs, None, jobHandler, **{'SampledVars':{'prefix':'None'}, 'additionalEdits':{}})
    while True:
      finishedJobs = jobHandler.getFinished()
      for finishedJob in finishedJobs:
        if finishedJob.getReturnCode() == 0:
          # if the return code is > 0 => means the system code crashed... we do not want to make the statistics poor => we discard this run
          for output in outputs:
            if not isinstance(output, OutStreamEntity):
              model.collectOutput(finishedJob, output)
            else:
              output.addOutput()
        else:
          self.raiseADebug('the job "'+finishedJob.identifier+'" has failed.')
          if self.failureHandling['fail']:
            #add run to a pool that can be sent to the sampler later
            self.failedRuns.append(copy.copy(finishedJob))
          else:
            if finishedJob.identifier not in self.failureHandling['jobRepetitionPerformed']:
              self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] = 1
            if self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] <= self.failureHandling['repetitions']:
              # we re-add the failed job
              jobHandler.reAddJob(finishedJob)
              self.raiseAWarning('As prescribed in the input, trying to re-submit the job "'+
                                 finishedJob.identifier+'". Trial '+
                               str(self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier]) +
                               '/'+str(self.failureHandling['repetitions']))
              self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] += 1
            else:
              #add run to a pool that can be sent to the sampler later
              self.failedRuns.append(copy.copy(finishedJob))
              self.raiseAWarning('The job "'+finishedJob.identifier+'" has been submitted '+
                                 str(self.failureHandling['repetitions'])+' times, failing all the times!!!')
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
        break
      time.sleep(self.sleepTime)
    if sampler is not None:
      sampler.handleFailedRuns(self.failedRuns)
    else:
      if len(self.failedRuns)>0:
        self.raiseAWarning('There were %i failed runs!' % len(self.failedRuns))

  def _localGetInitParams(self):
    """
      Place here a specialization of the exporting of what in the step is added to the initial parameters
      the printing format of paramDict is key: paramDict[key]
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}
#
#
#

class MultiRun(SingleRun):
  """
    this class implements one step of the simulation pattern' where several runs are needed
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._samplerInitDict = {} #this is a dictionary that gets sent as key-worded list to the initialization of the sampler
    self.counter          = 0  #just an handy counter of the runs already performed
    self.printTag = 'STEP MULTIRUN'

  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    SingleRun._localInputAndCheckParam(self,paramInput)
    if self.samplerType not in [item[0] for item in self.parList]:
      self.raiseAnError(IOError,'It is not possible a multi-run without a sampler or optimizer!')

  def _initializeSampler(self,inDictionary):
    """
      Method to initialize the sampler
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    if 'SolutionExport' in inDictionary.keys():
      self._samplerInitDict['solutionExport']=inDictionary['SolutionExport']

    inDictionary[self.samplerType].initialize(**self._samplerInitDict)
    self.raiseADebug('for the role of sampler the item of class '+inDictionary[self.samplerType].type+' and name '+inDictionary[self.samplerType].name+' has been initialized')
    self.raiseADebug('Sampler initialization dictionary: '+str(self._samplerInitDict))

  def _localInitializeStep(self,inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    SingleRun._localInitializeStep(self,inDictionary)
    # check that no input data objects are also used as outputs?
    for out in inDictionary['Output']:
      if out.type not in ['PointSet','HistorySet','DataSet']:
        continue
      for inp in inDictionary['Input']:
        if inp.type not in ['PointSet','HistorySet','DataSet']:
          continue
        if inp == out:
          self.raiseAnError(IOError,'The same data object should not be used as both <Input> and <Output> in the same MultiRun step! ' \
              + 'Step: "{}", DataObject: "{}"'.format(self.name,out.name))
    self.counter = 0
    self._samplerInitDict['externalSeeding'] = self.initSeed
    self._initializeSampler(inDictionary)
    #generate lambda function list to collect the output without checking the type
    self._outputCollectionLambda = []
    self._outputDictCollectionLambda = []
    # set up output collection lambdas
    for outIndex, output in enumerate(inDictionary['Output']):
      if not isinstance(output, OutStreamEntity):
        if 'SolutionExport' in inDictionary.keys() and output.name == inDictionary['SolutionExport'].name:
          self._outputCollectionLambda.append((lambda x:None, outIndex))
          self._outputDictCollectionLambda.append((lambda x:None, outIndex))
        else:
          self._outputCollectionLambda.append( (lambda x: inDictionary['Model'].collectOutput(x[0],x[1]), outIndex) )
          self._outputDictCollectionLambda.append( (lambda x: inDictionary['Model'].collectOutputFromDict(x[0],x[1]), outIndex) )
      else:
        self._outputCollectionLambda.append((lambda x: x[1].addOutput(), outIndex))
        self._outputDictCollectionLambda.append((lambda x: x[1].addOutput(), outIndex))
    self._registerMetadata(inDictionary)
    self.raiseADebug('Generating input batch of size '+str(inDictionary['jobHandler'].runInfoDict['batchSize']))
    # set up and run the first batch of samples
    # FIXME this duplicates a lot of code from _locatTakeAstepRun, which should be consolidated
    # first, check and make sure the model is ready
    model = inDictionary['Model']
    if isinstance(model,Models.ROM):
      if not model.amITrained:
        model.raiseAnError(RuntimeError,'ROM model "%s" has not been trained yet, so it cannot be sampled!' %model.name+\
                                        ' Use a RomTrainer step to train it.')
    for inputIndex in range(inDictionary['jobHandler'].runInfoDict['batchSize']):
      if inDictionary[self.samplerType].amIreadyToProvideAnInput():
        try:
          newInput = self._findANewInputToRun(inDictionary[self.samplerType], inDictionary['Model'], inDictionary['Input'], inDictionary['Output'], inDictionary['jobHandler'])
          if newInput is not None:
            inDictionary["Model"].submit(newInput, inDictionary[self.samplerType].type, inDictionary['jobHandler'], **copy.deepcopy(inDictionary[self.samplerType].inputInfo))
            self.raiseADebug('Submitted input '+str(inputIndex+1))
        except utils.NoMoreSamplesNeeded:
          self.raiseAMessage('Sampler returned "NoMoreSamplesNeeded".  Continuing...')
  @profile
  def _localTakeAstepRun(self,inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    jobHandler = inDictionary['jobHandler']
    model      = inDictionary['Model'     ]
    inputs     = inDictionary['Input'     ]
    outputs    = inDictionary['Output'    ]
    sampler    = inDictionary[self.samplerType]
    # check to make sure model can be run
    ## first, if it's a ROM, check that it's trained
    if isinstance(model,Models.ROM):
      if not model.amITrained:
        model.raiseAnError(RuntimeError,'ROM model "%s" has not been trained yet, so it cannot be sampled!' %model.name+\
                                        ' Use a RomTrainer step to train it.')
    # run step loop
    while True:
      # collect finished jobs
      finishedJobs = jobHandler.getFinished()

      ##FIXME: THE BATCH STRATEGY IS TOO INTRUSIVE. A MORE ELEGANT WAY NEEDS TO BE FOUND (E.G. REALIZATION OBJECT)
      for finishedJobObjs in finishedJobs:
        # NOTE: HERE WE RETRIEVE THE JOBS. IF BATCHING, THE ELEMENT IN finishedJobs is a LIST
        #       WE DO THIS in this way because:
        #           in case of BATCHING, the finalizeActualSampling method MUST BE called ONCE/BATCH
        #           otherwise, the finalizeActualSampling method MUST BE called ONCE/job
        #FIXME: This method needs to be improved since it is very intrusise
        if type(finishedJobObjs).__name__ in 'list':
          finishedJobList = finishedJobObjs
          self.raiseADebug('BATCHING: Collecting JOB batch named "{}".'.format(finishedJobList[0].groupId))
        else:
          finishedJobList = [finishedJobObjs]
        for finishedJob in finishedJobList:
          finishedJob.trackTime('step_collected')
          # update number of collected runs
          self.counter +=1
          # collect run if it succeeded
          if finishedJob.getReturnCode() == 0:
            for myLambda, outIndex in self._outputCollectionLambda:
              myLambda([finishedJob,outputs[outIndex]])
              self.raiseADebug('Just collected job {j:^8} and sent to output "{o}"'
                              .format(j=finishedJob.identifier,
                                      o=inDictionary['Output'][outIndex].name))
          # pool it if it failed, before we loop back to "while True" we'll check for these again
          else:
            self.raiseADebug('the job "{}" has failed.'.format(finishedJob.identifier))
            if self.failureHandling['fail']:
              # is this sampler/optimizer able to handle failed runs? If not, add the failed run in the pool
              if not sampler.ableToHandelFailedRuns:
                #add run to a pool that can be sent to the sampler later
                self.failedRuns.append(copy.copy(finishedJob))
            else:
              if finishedJob.identifier not in self.failureHandling['jobRepetitionPerformed']:
                self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] = 1
              if self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] <= self.failureHandling['repetitions']:
                # we re-add the failed job
                jobHandler.reAddJob(finishedJob)
                self.raiseAWarning('As prescribed in the input, trying to re-submit the job "'+finishedJob.identifier+'". Trial '+
                                 str(self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier]) +'/'+str(self.failureHandling['repetitions']))
                self.failureHandling['jobRepetitionPerformed'][finishedJob.identifier] += 1
              else:
                # is this sampler/optimizer able to handle failed runs? If not, add the failed run in the pool
                if not sampler.ableToHandelFailedRuns:
                  self.failedRuns.append(copy.copy(finishedJob))
                self.raiseAWarning('The job "'+finishedJob.identifier+'" has been submitted '+ str(self.failureHandling['repetitions'])+' times, failing all the times!!!')
            if sampler.ableToHandelFailedRuns:
              self.raiseAWarning('The sampler/optimizer "'+sampler.type+'" is able to handle failed runs!')
            #pop the failed job from the list
            finishedJobList.pop(finishedJobList.index(finishedJob))
        if type(finishedJobObjs).__name__ in 'list': # TODO: should be consistent, if no batching should batch size be 1 or 0 ?
          # if sampler claims it's batching, then only collect once, since it will collect the batch
          # together, not one-at-a-time
          # FIXME: IN HERE WE SEND IN THE INSTANCE OF THE FIRST JOB OF A BATCH
          # FIXME: THIS IS DONE BECAUSE CURRENTLY SAMPLERS/OPTIMIZERS RETRIEVE SOME INFO from the Runner instance but it can be
          # FIXME: dangerous if the sampler/optimizer requires info from each job. THIS MUST BE FIXED.
          sampler.finalizeActualSampling(finishedJobs[0][0],model,inputs)
        else:
          # sampler isn't intending to batch, so we send them in one-at-a-time as per normal
          for finishedJob in finishedJobList:
            # finalize actual sampler
            sampler.finalizeActualSampling(finishedJob,model,inputs)
        for finishedJob in finishedJobList:
          finishedJob.trackTime('step_finished')

        # terminate jobs as requested by the sampler, in case they're not needed anymore
        ## TODO is this a safe place to put this?
        ## If it's placed after adding new jobs and IDs are re-used i.e. for failed tests,
        ## -> then the new jobs will be killed if this is placed after new job submission!
        jobHandler.terminateJobs(sampler.getJobsToEnd(clear=True))

        # add new jobs, for DET-type samplers
        # put back this loop (do not take it away again. it is NEEDED for NOT-POINT samplers(aka DET)). Andrea
        # NOTE for non-DET samplers, this check also happens outside this collection loop
        if sampler.onlySampleAfterCollecting:
          self._addNewRuns(sampler, model, inputs, outputs, jobHandler, inDictionary)
      # END for each collected finished run ...
      ## If all of the jobs given to the job handler have finished, and the sampler
      ## has nothing else to provide, then we are done with this step.
      if jobHandler.isFinished() and not sampler.amIreadyToProvideAnInput():
        self.raiseADebug('Sampling finished with %d runs submitted, %d jobs running, and %d completed jobs waiting to be processed.' % (jobHandler.numSubmitted(),jobHandler.numRunning(),len(jobHandler.getFinishedNoPop())) )
        break
      if not sampler.onlySampleAfterCollecting:
        # NOTE for some reason submission outside collection breaks the DET
        # however, it is necessary i.e. batch sampling
        self._addNewRuns(sampler, model, inputs, outputs, jobHandler, inDictionary, verbose=False)
      time.sleep(self.sleepTime)
    # END while loop that runs the step iterations (collection and submission-for-DET)
    # if any collected runs failed, let the sampler treat them appropriately, and any other closing-out actions
    sampler.finalizeSampler(self.failedRuns)

  def _addNewRuns(self, sampler, model, inputs, outputs, jobHandler, inDictionary, verbose=True):
    """
      Checks for open spaces and adds new runs to jobHandler queue (via model.submit currently)
      @ In, sampler, Sampler, the sampler in charge of generating the sample
      @ In, model, Model, the model in charge of evaluating the sample
      @ In, inputs, object, the raven object used as the input in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, outputs, object, the raven object used as the output in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, jobHandler, object, the raven object used to handle jobs
      @ In, inDictionary, dict, additional step objects map
      @ In, verbose, bool, optional, if True print DEBUG statements
      @ Out, None
    """
    isEnsemble = isinstance(model, Models.EnsembleModel)
    ## In order to ensure that the queue does not grow too large, we will
    ## employ a threshold on the number of jobs the jobHandler can take,
    ## in addition, we cannot provide more jobs than the sampler can provide.
    ## So, we take the minimum of these two values.
    if verbose:
      self.raiseADebug('Testing if the sampler is ready to generate a new input')
    for _ in range(min(jobHandler.availability(isEnsemble), sampler.endJobRunnable())):
      if sampler.amIreadyToProvideAnInput():
        try:
          newInput = self._findANewInputToRun(sampler, model, inputs, outputs, jobHandler)
          if newInput is not None:
            model.submit(newInput, inDictionary[self.samplerType].type, jobHandler, **copy.deepcopy(sampler.inputInfo))
        except utils.NoMoreSamplesNeeded:
          self.raiseAMessage(' ... Sampler returned "NoMoreSamplesNeeded".  Continuing...')
          break
      else:
        if verbose:
          self.raiseADebug(' ... sampler has no new inputs currently.')
        break
    else:
      if verbose:
        self.raiseADebug(' ... no available JobHandler spots currently (or the Sampler is done.)')

  def _findANewInputToRun(self, sampler, model, inputs, outputs, jobHandler):
    """
      Repeatedly calls Sampler until a new run is found or "NoMoreSamplesNeeded" is raised.
      @ In, sampler, Sampler, the sampler in charge of generating the sample
      @ In, model, Model, the model in charge of evaluating the sample
      @ In, inputs, object, the raven object used as the input in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, outputs, object, the raven object used as the output in this step
        (i.e., a DataObject, File, or Database, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, jobHandler, object, the raven object used to handle jobs
      @ Out, newInp, list, list containing the new inputs (or None if a restart)
    """
    #The value of "found" determines what the Sampler is ready to provide.
    #  case 0: a new sample has been discovered and can be run, and newInp is a new input list.
    #  case 1: found the input in restart, and newInp is a realization dictionary of data to use
    found, newInp = sampler.generateInput(model,inputs)
    if found == 1:
      kwargs = copy.deepcopy(sampler.inputInfo)
      # "submit" the finished run
      jobHandler.addFinishedJob(newInp, metadata=kwargs)
      return None
      # NOTE: we return None here only because the Sampler's "counter" is not correctly passed
      # through if we add several samples at once through the restart. If we actually returned
      # a Realization object from the Sampler, this would not be a problem. - talbpaul
    return newInp

#
#
#

class PostProcess(SingleRun):
  """
    This is an alternate name for SingleRun
  """

#
#
#
class RomTrainer(Step):
  """
    This step type is used only to train a ROM
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'STEP ROM TRAINER'

  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    if [item[0] for item in self.parList].count('Input')!=1:
      self.raiseAnError(IOError,'Only one Input and only one is allowed for a training step. Step name: '+str(self.name))
    if [item[0] for item in self.parList].count('Output')<1:
      self.raiseAnError(IOError,'At least one Output is need in a training step. Step name: '+str(self.name))
    for item in self.parList:
      if item[0]=='Output' and item[2] not in ['ROM']:
        self.raiseAnError(IOError,'Only ROM output class are allowed in a training step. Step name: '+str(self.name))

  def _localGetInitParams(self):
    """
      Place here a specialization of the exporting of what in the step is added to the initial parameters
      the printing format of paramDict is key: paramDict[key]
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}

  def _localInitializeStep(self,inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    pass

  def _localTakeAstepRun(self,inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    #Train the ROM... It is not needed to add the trainingSet since it's already been added in the initialization method
    for ROM in inDictionary['Output']:
      ROM.train(inDictionary['Input'][0])
#
#
#
class IOStep(Step):
  """
    This step is used to extract or push information from/into a Database,
    or from a directory, or print out the data to an OutStream
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'STEP IOCOMBINED'
    self.fromDirectory = None

  def __getOutputs(self, inDictionary):
    """
      Utility method to get all the instances marked as Output
      @ In, inDictionary, dict, dictionary of all instances
      @ Out, outputs, list, list of Output instances
    """
    outputs         = []
    for out in inDictionary['Output']:
      if not isinstance(out, OutStreamEntity):
        outputs.append(out)
    return outputs

  def _localInitializeStep(self,inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    # check if #inputs == #outputs
    # collect the outputs without outstreams
    outputs = self.__getOutputs(inDictionary)
    databases = set()
    self.actionType = []
    errTemplate = 'In Step "{name}": When the Input is {inp}, this step accepts only {okay} as Outputs, ' +\
                  'but received "{received}" instead!'
    if len(inDictionary['Input']) != len(outputs) and len(outputs) > 0:
      self.raiseAnError(IOError,'In Step named ' + self.name + \
          ', the number of Inputs != number of Outputs, and there are Outputs. '+\
          'Inputs: %i Outputs: %i'%(len(inDictionary['Input']),len(outputs)) )
    #determine if this is a DATAS->Database, Database->DATAS or both.
    # also determine if this is an invalid combination
    for i in range(len(outputs)):
      # from Database to ...
      if isinstance(inDictionary['Input'][i], Database):
        ## ... dataobject
        if isinstance(outputs[i], DataObject.DataObject):
          self.actionType.append('Database-dataObjects')
        ## ... anything else
        else:
          self.raiseAnError(IOError,errTemplate.format(name = self.name,
                                                       inp = 'Database',
                                                       okay = 'DataObjects',
                                                       received = inDictionary['Output'][i].type))
      # from DataObject to ...
      elif  isinstance(inDictionary['Input'][i], DataObject.DataObject):
        ## ... Database
        if isinstance(outputs[i], Database):
          self.actionType.append('dataObjects-Database')
        ## ... anything else
        else:
          self.raiseAnError(IOError,errTemplate.format(name = self.name,
                                                       inp = 'DataObjects',
                                                       okay = 'Database',
                                                       received = inDictionary['Output'][i].type))
      # from ROM model to ...
      elif isinstance(inDictionary['Input'][i], Models.ROM):
        # ... file
        if isinstance(outputs[i],Files.File):
          self.actionType.append('ROM-FILES')
        # ... data object
        elif isinstance(outputs[i], DataObject.DataObject):
          self.actionType.append('ROM-dataObjects')
        # ... anything else
        else:
          self.raiseAnError(IOError,errTemplate.format(name = self.name,
                                                       inp = 'ROM',
                                                       okay = 'Files or DataObjects',
                                                       received = inDictionary['Output'][i].type))
      # from File to ...
      elif isinstance(inDictionary['Input'][i],Files.File):
        # ... ROM
        if isinstance(outputs[i],Models.ROM):
          self.actionType.append('FILES-ROM')
        # ... dataobject
        elif isinstance(outputs[i],DataObject.DataObject):
          self.actionType.append('FILES-dataObjects')
        # ... anything else
        else:
          self.raiseAnError(IOError,errTemplate.format(name = self.name,
                                                       inp = 'Files',
                                                       okay = 'ROM',
                                                       received = inDictionary['Output'][i].type))
      # from anything else to anything else
      else:
        self.raiseAnError(IOError,
                          'In Step "{name}": This step accepts only {okay} as Input. Received "{received}" instead!'
                          .format(name = self.name,
                                  okay = 'Database, DataObjects, ROM, or Files',
                                  received = inDictionary['Input'][i].type))
    # check actionType for fromDirectory
    if self.fromDirectory and len(self.actionType) == 0:
      self.raiseAnError(IOError,'In Step named ' + self.name + '. "fromDirectory" attribute provided but not conversion action is found (remove this atttribute for OutStream actions only"')
    #Initialize all the Database outputs.
    for i in range(len(outputs)):
      #if type(outputs[i]).__name__ not in ['str','bytes','unicode']:
      if isinstance(inDictionary['Output'][i], Database):
        if outputs[i].name not in databases:
          databases.add(outputs[i].name)
          outputs[i].initialize(self.name)
          self.raiseADebug('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(outputs[i].type,outputs[i].name))

    #if have a fromDirectory and are a dataObjects-*, need to load data
    if self.fromDirectory:
      for i in range(len(inDictionary['Input'])):
        if self.actionType[i].startswith('dataObjects-'):
          inInput = inDictionary['Input'][i]
          filename = os.path.join(self.fromDirectory, inInput.name)
          inInput.load(filename, style='csv')

    #Initialize all the OutStreams
    for output in inDictionary['Output']:
      if isinstance(output, OutStreamEntity):
        output.initialize(inDictionary)
        self.raiseADebug('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(output.type,output.name))
    # register metadata
    self._registerMetadata(inDictionary)

  def _localTakeAstepRun(self,inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    outputs = self.__getOutputs(inDictionary)
    for i in range(len(outputs)):
      if self.actionType[i] == 'Database-dataObjects':
        #inDictionary['Input'][i] is Database, outputs[i] is a DataObjects
        inDictionary['Input'][i].loadIntoData(outputs[i])
      elif self.actionType[i] == 'dataObjects-Database':
        #inDictionary['Input'][i] is a dataObjects, outputs[i] is Database
        outputs[i].saveDataToFile(inDictionary['Input'][i])

      elif self.actionType[i] == 'ROM-dataObjects':
        #inDictionary['Input'][i] is a ROM, outputs[i] is dataObject
        ## print information from the ROM to the data set or associated XML.
        romModel = inDictionary['Input'][i]
        # get non-pointwise data (to place in XML metadata of data object)
        ## TODO how can user ask for particular information?
        xml = romModel.writeXML(what='all')
        self.raiseADebug('Adding meta "{}" to output "{}"'.format(xml.getRoot().tag,outputs[i].name))
        outputs[i].addMeta(romModel.name, node = xml)
        # get pointwise data (to place in main section of data object)
        romModel.writePointwiseData(outputs[i])

      elif self.actionType[i] == 'ROM-FILES':
        #inDictionary['Input'][i] is a ROM, outputs[i] is Files
        ## pickle the ROM
        #check the ROM is trained first
        if not inDictionary['Input'][i].amITrained:
          self.raiseAnError(RuntimeError,'Pickled rom "%s" was not trained!  Train it before pickling and unpickling using a RomTrainer step.' %inDictionary['Input'][i].name)
        fileobj = outputs[i]
        fileobj.open(mode='wb+')
        cloudpickle.dump(inDictionary['Input'][i],fileobj)
        fileobj.flush()
        fileobj.close()
      elif self.actionType[i] == 'FILES-ROM':
        #inDictionary['Input'][i] is a Files, outputs[i] is ROM
        ## unpickle the ROM
        fileobj = inDictionary['Input'][i]
        unpickledObj = pickle.load(open(fileobj.getAbsFile(),'rb+'))
        ## DEBUGG
        # the following will iteratively check the size of objects being unpickled
        # this is quite useful for finding memory crashes due to parallelism
        # so I'm leaving it here for reference
        # print('CHECKING SIZE OF', unpickledObj)
        # target = unpickledObj# .supervisedEngine.supervisedContainer[0]._macroSteps[2025]._roms[0]
        # print('CHECKING SIZES')
        # from utils.Debugging import checkSizesWalk
        # checkSizesWalk(target, 1, str(type(target)), tol=2e4)
        # print('*'*80)
        # crashme
        ## /DEBUGG
        if not isinstance(unpickledObj,Models.ROM):
          self.raiseAnError(RuntimeError,'Pickled object in "%s" is not a ROM.  Exiting ...' %str(fileobj))
        if not unpickledObj.amITrained:
          self.raiseAnError(RuntimeError,'Pickled rom "%s" was not trained!  Train it before pickling and unpickling using a RomTrainer step.' %unpickledObj.name)
        # save reseeding parameters from pickledROM
        loadSettings = outputs[i].initializationOptionDict
        # train the ROM from the unpickled object
        outputs[i].train(unpickledObj)
        # reseed as requested
        outputs[i].setAdditionalParams(loadSettings)

      elif self.actionType[i] == 'FILES-dataObjects':
        #inDictionary['Input'][i] is a Files, outputs[i] is PointSet
        ## load a CSV from file
        infile = inDictionary['Input'][i]
        options = {'fileToLoad':infile}
        outputs[i].load(inDictionary['Input'][i].getPath(),'csv',**options)

      else:
        # unrecognized, and somehow not caught by the step reader.
        self.raiseAnError(IOError,"Unknown action type "+self.actionType[i])

    for output in inDictionary['Output']:
      if isinstance(output, OutStreamEntity):
        output.addOutput()

  def _localGetInitParams(self):
    """
      Place here a specialization of the exporting of what in the step is added to the initial parameters
      the printing format of paramDict is key: paramDict[key]
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    return paramDict # no inputs

  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    if 'fromDirectory' in paramInput.parameterValues:
      self.fromDirectory = paramInput.parameterValues['fromDirectory']



factory = EntityFactory('Step')
factory.registerAllSubtypes(Step)
