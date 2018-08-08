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
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import atexit
import time
import abc
import os
import sys
import itertools
if sys.version_info.major > 2:
  import pickle
else:
  import cPickle as pickle
import copy
import numpy as np
#import pickle as cloudpickle
import cloudpickle
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
import Files
from utils import utils
from utils import InputData
import Models
from OutStreams import OutStreamManager
from DataObjects import DataObject
#Internal Modules End--------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
class Step(utils.metaclass_insert(abc.ABCMeta,BaseType)):
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

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)
    self.parList    = []   # List of list [[role played in the step, class type, specialization, global name (user assigned by the input)]]
    self.sleepTime  = 0.005  # Waiting time before checking if a run is finished
    #If a step possess re-seeding instruction it is going to ask to the sampler to re-seed according
    #  re-seeding = a number to be used as a new seed
    #  re-seeding = 'continue' the use the already present random environment
    #If there is no instruction (self.initSeed = None) the sampler will reinitialize
    self.initSeed        = None
    self._knownAttribute += ['sleepTime','re-seeding','pauseAtEnd','fromDirectory','repeatFailureRuns']
    self._excludeFromModelValidation = ['SolutionExport']
    # how to handle failed runs. By default, the step fails.
    # If the attribute "repeatFailureRuns" is inputted, a certain number of repetitions are going to be performed
    self.failureHandling = {"fail":True, "repetitions":0, "perturbationFactor":0.0, "jobRepetitionPerformed":{}}
    self.printTag = 'STEPS'

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Step, cls).getInputSpecification()

    inputSpecification.addParam("sleepTime", InputData.FloatType)
    inputSpecification.addParam("re-seeding", InputData.StringType)
    inputSpecification.addParam("pauseAtEnd", InputData.StringType)
    inputSpecification.addParam("fromDirectory", InputData.StringType)
    inputSpecification.addParam("repeatFailureRuns", InputData.StringType)

    for stepPart in ["Input","Model","Sampler","Output","Optimizer","SolutionExport","Function"]:
      stepPartInput = InputData.parameterInputFactory(stepPart, contentType=InputData.StringType)
      stepPartInput.addParam("class", InputData.StringType, True)
      stepPartInput.addParam("type", InputData.StringType, True)
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
    if not set(paramInput.parameterValues.keys()).issubset(set(self._knownAttribute)):
      self.raiseAnError(IOError,'In step of type {0:15} and name {1:15} there are unknown attributes {2:100}'.format(self.type,self.name,str(paramInput.parameterValues.keys())))
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
    for child in paramInput.subparts:
      classType = child.parameterValues['class']
      classSubType = child.parameterValues['type']
      self.parList.append([child.getName(),classType,classSubType,child.value])

    self.pauseEndStep = False
    if 'pauseAtEnd' in paramInput.parameterValues:
      if   paramInput.parameterValues['pauseAtEnd'].lower() in utils.stringsThatMeanTrue():
        self.pauseEndStep = True
      elif paramInput.parameterValues['pauseAtEnd'].lower() in utils.stringsThatMeanFalse():
        self.pauseEndStep = False
      else:
        self.raiseAnError(IOError,printString.format(self.type,self.name,paramInput.parameterValues['pauseAtEnd'],'pauseAtEnd'))
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
    for role,entities in inDictionary.items():
      if isinstance(entities,list):
        for entity in entities:
          if hasattr(entity,'provideExpectedMetaKeys'):
            metaKeys = metaKeys.union(entity.provideExpectedMetaKeys())
      else:
        if hasattr(entities,'provideExpectedMetaKeys'):
          metaKeys = metaKeys.union(entities.provideExpectedMetaKeys())
    ## then give them to the output data objects
    for out in inDictionary['Output']+(inDictionary['TargetEvaluation'] if 'TargetEvaluation' in inDictionary else []):
      if 'addExpectedMeta' in dir(out):
        out.addExpectedMeta(metaKeys)

  def _endStepActions(self,inDictionary):
    """
      This method is intended for performing actions at the end of a step
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    if self.pauseEndStep:
      for i in range(len(inDictionary['Output'])):
        #if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
        if inDictionary['Output'][i].type in ['OutStreamPlot']:
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
    Step.__init__(self)
    self.samplerType    = 'Sampler'
    self.failedRuns     = []
    self.lockedFileName = "ravenLocked.raven"
    self.printTag       = 'STEP SINGLERUN'

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
        Models.validate(self.parList[modelIndex][2], role, toBeTested[role],self)
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
      try:
        os.mkdir(currentWorkingDirectory)
      except OSError:
        self.raiseAWarning('current working dir '+currentWorkingDirectory+' already exists, ' +
                           'this might imply deletion of present files')
        if utils.checkIfPathAreAccessedByAnotherProgram(currentWorkingDirectory,3.0):
          self.raiseAWarning('directory '+ currentWorkingDirectory + ' is likely used by another program!!! ')
        if utils.checkIfLockedRavenFileIsPresent(currentWorkingDirectory,self.lockedFileName):
          self.raiseAnError(RuntimeError, self, "another instance of RAVEN is running in the working directory "+ currentWorkingDirectory+". Please check your input!")
        # register function to remove the locked file at the end of execution
        atexit.register(utils.removeFile,os.path.join(currentWorkingDirectory,self.lockedFileName))
    inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'],modelInitDict)

    self.raiseADebug('for the role Model  the item of class {0:15} and name {1:15} has been initialized'.format(
      inDictionary['Model'].type,inDictionary['Model'].name))

    #HDF5 initialization
    for i in range(len(inDictionary['Output'])):
      #if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
      if 'HDF5' in inDictionary['Output'][i].type:
        inDictionary['Output'][i].initialize(self.name)
      elif inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']:
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
    model.submit(inputs, None, jobHandler, **{'SampledVars':{'prefix':'None'},'additionalEdits':{}})
    while True:
      finishedJobs = jobHandler.getFinished()
      for finishedJob in finishedJobs:
        if finishedJob.getReturnCode() == 0:
          # if the return code is > 0 => means the system code crashed... we do not want to make the statistics poor => we discard this run
          for output in outputs:
            if output.type not in ['OutStreamPlot','OutStreamPrint']:
              model.collectOutput(finishedJob,output)
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
    SingleRun.__init__(self)
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
      if output.type not in ['OutStreamPlot','OutStreamPrint']:
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
          newInput = self._findANewInputToRun(inDictionary[self.samplerType], inDictionary['Model'], inDictionary['Input'], inDictionary['Output'])
          inDictionary["Model"].submit(newInput, inDictionary[self.samplerType].type, inDictionary['jobHandler'], **copy.deepcopy(inDictionary[self.samplerType].inputInfo))
          self.raiseADebug('Submitted input '+str(inputIndex+1))
        except utils.NoMoreSamplesNeeded:
          self.raiseAMessage('Sampler returned "NoMoreSamplesNeeded".  Continuing...')

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
      for finishedJob in finishedJobs:
        finishedJob.trackTime('step_collected')
        # update number of collected runs
        self.counter +=1
        # collect run if it succeeded
        if finishedJob.getReturnCode() == 0:
          for myLambda, outIndex in self._outputCollectionLambda:
            myLambda([finishedJob,outputs[outIndex]])
            self.raiseADebug('Just collected output {0:2} of the input {1:6}'.format(outIndex+1,self.counter))
        # pool it if it failed, before we loop back to "while True" we'll check for these again
        else:
          self.raiseADebug('the job "'+finishedJob.identifier+'" has failed.')
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
        # finalize actual sampler
        sampler.finalizeActualSampling(finishedJob,model,inputs)
        finishedJob.trackTime('step_finished')
        # add new job

        isEnsemble = isinstance(model, Models.EnsembleModel)
        # put back this loop (do not take it away again. it is NEEDED for NOT-POINT samplers(aka DET)). Andrea
        ## In order to ensure that the queue does not grow too large, we will
        ## employ a threshold on the number of jobs the jobHandler can take,
        ## in addition, we cannot provide more jobs than the sampler can provide.
        ## So, we take the minimum of these two values.
        for _ in range(min(jobHandler.availability(isEnsemble),sampler.endJobRunnable())):
          self.raiseADebug('Testing if the sampler is ready to generate a new input')

          if sampler.amIreadyToProvideAnInput():
            try:
              newInput = self._findANewInputToRun(sampler, model, inputs, outputs)
              model.submit(newInput, inDictionary[self.samplerType].type, jobHandler, **copy.deepcopy(sampler.inputInfo))
            except utils.NoMoreSamplesNeeded:
              self.raiseAMessage('Sampler returned "NoMoreSamplesNeeded".  Continuing...')
              break
          else:
            break
      ## If all of the jobs given to the job handler have finished, and the sampler
      ## has nothing else to provide, then we are done with this step.
      if jobHandler.isFinished() and not sampler.amIreadyToProvideAnInput():
        self.raiseADebug('Finished with %d runs submitted, %d jobs running, and %d completed jobs waiting to be processed.' % (jobHandler.numSubmitted(),jobHandler.numRunning(),len(jobHandler.getFinishedNoPop())) )
        break
      time.sleep(self.sleepTime)
    # END while loop that runs the step iterations
    # if any collected runs failed, let the sampler treat them appropriately, and any other closing-out actions
    sampler.finalizeSampler(self.failedRuns)

  def _findANewInputToRun(self, sampler, model, inputs, outputs):
    """
      Repeatedly calls Sampler until a new run is found or "NoMoreSamplesNeeded" is raised.
      @ In, sampler, Sampler, the sampler in charge of generating the sample
      @ In, model, Model, the model in charge of evaluating the sample
      @ In, inputs, object, the raven object used as the input in this step
        (i.e., a DataObject, File, or HDF5, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ In, outputs, object, the raven object used as the output in this step
        (i.e., a DataObject, File, or HDF5, I guess? Maybe these should all
        inherit from some base "Data" so that we can ensure a consistent
        interface for these?)
      @ Out, newInp, list, list containing the new inputs
    """
    #The value of "found" determines what the Sampler is ready to provide.
    #  case 0: a new sample has been discovered and can be run, and newInp is a new input list.
    #  case 1: found the input in restart, and newInp is a realization dicitonary of data to use
    found = None
    while found != 0:
      found,newInp = sampler.generateInput(model,inputs)
      if found == 1:
        # loop over the outputs for this step and collect the data for each
        for collector, outIndex in self._outputDictCollectionLambda:
          collector([newInp,outputs[outIndex]])
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
    Step.__init__(self)
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
    Step.__init__(self)
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
      if not isinstance(out,OutStreamManager):
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
    outputs         = self.__getOutputs(inDictionary)
    databases       = set()
    self.actionType = []
    if len(inDictionary['Input']) != len(outputs) and len(outputs) > 0:
      self.raiseAnError(IOError,'In Step named ' + self.name + \
          ', the number of Inputs != number of Outputs, and there are Outputs. '+\
          'Inputs: %i Outputs: %i'%(len(inDictionary['Input']),len(outputs)) )
    #determine if this is a DATAS->HDF5, HDF5->DATAS or both.
    # also determine if this is an invalid combination
    for i in range(len(outputs)):
      if inDictionary['Input'][i].type == 'HDF5':
        if isinstance(outputs[i],DataObject.DataObject):
          self.actionType.append('HDF5-dataObjects')
        else:
          self.raiseAnError(IOError,'In Step named ' + self.name + '. This step accepts A DataObjects as Output only, when the Input is an HDF5. Got ' + inDictionary['Output'][i].type)
      elif  isinstance(inDictionary['Input'][i],DataObject.DataObject):
        if outputs[i].type == 'HDF5':
          self.actionType.append('dataObjects-HDF5')
        else:
          self.raiseAnError(IOError,'In Step named ' + self.name + '. This step accepts ' + 'HDF5' + ' as Output only, when the Input is a DataObjects. Got ' + inDictionary['Output'][i].type)
      elif isinstance(inDictionary['Input'][i],Models.ROM):
        if isinstance(outputs[i],Files.File):
          self.actionType.append('ROM-FILES')
        else:
          self.raiseAnError(IOError,'In Step named ' + self.name + '. This step accepts A Files as Output only, when the Input is a ROM. Got ' + inDictionary['Output'][i].type)
      elif isinstance(inDictionary['Input'][i],Files.File):
        if   isinstance(outputs[i],Models.ROM):
          self.actionType.append('FILES-ROM')
        elif isinstance(outputs[i],DataObject.DataObject):
          self.actionType.append('FILES-dataObjects')
        else:
          self.raiseAnError(IOError,'In Step named ' + self.name + '. This step accepts A ROM as Output only, when the Input is a Files. Got ' + inDictionary['Output'][i].type)
      else:
        self.raiseAnError(IOError,'In Step named ' + self.name + '. This step accepts DataObjects, HDF5, ROM and Files as Input only. Got ' + inDictionary['Input'][i].type)
    if self.fromDirectory and len(self.actionType) == 0:
      self.raiseAnError(IOError,'In Step named ' + self.name + '. "fromDirectory" attribute provided but not conversion action is found (remove this atttribute for OutStream actions only"')
    #Initialize all the HDF5 outputs.
    for i in range(len(outputs)):
      #if type(outputs[i]).__name__ not in ['str','bytes','unicode']:
      if 'HDF5' in inDictionary['Output'][i].type:
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

    #Initialize all the OutStreamPrint and OutStreamPlot outputs
    for output in inDictionary['Output']:
      if type(output).__name__ in ['OutStreamPrint','OutStreamPlot']:
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
      if self.actionType[i] == 'HDF5-dataObjects':
        #inDictionary['Input'][i] is HDF5, outputs[i] is a DataObjects
        allRealizations = inDictionary['Input'][i].allRealizations()
        ## TODO convert to load function when it can handle unstructured multiple realizations
        for rlz in allRealizations:
          outputs[i].addRealization(rlz)
      elif self.actionType[i] == 'dataObjects-HDF5':
        #inDictionary['Input'][i] is a dataObjects, outputs[i] is HDF5
        ## TODO convert to load function when it can handle unstructured multiple realizations
        for rlzNo in range(len(inDictionary['Input'][i])):
          rlz = inDictionary['Input'][i].realization(rlzNo, unpackXArray=True)
          rlz = dict((var,np.atleast_1d(val)) for var, val in rlz.items())
          outputs[i].addRealization(rlz)

      elif self.actionType[i] == 'ROM-FILES':
        #inDictionary['Input'][i] is a ROM, outputs[i] is Files
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
        fileobj = inDictionary['Input'][i]
        unpickledObj = pickle.load(open(fileobj.getAbsFile(),'rb+'))
        if not isinstance(unpickledObj,Models.ROM):
          self.raiseAnError(RuntimeError,'Pickled object in "%s" is not a ROM.  Exiting ...' %str(fileobj))
        if not unpickledObj.amITrained:
          self.raiseAnError(RuntimeError,'Pickled rom "%s" was not trained!  Train it before pickling and unpickling using a RomTrainer step.' %unpickledObj.name)
        # save reseeding parameter from pickledROM
        reseedInt = outputs[i].initializationOptionDict.get('reseedValue',None)
        # train the ROM from the unpickled object
        outputs[i].train(unpickledObj)
        # reseed as requested
        if reseedInt is not None:
          outputs[i].reseed(reseedInt)
      elif self.actionType[i] == 'FILES-dataObjects':
        #inDictionary['Input'][i] is a Files, outputs[i] is PointSet
        infile = inDictionary['Input'][i]
        options = {'fileToLoad':infile}
        outputs[i].load(inDictionary['Input'][i].getPath(),'csv',**options)
      else:
        self.raiseAnError(IOError,"Unknown action type "+self.actionType[i])
    for output in inDictionary['Output']:
      if output.type in ['OutStreamPrint','OutStreamPlot']:
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

#
#
#
__interFaceDict                      = {}
__interFaceDict['SingleRun'        ] = SingleRun
__interFaceDict['MultiRun'         ] = MultiRun
__interFaceDict['IOStep'           ] = IOStep
__interFaceDict['RomTrainer'       ] = RomTrainer
__interFaceDict['PostProcess'      ] = PostProcess
__base                               = 'Step'

def returnInstance(Type,caller):
  """
    Returns the instance of a Step
    @ In, Type, string, requested step
    @ In, caller, object, requesting object
    @ Out, __interFaceDict, instance, instance of the step
  """
  return __interFaceDict[Type]()
  caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
