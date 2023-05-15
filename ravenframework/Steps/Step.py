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
  Step module
  Module containing the base class of the Step entity
  Such Steps are called by the Simulation entity
  Created on May 6, 2021
  @author: alfoa
  supercedes Steps.py from alfoa (2/16/2013)
"""

#External Modules------------------------------------------------------------------------------------
import abc
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..BaseClasses import BaseEntity, InputDataUser
from ..utils import utils
from ..utils import InputData, InputTypes
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
    # If a step possess re-seeding instruction it is going to ask to the sampler to re-seed according
    #  re-seeding = a number to be used as a new seed
    #  re-seeding = 'continue' the use the already present random environment
    # If there is no instruction (self.initSeed = None) the sampler will reinitialize
    self.initSeed = None
    self._excludeFromModelValidation = ['SolutionExport']
    # how to handle failed runs. By default, the step fails.
    # If the attribute "repeatFailureRuns" is inputted, a certain number of repetitions are going to be performed
    self.failureHandling = {"fail": True, "repetitions": 0, "perturbationFactor": 0.0, "jobRepetitionPerformed": {}}
    self.printTag = 'STEPS'
    self._clearRunDir = None
    self.pauseEndStep = False

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
              If inputted, the RNG can be re-seeded. The value of this attribute
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
          self.raiseAnError(IOError, printString.format(self.type, self.name, self.initSeed, 're-seeding'))
    if 'sleepTime' in paramInput.parameterValues:
      self.sleepTime = paramInput.parameterValues['sleepTime']
    if os.environ.get('RAVENinterfaceCheck', None) == 'True':
      self._clearRunDir = False
    else:
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
        self.raiseAnError(IOError, printString.format(self.type, self.name, paramInput.parameterValues['pauseAtEnd'], 'pauseAtEnd'),
                                  f'expected one of {utils.boolThingsFull}')
    if 'repeatFailureRuns' in paramInput.parameterValues:
      failureSettings = str(paramInput.parameterValues['repeatFailureRuns']).split("|")
      self.failureHandling['fail'] = False
      if len(failureSettings) != 1:
        self.raiseAnError(IOError, 'repeatFailureRuns format error. Expecting the repetition number only ')
      self.failureHandling['repetitions'] = utils.intConversion(failureSettings[0])
      if self.failureHandling['repetitions'] is None:
        self.raiseAnError(IOError, f'In Step named {self.name} it was not possible to cast "repetitions" attribute into an integer!')
    self._localInputAndCheckParam(paramInput)
    if None in self.parList:
      self.raiseAnError(IOError, f'A problem was found in the definition of the step {self.name}')

  @abc.abstractmethod
  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """

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

  def _initializeStep(self, inDictionary):
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
  def _localInitializeStep(self, inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """

  @abc.abstractmethod
  def _localTakeAstepRun(self, inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """

  def _registerMetadata(self, inDictionary):
    """
      collects expected metadata keys and deliver them to output data objects
      @ In, inDictionary, dict, initialization dictionary
      @ Out, None
    """
    # first collect them
    metaKeys = set()
    metaParams = dict()
    for _, entities in inDictionary.items():
      if isinstance(entities, list):
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
    # then give them to the output data objects
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

  def flushStep(self):
    """
      Reset Step attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
