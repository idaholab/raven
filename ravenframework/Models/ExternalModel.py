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
Module where the base class and the specialization of different type of Model are
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
import inspect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Dummy import Dummy
from .. import CustomCommandExecuter
from ..utils import utils, InputData, InputTypes, mathUtils
from ..Decorators.Parallelization import Parallel
#Internal Modules End--------------------------------------------------------------------------------

class ExternalModel(Dummy):
  """
    External model class: this model allows to interface with an external python module
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(ExternalModel, cls).getInputSpecification()
    inputSpecification.setStrictMode(False) #External models can allow new elements
    inputSpecification.addParam("ModuleToLoad", InputTypes.StringType, False)
    inputSpecification.addSub(InputData.parameterInputFactory("variables", contentType=InputTypes.StringListType))
    inputSpecification.addSub(InputData.parameterInputFactory("inputs", contentType=InputTypes.StringListType))
    inputSpecification.addSub(InputData.parameterInputFactory("outputs", contentType=InputTypes.StringListType))
    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    #one data is needed for the input
    #cls.raiseADebug('think about how to import the roles to allowed class for the external model. For the moment we have just all')
    pass

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.sim = None
    self.modelVariableValues = {}     # dictionary of variable values for the external module imported at runtime
    self.modelVariableType = {}       # dictionary of variable types, used for consistency checks
    self.listOfRavenAwareVars = []    # list of variables RAVEN needs to be aware of
    self._availableVariableTypes = ['float','bool','int','ndarray',
                                    'c1darray','float16','float32','float64',
                                    'float128','int16','int32','int64','bool8'] # available data types
    self._availableVariableTypes = self._availableVariableTypes + ['numpy.'+item for item in self._availableVariableTypes]                   # as above
    self._availableVariableTypes.append('str') #not a numpy type
    self.printTag = 'EXTERNAL MODEL'  # label
    self.initExtSelf = utils.Object() # initial externalizable object
    self.workingDir = None            # RAVEN working dir
    self.pickled = False              # is this model pickled?
    self.constructed = True           # is this model constructed?

  def _copyModel(self, obj):
    """
      Set this instance to be a copy of the provided object.
      This is used to replace placeholder models with serialized objects
      during deserialization in IOStep.
      @ In, obj, instance, the instance of the object to copy from
      @ Out, None
    """
    super()._copyModel(obj)
    self.constructed = True

  def applyRunInfo(self, runInfo):
    """
      Take information from the RunInfo
      @ In, runInfo, dict, RunInfo info
      @ Out, None
    """
    self.workingDir = runInfo['WorkingDir']

  def initialize(self,runInfo,inputs,initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    for key in self.modelVariableType.keys():
      self.modelVariableType[key] = None
    if 'initialize' in dir(self.sim):
      self.sim.initialize(self.initExtSelf,runInfo,inputs)
    Dummy.initialize(self, runInfo, inputs)

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(kwargs)), tuple, return the new input in a tuple form
    """

    modelVariableValues = {}
    if 'createNewInput' in dir(self.sim):
      if 'SampledVars' in kwargs.keys():
        sampledVars = self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'],'input',False)
      extCreateNewInput = self.sim.createNewInput(self.initExtSelf,myInput,samplerType,**kwargs)
      if extCreateNewInput is None:
        self.raiseAnError(AttributeError,'in external Model '+self.ModuleToLoad+' the method createNewInput must return something. Got: None')
      if type(extCreateNewInput).__name__ != "dict":
        self.raiseAnError(AttributeError,'in external Model '+self.ModuleToLoad+ ' the method createNewInput must return a dictionary. Got type: ' +type(extCreateNewInput).__name__)
      if 'SampledVars' in kwargs.keys() and len(self.alias['input'].keys()) != 0:
        kwargs['SampledVars'] = sampledVars
      # add sampled vars
      if 'SampledVars' in kwargs:
        for key in kwargs['SampledVars']:
          if key not in extCreateNewInput:
            extCreateNewInput[key] =   kwargs['SampledVars'][key]

      newInput = ([(extCreateNewInput)],copy.deepcopy(kwargs))
    else:
      newInput =  Dummy.createNewInput(self, myInput,samplerType,**kwargs)
    if 'SampledVars' in kwargs:
      modelVariableValues.update(kwargs['SampledVars'])
    return newInput, copy.copy(modelVariableValues)

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    #Model._readMoreXML(self, xmlNode)
    paramInput = ExternalModel.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    if 'ModuleToLoad' in paramInput.parameterValues:
      self.ModuleToLoad = paramInput.parameterValues['ModuleToLoad']
      moduleToLoadString, self.ModuleToLoad = utils.identifyIfExternalModelExists(self, self.ModuleToLoad, self.workingDir)
      # load the external module and point it to self.sim
      self.sim = utils.importFromPath(moduleToLoadString,self.messageHandler.getDesiredVerbosity(self)>1)
    elif paramInput.parameterValues['subType'].strip() == 'pickledModel':
      self.pickled = True
      self.constructed = False
    ## NOTE we implicitly assume not having ModuleToLoad means you're a plugin or a known type.
    elif paramInput.parameterValues['subType'].strip() is not None:
      ExternalModel.plugins.loadPlugin("ExternalModel",paramInput.parameterValues['subType'])
      # We assume it is a plugin. Look for the type in the plugins class list
      if paramInput.parameterValues['subType'] not in ExternalModel.plugins.knownTypes():
        self.raiseAnError(IOError,('The "subType" named "{sub}" does not belong to any ' +
                                   'ExternalModel plugin available. Available plugins are: {plugs}')
                                  .format(sub=paramInput.parameterValues['subType'],
                                          plugs=', '.join(ExternalModel.plugins.knownTypes())))
      self.sim = ExternalModel.plugins.returnPlugin("ExternalModel",paramInput.parameterValues['subType'],self)
    else:
      self.raiseAnError(IOError,'"ModuleToLoad" attribute or "subType" not provided for Model "ExternalModel" named "'+self.name+'"!')
    if not self.pickled:
      # check if there are variables and, in case, load them
      for child in paramInput.subparts:
        if child.getName() == 'variables':
          self.raiseAWarning(DeprecationWarning ,'"variables" node inputted but has been deprecated!  Please list variables in the "inputs" and "outputs" nodes instead.  This Warning will result in an error in RAVEN 3.0!')
          self.modelVariableType = dict.fromkeys(child.value)
        elif child.getName() == 'inputs':
          self._setVariableList('input', child.value)
        elif child.getName() == 'outputs':
          self._setVariableList('output', child.value)

      if not self.modelVariableType:
        if not self._getVariableList('input') or not self._getVariableList('output'):
          self.raiseAnError(IOError, "<inputs> and <outputs> nodes must be specified in the ExternalModel input specifications!")
        self.modelVariableType = dict.fromkeys(self._getVariableList('input')+self._getVariableList('output'))
      # adjust model-aware variables based on aliases
      self._replaceVariablesNamesWithAliasSystem(self.modelVariableType,'inout')
      self.listOfRavenAwareVars.extend(self.modelVariableType.keys())
      # check if there are other information that the external module wants to load
      #TODO this needs to be converted to work with paramInput
      if '_readMoreXML' in dir(self.sim):
        self.sim._readMoreXML(self.initExtSelf,xmlNode)

  def evaluate(self, request: dict):
    """
      When the ExternalModel is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
      @ In, request, dict, feature coordinates (request)
      @ Out, outputEvaluation, dict, the dict containing the outputs for each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    outputEvaluation, _ = self._externalRun(request)
    return outputEvaluation

  def _externalRun(self, Input):
    """
      Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
      @ In, Input, dict, list of the inputs needed for running the model
      @ Out, (outcomes,self), tuple, tuple containing the dictionary of the results (pos 0) and the self (pos 1)
    """
    if self.pickled and not self.constructed:
      self.raiseAnError(IOError, 'The <pickledModel> "{}" has not been de-serialized (IOStep)!'.format(self.name))
    # instantiate an object to pass to the user on which to store I/O variables
    externalSelf = utils.Object()
    # build dict of variables mapped to values
    modelVariableValues = {}
    for key in self.modelVariableType.keys():
      modelVariableValues[key] = None
    for key, value in self.initExtSelf.__dict__.items():
      setattr(externalSelf, key, copy.copy(value))
      modelVariableValues[key] = copy.copy(value)
    for key in Input.keys():
      if key in modelVariableValues.keys():
        modelVariableValues[key] = copy.copy(Input[key])
    # collect updates from user's createNewInput if it's present
    if 'createNewInput' not in dir(self.sim):
      InputDict = {}
    else:
      InputDict = Input
    additionalKeys = []
    if '_indexMap' in Input.keys():
      additionalKeys.append('_indexMap')
    for key in Input.keys():
      if key in additionalKeys:
        modelVariableValues[key] = copy.copy(Input[key])
    # make sure the instance we pass to the user has the variables it needs
    for key in list(self.modelVariableType.keys()) + additionalKeys:
      # preference is to make it a member of the insantiated object
      if key.isidentifier():
        setattr(externalSelf, key, copy.copy(modelVariableValues[key]))
      # otherwise, just have it available in the dictionary instead. This feels redundant.
      else:
        self.raiseAWarning(f'Variable "{key}" could not be added to ExternalModel first argument due to complex name. ' +
                            'Find it in the second (dictionary) argument instead.')
    # only pass the variables and their values according to the model itself.
    for key in Input:
      if key in self.modelVariableType or key in additionalKeys:
        InputDict[key] = Input[key]

    # run the user's model
    if 'current_time' in Input:
      self.sim.runStep(externalSelf, InputDict)
    else:
      self.sim.run(externalSelf, InputDict)

    # collect the user results
    for key in self.modelVariableType:
      # preferentially obtain values from the instantiated object
      if key.isidentifier() and hasattr(externalSelf, key): # checks valid python variable name
        modelVariableValues[key] = copy.copy(getattr(externalSelf, key))
      # otherwise, try to get it from the dictionary. If not, save it as "None" so it reports back to the user.
      else:
        self.raiseAWarning(f'Variable "{key}" could not be read from ExternalModel first argument due to complex name. ' +
                            'Checking it in the second (dictionary) argument instead.')
        modelVariableValues[key] = copy.copy(InputDict.get(key, None))
    # in case initialization variables have been modified, update them.
    for key in self.initExtSelf.__dict__:
      setattr(self.initExtSelf, key, copy.copy(getattr(externalSelf, key)))
    # check for missing/bad data
    if None in self.modelVariableType.values():
      errorFound = False
      for key in self.modelVariableType:
        self.modelVariableType[key] = type(modelVariableValues[key]).__name__
        if self.modelVariableType[key] not in self._availableVariableTypes:
          if not errorFound:
            self.raiseADebug('Unsupported type found. Available ones are: '+ str(self._availableVariableTypes).replace('[','').replace(']', ''),verbosity='silent')
          errorFound = True
          self.raiseADebug('variable '+ key+' has an unsupported type -> '+ self.modelVariableType[key],verbosity='silent')
      if errorFound:
        self.raiseAnError(RuntimeError, 'Errors detected. See above!!')
    outcomes = dict((k, modelVariableValues[k]) for k in self.listOfRavenAwareVars)
    # check type consistency... This is needed in order to keep under control the external model... In order to avoid problems in collecting the outputs in our internal structures
    for key in self.modelVariableType:
      if not utils.typeMatch(outcomes[key], self.modelVariableType[key]):
        self.raiseAnError(RuntimeError, 'type of variable '+ key + ' is ' + str(type(outcomes[key]))+' and mismatches with respect to the input ones (' + self.modelVariableType[key] +')!!!')
    self._replaceVariablesNamesWithAliasSystem(outcomes, 'inout', True)
    # add the indexMap, if provided
    indexMap = getattr(externalSelf, '_indexMap', None)
    if indexMap:
      outcomes['_indexMap'] = indexMap
    # TODO slow conversion, but provides type consistency --> TODO this doesn't mach up well with other models!
    outcomes = dict((k, np.atleast_1d(val)) for k, val in outcomes.items())
    return outcomes, self

  @Parallel()
  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the 'SampledVars' that contains a dictionary {'name variable':value}
        @ Out, returnValue, tuple, This will hold two pieces of information,
          the first item will be the input data used to generate this sample,
          the second item will be the output of this model given the specified
          inputs
    """
    Input = self.createNewInput(myInput, samplerType, **kwargs)
    inRun = copy.copy(self._manipulateInput(Input[0][0]))
    # collect results from model run
    result,instSelf = self._externalRun(inRun,)
    evalIndexMap = result.get('_indexMap', [{}])[0]
    # build realization
    ## do it in this order to make sure only the right variables are overwritten
    ## first inRun, which has everything from self.* and Input[*]
    rlz = dict((var, np.atleast_1d(val)) for var, val in inRun.items())
    ## then result, which has the expected outputs and possibly changed inputs
    rlz.update(dict((var, np.atleast_1d(val)) for var, val in result.items()))
    ## then get the metadata from kwargs
    rlz.update(dict((var, np.atleast_1d(val)) for var, val in kwargs.items()))
    ## then get the inputs from SampledVars (overwriting any other entries)
    rlz.update(dict((var, np.atleast_1d(val)) for var, val in kwargs['SampledVars'].items()))
    if '_indexMap' in rlz:
      rlz['_indexMap'][0].update(evalIndexMap)
    return rlz

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    # TODO this is done in dummy, so don't do it here?, but need to check before checking history lengths)
    # OLD instanciatedSelf = evaluation['RAVEN_instantiated_self']
    # OLD outcomes         = evaluatedOutput[0]

    # TODO move this check to the data object instead.
    if output.type in ['HistorySet']:
      outputSize = -1
      for key in output.getVars('output'):
        # OLD ? if key in instanciatedSelf.modelVariableType.keys(): #TODO why would it not be in this dict?
        if outputSize == -1:
          outputSize = len(np.atleast_1d(evaluation[key]))
        if not mathUtils.sizeMatch(evaluation[key],outputSize):
          self.raiseAnError(Exception,"the time series size needs to be the same for the output space in a HistorySet! Variable:"+key+". Size in the HistorySet="+str(outputSize)+".Size outputed="+str(outputSize))

    Dummy.collectOutput(self, finishedJob, output, options)

  def getSerializationFiles(self):
    """
      Returns a list of any files that this needs if it is serialized
      @ In, None
      @ Out, getSerializationFiles, set, set of filenames that are needed
    """
    return {self.sim.__file__}
