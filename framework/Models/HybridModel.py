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
Created on September, 2017

@author: wangc
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
from numpy import linalg
import time
import itertools
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from .Dummy import Dummy
import Models
import Files
from utils import InputData
from utils import utils
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class HybridModel(Dummy):
  """
    HybridModel Class. This class is aimed to automatically select the model to run among different models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for the class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    inputSpecification = super(HybridModel, cls).getInputSpecification()
    modelInput = InputData.parameterInputFactory("Model", contentType=InputData.StringType)
    modelInput.addParam("class", InputData.StringType)
    modelInput.addParam("type", InputData.StringType)
    inputSpecification.addSub(modelInput)
    romInput = InputData.parameterInputFactory("ROM", contentType=InputData.StringType)
    romInput.addParam("class", InputData.StringType)
    romInput.addParam("type", InputData.StringType)
    inputSpecification.addSub(romInput)
    targetEvaluationInput = InputData.parameterInputFactory("TargetEvaluation", contentType=InputData.StringType)
    targetEvaluationInput.addParam("class", InputData.StringType)
    targetEvaluationInput.addParam("type", InputData.StringType)
    inputSpecification.addSub(targetEvaluationInput)

    # add settings block
    tolInput = InputData.parameterInputFactory("tolerance", contentType=InputData.FloatType)
    maxTrainStepInput = InputData.parameterInputFactory("maxTrainSize", contentType=InputData.IntegerType)
    initialTrainStepInput = InputData.parameterInputFactory("minInitialTrainSize", contentType=InputData.IntegerType)
    settingsInput = InputData.parameterInputFactory("settings", contentType=InputData.StringType)
    settingsInput.addSub(tolInput)
    settingsInput.addSub(maxTrainStepInput)
    settingsInput.addSub(initialTrainStepInput)
    inputSpecification.addSub(settingsInput)
    # add validationMethod block
    threshold = InputData.parameterInputFactory("threshold", contentType=InputData.FloatType)
    validationMethodInput = InputData.parameterInputFactory("validationMethod", contentType=InputData.StringType)
    validationMethodInput.addParam("name", InputData.StringType)
    validationMethodInput.addSub(threshold)
    inputSpecification.addSub(validationMethodInput)

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    pass

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self,runInfoDict)
    self.modelInstance            = None             # Instance of given model
    self.targetEvaluationInstance = None             # Instance of data object used to store the inputs and outputs of HybridModel
    self.tempTargetEvaluation     = None             # Instance of data object that are used to store the training set
    self.romsDictionary        = {}                  # dictionary of models that is going to be employed, i.e. {'romName':Instance}
    self.romTrainStartSize     = 10                  # the initial size of training set
    self.romTrainMaxSize       = 1.0e6               # the maximum size of training set
    self.romValidateSize       = 10                  # the size of rom validation set
    self.romTrained            = False               # True if all roms are trained
    self.sleepTime             = 0.005               # waiting time before checking if a run is finished.
    self.romConverged          = False               # True if all roms are converged
    self.romValid              = False               # True if all roms are valid for given input data
    self.romConvergence        = 0.01                # The criterion used to check ROM convergence
    self.validationMethod      = {}                  # dict used to store the validation methods and their settings
    self.existTrainSize        = 0                   # The size of existing training set in the provided data object via 'TargetEvaluation'
    self.printTag              = 'HYBRIDMODEL MODEL' # print tag
    self.createWorkingDir      = False               # If the type of model is 'Code', this will set to true
    self.tempOutputs           = {}                  # Indicators used to collect model inputs/outputs for rom training
    self.oldTrainingSize       = 0                   # The size of training set that is previous used to train the rom
    self.modelIndicator        = {}                  # a dict i.e. {jobPrefix: 1 or 0} used to indicate the runs: model or rom. '1' indicates ROM run, and '0' indicates Code run
    self.crowdingDistance      = None
    self.metricCategories      = {'find_min':['explained_variance_score', 'r2_score'], 'find_max':['median_absolute_error', 'mean_squared_error', 'mean_absolute_error']}
    # assembler objects to be requested
    self.addAssemblerObject('Model','1',True)
    self.addAssemblerObject('ROM','n')
    self.addAssemblerObject('TargetEvaluation','1')

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy.localInputAndChecks(self, xmlNode)
    paramInput = HybridModel.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
      if child.getName() == 'Model':
        self.modelInstance = child.value.strip()
        if child.parameterValues['type'] == 'Code':
          self.createWorkingDir = True
      if child.getName() == 'TargetEvaluation':
        self.targetEvaluationInstance = child.value.strip()
      if child.getName() == 'ROM':
        romName = child.value.strip()
        self.romsDictionary[romName] = {'Instance': None, 'Converged': False, 'Valid': False}
      if child.getName() == 'settings':
        for childChild in child.subparts:
          if childChild.getName() == 'maxTrainSize':
            self.romTrainMaxSize = utils.intConversion(childChild.value)
          if childChild.getName() == 'minInitialTrainSize':
            self.romTrainStartSize = utils.intConversion(childChild.value)
          if childChild.getName() == 'tolerance':
            self.romConvergence = utils.floatConversion(childChild.value)
      if child.getName() == 'validationMethod':
        name = child.parameterValues['name']
        self.validationMethod[name] = {}
        for childChild in child.subparts:
          if childChild.getName() == 'threshold':
            self.validationMethod[name]['threshold'] = utils.floatConversion(childChild.value)
        if name != 'CrowdingDistance':
          self.raiseAnError(IOError, "Validation method ", name, " is not implemented yet!")

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize this model class
      @ In, runInfo, dict, is the run info from the jobHandler
      @ In, inputs, list, is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    if isinstance(self.modelInstance, Models.Model):
      self.raiseAnError(IOError, "HybridModel has already been initialized, and it can not be initialized again!")
    self.modelInstance = self.retrieveObjectFromAssemblerDict('Model', self.modelInstance)
    if self.modelInstance.type == 'Code':
      codeInput = []
      for elem in inputs:
        if isinstance(elem, Files.File):
          codeInput.append(elem)
      self.modelInstance.initialize(runInfo, codeInput, initDict)

    self.targetEvaluationInstance = self.retrieveObjectFromAssemblerDict('TargetEvaluation', self.targetEvaluationInstance)
    if len(self.targetEvaluationInstance):
      self.raiseAWarning("The provided TargetEvaluation data object is not empty, the existing data will also be used to train the ROMs!")
      self.existTrainSize = len(self.targetEvaluationInstance)
    self.tempTargetEvaluation = copy.deepcopy(self.targetEvaluationInstance)
    if self.modelInstance is None:
      self.raiseAnError(IOError,'Model XML block needs to be inputted!')
    if self.targetEvaluationInstance is None:
      self.raiseAnError(IOError, 'TargetEvaluation XML block needs to be inputted!')
    for romName, romInfo in self.romsDictionary.items():
      romInfo['Instance'] = self.retrieveObjectFromAssemblerDict('ROM', romName)
      if romInfo['Instance']  is None:
        self.raiseAnError(IOError, 'ROM XML block needs to be inputted!')
    modelInputs = self.targetEvaluationInstance.getVars("input")
    modelOutputs = self.targetEvaluationInstance.getVars("output")
    modelName = self.modelInstance.name
    totalRomOutputs = []
    for romInfo in self.romsDictionary.values():
      romIn = romInfo['Instance']
      if romIn.amITrained:
        self.raiseAWarning("The provided rom ", romIn.name, " is already trained, we will reset it!")
        romIn.reset()
      romIn.initialize(runInfo, inputs, initDict)
      romInputs = romIn.getInitParams()['Features']
      romOutputs = romIn.getInitParams()['Target']
      totalRomOutputs.extend(romOutputs)
      unknownList = utils.checkIfUnknowElementsinList(modelInputs, romInputs)
      if unknownList:
        self.raiseAnError(IOError, 'Input Parameters: "', ','.join(str(e) for e in unknownList), '" used in ROM ', romIn.name, ' can not found in Model ', modelName)
      unknownList = utils.checkIfUnknowElementsinList(romInputs, modelInputs)
      if unknownList:
        self.raiseAnError(IOError, 'Input Parameters: "', ','.join(str(e) for e in unknownList), '" used in Model ', modelName, ', but not used in ROM ', romIn.name)
      unknownList = utils.checkIfUnknowElementsinList(modelOutputs, romOutputs)
      if unknownList:
        self.raiseAnError(IOError, 'Output Parameters: "', ','.join(str(e) for e in unknownList), '" used in ROM ', romIn.name, ' can not found in Model ', modelName)
      if romIn.amITrained:
        # Only untrained roms are allowed
        self.raiseAnError(IOError,'HybridModel only accepts untrained ROM, but rom "', romIn.name, '" is already trained')
    # check: we require that the union of ROMs outputs is the same as the paired model in order to use the ROM
    # to replace the paired model.
    if len(set(totalRomOutputs)) != len(totalRomOutputs):
      dup = []
      for elem in set(totalRomOutputs):
        if totalRomOutputs.count(elem) > 1:
          dup.append(elem)
      # we assume there is no duplicate outputs among the roms
      self.raiseAnError(IOError, 'The following outputs ', ','.join(str(e) for e in dup), "are found in the outputs of multiple roms!")
    unknownList = utils.checkIfUnknowElementsinList(totalRomOutputs,modelOutputs)
    if unknownList:
      self.raiseAnError(IOError, "The following outputs: ", ','.join(str(e) for e in unknownList), " used in Model: ", modelName, "but not used in the paired ROMs.")
    self.tempOutputs['uncollectedJobIds'] = []

  def getInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, tempDict, dict, dictionary to be updated. {'attribute name':value}
    """
    tempDict = OrderedDict()
    tempDict['ROMs contained in HybridModel are '] = self.romsDictionary.keys()
    return tempDict

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input. In this case, it calls all the getAdditionalInputEdits methods
      of the sub-models
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    self.modelInstance.getAdditionalInputEdits(inputInfo)

  def __selectInputSubset(self,romName, kwargs):
    """
      Method aimed to select the input subset for a certain model
      @ In, romName, string, the rom name
      @ In, kwargs , dict, the kwarded dictionary where the sampled vars are stored
      @ Out, selectedKwargs , dict, the subset of variables (in a swallow copy of the kwargs dict)
    """
    selectedKwargs = copy.copy(kwargs)
    selectedKwargs['SampledVars'], selectedKwargs['SampledVarsPb'] = {}, {}
    featsList = self.romsDictionary[romName]['Instance'].getInitParams()['Features']
    selectedKwargs['SampledVars'] = {key: kwargs['SampledVars'][key] for key in featsList}
    if 'SampledVarsPb' in kwargs.keys():
      selectedKwargs['SampledVarsPb'] = {key: kwargs['SampledVarsPb'][key] for key in featsList}
    else:
      selectedKwargs['SampledVarsPb'] = {key: 1.0 for key in featsList}
    return selectedKwargs

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """
    self.raiseADebug("Create New Input")
    useROM = kwargs['useROM']
    if useROM:
      identifier = kwargs['prefix']
      newKwargs = {'prefix':identifier, 'useROM':useROM}
      for romName in self.romsDictionary.keys():
        newKwargs[romName] = self.__selectInputSubset(romName, kwargs)
        newKwargs[romName]['prefix'] = romName+utils.returnIdSeparator()+identifier
        newKwargs[romName]['uniqueHandler'] = self.name+identifier
    else:
      newKwargs = copy.deepcopy(kwargs)
    if self.modelInstance.type == 'Code':
      codeInput = []
      romInput = []
      for elem in myInput:
        if isinstance(elem, Files.File):
          codeInput.append(elem)
        elif elem.type in ['PointSet', 'HistorySet']:
          romInput.append(elem)
        else:
          self.raiseAnError(IOError, "The type of input ", elem.name, " can not be accepted!")
      if useROM:
        return (romInput, samplerType, newKwargs)
      else:
        return (codeInput, samplerType, newKwargs)
    return (myInput, samplerType, newKwargs)

  def trainRom(self, samplerType, kwargs):
    """
      This function will train all ROMs if they are not converged
      @ In, samplerType, string, the type of sampler
      @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
        a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, None
    """
    self.raiseADebug("Start to train roms")
    for romInfo in self.romsDictionary.values():
      cvMetrics = romInfo['Instance'].convergence(self.tempTargetEvaluation)
      if cvMetrics is not None:
        converged = self.isRomConverged(cvMetrics)
        romInfo['Converged'] = converged
        if converged:
          romInfo['Instance'].reset()
          romInfo['Instance'].train(self.tempTargetEvaluation)
          self.raiseADebug("ROM ", romInfo['Instance'].name, " is converged!")
      else:
        self.raiseAMessage("Minimum initial training size is met, but the training size is not enough to be used to perform cross validation")
    self.oldTrainingSize = len(self.tempTargetEvaluation)

  def isRomConverged(self, outputDict):
    """
      This function will check the convergence of rom
      @ In, outputDict, dict, dictionary contains the metric information
        e.g. {targetName:{metricName:List of metric values}}, this dict is coming from results of cross validation
      @ Out, converged, bool, True if the rom is converged
    """
    converged = True
    # very temporary solution
    for romName, metricInfo in outputDict.items():
      converged = self.checkErrors(metricInfo[0], metricInfo[1])
    return converged

  def checkErrors(self, metricType, metricResults):
    """
      This function is used to compare the metric outputs with the tolerance for the rom convergence
      @ In, metricType, list, the list of metrics
      @ In, metricResults, list or dict
      @ Out, converged, bool, True if the metric outputs are less than the tolerance
    """
    if type(metricResults) == list or isinstance(metricResults,np.ndarray):
      errorList = np.atleast_1d(metricResults)
    elif type(metricResults) == dict:
      errorList = np.atleast_1d(metricResults.values())
    else:
      self.raiseAnError(IOError, "The outputs generated by the cross validation '", self.cvInstance.name, "' can not be processed by HybridModel '", self.name, "'!")
    converged = False
    error = None
    # we only allow to define one metric in the cross validation PP
    for key, metricList in self.metricCategories.items():
      if metricType[1] in metricList:
        if key == 'find_min':
          # use displacement from the optimum to indicate tolerance
          error = 1.0 - np.amin(errorList)
        elif key == 'find_max':
          error = np.amax(errorList)
        converged = True if error <= self.romConvergence else False
        break
    if error is None:
      self.raiseAnError(IOError, "Metric %s used for cross validation can not be handled by the HybridModel." %metricType[1])
    if not converged:
      self.raiseADebug("The current error: ", str(error), " is not met with the given tolerance ", str(self.romConvergence))
    else:
      self.raiseADebug("The current error: ", str(error), " is met with the given tolerance ", str(self.romConvergence))
    return converged

  def checkRomConvergence(self):
    """
      This function will check the convergence of all ROMs
      @ In, None
      @ Out, bool, True if all ROMs are converged
    """
    converged = True
    for romInfo in self.romsDictionary.values():
      if not romInfo['Converged']:
        converged = False
    if converged:
      self.raiseADebug("All ROMs are converged")
    return converged

  def checkRomValidity(self, kwargs):
    """
      This function will check the validity of all roms
      @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
        a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, None
    """
    allValid = False
    for selectionMethod, params in self.validationMethod.items():
      if selectionMethod == 'CrowdingDistance':
        allValid = self.crowdingDistanceMethod(params, kwargs['SampledVars'])
      else:
        self.raiseAnError(IOError, "Unknown model selection method ", selectionMethod, " is given!")
    if allValid:
      self.raiseADebug("ROMs  are all valid for given model ", self.modelInstance.name)
    return allValid

  def crowdingDistanceMethod(self, settingDict, varDict):
    """
      This function will check the validity of all roms based on the crowding distance method
      @ In, settingDict, dict, stores the setting information for the crowding distance method
      @ In, varDict, dict,  is a dictionary that contains the information coming from the sampler,
        i.e. {'name variable':value}
      @ Out, allValid, bool, True if the  given sampled point is valid for all roms, otherwise False
    """
    allValid = True
    for romInfo in self.romsDictionary.values():
      valid = False
      # generate the data for input parameters
      paramsList = romInfo['Instance'].getInitParams()['Features']
      trainInput = self._extractInputs(romInfo['Instance'].trainingSet, paramsList)
      currentInput = self._extractInputs(varDict, paramsList)
      if self.crowdingDistance is None:
        self.crowdingDistance = self.computeCrowdingDistance(trainInput)
      sizeCD = len(self.crowdingDistance)
      if sizeCD != trainInput.shape[1]:
        self.crowdingDistance = self.updateCrowdingDistance(trainInput[:,0:sizeCD], trainInput[:,sizeCD:], self.crowdingDistance)
      crowdingDistance = self.updateCrowdingDistance(trainInput, currentInput, self.crowdingDistance)
      maxDist = np.amax(crowdingDistance)
      minDist = np.amin(crowdingDistance)
      if maxDist == minDist:
        coeffCD = 1.0
      else:
        coeffCD = (maxDist - crowdingDistance[-1])/(maxDist - minDist)
      self.raiseADebug("Crowding Distance Coefficient: ", coeffCD)
      if coeffCD >= settingDict['threshold']:
        valid = True
      romInfo['Valid'] = valid
      if valid:
        self.raiseADebug("ROM ",romInfo['Instance'].name, " is valid")
      else:
        allValid = False
    return allValid

  def _extractInputs(self,dataIn, paramsList):
    """
      Extract the the parameters in the paramsList from the given data object dataIn
      @ dataIn, Instance or Dict, data object or dictionary contains the input and output parameters
      @ paramsList, List, List of parameter names
      @ localInput, numpy.array, array contains the values of selected input and output parameters
    """
    localInput = []
    if type(dataIn) == dict:
      for elem in paramsList:
        if elem in dataIn.keys():
          localInput.append(np.atleast_1d(dataIn[elem]))
        else:
          self.raiseAnError(IOError, "Parameter ", elem, " is not found!")
    else:
      self.raiseAnError(IOError, "The input type '", inputType, "' can not be accepted!")
    return np.asarray(localInput)

  def computeCrowdingDistance(self, trainSet):
    """
      This function will compute the Crowding distance coefficients among the input parameters
      @ In, trainSet, numpy.array, array contains values of input parameters
      @ Out, crowdingDist, numpy.array, crowding distances for given input parameters
    """
    dim = trainSet.shape[1]
    distMat = np.zeros((dim, dim))
    for i in range(dim):
      for j in range(i):
        distMat[i,j] = linalg.norm(trainSet[:,i] - trainSet[:,j])
        distMat[j,i] = distMat[i,j]
    crowdingDist = np.sum(distMat,axis=1)
    return crowdingDist

  def updateCrowdingDistance(self, oldSet, newSet, crowdingDistance):
    """
      This function will compute the Crowding distance coefficients among the input parameters
      @ In, oldSet, numpy.array, array contains values of input parameters that have been already used
      @ In, newSet, numpy.array, array contains values of input parameters that will be used for computing the
      @ In, crowdingDistance, numpy.array, the crowding distances for oldSet
      @ Out, newCrowdingDistance, numpy.array, the updated crowding distances for both oldSet and newSet
    """
    oldSize = oldSet.shape[1]
    newSize = newSet.shape[1]
    totSize = oldSize + newSize
    if oldSize != crowdingDistance.size:
      self.raiseAnError(IOError, "The old crowding distances is not match the old data set!")
    newCrowdingDistance = np.zeros(totSize)
    distMatAppend = np.zeros((oldSize,newSize))
    for i in range(oldSize):
      for j in range(newSize):
        distMatAppend[i,j] = linalg.norm(oldSet[:,i] - newSet[:,j])
    distMatNew = self.computeCrowdingDistance(newSet)
    for i in range(oldSize):
      newCrowdingDistance[i] = crowdingDistance[i] + np.sum(distMatAppend[i,:])
    for i in range(newSize):
      newCrowdingDistance[i+oldSize] = distMatNew[i] + np.sum(distMatAppend[:,i])
    return newCrowdingDistance

  def amIReadyToTrainROM(self):
    """
      This will check the status of training data object, if the data object is updated,
      This function will return true
      @ In, None
      @ Out, ready, bool, is this HybridModel ready to retrain the ROM?
    """
    ready = False
    newGeneratedTrainingSize = len(self.tempTargetEvaluation) - self.existTrainSize
    if newGeneratedTrainingSize > self.romTrainMaxSize:
      self.raiseAMessage("Maximum training size is reached, ROMs will not be trained anymore!")
      return ready
    trainingStepSize = len(self.tempTargetEvaluation) - self.oldTrainingSize
    if newGeneratedTrainingSize >= self.romTrainStartSize and trainingStepSize > 0:
      ready = True
    return ready

  def submit(self,myInput,samplerType,jobHandler,**kwargs):
    """
      This will submit an individual sample to be evaluated by this model to a
      specified jobHandler as a client job. Note, some parameters are needed
      by createNewInput and thus descriptions are copied from there.
      @ In, myInput, list, the inputs (list) to start from to generate the new
        one
      @ In, samplerType, string, is the type of sampler that is calling to
        generate a new input
      @ In,  jobHandler, JobHandler instance, the global job handler instance
      @ In, **kwargs, dict,  is a dictionary that contains the information
        coming from the sampler, a mandatory key is the sampledVars' that
        contains a dictionary {'name variable':value}
      @ Out, None
    """
    for mm in utils.returnImportModuleString(jobHandler):
      if mm not in self.mods:
        self.mods.append(mm)
    prefix = kwargs['prefix']
    self.counter = prefix
    self.tempOutputs['uncollectedJobIds'].append(prefix)
    if self.amIReadyToTrainROM():
      self.trainRom(samplerType, kwargs)
      self.romConverged = self.checkRomConvergence()
    if self.romConverged:
      self.romValid = self.checkRomValidity(kwargs)
    else:
      self.romValid = False
    if self.romValid:
      self.modelIndicator[prefix] = 1
    else:
      self.modelIndicator[prefix] = 0
    ## Ensemble models need access to the job handler, so let's stuff it in our
    ## catch all kwargs where evaluateSample can pick it up, not great, but
    ## will suffice until we can better redesign this whole process.
    kwargs['jobHandler'] = jobHandler
    self.raiseADebug("Submit job with job identifier: {},  Runing ROM: {} ".format(kwargs['prefix'], self.romValid))
    kwargs['useROM'] = self.romValid
    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter
    jobHandler.addClientJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, prefix, kwargs)

  def evaluateSample(self, myInput, samplerType, kwargs):
    """
      This will evaluate an individual sample on this model. Note, parameters
      are needed by createNewInput and thus descriptions are copied from there.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
        a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, rlz, dict, This holds the output information of the evaluated sample.
    """
    self.raiseADebug("Evaluate Sample")
    kwargsKeys = list(kwargs.keys())
    kwargsKeys.pop(kwargsKeys.index("jobHandler"))
    kwargsToKeep = {keepKey: kwargs[keepKey] for keepKey in kwargsKeys}
    jobHandler = kwargs['jobHandler']
    newInput = self.createNewInput(myInput, samplerType, **kwargsToKeep)
    ## Unpack the specifics for this class, namely just the jobHandler
    result = self._externalRun(newInput,jobHandler)
    # assure rlz has all metadata
    rlz = dict((var,np.atleast_1d(kwargsToKeep[var])) for var in kwargsToKeep.keys())
    # update rlz with input space from inRun and output space from result
    rlz.update(dict((var,np.atleast_1d(kwargsToKeep['SampledVars'][var] if var in kwargs['SampledVars'] else result[var])) for var in set(itertools.chain(result.keys(),kwargsToKeep['SampledVars'].keys()))))

    return rlz

  def _externalRun(self,inRun, jobHandler):
    """
      Method that performs the actual run of the essembled model (separated from run method for parallelization purposes)
      @ In, inRun, tuple, tuple of Inputs (inRun[0] actual input, inRun[1] type of sampler,
        inRun[2] dictionary that contains information coming from sampler)
      @ In, jobHandler, instance, instance of jobHandler
      @ Out, exportDict, dict, dict of results from this hybrid model
    """
    self.raiseADebug("External Run")
    originalInput = inRun[0]
    samplerType = inRun[1]
    inputKwargs = inRun[2]
    identifier = inputKwargs.pop('prefix')
    useROM = inputKwargs.pop('useROM')
    uniqueHandler = self.name + identifier
    if useROM:
      # run roms
      exportDict = {}
      self.raiseADebug("Switch to ROMs")
      # submit all the roms
      for romName, romInfo in self.romsDictionary.items():
        inputKwargs[romName]['prefix'] = romName+utils.returnIdSeparator()+identifier
        nextRom = False
        while not nextRom:
          if jobHandler.availability() > 0:
            romInfo['Instance'].submit(originalInput, samplerType, jobHandler, **inputKwargs[romName])
            self.raiseADebug("Job ", romName, " with identifier ", identifier, " is submitted")
            nextRom = True
          else:
            time.sleep(self.sleepTime)
      # collect the outputs from the runs of ROMs
      while True:
        finishedJobs = jobHandler.getFinished(uniqueHandler=uniqueHandler)
        for finishedRun in finishedJobs:
          self.raiseADebug("collect job with identifier ", identifier)
          evaluation = finishedRun.getEvaluation()
          if isinstance(evaluation, Runners.Error):
            self.raiseAnError(RuntimeError, "The job identified by "+finishedRun.identifier+" failed!")
          # collect output in temporary data object
          tempExportDict = evaluation
          exportDict = self.__mergeDict(exportDict, tempExportDict)
        if jobHandler.areTheseJobsFinished(uniqueHandler=uniqueHandler):
          self.raiseADebug("Jobs with uniqueHandler ", uniqueHandler, "are collected!")
          break
        time.sleep(self.sleepTime)
      exportDict['prefix'] = identifier
    else:
      # run model
      inputKwargs['prefix'] = self.modelInstance.name+utils.returnIdSeparator()+identifier
      inputKwargs['uniqueHandler'] = self.name + identifier
      moveOn = False
      while not moveOn:
        if jobHandler.availability() > 0:
          self.modelInstance.submit(originalInput, samplerType, jobHandler, **inputKwargs)
          self.raiseADebug("Job submitted for model ", self.modelInstance.name, " with identifier ", identifier)
          moveOn = True
        else:
          time.sleep(self.sleepTime)
      while not jobHandler.isThisJobFinished(self.modelInstance.name+utils.returnIdSeparator()+identifier):
        time.sleep(self.sleepTime)
      self.raiseADebug("Job finished ", self.modelInstance.name, " with identifier ", identifier)
      finishedRun = jobHandler.getFinished(jobIdentifier = inputKwargs['prefix'], uniqueHandler = uniqueHandler)
      evaluation = finishedRun[0].getEvaluation()
      if isinstance(evaluation, Runners.Error):
        self.raiseAnError(RuntimeError, "The model "+self.modelInstance.name+" identified by "+finishedRun[0].identifier+" failed!")
      # collect output in temporary data object
      exportDict = evaluation
      self.raiseADebug("Create exportDict")
    # used in the collectOutput
    exportDict['useROM'] = useROM
    return exportDict

  def collectOutput(self,finishedJob,output):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, ClientRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError,"Job " + finishedJob.identifier +" failed!")
    useROM = evaluation['useROM']
    try:
      jobIndex = self.tempOutputs['uncollectedJobIds'].index(finishedJob.identifier)
      self.tempOutputs['uncollectedJobIds'].pop(jobIndex)
    except ValueError:
      jobIndex = None
    if jobIndex is not None and not useROM:
      self.tempTargetEvaluation.addRealization(evaluation)
      self.raiseADebug("ROM is invalid, collect ouptuts of Model with job identifier: {}".format(finishedJob.identifier))
    Dummy.collectOutput(self, finishedJob, output )

  def __mergeDict(self,exportDict, tempExportDict):
    """
      This function will combine two dicts into one
      @ In, exportDict, dict, dictionary stores the input, output and metadata
      @ In, tempExportDict, dict, dictionary stores the input, output and metadata
      @ Out,
    """
    if not exportDict:
      outputDict = copy.deepcopy(tempExportDict)
    else:
      outputDict = copy.deepcopy(exportDict)
      inKey = 'inputSpaceParams'
      outKey = 'outputSpaceParams'
      for key, value in tempExportDict[inKey].items():
        outputDict[inKey][key] = value
      for key, value in tempExportDict[outKey].items():
        output[outKey][key] = value
      for key, value in tempExportDict['metadata'].items():
        output['metadata'][key] = value
    self.raiseADebug("The exportDict has been updated")
    return outputDict
