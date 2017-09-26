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
    cvInput = InputData.parameterInputFactory("CV", contentType=InputData.StringType)
    cvInput.addParam("class", InputData.StringType)
    cvInput.addParam("type", InputData.StringType)
    inputSpecification.addSub(cvInput)
    # add settings block
    tolInput = InputData.parameterInputFactory("tolerance", contentType=InputData.FloatType)
    maxTrainStepInput = InputData.parameterInputFactory("maxTrainSize", contentType=InputData.IntegerType)
    initialTrainStepInput = InputData.parameterInputFactory("minInitialTrainSize", contentType=InputData.IntegerType)
    selectionMethod = InputData.parameterInputFactory("selectionMethod", contentType=InputData.StringType)
    settingsInput = InputData.parameterInputFactory("settings", contentType=InputData.StringType)
    settingsInput.addSub(tolInput)
    settingsInput.addSub(maxTrainStepInput)
    settingsInput.addSub(initialTrainStepInput)
    settingsInput.addSub(selectionMethod)
    inputSpecification.addSub(settingsInput)

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
    self.modelInstance            = None
    self.cvInstance               = None
    self.targetEvaluationInstance = None
    self.tempTargetEvaluation     = None
    self.romsDictionary        = {}      # dictionary of models that is going to be employed, i.e. {'romName':Instance}
    self.romTrainStartSize     = 10       # the initial size of training set
    self.romTrainMaxSize       = 1.0e6     # the maximum size of training set
    self.romValidateSize       = 10       # the size of rom validation set
    self.romTrained            = False    # True if all roms are trained
    self.sleepTime             = 0.005    # waiting time before checking if a run is finished.
    self.romConverged          = False    # True if all roms are converged
    self.romValid              = False    # True if all roms are valid for given input data
    self.romConvergence        = 0.01
    self.modelSelection        = 'CrowdingDistance'
    self.existTrainSize        = 0
    self.threshold             = 0.2       # the cut off validation criterion of new point using the CrowdingDistance
    self.printTag              = 'HYBRIDMODEL MODEL' # print tag
    self.createWorkingDir      = False
    self.tempOutputs           = {}
    self.oldTrainingSize       = 0
    # assembler objects to be requested
    self.addAssemblerObject('Model','1',True)
    self.addAssemblerObject('ROM','n')
    self.addAssemblerObject('CV','1')
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
      if child.getName() == 'CV':
        self.cvInstance = child.value.strip()
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
          if childChild.getName == 'selectionMethod':
            self.modelSelection = childChild.value.strip()

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
    self.cvInstance = self.retrieveObjectFromAssemblerDict('CV', self.cvInstance)
    self.cvInstance.initialize(runInfo, inputs, initDict)
    self.targetEvaluationInstance = self.retrieveObjectFromAssemblerDict('TargetEvaluation', self.targetEvaluationInstance)
    if not self.targetEvaluationInstance.isItEmpty():
      self.raiseAWarning("The provided TargetEvaluation data object is not empty, the existing data will also be used to train the ROMs!")
      self.existTrainSize = len(self.targetEvaluationInstance)
    self.tempTargetEvaluation = copy.deepcopy(self.targetEvaluationInstance)
    if self.modelInstance is None:
      self.raiseAnError(IOError,'Model XML block needs to be inputted!')
    if self.cvInstance is None:
      self.raiseAnError(IOError, 'CV XML block needs to be inputted!')
    if self.targetEvaluationInstance is None:
      self.raiseAnError(IOError, 'TargetEvaluation XML block needs to be inputted!')

    for romName, romInfo in self.romsDictionary.items():
      romInfo['Instance'] = self.retrieveObjectFromAssemblerDict('ROM', romName)
      if romInfo['Instance']  is None:
        self.raiseAnError(IOError, 'ROM XML block needs to be inputted!')

    modelInputs = self.targetEvaluationInstance.getParaKeys("inputs")
    modelOutputs = self.targetEvaluationInstance.getParaKeys("outputs")
    modelName = self.modelInstance.name
    totalRomOutputs = []

    for romInfo in self.romsDictionary.values():
      romIn = romInfo['Instance']
      if romIn.amITrained:
        self.raiseAWarning("The provided rom ", romIn.name, " is already trained, we will reset it!")
        romIn.reset()
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
      # Should we check the size of existing data object self.tempTargetEvaluation
      # and compared to the size of trainingSet in the ROM, the reason is that
      # we may end up using the same data to train the rom, the outputs may not be
      # collected yet!

      # reset the rom
      romInfo['Instance'].amITrained = False
      # always train the rom even if the rom is converged, we assume the cross validation and rom train are relative cheap
      outputMetrics = self.cvInstance.evaluateSample([romInfo['Instance'], self.tempTargetEvaluation], samplerType, kwargs)[1]
      converged = self.isRomConverged(outputMetrics)
      romInfo['Converged'] = converged
      if converged:
        romInfo['Instance'].train(self.tempTargetEvaluation)
        self.raiseADebug("ROM ", romInfo['Instance'].name, " is converged!")
      self.oldTrainingSize = len(self.tempTargetEvaluation)

  def isRomConverged(self, outputDict):
    """
      This function will check the convergence of rom
      @ In, outputDict, dict, dictionary contains the metric information
        e.g. {targetName:{metricName:List of metric values}}, this dict is coming from results of cross validation
      @ Out, None
    """
    converged = True
    for targetName, metricInfo in outputDict.items():
      for metricName, metricValues in metricInfo.items():
        if max(metricValues) > self.romConvergence:
          self.raiseADebug("maximum error: ", max(metricValues), " is greater than the given tolerance")
          converged = False
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
    if self.modelSelection == 'CrowdingDistance':
      allValid = self.crowdingDistanceMethod(kwargs['SampledVars'])
    else:
      self.raiseAnError(IOError, "Unknown model selection method ", self.modelSelection, " is given!")

    if allValid:
      self.raiseADebug("ROMs  are all valid for given model ", self.modelInstance.name)

    return allValid

  def crowdingDistanceMethod(self, varDict):
    """
      This function will check the validity of all roms based on the crowding distance method
      @ In, varDict, dict,  is a dictionary that contains the information coming from the sampler,
        i.e. {'name variable':value}
      @ Out, None
    """
    allValid = True
    for romInfo in self.romsDictionary.values():
      valid = False
      # generate the data for input parameters
      paramsList = romInfo['Instance'].getInitParams()['Features']
      trainInput = np.asarray(self._extractInputs(romInfo['Instance'].trainingSet, paramsList).values())
      currentInput = np.asarray(self._extractInputs(varDict, paramsList).values())
      coeffCD = self.computeCDCoefficient(trainInput, currentInput)
      self.raiseADebug("Crowding Distance Coefficient: ", coeffCD)
      if coeffCD > self.threshold:
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
      @ localInput, dict, dictionary contains the selected input and output parameters and their values
    """
    localInput = dict.fromkeys(paramsList, None)
    if type(dataIn) == dict:
      for elem, value in dataIn.items():
        if elem in paramsList:
          localInput[elem] = np.atleast_1d(value)
    else:
      self.raiseAnError(IOError, "The input type '", inputType, "' can not be accepted!")

    return localInput

  def computeCDCoefficient(self, trainSet, newSet):
    """
      This function will compute the Crowding distance coefficients among the input parameters
      @ In, trainSet, numpy.array, array contains values of previous generated input parameters
      @ In, newSet, numpy.array, array contains values of current generated input parameters
      @ Out, crowdingDistCoeff, float, the coefficient of crowding distance
    """
    totalSet = np.concatenate((trainSet,newSet), axis=1)
    dim = totalSet.shape[1]
    distMat = np.zeros((dim, dim))
    for i in range(dim):
      for j in range(dim):
        distMat[i,j] = linalg.norm(totalSet[:,i] - totalSet[:,j])
    crowdingDist = np.sum(distMat,axis=1)
    maxDist = np.amax(crowdingDist)
    minDist = np.amin(crowdingDist)
    if maxDist == minDist:
      crowdingDistCoeff = 1.0
    else:
      crowdingDistCoeff = (maxDist - crowdingDist[-1])/(maxDist - minDist)

    return crowdingDistCoeff

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
      @ Out, returnValue, dict, This holds the output information of the evaluated sample.
    """
    self.raiseADebug("Evaluate Sample")

    kwargsKeys = kwargs.keys()
    kwargsKeys.pop(kwargsKeys.index("jobHandler"))
    kwargsToKeep = {keepKey: kwargs[keepKey] for keepKey in kwargsKeys}
    jobHandler = kwargs['jobHandler']
    newInput = self.createNewInput(myInput, samplerType, **kwargsToKeep)
    ## Unpack the specifics for this class, namely just the jobHandler
    returnValue = (newInput,self._externalRun(newInput,jobHandler))
    return returnValue

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
          tempExportDict = self.createExportDictionaryFromFinishedJob(finishedRun, False)
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
      exportDict = self.modelInstance.createExportDictionaryFromFinishedJob(finishedRun[0], True)
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
    exportDict = evaluation[1]
    useROM = exportDict['useROM']

    try:
      jobIndex = self.tempOutputs['uncollectedJobIds'].index(finishedJob.identifier)
      self.tempOutputs['uncollectedJobIds'].pop(jobIndex)
    except ValueError:
      jobIndex = None

    if jobIndex is not None and not useROM:
      self.collectOutputFromDict(exportDict, self.tempTargetEvaluation)
      self.raiseADebug("ROM is invalid, collect ouptuts of Model with job identifier: {}".format(finishedJob.identifier))

    Dummy.collectOutput(self, finishedJob, output, options = {'exportDict':exportDict})

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
