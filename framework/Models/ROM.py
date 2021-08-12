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
#External Modules------------------------------------------------------------------------------------
import copy
import itertools
import numpy as np
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Dummy import Dummy
import Decorators
from SupervisedLearning import factory
from utils import utils, xmlUtils
from utils import InputData, InputTypes
from Decorators.Parallelization import Parallel
import LearningGate
#Internal Modules End--------------------------------------------------------------------------------

# set enviroment variable to avoid parallelim degradation in some surrogate models
os.environ["MKL_NUM_THREADS"]="1"

class ROM(Dummy):
  """
    ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome
  """
  interfaceFactory = factory

  @classmethod
  def getInputSpecification(cls, xml=None):
    """
      Method to get a reference to a class that specifies the input data for
      class cls. This one seems a bit excessive, are all of these for this class?
      @ In, cls, the class for which we are retrieving the specification
      @ In, xml, xml.etree.ElementTree.Element, optional, if given then only get specs for
          corresponding subType requested by the node
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addParam('subType', required=True, param_type=InputTypes.StringType)
    ######################
    # dynamically loaded #
    ######################
    assert xml is not None
    subType = xml.attrib.get('subType')
    validClass = cls.interfaceFactory.returnClass(subType)
    validSpec = validClass.getInputSpecification()
    inputSpecification.mergeSub(validSpec)

    cvInput = InputData.parameterInputFactory("CV", contentType=InputTypes.StringType)
    cvInput.addParam("class", InputTypes.StringType)
    cvInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(cvInput)


    ## wangc: I think we should avoid loading all inputSpecifications
    # for typ in SupervisedLearning.factory.knownTypes():
    #   obj = SupervisedLearning.factory.returnClass(typ)
    #   if hasattr(obj, 'getInputSpecifications'):
    #     subspecs = obj.getInputSpecifications()
    #     inputSpecification.mergeSub(subspecs)

    ## TODO: remove
    # CriterionInputType = InputTypes.makeEnumType("criterion", "criterionType", ["bic","aic","gini","entropy","mse"])


    ### TODO: Move to ROMCollection Class
    # ####################
    # # manually entered #
    # ####################
    # # segmenting and clustering
    # segment = InputData.parameterInputFactory("Segment", strictMode=True)
    # segmentGroups = InputTypes.makeEnumType('segmentGroup', 'sesgmentGroupType', ['segment', 'cluster', 'interpolate'])
    # segment.addParam('grouping', segmentGroups)
    # subspace = InputData.parameterInputFactory('subspace', contentType=InputTypes.StringType)
    # subspace.addParam('divisions', InputTypes.IntegerType, False)
    # subspace.addParam('pivotLength', InputTypes.FloatType, False)
    # subspace.addParam('shift', InputTypes.StringType, False)
    # segment.addSub(subspace)
    # clusterEvalModeEnum = InputTypes.makeEnumType('clusterEvalModeEnum', 'clusterEvalModeType', ['clustered', 'truncated', 'full'])
    # segment.addSub(InputData.parameterInputFactory('evalMode', strictMode=True, contentType=clusterEvalModeEnum))
    # segment.addSub(InputData.parameterInputFactory('evaluationClusterChoice', strictMode=True, contentType=InputTypes.makeEnumType('choiceGroup', 'choiceGroupType', ['first', 'random', 'centroid'])))
    # ## clusterFeatures
    # segment.addSub(InputData.parameterInputFactory('clusterFeatures', contentType=InputTypes.StringListType))
    # ## max cycles (for Interpolated ROMCollection)
    # segment.addSub(InputData.parameterInputFactory('maxCycles', contentType=InputTypes.IntegerType))
    # ## classifier
    # clsfr = InputData.parameterInputFactory('Classifier', strictMode=True, contentType=InputTypes.StringType)
    # clsfr.addParam('class', InputTypes.StringType, True)
    # clsfr.addParam('type', InputTypes.StringType, True)
    # segment.addSub(clsfr)
    # ## metric
    # metric = InputData.parameterInputFactory('Metric', strictMode=True, contentType=InputTypes.StringType)
    # metric.addParam('class', InputTypes.StringType, True)
    # metric.addParam('type', InputTypes.StringType, True)
    # segment.addSub(metric)
    # segment.addSub(InputData.parameterInputFactory('macroParameter', contentType=InputTypes.StringType))
    # inputSpecification.addSub(segment)
    # ##### END ROMCollection
    #
    # inputSpecification.addSub(InputData.parameterInputFactory('clusterEvalMode', contentType=clusterEvalModeEnum))
    # inputSpecification.addSub(InputData.parameterInputFactory('maxCycles', contentType=InputTypes.IntegerType)) # for Interpolated ROMCollection

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['PointSet', 'HistorySet', 'DataSet']

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.initializationOptionDict = {}    # ROM initialization options
    self.amITrained = False               # boolean flag, is the ROM trained?
    self.supervisedEngine = None          # dict of ROM instances (== number of targets => keys are the targets)
    self.printTag = 'ROM MODEL'           # label
    self.cvInstance = None                # Instance of provided cross validation
    self._estimator = None                # Instance of provided estimator (ROM)
    self._interfaceROM = None             # Instance of provided ROM
    # for Clustered ROM
    self.addAssemblerObject('Classifier', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('Metric', InputData.Quantity.zero_to_infinity)
    self.addAssemblerObject('CV', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('estimator', InputData.Quantity.zero_to_one)

  def __getstate__(self):
    """
      Method for choosing what gets serialized in this class
      @ In, None
      @ Out, d, dict, things to serialize
    """
    d = copy.copy(self.__dict__)
    # NOTE assemblerDict isn't needed if ROM already trained, but it can create an infinite recursion
    ## for the ROMCollection if left in, so remove it on getstate.
    del d['assemblerDict']
    return d

  def __setstate__(self, d):
    """
      Method for unserializing.
      @ In, d, dict, things to unserialize
      @ Out, None
    """
    # default setstate behavior
    self.__dict__ = d
    # since we pop this out during saving state, initialize it here
    self.assemblerDict = {}

  def applyRunInfo(self, runInfo):
    """
      Take information from the RunInfo
      @ In, runInfo, dict, RunInfo info
      @ Out, None
    """
    self.initializationOptionDict['NumThreads'] = runInfo.get('NumThreads', 1)

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    super()._readMoreXML(xmlNode)
    paramInput = self.getInputSpecification(xml=xmlNode)()
    paramInput.parseNode(xmlNode)
    cvNode = paramInput.findFirst('CV')
    self.cvInstance = cvNode.values if cvNode is not None else None
    estimatorNode = paramInput.findFirst('estimator')
    self._estimator = estimatorNode.values if estimatorNode is not None else None
    ##
    self._interfaceROM = self.interfaceFactory.returnInstance(self.subType)
    self._interfaceROM._readMoreXML(xmlNode)
    ## TODO: how to handle 'estimator' node?

    self.initializationOptionDict['name'] = self.name
    self.initializationOptionDict['modelInstance'] = self._interfaceROM
    # if working with a pickled ROM, send along that information
    if self.subType == 'pickledROM':
      self.initializationOptionDict['pickled'] = True

    pivot = paramInput.findFirst('pivotParameter')
    if pivot is not None:
      self.initializationOptionDict['pivotParameter'] = pivot.value
    else:
      self.initializationOptionDict['pivotParameter'] = 'time'

    self._initializeSupervisedGate(**self.initializationOptionDict)
    #the ROM is instanced and initialized

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize this class
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    # retrieve cross validation object
    if self.cvInstance is not None:
      self.cvInstance = self.retrieveObjectFromAssemblerDict('CV', self.cvInstance)
      self.cvInstance.initialize(runInfo, inputs, initDict)
    if self._estimator is not None:
      self._estimator = self.retrieveObjectFromAssemblerDict('estimator', self._estimator)
      self._estimator.initialize(runInfo, inputs, initDict)

  def _initializeSupervisedGate(self,**initializationOptions):
    """
      Method to initialize the supervisedGate class
      @ In, initializationOptions, dict, the initialization options
      @ Out, None
    """
    self.supervisedEngine = LearningGate.factory.returnInstance('SupervisedGate', self.subType, **initializationOptions)

  def reset(self):
    """
      Reset the ROM
      @ In,  None
      @ Out, None
    """
    self.supervisedEngine.reset()
    self.amITrained   = False

  def reseed(self,seed):
    """
      Used to reset the seed of the underlying ROM.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    self.supervisedEngine.reseed(seed)

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = self.supervisedEngine.getInitParams()
    return paramDict

  def provideExpectedMetaKeys(self):
    """
      Overrides the base class method to assure child engine is also polled for its keys.
      @ In, None
      @ Out, metaKeys, set(str), names of meta variables being provided
      @ Out, metaParams, dict, the independent indexes related to expected keys
    """
    # load own keys and params
    metaKeys, metaParams = Dummy.provideExpectedMetaKeys(self)
    # add from engine
    keys, params = self.supervisedEngine.provideExpectedMetaKeys()
    metaKeys = metaKeys.union(keys)
    metaParams.update(params)
    return metaKeys, metaParams

  def train(self,trainingSet):
    """
      This function train the ROM
      @ In, trainingSet, dict or PointSet or HistorySet, data used to train the ROM; if an HistorySet is provided the a list of ROM is created in order to create a temporal-ROM
      @ Out, None
    """
    if type(trainingSet).__name__ == 'ROM':
      self.initializationOptionDict = copy.deepcopy(trainingSet.initializationOptionDict)
      self.trainingSet              = copy.copy(trainingSet.trainingSet)
      self.amITrained               = copy.deepcopy(trainingSet.amITrained)
      self.supervisedEngine         = copy.deepcopy(trainingSet.supervisedEngine)
    else:
      # TODO: The following check may need to be moved to Dummy Class -- wangc 7/30/2018
      if type(trainingSet).__name__ != 'dict' and trainingSet.type == 'HistorySet':
        pivotParameterId = self.supervisedEngine.pivotParameterId
        if not trainingSet.checkIndexAlignment(indexesToCheck=pivotParameterId):
          self.raiseAnError(IOError, "The data provided by the data object", trainingSet.name, "is not synchonized!",
                  "The time-dependent ROM requires all the histories are synchonized!")
      self.trainingSet = copy.copy(self._inputToInternal(trainingSet))
      self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
      self.supervisedEngine.train(self.trainingSet, self.assemblerDict)
      self.amITrained = self.supervisedEngine.amITrained

  def confidence(self,request,target = None):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, datatype, feature coordinates (request)
      @ Out, confidenceDict, dict, the dict containing the confidence on each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    inputToROM = self._inputToInternal(request)
    confidenceDict = self.supervisedEngine.confidence(inputToROM)
    return confidenceDict

  @Decorators.timingProfile
  def evaluate(self, request):
    """
      When the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
      @ In, request, datatype, feature coordinates (request)
      @ Out, outputEvaluation, dict, the dict containing the outputs for each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    inputToROM       = self._inputToInternal(request)
    outputEvaluation = self.supervisedEngine.run(inputToROM)
    # assure numpy array formatting # TODO can this be done in the supervised engine instead?
    for k,v in outputEvaluation.items():
      outputEvaluation[k] = np.atleast_1d(v)
    return outputEvaluation

  def _externalRun(self,inRun):
    """
      Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
      @ In, inRun, datatype, feature coordinates
      @ Out, returnDict, dict, the return dictionary containing the results
    """
    returnDict = self.evaluate(inRun)
    self._replaceVariablesNamesWithAliasSystem(returnDict, 'output', True)
    self._replaceVariablesNamesWithAliasSystem(inRun, 'input', True)
    return returnDict

  @Parallel()
  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, rlz, dict, This will hold two pieces of information,
          the first will be the input data used to generate this sample,
          the second will be the output of this model given the specified
          inputs
    """
    Input = self.createNewInput(myInput, samplerType, **kwargs)
    inRun = self._manipulateInput(Input[0])
    # collect results from model run
    result = self._externalRun(inRun)
    # build realization
    # assure rlz has all metadata
    self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'] ,'input',True)
    rlz = dict((var,np.atleast_1d(kwargs[var])) for var in kwargs.keys())
    # update rlz with input space from inRun and output space from result
    rlz.update(dict((var,np.atleast_1d(inRun[var] if var in kwargs['SampledVars'] else result[var])) for var in set(itertools.chain(result.keys(),inRun.keys()))))
    return rlz

  def setAdditionalParams(self, params):
    """
      Used to set parameters at a time other than initialization (such as deserializing).
      @ In, params, dict, new params to set (internals depend on ROM)
      @ Out, None
    """
    self.supervisedEngine.setAdditionalParams(params)

  def convergence(self,trainingSet):
    """
      This is to get the cross validation score of ROM
      @ In, trainingSize, int, the size of current training size
      @ Out, cvScore, dict, the dict containing the score of cross validation
    """
    if self.subType.lower() not in ['scikitlearn','ndinvdistweight']:
      self.raiseAnError(IOError, 'convergence calculation is not Implemented for ROM', self.name, 'with type', self.subType)
    cvScore = self._crossValidationScore(trainingSet)
    return cvScore

  def _crossValidationScore(self, trainingSet):
    """
      The function calculates the cross validation score on ROMs
      @ In, trainingSize, int, the size of current training size
      @ Out, cvMetrics, dict, the calculated cross validation metrics
    """
    if len(self.supervisedEngine.supervisedContainer) > 1:
      self.raiseAnError(IOError, "Cross Validation Method is not implemented for Clustered ROMs")
    cvMetrics = None
    if self._checkCV(len(trainingSet)):
      # reset the ROM before perform cross validation
      cvMetrics = {}
      self.reset()
      outputMetrics = self.cvInstance._pp.run([self, trainingSet])
      exploredTargets = []
      for cvKey, metricValues in outputMetrics.items():
        info = self.cvInstance._pp._returnCharacteristicsOfCvGivenOutputName(cvKey)
        if info['targetName'] in exploredTargets:
          self.raiseAnError(IOError, "Multiple metrics are used in cross validation '", self.cvInstance.name, "' for ROM '", rom.name,  "'!")
        exploredTargets.append(info['targetName'])
        cvMetrics[self.name] = (info['metricType'], metricValues)
    return cvMetrics

  def _checkCV(self, trainingSize):
    """
      The function will check whether we can use Cross Validation or not
      @ In, trainingSize, int, the size of current training size
      @ Out, None
    """
    useCV = True
    initDict =  self.cvInstance._pp.initializationOptionDict
    if 'SciKitLearn' in initDict.keys() and 'n_splits' in initDict['SciKitLearn'].keys():
      if trainingSize < utils.intConversion(initDict['SciKitLearn']['n_splits']):
        useCV = False
    else:
      useCV = False
    return useCV

  def writePointwiseData(self, writeTo):
    """
      Called by the OutStreamPrint object to cause the ROM to print information about itself
      @ In, writeTo, DataObject, data structure to add data to
      @ Out, None
    """
    # TODO handle statepoint ROMs (dynamic, but rom doesn't handle intrinsically)
    ## should probably let the LearningGate handle this! It knows how to stitch together pieces, sort of.
    engines = self.supervisedEngine.supervisedContainer
    for engine in engines:
      engine.writePointwiseData(writeTo)

  def writeXML(self, what='all'):
    """
      Called by the OutStreamPrint object to cause the ROM to print itself
      @ In, what, string, optional, keyword requesting what should be printed
      @ Out, xml, xmlUtils.StaticXmlElement, written meta
    """
    #determine dynamic or static
    dynamic = self.supervisedEngine.isADynamicModel
    # determine if it can handle dynamic data
    handleDynamicData = self.supervisedEngine.canHandleDynamicData
    # get pivot parameter
    pivotParameterId = self.supervisedEngine.pivotParameterId
    # find some general settings needed for either dynamic or static handling
    ## get all the targets the ROMs have
    ROMtargets = self.supervisedEngine.supervisedContainer[0].target
    ## establish requested targets
    targets = ROMtargets if what=='all' else what.split(',')
    ## establish sets of engines to work from
    engines = self.supervisedEngine.supervisedContainer
    # if the ROM is "dynamic" (e.g. time-dependent targets), then how we print depends
    #    on whether the engine is naturally dynamic or whether we need to handle that part.
    if dynamic and not handleDynamicData:
      # time-dependent, but we manage the output (chopped)
      xml = xmlUtils.DynamicXmlElement('ROM', pivotParam = pivotParameterId)
      ## pre-print printing
      engines[0].writeXMLPreamble(xml) #let the first engine write the preamble
      for s,rom in enumerate(engines):
        pivotValue = self.supervisedEngine.historySteps[s]
        #for target in targets: # should be handled by SVL engine or here??
        #  #skip the pivot param
        #  if target == pivotParameterId:
        #    continue
        #otherwise, call engine's print method
        self.raiseAMessage('Printing time-like',pivotValue,'ROM XML')
        subXML = xmlUtils.StaticXmlElement(self.supervisedEngine.supervisedContainer[0].printTag)
        rom.writeXML(subXML, skip = [pivotParameterId])
        for element in subXML.getRoot():
          xml.addScalarNode(element, pivotValue)
        #xml.addScalarNode(subXML.getRoot(), pivotValue)
    else:
      # directly accept the results from the engine
      xml = xmlUtils.StaticXmlElement(self.name)
      ## pre-print printing
      engines[0].writeXMLPreamble(xml)
      engines[0].writeXML(xml)
    return xml
