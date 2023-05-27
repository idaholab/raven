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
  Created on May 17, 2017
  @author: alfoa, wangc
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
from .. import Decorators
from ..SupervisedLearning import factory
from ..utils import utils, xmlUtils, mathUtils
from ..utils import InputData, InputTypes
from ..Decorators.Parallelization import Parallel
#Internal Modules End--------------------------------------------------------------------------------

# set enviroment variable to avoid parallelim degradation in some surrogate models
os.environ["MKL_NUM_THREADS"]="1"

class ROM(Dummy):
  """
    ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome
  """
  interfaceFactory = factory
  segmentNameToClass = {'segment': 'Segments',
                 'cluster': 'Clusters',
                 'interpolate': 'Interpolated'}
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
    inputSpecification.description = r"""A Reduced Order Model (ROM) is a mathematical model consisting of a fast
                                        solution trained to predict a response of interest of a physical system.
                                        The ``training'' process is performed by sampling the response of a physical
                                        model with respect to variations of its parameters subject, for example, to
                                        probabilistic behavior.
                                        The results (outcomes of the physical model) of the sampling are fed into the
                                        algorithm representing the ROM that tunes itself to replicate those results.
                                        RAVEN supports several different types of ROMs, both internally developed and
                                        imported through an external library called ``scikit-learn''~\cite{SciKitLearn}.
                                        Currently in RAVEN, the user can use the \xmlAttr{subType} to select the ROM.
                                      """
    inputSpecification.addParam('subType', required=True, param_type=InputTypes.StringType,
        descr=r"""specify the type of ROM that will be used""")
    ######################
    # dynamically loaded #
    ######################
    # assert xml is not None
    if xml is not None:
      subType = xml.attrib.get('subType')
      validClass = cls.interfaceFactory.returnClass(subType)
      validSpec = validClass.getInputSpecification()
      inputSpecification.mergeSub(validSpec)
      ## Add segment input specifications
      segment = xml.find('Segment')
      if segment is not None:
        segType = segment.attrib.get('grouping', 'segment')
        validClass = cls.interfaceFactory.returnClass(cls.segmentNameToClass[segType])
        validSpec = validClass.getInputSpecification()
        inputSpecification.mergeSub(validSpec)

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
    self.amITrained = False               # boolean flag, is the ROM trained?
    self.printTag = 'ROM MODEL'           # label
    self.cvInstanceName = None            # the name of Cross Validation instance
    self.cvInstance = None                # Instance of provided cross validation
    self._estimatorNameList = []          # the name list of estimator instance
    self._estimatorList = []              # List of instances of provided estimators (ROM)
    self._interfaceROM = None             # Instance of provided ROM

    self.pickled = False # True if ROM comes from a pickled rom
    self.pivotParameterId = 'time' # The name of pivot parameter
    self.canHandleDynamicData = False # check if the model can autonomously handle the time-dependency
                                      # if not and time-dep data are passed in, a list of ROMs are constructed
    self.isADynamicModel = False # True if the ROM is time-dependent
    self.supervisedContainer = [] # List ROM instances
    self.historySteps = [] # The history steps of pivot parameter
    self.segment = False # True if segmenting/clustering/interpolating is requested
    self.numThreads = 1 # number of threads used by the ROM
    self.seed = None # seed information
    self._segmentROM = None # segment rom instance
    self._paramInput = None # the parsed xml input

    # for Clustered ROM
    self.addAssemblerObject('Classifier', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('Metric', InputData.Quantity.zero_to_infinity)
    self.addAssemblerObject('CV', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('estimator', InputData.Quantity.zero_to_infinity)

  def _localWhatDoINeed(self):
    """
      Used to obtain necessary objects.
      @ In, None
      @ Out, need, dict, the dict listing the needed objects
    """
    need = {}
    if utils.first(self.supervisedContainer).requireJobHandler:
      need =   {'internal':[(None,'jobHandler')]}
    return need

  def _localGenerateAssembler(self, assemblerObjects):
    """
      Used to obtain necessary objects.
      @ In, assemblerObjects, dict, the assembler objects
      @ Out, None
    """
    if utils.first(self.supervisedContainer).requireJobHandler:
      self.assemblerDict['jobHandler'] = assemblerObjects['internal']['jobHandler']

  def __getstate__(self):
    """
      Method for choosing what gets serialized in this class
      @ In, None
      @ Out, d, dict, things to serialize
    """
    d = copy.copy(self.__dict__)
    if not self.amITrained:
      supervisedEngineObj = d.pop("supervisedContainer")
      del supervisedEngineObj
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
    self.__dict__.update(d)
    if not d['amITrained']:
      # NOTE this will fail if the ROM requires the paramInput spec! Fortunately, you shouldn't pickle untrained.
      modelInstance = self.interfaceFactory.returnInstance(self.subType)
      self.supervisedContainer  = [modelInstance]
    # since we pop this out during saving state, initialize it here
    self.assemblerDict = {}

  def applyRunInfo(self, runInfo):
    """
      Take information from the RunInfo
      @ In, runInfo, dict, RunInfo info
      @ Out, None
    """
    self.numThreads = runInfo.get('NumThreads', 1)

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet, only the last set of entries are copied
      The copied values are returned as a dictionary back
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(kwargs)), tuple, return the new input in a tuple form
    """
    if len(myInput)>1:
      self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name'+self.name)
    [(inputDict)],kwargs = super().createNewInput(myInput,samplerType,**kwargs)
    return ([(inputDict)],kwargs)

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
    self._paramInput = paramInput
    cvNode = paramInput.findFirst('CV')
    if cvNode is not None:
      self.cvInstanceName = cvNode.value
    estimatorNodeList = paramInput.findAll('estimator')
    self._estimatorNameList = [estimatorNode.value for estimatorNode in estimatorNodeList] if len(estimatorNodeList) > 0 else []

    self._interfaceROM = self.interfaceFactory.returnInstance(self.subType)
    segmentNode = paramInput.findFirst('Segment')
    ## remove Segment node before passing input xml to SupervisedLearning ROM
    if segmentNode is not None:
      self.segment = True
      # determine type of segment to load -> limited by InputData to specific options
      segType = segmentNode.parameterValues.get('grouping', 'segment')
      self._segmentROM =  self.interfaceFactory.returnInstance(self.segmentNameToClass[segType])
      segment = xmlNode.find('Segment')
      romXml = copy.deepcopy(xmlNode)
      romXml.remove(segment)
    else:
      romXml = xmlNode
    self._interfaceROM._readMoreXML(romXml)

    if self.segment:
      romInfo = {'name':self.name, 'modelInstance': self._interfaceROM}
      self._segmentROM.setTemplateROM(romInfo)
      self._segmentROM._handleInput(paramInput)
      self.supervisedContainer = [self._segmentROM]
    else:
      self.supervisedContainer = [self._interfaceROM]
    #the variable list is needed by things like FMU export
    self._setVariableList("input", self.supervisedContainer[0].features)
    self._setVariableList("output", self.supervisedContainer[0].target)
    # if working with a pickled ROM, send along that information
    if self.subType == 'pickledROM':
      self.pickled = True

    pivot = paramInput.findFirst('pivotParameter')
    if pivot is not None:
      self.pivotParameterId = pivot.value

    self.canHandleDynamicData = self._interfaceROM.isDynamic()

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize this class
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    # retrieve cross validation object
    if self.cvInstance is None and self.cvInstanceName is not None:
      self.cvInstance = self.retrieveObjectFromAssemblerDict('CV', self.cvInstanceName)
      self.cvInstance.initialize(runInfo, inputs, initDict)

    # only initialize once
    if len(self._estimatorList) == 0 and len(self._estimatorNameList) > 0:
      self._estimatorList = [self.retrieveObjectFromAssemblerDict('estimator', estimatorName) for estimatorName in self._estimatorNameList]
      self._interfaceROM.setEstimator(self._estimatorList)

  def reset(self):
    """
      Reset the ROM
      @ In,  None
      @ Out, None
    """
    for rom in self.supervisedContainer:
      rom.reset()
    self.amITrained = False

  def reseed(self,seed):
    """
      Used to reset the seed of the underlying ROM.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    for rom in self.supervisedContainer:
      rom.reseed(seed)

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = self.supervisedContainer[-1].returnInitialParameters()
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
    # add from specific rom
    keys, params = self.supervisedContainer[-1].provideExpectedMetaKeys()
    metaKeys = metaKeys.union(keys)
    metaParams.update(params)
    return metaKeys, metaParams

  def _copyModel(self, obj):
    """
      Set this instance to be a copy of the provided object.
      This is used to replace placeholder models with serialized objects
      during deserialization in IOStep.
      Also train this model.
      @ In, obj, instance, the instance of the object to copy from
      @ Out, None
    """
    # save reseeding parameters from pickledROM
    loadSettings = {'seed': self.seed, 'paramInput': self._paramInput}
    # train the ROM from the unpickled object
    self.train(obj)
    self.setAdditionalParams(loadSettings)
    self.pickled = False

  def train(self,trainingSet):
    """
      This function train the ROM
      @ In, trainingSet, dict or PointSet or HistorySet, data used to train the ROM; if an HistorySet is provided the a list of ROM is created in order to create a temporal-ROM
      @ Out, None
    """
    if type(trainingSet).__name__ == 'ROM':
      self.trainingSet              = copy.copy(trainingSet.trainingSet)
      self.amITrained               = copy.deepcopy(trainingSet.amITrained)
      self.supervisedContainer      = copy.deepcopy(trainingSet.supervisedContainer)
      self.seed = trainingSet.seed
    else:
      # TODO: The following check may need to be moved to Dummy Class -- wangc 7/30/2018
      if type(trainingSet).__name__ != 'dict' and trainingSet.type == 'HistorySet':
        if not trainingSet.checkIndexAlignment(indexesToCheck=self.pivotParameterId):
          self.raiseAnError(IOError, "The data provided by the data object", trainingSet.name, "is not synchonized!",
                  "The time-dependent ROM requires all the histories are synchonized!")
      self.trainingSet = copy.copy(self._inputToInternal(trainingSet))
      self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
      if not self.supervisedContainer[0].requireJobHandler and 'jobHandler' in self.assemblerDict:
        self.assemblerDict.pop("jobHandler")

      self.supervisedContainer[0].setAssembledObjects(self.assemblerDict)
      # if training using ROMCollection, special treatment
      if self.segment:
        self.supervisedContainer[0].train(self.trainingSet)
      else:
        # not a collection # TODO move time-dependent snapshots to collection!
        ## time-dependent or static ROM?
        if any(type(x).__name__ == 'list' for x in self.trainingSet.values()):
          # we need to build a "time-dependent" ROM
          self.isADynamicModel = True
          if self.pivotParameterId not in list(self.trainingSet.keys()):
            self.raiseAnError(IOError, 'The pivot parameter "{}" is not present in the training set.'.format(self.pivotParameterId),
                              'A time-dependent-like ROM cannot be created!')
          if type(self.trainingSet[self.pivotParameterId]).__name__ != 'list':
            self.raiseAnError(IOError, 'The pivot parameter "{}" is not a list.'.format(self.pivotParameterId),
                              " Are you sure it is part of the output space of the training set?")
          self.historySteps = self.trainingSet.get(self.pivotParameterId)[-1]
          if not len(self.historySteps):
            self.raiseAnError(IOError, "the training set is empty!")
          # intrinsically time-dependent or does the Gate need to handle it?
          if self.canHandleDynamicData:
            # the ROM is able to manage the time dependency on its own
            self.supervisedContainer[-1].train(self.trainingSet)
          else:
            # TODO we can probably migrate this time-dependent handling to a type of ROMCollection!
            # we need to construct a chain of ROMs
            # the check on the number of time steps (consistency) is performed inside the historySnapShoots method
            # get the time slices
            newTrainingSet = mathUtils.historySnapShoots(self.trainingSet, len(self.historySteps))
            assert type(newTrainingSet).__name__ == 'list'
            # copy the original ROM
            originalROM = self.supervisedContainer[0]
            # start creating and training the time-dep ROMs
            self.supervisedContainer = [copy.deepcopy(originalROM) for _ in range(len(self.historySteps))]
            # train
            for ts in range(len(self.historySteps)):
              self.supervisedContainer[ts].train(newTrainingSet[ts])
        # if a static ROM ...
        else:
          #self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
          self.supervisedContainer[0].train(self.trainingSet)
      # END if ROMCollection
      self.amITrained = True

  def confidence(self,request,target = None):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, datatype, feature coordinates (request)
      @ Out, confidenceDict, dict, the dict containing the confidence on each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    request = self._inputToInternal(request)
    if not self.amITrained:
      self.raiseAnError(RuntimeError, "ROM "+self.name+" has not been trained yet and, consequentially, can not be evaluated!")
    confidenceDict = {}
    for rom in self.supervisedContainer:
      sliceEvaluation = rom.confidence(request)
      if len(list(confidenceDict.keys())) == 0:
        confidenceDict.update(sliceEvaluation)
      else:
        for key in confidenceDict.keys():
          confidenceDict[key] = np.append(confidenceDict[key],sliceEvaluation[key])
    return confidenceDict

  @Decorators.timingProfile
  def evaluate(self, request):
    """
      When the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
      @ In, request, datatype, feature coordinates (request)
      @ Out, resultsDict, dict, the dict containing the outputs for each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    request = self._inputToInternal(request)
    if self.pickled:
      self.raiseAnError(RuntimeError,'ROM "', self.name, '" has not been loaded yet!  Use an IOStep to load it.')
    if not self.amITrained:
      self.raiseAnError(RuntimeError, "ROM ", self.name, " has not been trained yet and, consequentially, can not be evaluated!")
    resultsDict = {}
    if self.segment:
      resultsDict = self.supervisedContainer[0].run(request)
    else:
      for rom in self.supervisedContainer:
        sliceEvaluation = rom.run(request)
        if len(list(resultsDict.keys())) == 0:
          resultsDict.update(sliceEvaluation)
        else:
          for key in resultsDict.keys():
            resultsDict[key] = np.append(resultsDict[key],sliceEvaluation[key])
    # assure numpy array formatting # TODO can this be done in the supervised engine instead?
    for k,v in resultsDict.items():
      resultsDict[k] = np.atleast_1d(v)
    return resultsDict

  def derivatives(self, request, feats = None, order=1):
    """
      This method is aimed to evaluate the derivatives using this ROM
      The differentiation method depends on the specific ROM. Numerical
      differentiation is available by default for any ROM. Some ROMs (e.g.
      neural networks) use automatic differentiation.
      @ In, request, dict, coordinate of the central point
      @ In, feats, list, optional, list of features we need to compute the
                                   derivative for (default all)
      @ In, order, int, order of the derivative (1 = first, 2 = second, etc.). Max 10
      @ Out, derivatives, dict, the dict containing the derivatives for each target
                                ({'d(feature1)/d(target1)':np.array(size 1 or n_ts),
                                  'd(feature1)/d(target1)':np.array(...)}. For example,
                                {"x1/y1":np.array(...),"x2/y1":np.array(...),etc.}
    """
    if self.pickled:
      self.raiseAnError(RuntimeError,'ROM "', self.name, '" has not been loaded yet!  Use an IOStep to load it.')
    if not self.amITrained:
      self.raiseAnError(RuntimeError, "ROM ", self.name, " has not been trained yet and, consequentially, can not be evaluated!")
    derivatives = {}
    if self.segment:
      derivatives = mathUtils.derivatives(self.supervisedContainer[0].evaluate, request, var=feats, n=order)
    else:
      for rom in self.supervisedContainer:
        sliceEvaluation = mathUtils.derivatives(rom.evaluate, request, var=feats, n=order)
        if len(list(derivatives.keys())) == 0:
          derivatives.update(sliceEvaluation)
        else:
          for key in derivatives.keys():
            derivatives[key] = np.append(derivatives[key],sliceEvaluation[key])
    return derivatives

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
    for rom in self.supervisedContainer:
      rom.setAdditionalParams(params)

  def convergence(self,trainingSet):
    """
      This is to get the cross validation score of ROM
      @ In, trainingSize, int, the size of current training size
      @ Out, cvScore, dict, the dict containing the score of cross validation
    """
    cvScore = self._crossValidationScore(trainingSet)
    return cvScore

  def _crossValidationScore(self, trainingSet):
    """
      The function calculates the cross validation score on ROMs
      @ In, trainingSize, int, the size of current training size
      @ Out, cvMetrics, dict, the calculated cross validation metrics
    """
    if len(self.supervisedContainer) > 1:
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
    for engine in self.supervisedContainer:
      engine.writePointwiseData(writeTo)

  def writeXML(self, what='all'):
    """
      Called by the OutStreamPrint object to cause the ROM to print itself
      @ In, what, string, optional, keyword requesting what should be printed
      @ Out, xml, xmlUtils.StaticXmlElement, written meta
    """
    #determine dynamic or static
    dynamic = self.isADynamicModel
    # determine if it can handle dynamic data
    handleDynamicData = self.canHandleDynamicData
    # get pivot parameter
    pivotParameterId = self.pivotParameterId
    # find some general settings needed for either dynamic or static handling
    ## get all the targets the ROMs have
    romTargets = self.supervisedContainer[0].target
    ## establish requested targets
    targets = romTargets if what=='all' else what.split(',')
    ## establish sets of engines to work from
    engines = self.supervisedContainer
    # if the ROM is "dynamic" (e.g. time-dependent targets), then how we print depends
    #    on whether the engine is naturally dynamic or whether we need to handle that part.
    if dynamic and not handleDynamicData:
      # time-dependent, but we manage the output (chopped)
      xml = xmlUtils.DynamicXmlElement('ROM', pivotParam = pivotParameterId)
      ## pre-print printing
      engines[0].writeXMLPreamble(xml) #let the first engine write the preamble
      for s,rom in enumerate(engines):
        pivotValue = self.historySteps[s]
        #for target in targets: # should be handled by SVL engine or here??
        #  #skip the pivot param
        #  if target == pivotParameterId:
        #    continue
        #otherwise, call engine's print method
        self.raiseAMessage('Printing time-like',pivotValue,'ROM XML')
        subXML = xmlUtils.StaticXmlElement(self.supervisedContainer[0].printTag)
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

  def getSolutionMetadata(self):
    """
      Get ROM solution metadata
      @ In, None
      @ Out, solutionMetadata, dict, dictionary containing
                                     ROM-specific solution metadata
    """
    solutionMetadata = {}
    #determine dynamic or static
    dynamic = self.isADynamicModel
    # determine if it can handle dynamic data
    handleDynamicData = self.canHandleDynamicData
    # find some general settings needed for either dynamic or static handling
    ## establish sets of engines to work from
    engines = self.supervisedContainer
    # if the ROM is "dynamic" (e.g. time-dependent targets), then how we print depends
    #    on whether the engine is naturally dynamic or whether we need to handle that part.
    if dynamic and handleDynamicData:
      ## pre-print printing
      if hasattr(engines[0],"getSolutionMetadata"):
        solutionMetadata = engines[0].getSolutionMetadata()
    return solutionMetadata

  def writePyomoGreyModel(self):
    """
      @ In, None, called by the OutStreamPrint object to cause the ROM to print itself
      @ Out, xml, xmlUtils.StaticXmlElement, written meta
    """
    template = r"""
# MODEL GENERATED BY RAVEN (raven.inl.gov)
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
from pyomo.contrib.pynumero.dependencies import (numpy as np)
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver, CyIpoptNLP
import os, sys, pickle
from contextlib import redirect_stdout
# RAVEN ROM PYOMO GREY MODEL CLASS

class ravenROM(ExternalGreyBoxModel):

  def __init__(self, **kwargs):
    self._rom_file = kwargs.get("rom_file")
    self._raven_framework = kwargs.get("raven_framework")
    self._order = kwargs.get("order")
    self._raven_framework = os.path.abspath(self._raven_framework)
    if not os.path.exists(self._raven_framework):
      raise IOError('The RAVEN framework directory does not exist in location "' + str(self._raven_framework)+'" !')
    if os.path.isdir(os.path.join(self._raven_framework,"ravenframework")):
      sys.path.append(self._raven_framework)
    else:
      raise IOError('The RAVEN framework directory does not exist in location "' + str(self._raven_framework)+'" !')
    from ravenframework.CustomDrivers import DriverUtils as dutils
    dutils.doSetup()
    # de-serialize the ROM
    self._rom_file = os.path.abspath(self._rom_file)
    if not os.path.exists(self._rom_file):
      raise IOError('The serialized (binary) file has not been found in location "' + str(self._rom_file)+'" !')
    self.rom = pickle.load(open(self._rom_file, mode='rb'))
    #
    self.settings = self.rom.getInitParams()
    # get input names
    self._input_names = self.settings.get('Features')
    # n_inputs
    self._n_inputs = len(self._input_names)
    # get output names
    self._output_names = self.settings.get('Target')
    # n_outputs
    self._n_outputs = len(self._output_names)
    # input values storage
    self._input_values = np.zeros(self._n_inputs, dtype=np.float64)

  def return_train_values(self, feat):
    return self.rom.trainingSet.get(feat)

  def input_names(self):
    return self._input_names

  def output_names(self):
    return self._output_names

  def set_input_values(self, input_values):
    assert len(input_values) == self._n_inputs
    np.copyto(self._input_values, input_values)

  def evaluate_equality_constraints(self):
      raise NotImplementedError('This method should not be called for this model.')

  def evaluate_outputs(self):
    request = {k:np.asarray(v) for k,v in zip(self._input_names,self._input_values)}
    outs = self.rom.evaluate(request)
    eval_outputs = np.asarray([outs[k].flatten() for k in self._output_names], dtype=np.float64)
    return eval_outputs.flatten()

  def evaluate_jacobian_outputs(self):
    request = {k:np.asarray(v) for k,v in zip(self._input_names,self._input_values)}
    derivatives = self.rom.derivatives(request, order=self._order)
    jac = np.zeros((self._n_outputs, self._n_inputs))
    for tc, target in enumerate(self._output_names):
      for fc, feature in enumerate(self._input_names):
        jac[tc,fc] = derivatives['d{}|d{}'.format(target, feature)]
    return jac

def pyomoModel(ex_model):
  m = pyo.ConcreteModel()
  m.egb = ExternalGreyBoxBlock()
  m.egb.set_external_model(ex_model)
  for inp in ex_model.input_names():
    m.egb.inputs[inp].value = np.mean(ex_model.return_train_values(inp))
    m.egb.inputs[inp].setlb(np.min(ex_model.return_train_values(inp)))
    m.egb.inputs[inp].setub(np.max(ex_model.return_train_values(inp)))
  for out in ex_model.output_names():
    m.egb.outputs[out].value = np.mean(ex_model.return_train_values(out))
    m.egb.outputs[out].setlb(np.min(ex_model.return_train_values(out)))
    m.egb.outputs[out].setub(np.max(ex_model.return_train_values(out)))
  m.obj = pyo.Objective(expr=m.egb.outputs[out])

  return m

if __name__ == '__main__':
  for cnt, item in enumerate(sys.argv):
    if item.lower() == "-r":
      rom_file = sys.argv[cnt+1]
    if item.lower() == "-f":
      raven_framework = sys.argv[cnt+1]
    if item.lower() == "-order":
      order = int(sys.argv[cnt+1])
    else:
      order = 1
  ext_model = ravenROM(**{'rom_file':rom_file,'raven_framework':raven_framework,'order':order})
  concreteModel = pyomoModel(ext_model)
  ### here you should implement the optimization problem
  ###
  solver = pyo.SolverFactory('cyipopt')
  solver.config.options['hessian_approximation'] = 'limited-memory'
  results = solver.solve(concreteModel)
  print(results)
  count = 0
  lines = str(results).split("\n")
  with open ("GreyModelOutput_cyipopt.txt", "w") as textfile:
    with redirect_stdout(textfile):
      print("---------------------< Output Summary >---------------------" + "\n")
      for element in lines:
        textfile.write(element + "\n")
      print("---------------------< Optimization Results >---------------------" + "\n")
      concreteModel.pprint()
"""
    return template
