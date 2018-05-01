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
import warnings
warnings.simplefilter('default', DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import inspect
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Dummy import Dummy
import SupervisedLearning
from utils import utils
from utils import InputData
import Files
import LearningGate

#Internal Modules End--------------------------------------------------------------------------------


class ROM(Dummy):
  """
    ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls. This one seems a bit excessive, are all of these for this class?
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(ROM, cls).getInputSpecification()

    IndexSetInputType = InputData.makeEnumType(
        "indexSet", "indexSetType",
        ["TensorProduct", "TotalDegree", "HyperbolicCross", "Custom"])
    CriterionInputType = InputData.makeEnumType(
        "criterion", "criterionType", ["bic", "aic", "gini", "entropy", "mse"])

    InterpolationInput = InputData.parameterInputFactory(
        'Interpolation', contentType=InputData.StringType)
    InterpolationInput.addParam("quad", InputData.StringType, False)
    InterpolationInput.addParam("poly", InputData.StringType, False)
    InterpolationInput.addParam("weight", InputData.FloatType, False)

    inputSpecification.addSub(
        InputData.parameterInputFactory(
            'Features', contentType=InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory(
            'Target', contentType=InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("IndexPoints", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("IndexSet", IndexSetInputType))
    inputSpecification.addSub(
        InputData.parameterInputFactory(
            'pivotParameter', contentType=InputData.StringType))
    inputSpecification.addSub(InterpolationInput)
    inputSpecification.addSub(
        InputData.parameterInputFactory("PolynomialOrder",
                                        InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("SobolOrder", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("SparseGrid", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("persistence", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("gradient", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("simplification", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("graph", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("beta", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("knn", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("partitionPredictor",
                                        InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("smooth", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("kernel", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("bandwidth", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("p", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("SKLtype", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("n_iter", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("tol", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("alpha_1", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("alpha_2", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("lambda_1", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("lambda_2", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("compute_score",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("threshold_lambda",
                                        InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("fit_intercept",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("normalize",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("verbose",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("alpha", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("l1_ratio", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("max_iter", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("warm_start",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("positive",
                                        InputData.StringType))  #bool?
    inputSpecification.addSub(
        InputData.parameterInputFactory("eps", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("n_alphas", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("precompute", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("n_nonzero_coefs",
                                        InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("fit_path",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("max_n_alphas", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("criterion", CriterionInputType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("penalty",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("dual", InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("C", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("intercept_scaling",
                                        InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("class_weight", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("random_state", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("cv", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("shuffle",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("loss", InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("epsilon", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("eta0", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("solver", InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("alphas", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("scoring", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("gcv_mode",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("store_cv_values",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("learning_rate",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("power_t", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("multi_class", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("kernel", InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("degree", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("gamma", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("coef0", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("probability",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("shrinking",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("cache_size", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("nu", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("code_size", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("fit_prior",
                                        InputData.StringType))  #bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("class_prior", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("binarize", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("n_neighbors", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("weights",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("algorithm",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("leaf_size", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("metric",
                                        InputData.StringType))  #enum?
    inputSpecification.addSub(
        InputData.parameterInputFactory("radius", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("outlier_label", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("shrink_threshold",
                                        InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("priors", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("reg_param", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("splitter",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("max_features", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("max_depth", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("min_samples_split",
                                        InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("min_samples_leaf",
                                        InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("max_leaf_nodes",
                                        InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("regr", InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("corr", InputData.StringType))  #enum?
    inputSpecification.addSub(
        InputData.parameterInputFactory("beta0", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("storage_mode",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("theta0", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("thetaL", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("thetaU", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("nugget", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("optimizer",
                                        InputData.StringType))  #enum
    inputSpecification.addSub(
        InputData.parameterInputFactory("random_start", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("Pmax", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("Pmin", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("Qmax", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("Qmin", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("outTruncation", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("Fourier", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("FourierOrder", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("reseedCopies", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("reseedValue", InputData.IntegerType))
    # inputs for neural_network
    inputSpecification.addSub(
        InputData.parameterInputFactory("hidden_layer_sizes",
                                        InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("activation", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("batch_size", InputData.StringType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("learning_rate_init",
                                        InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("momentum", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("nesterovs_momentum",
                                        InputData.StringType))  # bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("early_stopping",
                                        InputData.StringType))  # bool
    inputSpecification.addSub(
        InputData.parameterInputFactory("validation_fraction",
                                        InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("beta_1", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("beta_2", InputData.FloatType))
    # PolyExp
    inputSpecification.addSub(
        InputData.parameterInputFactory("maxNumberExpTerms",
                                        InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("numberExpTerms",
                                        InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("maxPolyOrder", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("polyOrder", InputData.IntegerType))
    coeffRegressorEnumType = InputData.makeEnumType(
        "coeffRegressor", "coeffRegressorType", ["poly", "spline", "nearest"])
    inputSpecification.addSub(
        InputData.parameterInputFactory(
            "coeffRegressor", contentType=coeffRegressorEnumType))
    # DMD
    inputSpecification.addSub(
        InputData.parameterInputFactory("rankSVD", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("energyRankSVD", InputData.FloatType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("rankTLSQ", InputData.IntegerType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("exactModes", InputData.BoolType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("optimized", InputData.BoolType))
    inputSpecification.addSub(
        InputData.parameterInputFactory("dmdType", InputData.StringType))

    #Estimators can include ROMs, and so because baseNode does a copy, this
    #needs to be after the rest of ROMInput is defined.
    EstimatorInput = InputData.parameterInputFactory(
        'estimator',
        contentType=InputData.StringType,
        baseNode=inputSpecification)
    EstimatorInput.addParam("estimatorType", InputData.StringType, False)
    #The next lines are to make subType and name not required.
    EstimatorInput.addParam("subType", InputData.StringType, False)
    EstimatorInput.addParam("name", InputData.StringType, False)
    inputSpecification.addSub(EstimatorInput)

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input'] = [cls.validateDict['Input'][0]]
    cls.validateDict['Input'][0]['required'] = True
    cls.validateDict['Input'][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'] = ['PointSet', 'HistorySet']

  def __init__(self, runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self, runInfoDict)
    self.initializationOptionDict = {}  # ROM initialization options
    self.amITrained = False  # boolean flag, is the ROM trained?
    self.supervisedEngine = None  # dict of ROM instances (== number of targets => keys are the targets)
    self.printTag = 'ROM MODEL'

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy._readMoreXML(self, xmlNode)
    self.initializationOptionDict['name'] = self.name
    paramInput = ROM.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    def tryStrParse(s):
      """
        Trys to parse if it is stringish
        @ In, s, string, possible string
        @ Out, s, string, original type, or possibly parsed string
      """
      return utils.tryParse(s) if type(s).__name__ in ['str', 'unicode'] else s

    for child in paramInput.subparts:
      if len(child.parameterValues) > 0:
        if child.getName() == 'alias':
          continue
        if child.getName() not in self.initializationOptionDict.keys():
          self.initializationOptionDict[child.getName()] = {}
        self.initializationOptionDict[child.getName()][
            child.value] = child.parameterValues
      else:
        if child.getName() == 'estimator':
          self.initializationOptionDict[child.getName()] = {}
          for node in child.subparts:
            self.initializationOptionDict[child.getName()][
                node.getName()] = tryStrParse(node.value)
        else:
          self.initializationOptionDict[child.getName()] = tryStrParse(
              child.value)
    # if working with a pickled ROM, send along that information
    if self.subType == 'pickledROM':
      self.initializationOptionDict['pickled'] = True
    self._initializeSupervisedGate(**self.initializationOptionDict)
    #the ROM is instanced and initialized
    self.mods = self.mods + list(
        set(
            utils.returnImportModuleString(
                inspect.getmodule(SupervisedLearning), True)) - set(self.mods))
    self.mods = self.mods + list(
        set(
            utils.returnImportModuleString(
                inspect.getmodule(LearningGate), True)) - set(self.mods))

  def _initializeSupervisedGate(self, **initializationOptions):
    """
      Method to initialize the supervisedGate class
      @ In, initializationOptions, dict, the initialization options
      @ Out, None
    """
    self.supervisedEngine = LearningGate.returnInstance(
        'SupervisedGate', self.subType, self, **initializationOptions)

  def printXML(self, options={}):
    """
      Called by the OutStreamPrint object to cause the ROM to print itself to file.
      @ In, options, dict, optional, the options to use in printing, including filename, things to print, etc.
      @ Out, None
    """
    #determine dynamic or static
    dynamic = self.supervisedEngine.isADynamicModel
    # determine if it can handle dynamic data
    handleDynamicData = self.supervisedEngine.canHandleDynamicData
    # get pivot parameter
    pivotParameterId = self.supervisedEngine.pivotParameterId
    # establish file
    if 'filenameroot' in options.keys():
      filenameLocal = options['filenameroot']
    else:
      filenameLocal = self.name + '_dump'
    if dynamic and not handleDynamicData:
      outFile = Files.returnInstance('DynamicXMLOutput', self)
    else:
      outFile = Files.returnInstance('StaticXMLOutput', self)
    outFile.initialize(filenameLocal + '.xml', self.messageHandler)
    outFile.newTree('ROM', pivotParam=pivotParameterId)
    #get all the targets the ROMs have
    ROMtargets = self.supervisedEngine.initializationOptions['Target'].split(
        ",")
    #establish targets
    targets = options['target'].split(
        ',') if 'target' in options.keys() else ROMtargets
    #establish sets of engines to work from
    engines = self.supervisedEngine.supervisedContainer
    #handle 'all' case
    if 'all' in targets:
      targets = ROMtargets
    # setup print
    engines[0].printXMLSetup(outFile, options)
    #this loop is only 1 entry long if not dynamic
    for s, rom in enumerate(engines):
      if dynamic and not handleDynamicData:
        pivotValue = self.supervisedEngine.historySteps[s]
      else:
        pivotValue = 0
      for target in targets:
        #for key,target in step.items():
        #skip the pivot param
        if target == pivotParameterId:
          continue
        #otherwise, if this is one of the requested keys, call engine's print method
        if target in ROMtargets:
          options['Target'] = target
          self.raiseAMessage('Printing time-like', pivotValue, 'target',
                             target, 'ROM XML')
          rom.printXML(outFile, pivotValue, options)
    self.raiseADebug('Writing to XML file...')
    outFile.writeFile()
    self.raiseAMessage('ROM XML printed to "' + filenameLocal + '.xml"')

  def reset(self):
    """
      Reset the ROM
      @ In,  None
      @ Out, None
    """
    self.supervisedEngine.reset()
    self.amITrained = False

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

  def train(self, trainingSet):
    """
      This function train the ROM
      @ In, trainingSet, dict or PointSet or HistorySet, data used to train the ROM; if an HistorySet is provided the a list of ROM is created in order to create a temporal-ROM
      @ Out, None
    """
    if type(trainingSet).__name__ == 'ROM':
      self.initializationOptionDict = copy.deepcopy(
          trainingSet.initializationOptionDict)
      self.trainingSet = copy.copy(trainingSet.trainingSet)
      self.amITrained = copy.deepcopy(trainingSet.amITrained)
      self.supervisedEngine = copy.deepcopy(trainingSet.supervisedEngine)
    else:
      self.trainingSet = copy.copy(self._inputToInternal(trainingSet))
      self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout',
                                                 False)
      self.supervisedEngine.train(self.trainingSet)
      self.amITrained = self.supervisedEngine.amITrained

  def confidence(self, request, target=None):
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

  def evaluate(self, request):
    """
      When the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
      @ In, request, datatype, feature coordinates (request)
      @ Out, outputEvaluation, dict, the dict containing the outputs for each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    inputToROM = self._inputToInternal(request)
    outputEvaluation = self.supervisedEngine.evaluate(inputToROM)
    # assure numpy array formatting # TODO can this be done in the supervised engine instead?
    for k, v in outputEvaluation.items():
      outputEvaluation[k] = np.atleast_1d(v)
    return outputEvaluation

  def _externalRun(self, inRun):
    """
      Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
      @ In, inRun, datatype, feature coordinates
      @ Out, returnDict, dict, the return dictionary containing the results
    """
    returnDict = self.evaluate(inRun)
    self._replaceVariablesNamesWithAliasSystem(returnDict, 'output', True)
    self._replaceVariablesNamesWithAliasSystem(inRun, 'input', True)
    return returnDict

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
    self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'], 'input',
                                               True)
    rlz = dict((var, np.atleast_1d(kwargs[var])) for var in kwargs.keys())
    # update rlz with input space from inRun and output space from result
    rlz.update(
        dict((var,
              np.atleast_1d(inRun[var]
                            if var in kwargs['SampledVars'] else result[var]))
             for var in set(result.keys() + inRun.keys())))
    return rlz

  def reseed(self, seed):
    """
      Used to reset the seed of the underlying ROM.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    self.supervisedEngine.reseed(seed)
