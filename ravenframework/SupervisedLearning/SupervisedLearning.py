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
  Created on May 8, 2018

  @author: alfoa, wangc

  Originally from ../SupervisedLearning.py, split in PR #650 in July 2018
  Base subclass definition for all supported type of ROM aka Surrogate Models etc
  Previous module notes:
  here we intend ROM as super-visioned learning,
  where we try to understand the underlying model by a set of labeled sample
  a sample is composed by (feature,label) that is easy translated in (input,output)
"""

#External Modules------------------------------------------------------------------------------------
import abc
import copy
import numpy as np
import sklearn
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import utils, mathUtils, xmlUtils
from ..utils import InputTypes, InputData
from ..BaseClasses import BaseInterface
from .FeatureSelection import factory as featureSelectionFactory
from .FeatureSelection import utils as featSelectUtils
#Internal Modules End--------------------------------------------------------------------------------

class SupervisedLearning(BaseInterface):
  """
    This is the general interface to any SupervisedLearning learning method.
    Essentially it contains a train method and an evaluate method
  """
  returnType       = ''    # this describe the type of information generated the possibility are
                           # 'boolean', 'integer', 'float'
  qualityEstType   = []    # this describe the type of estimator returned known type are 'distance', 'probability'.
                           # The values are returned by the self.__confidenceLocal__(Features)
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.addParam("subType", param_type=InputTypes.StringType, required=True,
        descr=r"""specify the type of ROM that will be used""")
    spec.addSub(InputData.parameterInputFactory('Features',contentType=InputTypes.StringListType,
        descr=r"""specifies the names of the features of this ROM.
        \nb These parameters are going to be requested for the training of this object
        (see Section~\ref{subsec:stepRomTrainer})"""))
    spec.addSub(InputData.parameterInputFactory('Target',contentType=InputTypes.StringListType,
        descr=r"""contains a comma separated list of the targets of this ROM. These parameters
        are the Figures of Merit (FOMs) this ROM is supposed to predict.
        \nb These parameters are going to be requested for the training of this
        object (see Section \ref{subsec:stepRomTrainer})."""))
    spec.addSub(InputData.parameterInputFactory('pivotParameter',contentType=InputTypes.StringType,
        descr=r"""If a time-dependent ROM is requested, please specifies the pivot
        variable (e.g. time, etc) used in the input HistorySet.""", default='time'))
    ######################
    # feature selections #
    # dynamically loaded #
    ######################
    featureSelection = InputData.parameterInputFactory("featureSelection",
                                                       descr=r"""Apply feature selection algorithm""")
    for subType in featureSelectionFactory.knownTypes():
      validClass = featureSelectionFactory.returnClass(subType)
      validSpec = validClass.getInputSpecification()
      featureSelection.addSub(validSpec)
    spec.addSub(featureSelection)
    # Feature space transformation?
    featureSpaceTransformation = InputData.parameterInputFactory('featureSpaceTransformation',
                                                   descr=r"""Use dimensionality reduction technique to perform a trasformation of the training dataset
                                                  into an uncorrelated one. The dimensionality of the problem will not be reduced but
                                                  the data will be transformed in the transformed space. E.g if the number of features
                                                  are 5, the method projects such features into a new uncorrelated space (still 5-dimensional).
                                                  In case of time-dependent ROMs, all the samples are concatenated in a global 2D matrix
                                                  (n_samples*n_timesteps,n_features) before applying the transformation and then reconstructed
                                                  back into the original shape (before fitting the model).
                                                   """)
    transformationMethodType = InputTypes.makeEnumType("transformationMethod",
                                                       "transformationMethodeType",
                                                       ['PCA', 'KernelLinearPCA','KernelPolyPCA','KernelRbfPCA','KernelSigmoidPCA','KernelCosinePCA', 'ICA'])
    featureSpaceTransformation.addSub(InputData.parameterInputFactory('transformationMethod',contentType=transformationMethodType,
        descr=r"""Transformation method to use. Eight options (5 Kernel PCAs) are available:
                  \begin{itemize}
                    \item \textit{PCA}, Principal Component Analysis;
                    \item \textit{KernelLinearPCA}, Kernel (Linear) Principal component analysis;
                    \item \textit{KernelPolyPCA}, Kernel (Poly) Principal component analysis;
                    \item \textit{KernelRbfPCA}, Kernel(Rbf) Principal component analysis;
                    \item \textit{KernelSigmoidPCA}, Kernel (Sigmoid) Principal component analysis;
                    \item \textit{KernelCosinePCA}, Kernel (Cosine) Principal component analysis;
                    \item \textit{ICA}, Independent component analysis;
                   \end{itemize}

        """, default="PCA"))
    featureSpaceTransformation.addSub(InputData.parameterInputFactory('parametersToInclude',contentType=InputTypes.StringListType,
        descr=r"""List of IDs of features/variables to include in the transformation process.""", default=None))

    spaceEnum = InputTypes.makeEnumType("spaceEnum","spaceEnumType",['Feature','feature', 'Target','target'])
    featureSpaceTransformation.addSub(InputData.parameterInputFactory('whichSpace',contentType=spaceEnum,
        descr=r"""Which space to search? Target or Feature?""", default="Feature"))
    spec.addSub(featureSpaceTransformation)

    cvInput = InputData.parameterInputFactory("CV", contentType=InputTypes.StringType,
        descr=r"""The text portion of this node needs to contain the name of the \xmlNode{PostProcessor} with \xmlAttr{subType}
        ``CrossValidation``.""")
    cvInput.addParam("class", InputTypes.StringType, descr=r"""should be set to \xmlString{Model}""")
    cvInput.addParam("type", InputTypes.StringType, descr=r"""should be set to \xmlString{PostProcessor}""")
    spec.addSub(cvInput)
    AliasInput = InputData.parameterInputFactory("alias", contentType=InputTypes.StringType,
        descr=r"""specifies alias for
        any variable of interest in the input or output space. These aliases can be used anywhere in the RAVEN input to
        refer to the variables. In the body of this node the user specifies the name of the variable that the model is going to use
        (during its execution).""")
    AliasInput.addParam("variable", InputTypes.StringType, True, descr=r"""define the actual alias, usable throughout the RAVEN input""")
    AliasTypeInput = InputTypes.makeEnumType("aliasType","aliasTypeType",["input","output"])
    AliasInput.addParam("type", AliasTypeInput, True, descr=r"""either ``input'' or ``output''.""")
    spec.addSub(AliasInput)
    return spec

  @staticmethod
  def checkArrayConsistency(arrayIn, isDynamic=False):
    """
      This method checks the consistency of the in-array
      @ In, arrayIn, object,  It should be an array
      @ In, isDynamic, bool, optional, is Dynamic?
      @ Out, (consistent, 'error msg'), tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    """
    #checking if None provides a more clear message about the problem
    if arrayIn is None:
      return (False,' The object is None, and contains no entries!')
    if type(arrayIn).__name__ == 'list':
      if isDynamic:
        for cnt, elementArray in enumerate(arrayIn):
          resp = SupervisedLearning.checkArrayConsistency(elementArray)
          if not resp[0]:
            return (False,' The element number '+str(cnt)+' is not a consistent array. Error: '+resp[1])
      else:
        return (False,' The list type is allowed for dynamic ROMs only')
    else:
      if type(arrayIn).__name__ not in ['ndarray','c1darray']:
        return (False,' The object is not a numpy array. Got type: '+type(arrayIn).__name__)
      if len(np.asarray(arrayIn).shape) > 1:
        return(False, ' The array must be 1-d. Got shape: '+str(np.asarray(arrayIn).shape))
    return (True,'')

  def __init__(self):
    """
      A constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    #Note: self.saveParam (BaseInterface class) is set at the _handleInput stage
    super().__init__()
    self.printTag = type(self).__name__
    # "inputs" to this model
    self.features = None
    # "outputs" of this model
    self.target = None
    # "True" if the ROM is already trained
    self.amITrained = False
    # time-like dependence in the model?
    self._dynamicHandling = False
    # time-like dependence in the feature space? FIXME: this is not the right design
    self.dynamicFeatures = False
    # feature selection algorithm container
    self.featureSelectionAlgo = None
    # the feature selection has been performed already?
    self.doneSelectionFeatures = False
    # should a feature space transformation be performed?
    self.performFeatureSpaceTransformation = False
    # container of the feature space transformation settings
    self.featureSpaceTransformationSettings = {}
    # objects assembled by the ROM Model, passed through.
    self._assembledObjects = None
    #average value and sigma are used for normalization of the feature data
    #a dictionary where for each feature a tuple (average value, sigma)
    #these need to be declared in the child classes!!!!
    # normalization parameters
    self.muAndSigmaFeatures = {}
    # keys that can be passed to DataObject as meta information
    self.metadataKeys = set()
    # indexMap for metadataKeys to pass to a DataObject as meta dimensionality
    self.metadataParams = {}
    # This parameter is set at the initialization of the model
    # If True, the importances are computed if no 'feature_importances_' and
    # 'coef_' are set by the estimator (see def initializeModel)
    # After the computation, the importances are set as attribute of the self.model
    # variable and called 'feature_importances_' and accessable as self.model.feature_importances_
    self.computeImportances = False

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    state = copy.copy(self.__dict__)

    if state.get('featureSelectionAlgo') is not None:
      del state['featureSelectionAlgo']
    return state

  def __setstate__(self, d):
    """
      Initialize the ROM with the data contained in newstate
      @ In, d, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(d)
    if self.saveParams:
      fs = self.paramInput.findFirst("featureSelection")
      if  fs is not None:
        self.featureSelectionAlgo = featureSelectionFactory.returnInstance(fs.subparts[0].getName())
        self.featureSelectionAlgo._handleInput(fs.subparts[0])

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    # check if paramInput must be saved.
    # currently they are saved if and only if a feature selection
    # algorithm is activated
    # Consequentially we check for the presence of feature selection
    # as first thing before processing the input data
    featSelection = paramInput.findFirst("featureSelection")
    self.saveParams = featSelection is not None
    # now we can proceed with the input handling
    super()._handleInput(paramInput)
    nodes, notFound = paramInput.findNodesAndExtractValues(['Features', 'Target', 'pivotParameter'])
    assert(not notFound)
    self.features = nodes['Features']
    self.target = nodes['Target']
    self.pivotID = nodes['pivotParameter']

    if featSelection is not None:
      if len(featSelection.subparts) > 1:
        self.raiseAnError(IOError, "Only one feature selection algorithm is allowed in the ROM")
      featAlgo = featSelection.subparts[0]
      self.featureSelectionAlgo = featureSelectionFactory.returnInstance(featAlgo.getName())
      # handle input
      self.featureSelectionAlgo._handleInput(featAlgo)
      # if the feature selection algorithm is set, we should always have a mean to compute
      # the feature importances (e.g. either the model can provide them or we use permutation)
      self.computeImportances = True
    # dim reduction to transform the training space?
    featureSpaceTransformation = paramInput.findFirst("featureSpaceTransformation")
    if featureSpaceTransformation is not None:
      self.performFeatureSpaceTransformation = True
      nodesFeatureTransformation, notFound = featureSpaceTransformation.findNodesAndExtractValues(['parametersToInclude', 'whichSpace', 'transformationMethod'])
      if nodesFeatureTransformation['parametersToInclude'] is None:
        self.raiseAnError(IOError, '"parametersToInclude" not found. It must be inputted in Feature Space Transformation settings!' )
      self.featureSpaceTransformationSettings.update(nodesFeatureTransformation)
      self.featureSpaceTransformationSettings['whichSpace'] = self.featureSpaceTransformationSettings['whichSpace'].lower()

    dups = set(self.target).intersection(set(self.features))
    if len(dups) != 0:
      self.raiseAnError(IOError, 'The target(s) "{}" is/are also among the given features!'.format(', '.join(dups)))

  ## This method is used when the SupervisedLearning ROM is directly initiated within another module
  def initializeFromDict(self, inputDict):
    """
      Function which initializes the ROM given a the information contained in inputDict
      @ In, inputDict, dict, dictionary containing the values required to initialize the ROM
      @ Out, None
    """
    self.features = inputDict.get('Features', None)
    self.target = inputDict.get('Target', None)
    self.featureSelectionAlgo = inputDict.get('featureSelectionAlgorithm', None)
    self.verbosity = inputDict.get('verbosity', None)

  def setEstimator(self, estimatorList):
    """
      Initialization method
      @ In, estimatorList, list of ROM instances/estimators used by ROM
      @ Out, None
    """
    pass

  ## TODO: we may not need the set and read AssembleObjects
  ## currently only used by ROMCollection
  def setAssembledObjects(self, assembledObjects):
    """
      Allows providing entities from the Assembler to be used in supervised learning algorithms.
      @ In, assembledObjects, dict, assembled objects that the ROM model requested as an Assembler.
      @ Out, None
    """
    self._assembledObjects = assembledObjects

  def readAssembledObjects(self):
    """
      Collects the entities from the Assembler as needed.
      In general, SVL don't need any assembled objects.
      @ In, None
      @ Out, None
    """
    pass

  @property
  def featureImportances_(self):
    """
      Method to return the features' importances
      @ In, None
      @ Out, featureImportances_, dict of dicts, dict of importances {'target1':{feature1:importance, feature1:importance,...},...}
    """
    return dict.fromkeys(self.target,dict.fromkeys(self.features,1.))

  @property
  def requireJobHandler(self):
    """
      Property setting the requirement for job handler (internal parallelization)
      If JobHandler is required, it will be stored in the assembler object container
      (i.e. self._assembledObjects['jobHandler'])
      @ In, None
      @ Out, requireJobHandler, bool, True if jobhandler is required
    """
    return self.featureSelectionAlgo is not None and not self.doneSelectionFeatures

  def train(self, tdict, indexMap=None):
    """
      Method to perform the training of the SupervisedLearning algorithm
      NB.the SupervisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, dict, training dictionary
      @ In, indexMap, dict, mapping of variables to their dependent indices, if any
      @ Out, None
    """
    if type(tdict) != dict:
      self.raiseAnError(TypeError,'In method "train", the training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = list(tdict.keys()), list(tdict.values())
    ## This is for handling the special case needed by skl *MultiTask* that
    ## requires multiple targets.
    targetValues = []
    for target in self.target:
      if target in names:
        targetValues.append(values[names.index(target)])
      else:
        self.raiseAnError(IOError,'The target '+target+' is not in the training set')

    # stack targets
    targetValues = np.stack(targetValues, axis=-1)
    # construct the evaluation matrixes
    ## add the indices if they're not present
    needFeatures = copy.deepcopy(self.features)
    needTargets = copy.deepcopy(self.target)
    if indexMap:
      for feat in self.features:
        for index in indexMap.get(feat, []):
          if index not in needFeatures and index not in needTargets:
            needFeatures.append(feat)
    if self.dynamicFeatures:
      featLen = 0
      for cnt, feat in enumerate(self.features):
        featLen = max(values[names.index(feat)][0].size, featLen)
      featureValues = np.zeros(shape=(len(targetValues), featLen,len(self.features)))
    else:
      featureValues = np.zeros(shape=(len(targetValues), len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names:
        self.raiseAnError(IOError,'The feature sought '+feat+' is not in the training set')
      else:
        valueToUse = values[names.index(feat)]
        resp = self.checkArrayConsistency(valueToUse, self.isDynamic())
        if not resp[0]:
          self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        valueToUse = np.asarray(valueToUse)
        if len(valueToUse) != featureValues.shape[0]:
          self.raiseAWarning('feature values:',featureValues.shape[0],tag='ERROR')
          self.raiseAWarning('target values:',len(valueToUse),tag='ERROR')
          self.raiseAnError(IOError,'In training set, the number of values provided for feature '+feat+' are != number of target outcomes!')
        self._localNormalizeData(values,names,feat)
        # valueToUse can be either a matrix (for who can handle time-dep data) or a vector (for who can not)
        if self.dynamicFeatures:
          featureValues[:, :, cnt] = (valueToUse[:, :]- self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
        else:
          featureValues[:,cnt] = ( (valueToUse[:,0] if len(valueToUse.shape) > 1 else valueToUse[:]) - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]

    if self.performFeatureSpaceTransformation:
      # nsamples, timeStep, nFeatures
      nComponents = len(self.featureSpaceTransformationSettings['parametersToInclude'])
      if self.featureSpaceTransformationSettings['transformationMethod'] == 'PCA':
        self.transformationEngine = sklearn.decomposition.IncrementalPCA(n_components=nComponents, whiten=True)
      elif self.featureSpaceTransformationSettings['transformationMethod'].startswith("Kernel"):
        # kernel PCA
        kernel = self.featureSpaceTransformationSettings['transformationMethod'].lower().replace("kernel", "").replace("pca", "")
        self.transformationEngine = sklearn.decomposition.KernelPCA(n_components=nComponents, kernel= kernel,  fit_inverse_transform=True, random_state=0)
      elif self.featureSpaceTransformationSettings['transformationMethod'] == 'ICA':
        self.transformationEngine = sklearn.decomposition.FastICA(n_components=nComponents, random_state=0)
      # This should be activated when the scaler is avaialable
      #else:
      #  self.transformationEngine = sklearn.decomposition.NMF(n_components=nComponents)
      #  setattr(self.transformationEngine, "scaler", sklearn.preprocessing.MinMaxScaler())

      params = self.featureSpaceTransformationSettings['parametersToInclude']
      space = self.featureSpaceTransformationSettings['whichSpace']
      if space == 'feature':
        indeces = np.asarray([i for i, e in enumerate(self.features) if e in params])
      else:
        indeces = np.asarray([i for i, e in enumerate(self.target) if e in params])
      if self.dynamicFeatures:
        # we use reshape the training matrix into a (n_samples*n_timesteps,n_features)
        # to come up with a global transfomation for all the samples
        # FIXME: this approach is not rigorous and should be replaced by
        # the application of FPCA (Functional Principal Component Analysis)
        if space == 'feature':
          shape = featureValues.shape
          newSpace = self.transformationEngine.fit_transform(featureValues.reshape(-1,shape[2])[:, indeces].T)
          featureValues = newSpace.reshape(shape)
        else:
          shape = targetValues.shape
          newSpace = self.transformationEngine.fit_transform(targetValues.reshape(-1,shape[2])[:, indeces].T).T
          targetValues = newSpace.reshape(shape)
      else:
        if space == 'feature':
          featureValues[ :, indeces] = self.transformationEngine.fit_transform(featureValues[:, indeces])
        else:
          targetValues[:, indeces] = self.transformationEngine.fit_transform(targetValues[ :, indeces]).T

    if self.featureSelectionAlgo is not None and not self.doneSelectionFeatures:
      if self.featureSelectionAlgo.needROM:
        self.featureSelectionAlgo.setEstimator(self)
      newFeatures, support = self.featureSelectionAlgo.run(self.features, self.target, featureValues, targetValues)
      # identify parameters to remove
      space =  self.featureSelectionAlgo.var("whichSpace")
      # support here is the support vector on the global space (not just the subspace on which the selection has been performed)
      # for this reason, the list of parameters to send are the full target or features
      vals = featSelectUtils.screenInputParams(support, self.paramInput, self.target if 'target' in space else self.features)
      if space == 'feature' and np.sum(support) != len(self.features):
        self.removed = set(self.features) - set(np.asarray(self.features)[newFeatures].tolist())
        self.raiseAMessage("Feature Selection removed the following features: {}".format(', '.join(self.removed)))
        self.raiseAMessage("Old feature space for surrogate model was       : {}".format(', '.join(self.features)))
        self.raiseAMessage("New feature space for surrogate model is now    : {}".format(', '.join(np.asarray(self.features)[newFeatures].tolist())))
      elif space == 'target' and np.sum(support) != len(self.target):
        self.removed = set(self.target) - set(np.asarray(self.target)[newFeatures].tolist())
        self.raiseAMessage("Feature Selection removed the following features (from target space): {}".format(', '.join(self.removed)))
        self.raiseAMessage("Old feature space (in the target space) for surrogate model was     : {}".format(', '.join(self.target)))
        self.raiseAMessage("New feature space (in the target space) for surrogate model is now  : {}".format(', '.join(np.asarray(self.target)[newFeatures].tolist())))
      else:
        self.raiseAMessage("Feature Selection DID NOT remove any feature since all all needed to maximize the performance of the surrogate model!")

      # re-update parameters
      self.paramInput.findNodesAndSetValues(vals)
      self._handleInput(self.paramInput)
      if self.dynamicFeatures:
        if space == 'feature':
          featureValues = featureValues[:,:,support]
        else:
          targetValues = targetValues[:,:,support]
      else:
        if space == 'feature':
          featureValues = featureValues[:,support]
        else:
          targetValues = targetValues[:,support]
      self.doneSelectionFeatures = True
    self._train(featureValues,targetValues)
    self.amITrained = True

  def _localNormalizeData(self,values,names,feat):
    """
      Method to normalize data based on the mean and standard deviation.  If undesired for a particular ROM,
      this method can be overloaded to simply pass (see, e.g., GaussPolynomialRom).
      @ In, values, list, list of feature values (from tdict)
      @ In, names, list, names of features (from tdict)
      @ In, feat, list, list of features (from ROM)
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = mathUtils.normalizationFactors(values[names.index(feat)])

  def confidence(self, edict):
    """
      This call is used to get an estimate of the confidence in the prediction.
      The base class self.confidence will translate a dictionary into numpy array, then call the local confidence
      @ In, edict, dict, evaluation dictionary
      @ Out, confidence, float, the confidence
    """
    if type(edict) != dict:
      self.raiseAnError(IOError,'method "confidence". The inquiring set needs to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values   = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index], self.isDynamic())
      if not resp[0]:
        self.raiseAnError(IOError,'In evaluate request for feature '+names[index]+':'+resp[1])

    if self.dynamicFeatures:
      featureValues = np.zeros(shape=(values[0].size, self.featureShape[1], len(self.features)))
    else:
      featureValues = np.zeros(shape=(values[0].size, len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names:
        self.raiseAnError(IOError,'The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)], self.isDynamic())
        if not resp[0]:
          self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = values[names.index(feat)]
    return self.__confidenceLocal__(featureValues)

  # compatibility with BaseInterface requires having a "run" method
  # TODO during SVL rework, "run" should probably replace "evaluate", maybe?
  def run(self, edict):
    """
      Method to perform the evaluation of a point or a set of points through the previous trained SupervisedLearning algorithm
      NB.the SupervisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluate, dict, {target: evaluated points}
    """
    return self.evaluate(edict)

  def evaluate(self,edict):
    """
      Method to perform the evaluation of a point or a set of points through the previous trained SupervisedLearning algorithm
      NB.the SupervisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluate, dict, {target: evaluated points}
    """
    if type(edict) != dict:
      self.raiseAnError(IOError,'method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values  = list(edict.keys()), list(edict.values())
    stepInFeatures = 0
    for index in range(len(values)):
      # If value is a float or int, convert to numpy array for evaluation
      if isinstance(values[index], (float, int, np.number)):
        values[index] = np.array([values[index]])
      resp = self.checkArrayConsistency(values[index], self.isDynamic())
      if not resp[0]:
        self.raiseAnError(IOError,'In evaluate request for feature '+names[index]+':'+resp[1])
      if self.dynamicFeatures:
        stepInFeatures = max(stepInFeatures,values[index].shape[-1])
    # construct the evaluation matrix
    if self.dynamicFeatures:
      featureValues = np.zeros(shape=(values[0].size, stepInFeatures, len(self.features)))
    else:
      featureValues = np.zeros(shape=(values[0].size, len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names:
        self.raiseAnError(IOError,'The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)], self.isDynamic())
        if not resp[0]:
          self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        if self.dynamicFeatures:
          featureValues[:, :, cnt] = ((values[names.index(feat)] - self.muAndSigmaFeatures[feat][0]))/self.muAndSigmaFeatures[feat][1]
        else:
          featureValues[:,cnt] = ((values[names.index(feat)] - self.muAndSigmaFeatures[feat][0]))/self.muAndSigmaFeatures[feat][1]
    if self.performFeatureSpaceTransformation:
      params = self.featureSpaceTransformationSettings['parametersToInclude']
      space = self.featureSpaceTransformationSettings['whichSpace'].lower()
      if space == 'feature':
        indeces = np.asarray([i for i, e in enumerate(self.features) if e in params])
      else:
        indeces = np.asarray([i for i, e in enumerate(self.target) if e in params])
        sh = featureValues.shape
        reconstructed = np.zeros((sh[1],len(self.target))) if self.dynamicFeatures else np.zeros((len(self.target)))
      if space == 'feature':
        if self.dynamicFeatures:
          shape = featureValues.shape
          newSpace = self.transformationEngine.transform(featureValues.reshape(-1,shape[2])[:, indeces].T)
          featureValues = newSpace.reshape(shape)
        else:
          featureValues[:, indeces] = self.transformationEngine.transform(featureValues[:, indeces])
    # now evaluate
    evaluation = self.__evaluateLocal__(featureValues)
    # if transformation in the target space
    if self.performFeatureSpaceTransformation and space == 'target':
      for idx in indeces:
        if self.dynamicFeatures:
          reconstructed[:,idx] = evaluation[np.asarray(self.target)[idx]][:]
        else:
          reconstructed[idx] = evaluation[np.asarray(self.target)[idx]]
      reconstructed = self.transformationEngine.inverse_transform(reconstructed)
      for idx in indeces:
        if self.dynamicFeatures:
          evaluation[np.asarray(self.target)[idx]] = reconstructed[:,idx]
        else:
          evaluation[np.asarray(self.target)[idx]] = reconstructed[idx]

    if self.doneSelectionFeatures and self.removed:
      dummy = np.empty(list(evaluation.values())[0].shape)
      dummy[:] = np.NaN
      for rem in self.removed:
        evaluation[rem] = dummy
    return evaluation

  def reset(self):
    """
      Reset ROM
    """
    self.amITrained = False
    self.__resetLocal__()

  def returnInitialParameters(self):
    """
      override this method to return the fix set of parameters of the ROM
      @ In, None
      @ Out, iniParDict, dict, initial parameter dictionary
    """
    iniParDict = dict(list({'returnType':self.__class__.returnType,'qualityEstType':self.__class__.qualityEstType,'Features':self.features,
                      'Target':self.target,'returnType':self.__class__.returnType}.items()) + list(self.__returnInitialParametersLocal__().items()))
    return iniParDict

  def returnCurrentSetting(self):
    """
      return the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, currParDict, dict, current parameter dictionary
    """
    currParDict = dict({'Trained':self.amITrained}.items() + self.__CurrentSettingDictLocal__().items())
    return currParDict

  def writeXMLPreamble(self, writeTo, targets=None):
    """
      Allows the SVE to put whatever it wants into an XML file only once (right before calling pringXML)
      Extend in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement instance, Element to write to
      @ In, targets, list, list of targets for whom information should be written
      @ Out, None
    """
    # different calls depending on if it's static or dynamic
    if isinstance(writeTo, xmlUtils.DynamicXmlElement):
      writeTo.addScalar('ROM', "type", self.printTag, None, general = True)
      # write some common parameters (e.g. features, targets, etc.)
      writeTo.addScalar('ROM', "Features", ",".join(self.features), None, general = True)
      writeTo.addScalar('ROM', "Targets", ",".join(self.target), None, general = True)
    else:
      writeTo.addScalar('ROM', "type", self.printTag)
      # write some common parameters (e.g. features, targets, etc.)
      writeTo.addScalar('ROM', "Features", ",".join(self.features))
      writeTo.addScalar('ROM', "Targets", ",".join(self.target))


  def writePointwiseData(self, *args):
    """
      Allows the SVE to add data to a DataObject
      Overload in subclasses.
      @ In, args, list, unused arguments
      @ Out, None
    """
    # by default, nothing to write!
    self.raiseAMessage('Writing ROM "{}", but no pointwise data found. Moving on ...')

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, StaticXmlElement to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    writeTo.addScalar('ROM',"noInfo",'ROM has no special output options yet.')

  def isDynamic(self):
    """
      This method is a utility function that tells if the relative ROM is able to
      treat dynamic data (e.g. time-series) on its own or not (Primarily called by LearningGate)
      @ In, None
      @ Out, isDynamic, bool, True if the ROM is able to treat dynamic data, False otherwise
    """
    return self._dynamicHandling

  def reseed(self,seed):
    """
      Used to reset the seed of the ROM.  By default does nothing; overwrite in the inheriting classes as needed.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    return

  def setAdditionalParams(self, params):
    """
      Sets parameters aside from initialization, such as during deserialization.
      @ In, params, dict, parameters to set (dependent on ROM)
      @ Out, None
    """
    # reseeding is common to many
    seed = params.pop('seed', None)
    if seed:
      self.reseed(seed)
    # overload this method in subclasses to load other parameters

  ### ROM Clustering (see ROMCollection.py) ###
  def isClusterable(self):
    """
      Allows ROM to declare whether it has methods for clustring. Default is no.
      @ In, None
      @ Out, isClusterable, bool, if True then has clustering mechanics.
    """
    # only true if overridden.
    return False

  def checkRequestedClusterFeatures(self, request):
    """
      Takes the user-requested features (sometimes "all") and interprets them for this ROM.
      @ In, request, dict(list), as from ROMColletion.Cluster._extrapolateRequestedClusterFeatures
      @ Out, interpreted, dict(list), interpreted features
    """
    self.raiseAnError(NotImplementedError, 'This ROM is not prepared to handle feature cluster requests!')

  def getLocalRomClusterFeatures(self, *args, **kwargs):
    """
      Provides metrics aka features on which clustering compatibility can be measured.
      This is called on LOCAL subsegment ROMs, not on the GLOBAL template ROM
      @ In, featureTemplate, str, format for feature inclusion
      @ In, settings, dict, as per getGlobalRomSegmentSettings
      @ In, picker, slice, indexer for segmenting data
      @ In, kwargs, dict, arbitrary keyword arguments
      @ Out, features, dict, {target_metric: np.array(floats)} features to cluster on
    """
    # TODO can we do a generic basic statistics clustering on mean, std for all roms?
    self.raiseAnError(NotImplementedError, 'Clustering capabilities not yet implemented for "{}" ROM!'.format(self.__class__.__name__))

  def getGlobalRomSegmentSettings(self, trainingDict, divisions):
    """
      Allows the ROM to perform some analysis before segmenting.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL subsegment ROMs!
      @ In, trainingDict, dict, data for training
      @ In, divisions, tuple, (division slice indices, unclustered spaces)
      @ Out, settings, object, arbitrary information about ROM clustering settings
      @ Out, trainingDict, dict, adjusted training data (possibly unchanged)
    """
    # by default, do nothing
    return None, trainingDict

  def adjustLocalRomSegment(self, settings, picker):
    """
      Adjusts this ROM to account for it being a segment as a part of a larger ROM collection.
      Call this before training the subspace segment ROMs
      Note this is called on the LOCAL subsegment ROMs, NOT on the GLOBAL templateROM from the ROMcollection!
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, picker, slice, slice object for selecting the desired segment
      @ Out, None
    """
    # by default, do nothing
    pass

  def finalizeLocalRomSegmentEvaluation(self, settings, evaluation, picker):
    """
      Allows global settings in "settings" to affect a LOCAL evaluation of a LOCAL ROM
      Note this is called on the LOCAL subsegment ROM and not the GLOBAL templateROM.
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, preliminary evaluation from the local segment ROM as {target: [values]}
      @ In, picker, slice, indexer for data range of this segment
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    return evaluation

  def finalizeGlobalRomSegmentEvaluation(self, settings, evaluation, weights, slicer):
    """
      Allows any global settings to be applied to the signal collected by the ROMCollection instance.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL supspace segment ROMs!
      @ In, evaluation, dict, {target: np.ndarray} evaluated full (global) signal from ROMCollection
      TODO finish docs
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    return evaluation
  ### END ROM Clustering ###

  @abc.abstractmethod
  def _train(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """

  @abc.abstractmethod
  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      This could be distance or probability or anything else, the type needs to be declared in the variable cls.qualityEstType
      @ In, featureVals, 2-D numpy array , [n_samples,n_features]
      @ Out, __confidenceLocal__, float, the confidence
    """

  @abc.abstractmethod
  def __evaluateLocal__(self,featureVals):
    """
      Local evaluation method
      @ In,  featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals , np.array, 1-D numpy array [n_samples]
    """

  def _evaluateLocal(self,featureVals):
    """
      Method accessable outside ROM class to perform direct evaluation
      @ In,  featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals , np.array, 1-D numpy array [n_samples]
    """
    return self.__evaluateLocal__(featureVals)

  @abc.abstractmethod
  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """

  @abc.abstractmethod
  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """

  @abc.abstractmethod
  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
