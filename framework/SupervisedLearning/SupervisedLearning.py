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

  @author: alfoa

  Originally from ../SupervisedLearning.py, split in PR #650 in July 2018
  Base subclass definition for all supported type of ROM aka Surrogate Models etc
  Previous module notes:
  here we intend ROM as super-visioned learning,
  where we try to understand the underlying model by a set of labeled sample
  a sample is composed by (feature,label) that is easy translated in (input,output)
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import copy
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils, mathUtils, xmlUtils
from BaseClasses import MessageUser
#Internal Modules End--------------------------------------------------------------------------------

class supervisedLearning(utils.metaclass_insert(abc.ABCMeta), MessageUser):
  """
    This is the general interface to any supervisedLearning learning method.
    Essentially it contains a train method and an evaluate method
  """
  returnType       = ''    # this describe the type of information generated the possibility are 'boolean', 'integer', 'float'
  qualityEstType   = []    # this describe the type of estimator returned known type are 'distance', 'probability'. The values are returned by the self.__confidenceLocal__(Features)
  ROMtype          = ''    # the broad class of the interpolator
  ROMtimeDependent = False # is this ROM able to treat time-like (any monotonic variable) explicitly in its formulation?

  @staticmethod
  def checkArrayConsistency(arrayIn,isDynamic=False):
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
          resp = supervisedLearning.checkArrayConsistency(elementArray)
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

  def __init__(self, **initDict):
    """
      A constructor that will appropriately initialize a supervised learning object
      @ In, initDict, dict, an arbitrary list of kwargs
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Supervised'
    self.features = None           # "inputs" to this model
    self.target = None             # "outputs" of this model
    self.amITrained = False
    self._dynamicHandling = False  # time-like dependence in the model?
    self._assembledObjects = None  # objects assembled by the ROM Model, passed through.
    self.numThreads = None         # threading for run
    self.initOptionDict = None     # construction variables
    self.verbosity = None          # printing verbosity
    self.kerasROMDict = None       # dictionary for ROM builded by Keras
    #average value and sigma are used for normalization of the feature data
    #a dictionary where for each feature a tuple (average value, sigma)
    #these need to be declared in the child classes!!!!
    self.muAndSigmaFeatures = {}   # normalization parameters
    self.metadataKeys = set()      # keys that can be passed to DataObject as meta information
    self.metadataParams = {}       # indexMap for metadataKeys to pass to a DataObject as meta dimensionality
    self.readInitDict(initDict)

  def readInitDict(self, initDict):
    """
      Reads in the initialization dict to initialize this instance
      @ In, initDict, dict, keywords passed to constructor
      @ Out, None
    """
    #booleanFlag that controls the normalization procedure. If true, the normalization is performed. Default = True
    self.numThreads = initDict.pop('NumThreads', None)
    self.initOptionDict = {} if initDict is None else initDict
    if 'Features' not in self.initOptionDict.keys():
      self.raiseAnError(IOError,'Feature names not provided')
    if 'Target' not in self.initOptionDict.keys():
      self.raiseAnError(IOError,'Target name not provided')
    self.features = self.initOptionDict.pop('Features')
    self.target = self.initOptionDict.pop('Target')
    self.verbosity = self.initOptionDict['verbosity'] if 'verbosity' in self.initOptionDict else None
    for target in self.target:
      if target in self.features:
        self.raiseAnError(IOError, f'The target "{target}" is also in the features!')
    self.kerasROMDict = self.initOptionDict.pop('KerasROMDict', None)

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    state = copy.copy(self.__dict__)
    state['initOptionDict'].pop('paramInput', None)
    ## capture what is normally pickled
    if not self.amITrained:
      supervisedEngineObj = state.pop("supervisedContainer", None)
      del supervisedEngineObj
    return state

  def __setstate__(self, d):
    """
      Initialize the ROM with the data contained in newstate
      @ In, d, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(d)

  def addMetaKeys(self, args, params=None):
    """
      Adds keywords to a list of expected metadata keys.
      @ In, args, list(str), keywords to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    if params is None:
      params = {}
    self.metadataKeys = self.metadataKeys.union(set(args))
    self.metadataParams.update(params)

  def removeMetaKeys(self, args):
    """
      Removes keywords to a list of expected metadata keys.
      @ In, args, list(str), keywords to de-register
      @ Out, None
    """
    self.metadataKeys = self.metadataKeys - set(args)
    for arg in set(args):
      self.metadataParams.pop(arg, None)

  def provideExpectedMetaKeys(self):
    """
      Provides the registered list of metadata keys for this entity.
      @ In, None
      @ Out, meta,tuple, (list(str),dict), expected keys (empty if none) and expected indexes related to expected keys
    """
    return self.metadataKeys, self.metadataParams

  def initialize(self, idict):
    """
      Initialization method
      @ In, idict, dict, dictionary of initialization parameters
      @ Out, None
    """
    pass #Overloaded by (at least) GaussPolynomialRom

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

  def train(self, tdict, indexMap=None):
    """
      Method to perform the training of the supervisedLearning algorithm
      NB.the supervisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, dict, training dictionary
      @ In, indexMap, dict, mapping of variables to their dependent indices, if any
      @ Out, None
    """
    if type(tdict) != dict:
      self.raiseAnError(TypeError,'In method "train", the training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = list(tdict.keys()), list(tdict.values())
    ## This is for handling the special case needed by SKLtype=*MultiTask* that
    ## requires multiple targets.

    targetValues = []
    for target in self.target:
      if target in names:
        targetValues.append(values[names.index(target)])
      else:
        self.raiseAnError(IOError,'The target '+target+' is not in the training set')

    #FIXME: when we do not support anymore numpy <1.10, remove this IF STATEMENT
    if int(np.__version__.split('.')[1]) >= 10:
      targetValues = np.stack(targetValues, axis=-1)
    else:
      sl = (slice(None),) * np.asarray(targetValues[0]).ndim + (np.newaxis,)
      targetValues = np.concatenate([np.asarray(arr)[sl] for arr in targetValues], axis=np.asarray(targetValues[0]).ndim)

    # construct the evaluation matrixes
    ## add the indices if they're not present
    needFeatures = copy.deepcopy(self.features)
    needTargets = copy.deepcopy(self.target)
    if indexMap:
      for feat in self.features:
        for index in indexMap.get(feat, []):
          if index not in needFeatures and index not in needTargets:
            needFeatures.append(feat)

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
        if len(valueToUse) != featureValues[:,0].size:
          self.raiseAWarning('feature values:',featureValues[:,0].size,tag='ERROR')
          self.raiseAWarning('target values:',len(valueToUse),tag='ERROR')
          self.raiseAnError(IOError,'In training set, the number of values provided for feature '+feat+' are != number of target outcomes!')
        self._localNormalizeData(values,names,feat)
        # valueToUse can be either a matrix (for who can handle time-dep data) or a vector (for who can not)
        featureValues[:,cnt] = ( (valueToUse[:,0] if len(valueToUse.shape) > 1 else valueToUse[:]) - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
    self.__trainLocal__(featureValues,targetValues)
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

  def confidence(self,edict):
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
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
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
      Method to perform the evaluation of a point or a set of points through the previous trained supervisedLearning algorithm
      NB.the supervisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluate, dict, {target: evaluated points}
    """
    return self.evaluate(edict)

  def evaluate(self,edict):
    """
      Method to perform the evaluation of a point or a set of points through the previous trained supervisedLearning algorithm
      NB.the supervisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluate, dict, {target: evaluated points}
    """
    if type(edict) != dict:
      self.raiseAnError(IOError,'method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values  = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index], self.isDynamic())
      if not resp[0]:
        self.raiseAnError(IOError,'In evaluate request for feature '+names[index]+':'+resp[1])
    # construct the evaluation matrix
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names:
        self.raiseAnError(IOError,'The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)], self.isDynamic())
        if not resp[0]:
          self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = ((values[names.index(feat)] - self.muAndSigmaFeatures[feat][0]))/self.muAndSigmaFeatures[feat][1]
    return self.__evaluateLocal__(featureValues)

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
    iniParDict = dict(list(self.initOptionDict.items()) + list({'returnType':self.__class__.returnType,'qualityEstType':self.__class__.qualityEstType,'Features':self.features,
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
    else:
      writeTo.addScalar('ROM', "type", self.printTag)

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
    writeTo.addScalar('ROM',"noInfo",'ROM has no special output options.')

  def isDynamic(self):
    """
      This method is a utility function that tells if the relative ROM is able to
      treat dynamic data (e.g. time-series) on its own or not (Primarly called by LearningGate)
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

  def adjustLocalRomSegment(self, settings):
    """
      Adjusts this ROM to account for it being a segment as a part of a larger ROM collection.
      Call this before training the subspace segment ROMs
      Note this is called on the LOCAL subsegment ROMs, NOT on the GLOBAL templateROM from the ROMcollection!
      @ In, settings, dict, as from getGlobalRomSegmentSettings
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

  def finalizeGlobalRomSegmentEvaluation(self, settings, evaluation):
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
  def __trainLocal__(self,featureVals,targetVals):
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
      @ In,  featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals , np.array, 1-D numpy array [n_samples]
    """

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
