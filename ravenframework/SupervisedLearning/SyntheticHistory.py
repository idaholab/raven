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
  Created on Jan 5, 2020

  @author: talbpaul, wangc
  Originally from ARMA.py, split for modularity
  Uses TimeSeriesAnalysis (TSA) algorithms to train then generate synthetic histories
"""
import numpy as np
import collections
import copy

from ..utils import InputData, xmlUtils
from ..TSA import TSAUser
from ..TSA.Factory import factory

from .SupervisedLearning import SupervisedLearning

class SyntheticHistory(SupervisedLearning, TSAUser):
  """
    Leverage TSA algorithms to train then generate synthetic signals.
  """
  # class attribute
  ## define the clusterable features for this ROM.
  # _clusterableFeatures = TODO # get from TSA
  @classmethod
  def getInputSpecification(cls):
    """
      Establish input specs for this class.
      @ In, None
      @ Out, spec, InputData.ParameterInput, class for specifying input template
    """
    specs = super().getInputSpecification()
    specs.description = r"""A ROM for characterizing and generating synthetic histories. This ROM makes use of
        a variety of TimeSeriesAnalysis (TSA) algorithms to characterize and generate new
        signals based on training signal sets. It is a more general implementation of the ARMA ROM. The available
        algorithms are discussed in more detail below. The SyntheticHistory ROM uses the TSA algorithms to
        characterize then reproduce time series in sequence; for example, if using Fourier then ARMA, the
        SyntheticHistory ROM will characterize the Fourier properties using the Fourier TSA algorithm on a
        training signal, then send the residual to the ARMA TSA algorithm for characterization. Generating
        new signals works in reverse, first generating a signal using the ARMA TSA algorithm then
        superimposing the Fourier TSA algorithm.
        //
        In order to use this Reduced Order Model, the \xmlNode{ROM} attribute
        \xmlAttr{subType} needs to be \xmlString{SyntheticHistory}."""
    specs = cls.addTSASpecs(specs)
    return specs

  ### INHERITED METHODS ###
  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
                           and printing messages
      @ In, kwargs: an arbitrary dictionary of keywords and values
    """
    # general infrastructure
    SupervisedLearning.__init__(self)
    TSAUser.__init__(self)
    self.printTag = 'SyntheticHistoryROM'
    self._dynamicHandling = True # This ROM is able to manage the time-series on its own.

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    SupervisedLearning._handleInput(self, paramInput)
    self.readTSAInput(paramInput)
    if len(self._tsaAlgorithms)==0:
      self.raiseAWarning("No Segmenting algorithms were requested.")

  def _train(self, featureVals, targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
      @ Out, None
    """
    self.raiseADebug('Training...')
    self.trainTSASequential(targetVals)

  def __evaluateLocal__(self, featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, rlz, dict, realization dictionary of values for each target
    """
    rlz = self.evaluateTSASequential()
    return rlz


  def getGlobalRomSegmentSettings(self, trainingDict, divisions):
    """
      Allows the ROM to perform some analysis before segmenting.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL subsegment ROMs!
      @ In, trainingDict, dict, data for training, full and unsegmented
      @ In, divisions, tuple, (division slice indices, unclustered spaces)
      @ Out, settings, object, arbitrary information about ROM clustering settings
      @ Out, trainingDict, dict, adjusted training data (possibly unchanged)
    """
    self.raiseADebug('Training Global...')
    # extracting info from training Dict, convert all signals to single array
    trainingDict = copy.deepcopy(trainingDict)
    names, values  = list(trainingDict.keys()), list(trainingDict.values())
    ## This is for handling the special case needed by skl *MultiTask* that
    ## requires multiple targets.
    targetValues = []
    targetNames = []
    for target in self.target:
      if target in names:
        targetValues.append(values[names.index(target)])
        targetNames.append(target)
      else:
        self.raiseAnError(IOError,'The target '+target+' is not in the training set')
    # stack targets
    targetValues = np.stack(targetValues, axis=-1)
    self.trainTSASequential(targetValues, trainGlobal=True)
    settings = self.getGlobalTSARomSettings()
    # update targets in trainingDict
    for i,target in enumerate(targetNames):
      trainingDict[target] = targetValues[:,:,i]
    return settings, trainingDict

  def finalizeGlobalRomSegmentEvaluation(self, settings, evaluation, weights, slicer):
    """
      Allows any global settings to be applied to the signal collected by the ROMCollection instance.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL supspace segment ROMs!
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, {target: np.ndarray} evaluated full (global) signal from ROMCollection
      @ In, weights, np.array(float), optional, if included then gives weight to histories for CDF preservation
      @ In, slicer, slice, indexer for data range of this segment FROM GLOBAL SIGNAL
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    if len(self._tsaGlobalAlgorithms)>0:
      rlz = self.evaluateTSASequential(evalGlobal=True, evaluation=evaluation, slicer=slicer)
      for key,val in rlz.items():
        evaluation[key] = val
    return evaluation

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    pass # TODO

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused (kept for compatability)
      @ In, skip, list, optional, unused (kept for compatability)
      @ Out, None
    """
    self.writeTSAtoXML(writeTo)

  ### Segmenting and Clustering ###
  def isClusterable(self):
    """
      Allows ROM to declare whether it has methods for clustring. Default is no.
      @ In, None
      @ Out, isClusterable, bool, if True then has clustering mechanics.
    """
    # clustering methods have been added
    return True

  def checkRequestedClusterFeatures(self, request):
    """
      Takes the user-requested features (sometimes "all") and interprets them for this ROM.
      @ In, request, dict(list), as from ROMColletion.Cluster._extrapolateRequestedClusterFeatures
      @ Out, interpreted, dict(list), interpreted features
    """
    # if no specific request, cluster on everything
    if request is None:
      return self._getClusterableFeatures()
    # if request given, iterate through and check them
    errMsg = []
    badAlgos = []  # keep a list of flagged bad algo names
    badFeatures = collections.defaultdict(list)  # keep a record of flagged bad features (but ok algos)
    for algoName, feature in request.items():
      if algoName not in badFeatures and algoName not in factory.knownTypes():
        errMsg.append(f'Unrecognized TSA algorithm while searching for cluster features: "{algoName}"! Expected one of: {factory.knownTypes()}')
        badAlgos.append(algoName)
        continue
      algo = factory.returnClass(algoName, self)
      if not algo.canCharacterize():
        errMsg.append(f'Cannot cluster on TSA algorithm "{algoName}"!  It does not support clustering.')
        badAlgos.append(algoName)
        continue
      if feature not in algo._features:
        badFeatures[algoName].append(feature)
    if badFeatures:
      for algoName, features in badFeatures.items():
        algo = factory.returnClass(algoName, self)
        errMsg.append(f'Unrecognized clusterable features for TSA algorithm "{algoName}": {features}. Acceptable features are: {algo._features}')
    if errMsg:
      self.raiseAnError(IOError, 'The following errors occured while building clusterable features:\n' +
                        '\n  '.join(errMsg))
    return request

  def _getClusterableFeatures(self):
    """
      Provides a list of clusterable features.
      For this ROM, these are as "TSA_algorith|feature" such as "fourier|amplitude"
      @ In, None
      @ Out, features, dict(list(str)), clusterable features by algorithm
    """
    features = {}
    # check: is it possible tsaAlgorithms isn't populated by now?
    for algo in self._tsaAlgorithms:
      if algo.canCharacterize():
        features[algo.name] = algo._features
      else:
        features[algo.name] = []
    return features

  def getLocalRomClusterFeatures(self, featureTemplate, settings, request, picker=None):
    """
      Provides metrics aka features on which clustering compatibility can be measured.
      This is called on LOCAL subsegment ROMs, not on the GLOBAL template ROM
      @ In, featureTemplate, str, format for feature inclusion
      @ In, settings, dict, as per getGlobalRomSegmentSettings
      @ In, request, dict(list), requested features to cluster on (by feature set)
      @ In, picker, slice, indexer for segmenting data
      @ Out, features, dict, {target_metric: np.array(floats)} features to cluster on
    """
    features = {}
    for algo in self._tsaAlgorithms:
      if algo.name not in request or not algo.canCharacterize():
        continue
      algoReq = request[algo.name] if request is not None else None
      algoFeatures = algo.getClusteringValues(featureTemplate, algoReq, self._tsaTrainedParams[algo])
      features.update(algoFeatures)
    return features

  def setLocalRomClusterFeatures(self, settings):
    """
      Forcibly set the parameters of this ROM based on those in "settings".
      Settings will have naming conventions as from getLocalRomClusterFeatures.
      @ In, settings, dict, parameters to set
    """
    byAlgo = collections.defaultdict(list)
    for feature, values in settings.items():
      target, algoName, ident = feature.split('|', maxsplit=2)
      byAlgo[algoName].append((target, ident, values))
    for algo in self._tsaAlgorithms:
      settings = byAlgo.get(algo.name, None)
      if settings:
        params = algo.setClusteringValues(settings, self._tsaTrainedParams[algo])
        self._tsaTrainedParams[algo] = params

  def findAlgoByName(self, name):
    """
      Find the corresponding algorithm by name
      @ In, name, str, name of algorithm
      @ Out, algo, TSA.TimeSeriesAnalyzer, algorithm
    """
    for algo in self._tsaAlgorithms:
      if algo.name == name:
        return algo
    return None

  def getFundamentalFeatures(self, requestedFeatures, featureTemplate=None):
    """
      Collect the fundamental parameters for this ROM
      Used for writing XML, interpolating, clustering, etc
      NOTE: This is originally copied over from SupervisedLearning!
         Primarily using this for interpolation.
      @ In, requestedFeatures, dict(list), featureSet and features to collect (may be None)
      @ In, featureTemplate, str, optional, templated string for naming features (probably leave as None)
      @ Out, features, dict, features to cluster on with shape {target_metric: np.array(floats)}
    """
    # NOTE: this should match the clustered features template.
    if featureTemplate is None:
      featureTemplate = '{target}|{metric}|{id}' # TODO this kind of has to be the format currently

    requests = self._getClusterableFeatures()
    features = self.getLocalRomClusterFeatures(featureTemplate, {}, requests, picker=None)

    return features

  def readFundamentalFeatures(self, features):
    """
      Reads in the requested ARMA model properties from a feature dictionary
      @ In, features, dict, dictionary of fundamental features
      @ Out, readFundamentalFeatures, dict, more clear list of features for construction
    """
    return features

  def setFundamentalFeatures(self, features):
    """
      opposite of getFundamentalFeatures, expects results as from readFundamentalFeatures
      Constructs this ROM by setting fundamental features from "features"
      @ In, features, dict, dictionary of info as from readFundamentalFeatures
      @ Out, None
    """
    # NOTE: we deepcopy'd a ROM to get here... so any non-clusterable features have
    #       been copied over. For example: ARMA 'results', 'model', 'initials'
    # TODO: these attributes should be overloaded in some fashion in the future
    self.setLocalRomClusterFeatures(features)
    self.amITrained = True

  def parametrizeGlobalRomFeatures(self, featureDict):
    """
      Parametrizes the GLOBAL features of the ROM (assumes this is the templateROM and segmentation is active)
      @ In, featureDict, dict, dictionary of features to parametrize
      @ Out, params, dict, dictionary of collected parametrized features
    """
    # NOTE: only used during interpolation for global features! returning empty dict...
    params = {}
    return params

  def setGlobalRomFeatures(self, params, pivotValues):
    """
      Sets global ROM properties for a templateROM when using segmenting
      Returns settings rather than "setting" them for use in ROMCollection classes
      @ In, params, dict, dictionary of parameters to set
      @ In, pivotValues, np.array, values of time parameter
      @ Out, results, dict, global ROM feature set
    """
    # NOTE: only used during interpolation for global features! returning empty dict...
    results = {}
    return results

  ### ESSENTIALLY UNUSED ###
  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure, since we do not desire normalization in this implementation.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __confidenceLocal__(self,featureVals):
    """
      This method is currently not needed for ARMA
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      Currently not implemented for ARMA
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      there are no possible default parameters to report
    """
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    """
      override this method to pass the set of parameters of the ROM that can change during simulation
      Currently not implemented for ARMA
    """
    pass
