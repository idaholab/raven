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
  Created on Jan 21, 2020

  @author: alfoa, wangc
  Support Vector Regression

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
from ...utils.importerUtils import importModuleLazy
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
np = importModuleLazy("numpy")
import ast
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..SupervisedLearning import SupervisedLearning
from ...utils import utils
#Internal Modules End--------------------------------------------------------------------------------

class ScikitLearnBase(SupervisedLearning):
  """
    Base Class for Scikitlearn-based surrogate models (classifiers and regressors)
  """
  info = {'problemtype':None, 'normalize':None}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    #Subclasses can define these to specify what is exported with writeXML
    self._vectorWriteList = []
    self._scalarWriteList = []

    self.uniqueVals = None # flag to indicate targets only have a single unique value
    self.settings = None # initial settings for the ROM
    self.model = None # Scikitlearn estimator/model
    self.multioutputWrapper = True # If True, use MultiOutputRegressor or MultiOutputClassifier to wrap self.model else
                                   # the self.model can handle multioutput/multi-targets prediction

  def updateSettings(self, settings):
    """
      Update the parameters of the self.model if the model is wrapper by sklearn.multioutput class
      @ In, settings, dict, dictionary of user-defined settings for the model
      @ Out, out, dict, the updated dictionary based on user-defined settings
    """
    out = dict()
    if self.multioutputWrapper:
      params = self.model.get_params()
      for key, val in params.items():
        out[key] = settings[key] if key in settings else val
        if '__' in key:
          deepKey = key.split('__')[-1]
          out[key] = settings[deepKey] if deepKey in settings else val
    else:
      out = settings
    return out

  def initializeModel(self, settings):
    """
      Method to initialize the surrogate model with a settings dictionary
      @ In, settings, dict, the dictionary containin the parameters/settings to instanciate the model
      @ Out, None
    """
    if self.settings is None:
      self.settings = settings
    self.model = self.model(**settings)
    if self.multioutputWrapper:
      self.multioutput(self.info['problemtype'])

  def multioutput(self, type='regression'):
    """
      Method to extend ScikitLearn ROM that do not natively support multi-target regression/classification
      @ In, type, str, either regression or classification
      @ Out, None
    """
    import sklearn.multioutput
    if type == 'regression':
      self.model = sklearn.multioutput.MultiOutputRegressor(self.model)
    elif type == 'classification':
      self.model = sklearn.multioutput.MultiOutputClassifier(self.model)
    else:
      self.raiseAnError(IOError, 'The "type" param for function "multioutput" should be either "regression" or "classification"! but got',
                        type)

  def setEstimator(self, estimatorList):
    """
      Initialization method
      @ In, estimatorList, list of ROM instances/estimators used by ROM
      @ Out, None
    """
    for estimator in estimatorList:
      interfaceRom = estimator._interfaceROM
      if not isinstance(interfaceRom, ScikitLearnBase):
        self.raiseAnError(IOError, 'ROM', estimator.name, 'can not be used as estimator for ROM', self.name)
      if not callable(getattr(interfaceRom.model, "fit", None)):
        self.raiseAnError(IOError, 'estimator:', estimator.name, 'can not be used! Please change to a different estimator')
      else:
        self.raiseADebug('A valid estimator', estimator.name, 'is provided!')

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples,n_targets], an array of output target
        associated with the corresponding points in featureVals
    """
    # check if all targets only have a single unique value, just store that value, no need to fit/train
    if all([len(np.unique(targetVals[:,index])) == 1 for index in range(targetVals.shape[1])]):
      self.uniqueVals = [np.unique(targetVals[:,index])[0] for index in range(targetVals.shape[1]) ]
    else:
      # the multi-target is handled by the internal wrapper
      self.uniqueVals = None
      self.model.fit(featureVals,targetVals)

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidenceDict, dict, dict of the dictionary for each target
    """
    confidenceDict = {}
    if callable(getattr(self.model, "predict_proba", None)):
      outcomes = self.model.predict_proba(featureVals)
      confidenceDict = {key:value for (key,value) in zip(self.target,outcomes)}
    else:
      self.raiseAnError(IOError,'the ROM '+str(self.name)+'has not the an method to evaluate the confidence of the prediction')
    return confidenceDict

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, np.array, list of values at which to evaluate the ROM
      @ Out, returnDict, dict, dict of all the target results
    """
    if self.uniqueVals is not None:
      outcomes =  self.uniqueVals
    else:
      outcomes = self.model.predict(featureVals)
    outcomes = np.atleast_1d(outcomes)
    if len(outcomes.shape) == 1:
      returnDict = {key:value for (key,value) in zip(self.target,outcomes)}
    else:
      returnDict = {key: outcomes[:, i] for i, key in enumerate(self.target)}
    return returnDict

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    settings = self.updateSettings(self.settings)
    self.model.set_params(**settings)

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = self.settings
    return params

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    pass

  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, list(float), unused
      @ In, names, list(string), unused
      @ In, feat, string, feature to (not) normalize
      @ Out, None
    """
    if not self.info['normalize']:
      self.muAndSigmaFeatures[feat] = (0.0,1.0)
    else:
      super()._localNormalizeData(values,names,feat)

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows writing out ROM information.
      For any SciKitLearn that wants to use this, it should create
      lists _vectorWriteList and _scalarWriteList for which data should be
      written.  (See LinearModel/LinearRegression.py for example)
      @ In, writeTo, xmlUtils.StaticXmlElement, StaticXmlElement to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    if self.multioutputWrapper:
      for index, targetName in enumerate(self.target):
        for vectorName in self._vectorWriteList:
          writeTo.addVector("ROM", vectorName+targetName, ",".join([str(x) for x in getattr(self.model.estimators_[index], vectorName)]))
        for scalarName in self._scalarWriteList:
          writeTo.addScalar("ROM", scalarName+targetName, str(getattr(self.model.estimators_[index], scalarName)))
    else:
      for vectorName in self._vectorWriteList:
        writeTo.addVector("ROM", vectorName, ",".join([str(x) for x in getattr(self.model, vectorName)]))
      for scalarName in self._scalarWriteList:
        writeTo.addScalar("ROM", scalarName, str(getattr(self.model, scalarName)))

