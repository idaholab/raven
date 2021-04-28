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
  Created on Dec. 20, 2018
  @author: wangc
  base class for tensorflow and keras used for deep neural network
  i.e. Multi-layer perceptron classifier, CNN, LSTM
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
import random as rn
import matplotlib
import platform
from scipy import stats
import os
import utils.importerUtils
tf = utils.importerUtils.importModuleLazyRenamed("tf", globals(), "tensorflow")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SupervisedLearning import supervisedLearning
from .KerasBase import KerasBase
#Internal Modules End--------------------------------------------------------------------------------

class KerasClassifier(KerasBase):
  """
    Multi-layer perceptron classifier constructed using Keras API in TensorFlow
  """
  ROMType = 'KerasClassifier'

  def __init__(self, **kwargs):
    """
      A constructor that will appropriately intialize a keras deep neural network object
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    super().__init__(**kwargs)

  def readInitDict(self, initDict):
    """
      Reads in the initialization dict to initialize this instance
      @ In, initDict, dict, keywords passed to constructor
      @ Out, None
    """
    super().readInitDict(initDict)
    self.printTag = 'KerasClassifier'

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    state = supervisedLearning.__getstate__(self)
    tf.keras.models.save_model(self._ROM, KerasClassifier.tempModelFile)
    # another method to save the TensorFlow model
    # self._ROM.save(KerasClassifier.tempModelFile)
    with open(KerasClassifier.tempModelFile, "rb") as f:
      serialModelData = f.read()
    state[KerasClassifier.modelAttr] = serialModelData
    os.remove(KerasClassifier.tempModelFile)
    del state["_ROM"]
    state['initOptionDict'].pop('paramInput',None)
    return state

  def __setstate__(self, d):
    """
      Initialize the ROM with the data contained in newstate
      @ In, d, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    with open(KerasClassifier.tempModelFile, "wb") as f:
      f.write(d[KerasClassifier.modelAttr])
    del d[KerasClassifier.modelAttr]
    tf.keras.backend.set_session(self._session)
    self._ROM = tf.keras.models.load_model(KerasClassifier.tempModelFile)
    os.remove(KerasClassifier.tempModelFile)
    self.__dict__.update(d)

  def _checkLayers(self):
    """
      Method used to check layers setups for KERAS model
      @ In, None
      @ Out, None
    """
    pass

  def _addHiddenLayers(self):
    """
      Method used to add hidden layers for KERAS model
      @ In, None
      @ Out, None
    """
    # start to build the ROM
    self._ROM = tf.keras.models.Sequential()
    # loop over layers
    for index, layerName in enumerate(self.layerLayout[:-1]):
      layerDict = copy.deepcopy(self.initOptionDict[layerName])
      layerType = layerDict.pop('type').lower()
      if layerType not in self.allowedLayers:
        self.raiseAnError(IOError,'Layers',layerName,'with type',layerType,'is not allowed in',self.printTag)
      layerSize = layerDict.pop('dim_out',None)
      layerInstant = self.__class__.availLayer[layerType]
      dropoutRate = layerDict.pop('rate',0.0)
      if layerSize is not None:
        if index == 0:
          self._ROM.add(layerInstant(layerSize,input_shape=self.featv.shape[1:], **layerDict))
        else:
          self._ROM.add(layerInstant(layerSize,**layerDict))
      else:
        if layerType == 'dropout':
          self._ROM.add(layerInstant(dropoutRate))
        else:
          self._ROM.add(layerInstant(**layerDict))

  def _addOutputLayers(self):
    """
      Method used to add last output layers for KERAS model
      @ In, None
      @ Out, None
    """
    layerName = self.layerLayout[-1]
    layerDict = self.initOptionDict.pop(layerName)
    layerType = layerDict.pop('type').lower()
    layerSize = layerDict.pop('dim_out',None)
    if layerSize is not None and layerSize != self.numClasses:
      self.raiseAWarning('The "dim_out" of last output layer: ', layerName, 'will be resetted to values provided in "num_classes", i.e.', self.numClasses)
    if layerType not in ['dense']:
      self.raiseAnError(IOError,'The last layer should always be Dense layer, but',layerType,'is provided!')
    layerInstant = self.__class__.availLayer[layerType]
    self._ROM.add(layerInstant(self.numClasses,**layerDict))

  def train(self,tdict):
    """
      Method to perform the training of the deep neural network algorithm
      NB.the KerasClassifier object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, dict, training dictionary
      @ Out, None
    """
    if type(tdict) != dict:
      self.raiseAnError(TypeError,'In method "train", the training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = zip(*tdict.items())
    # Currently, deep neural networks (DNNs) are only used for classification.
    # Targets for deep neural network should be labels only (i.e. integers only)
    # For both static  and time-dependent case, the targetValues are 2D arrays, i.e. [numSamples, numTargets]
    # For time-dependent case, the time-dependency is removed from the targetValues
    # Features can be 2D array, i.e. [numSamples, numFeatures], or 3D array,
    # i.e. [numSamples, numTimeSteps, numFeatures]
    # TODO: currently we only accept single target, we may extend to multi-targets by looping over targets
    # Another options is to use Keras Function APIs to directly build multi-targets models, two examples:
    # https://keras.io/models/model/
    # https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
    if len(self.target) > 1:
      self.raiseAnError(IOError, "Only single target is permitted by", self.printTag)
    if self.target[0] not in names:
      self.raiseAnError(IOError,'The target', self.target[0], 'is not in the training set')
    tval = np.asarray(values[names.index(self.target[0])])
    # FIXME: a better method may need to be added to process the labels
    # retrieve the most often used label
    targetValues = stats.mode(tval,axis=len(tval.shape)-1)[0] if len(tval.shape) > 1 else tval
    #targetValues = list(val[-1] if len(tval.shape) > 1 else val for val in tval)

    # We need to 'one-hot-encode' our target variable if multi-classes are requested
    # This means that a column will be created for each output category and a binary variable is inputted for
    # each category.
    if self.numClasses > 1 and 'categorical_crossentropy' in self.lossFunction:
      # Transform the labels (i.e. numerical or non-numerical) to normalized numerical labels
      targetValues = self.labelEncoder.fit_transform(targetValues.ravel())
      targetValues = tf.keras.utils.to_categorical(targetValues)
      if self.numClasses != targetValues.shape[-1]:
        self.raiseAWarning('The num_classes:',self.numClasses, 'specified by the user is not equal number of classes',
                           targetValues.shape[-1], ' in the provided data!')
        self.raiseAWarning('Reset the num_classes to be consistent with the data!')
        self.numClasses = targetValues.shape[-1]

    featureValues = []
    featureValuesShape = None
    for feat in self.features:
      if feat in names:
        fval = values[names.index(feat)]
        resp = self.checkArrayConsistency(fval, self.isDynamic())
        if not resp[0]:
          self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        fval = np.asarray(fval)
        if featureValuesShape is None:
          featureValuesShape = fval.shape
        if featureValuesShape != fval.shape:
          self.raiseAnError(IOError,'In training set, the number of values provided for feature '+feat+' are not consistent to other features!')
        self._localNormalizeData(values,names,feat)
        fval = (fval - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
        featureValues.append(fval)
      else:
        self.raiseAnError(IOError,'The feature ',feat,' is not in the training set')

    #FIXME: when we do not support anymore numpy <1.10, remove this IF STATEMENT
    if int(np.__version__.split('.')[1]) >= 10:
      featureValues = np.stack(featureValues, axis=-1)
    else:
      sl = (slice(None),) * np.asarray(featureValues[0]).ndim + (np.newaxis,)
      featureValues = np.concatenate([np.asarray(arr)[sl] for arr in featureValues], axis=np.asarray(featureValues[0]).ndim)

    self.__trainLocal__(featureValues,targetValues)
    self.amITrained = True

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature or shape=[numSamples, numTimeSteps, numFeatures]
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    self.featv = featureVals
    self.targv = targetVals
    # check layers
    self._checkLayers()
    # hidden layers
    self._addHiddenLayers()
    #output layer
    self._addOutputLayers()
    self._ROM.compile(loss=self.lossFunction, optimizer=self.optimizer, metrics=self.metrics)
    self._ROM._make_predict_function() # have to initialize before threading
    self._romHistory = self._ROM.fit(featureVals, targetVals, epochs=self.epochs, batch_size=self.batchSize, validation_split=self.validationSplit)
    # The following requires pydot-ng and graphviz to be installed (See the manual)
    # https://github.com/keras-team/keras/issues/3210
    if self.plotModel:
      tf.keras.utils.plot_model(self._ROM,to_file=self.plotModelFilename,show_shapes=True)

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals,numpy.array, 2-D or 3-D numpy array, [n_samples,n_features]
        or shape=[numSamples, numTimeSteps, numFeatures]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError,'KerasClassifier   : __confidenceLocal__ method must be implemented!')

  def __evaluateLocal__(self,featureVals):
    """
      Perform regression on samples in featureVals.
      classification labels will be returned based on num_classes
      @ In, featureVals, numpy.array, 2-D for static case and 3D for time-dependent case, values of features
      @ Out, prediction, dict, predicted values
    """
    featureVals = self._preprocessInputs(featureVals)
    prediction = {}
    with self.graph.as_default():
      tf.keras.backend.set_session(self._session)
      outcome = self._ROM.predict(featureVals)
    if self.numClasses > 1 and self.lossFunction in ['categorical_crossentropy']:
      outcome = np.argmax(outcome,axis=1)
      # Transform labels back to original encoding
      outcome = self.labelEncoder.inverse_transform(outcome)
      # TODO, extend to multi-targets, currently we only accept one target
      prediction[self.target[0]] = outcome
    else:
      prediction[self.target[0]] = [round(val[0]) for val in outcome]
    return prediction

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self._initGraph()
    self._ROM = None
    self.featv = None
    self.targv = None

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = self.initDict
    return params

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      The model can be reinstantiated from its config via:
      config = model.get_config()
      self._ROM = tf.keras.models.Sequential.from_config(config)
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    params = self._ROM.get_config()
    return params

