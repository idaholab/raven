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

#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
import random as rn
import matplotlib
import platform
from scipy import stats
import os
from ..utils import importerUtils
tf = importerUtils.importModuleLazyRenamed("tf", globals(), "tensorflow")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .KerasBase import KerasBase
#Internal Modules End--------------------------------------------------------------------------------

class KerasClassifier(KerasBase):
  """
    Multi-layer perceptron classifier constructed using Keras API in TensorFlow
  """
  info = {'problemtype':'regression', 'normalize':True}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlNode{KerasClassifier}
                        """
    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a keras deep neural network object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'KerasClassifier'

  def _getFirstHiddenLayer(self, layerInstant, layerSize, layerDict):
    """
      Creates the first hidden layer
      @ In, layerInstant, class, layer type from tensorflow.python.keras.layers
      @ In, layerSize, int, nodes in layer
      @ In, layerDict, dict, layer details
      @ Out, layer, tensorflow.python.keras.layers, new layer
    """
    return layerInstant(layerSize,input_shape=self.featv.shape[1:], **layerDict)

  def _getLastLayer(self, layerInstant, layerDict):
    """
      Creates the last layer
      @ In, layerInstant, class, layer type from tensorflow.python.keras.layers
      @ In, layerSize, int, nodes in layer
      @ In, layerDict, dict, layer details
      @ Out, layer, tensorflow.python.keras.layers, new layer
    """
    return layerInstant(self.numClasses,**layerDict)

  def _getTrainingTargetValues(self, names, values):
    """
      Gets the target values to train with, which differs depending
      on if this is a regression or classifier.
      @ In, names, list of names
      @ In, values, list of values
      @ Out, targetValues, list, list of catagorized target values
    """
    # This class uses deep neural networks (DNNs) for classification.
    # Targets for Classifier deep neural network should be labels only (i.e. integers only)
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
    return targetValues


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
