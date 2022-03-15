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
  Created on Apr. 13, 2021
  @author: cogljj, wangc
  base class for tensorflow and keras regression used for deep neural network
  i.e. Multi-layer perceptron regression, CNN, LSTM
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

class KerasRegression(KerasBase):
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
    specs.description = r"""The \xmlNode{KerasRegression}
                        """
    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a keras deep neural network object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'KerasRegression'

  def _getFirstHiddenLayer(self, layerInstant, layerSize, layerDict):
    """
      Creates the first hidden layer
      @ In, layerInstant, class, layer type from tensorflow.python.keras.layers
      @ In, layerSize, int, nodes in layer
      @ In, layerDict, dict, layer details
      @ Out, layer, tensorflow.python.keras.layers, new layer
    """
    return layerInstant(layerSize,input_shape=[self.featv.shape[-1]], **layerDict)

  def _getLastLayer(self, layerInstant, layerDict):
    """
      Creates the last layer
      @ In, layerInstant, class, layer type from tensorflow.python.keras.layers
      @ In, layerSize, int, nodes in layer
      @ In, layerDict, dict, layer details
      @ Out, layer, tensorflow.python.keras.layers, new layer
    """
    return layerInstant(self.targv.shape[-1],**layerDict)

  def _getTrainingTargetValues(self, names, values):
    """
      Gets the target values to train with, which differs depending
      on if this is a regression or classifier.
      @ In, names, list of names
      @ In, values, list of values
      @ Out, targetValues, numpy.ndarray of shape (numSamples, numTimesteps, numFeatures) or shape (numSamples, numFeatures)
    """
    # Features must be 3d i.e. [numSamples, numTimeSteps, numFeatures]

    for target in self.target:
      if target not in names:
        self.raiseAnError(IOError,'The target '+target+' is not in the training set')

    firstTarget = values[names.index(self.target[0])]
    if type(firstTarget) == type(np.array(1)) and len(firstTarget.shape) == 1:
      targetValues = np.zeros((len(firstTarget), len(self.target)))
      for i, target in enumerate(self.target):
        self._localNormalizeData(values, names, target)
        targetValues[:, i] = self._scaleToNormal(values[names.index(target)], target)
    else:
      targetValues = np.zeros((len(firstTarget), len(firstTarget[0]),
                               len(self.target)))
      for i, target in enumerate(self.target):
        self._localNormalizeData(values, names, target)
        targetValues[:, :, i] = self._scaleToNormal(values[names.index(target)], target)
    return targetValues


  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals,numpy.array, 2-D or 3-D numpy array, [n_samples,n_features]
        or shape=[numSamples, numTimeSteps, numFeatures]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError,'KerasRegression   : __confidenceLocal__ method must be implemented!')

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
        fval = self._scaleToNormal(fval, feat)
        featureValues.append(fval)
      else:
        self.raiseAnError(IOError,'The feature ',feat,' is not in the training set')
    featureValues = np.stack(featureValues, axis=-1)

    result = self.__evaluateLocal__(featureValues)
    pivotParameter = self.pivotID
    if pivotParameter not in edict:
      pass #don't need to do anything
    elif type(edict[pivotParameter]) == type([]):
      #XXX this should not be needed since sampler should just provide the numpy array.
      #Currently the CustomSampler provides all the pivot parameter values instead of the current one.
      self.raiseAWarning("Adjusting pivotParameter because incorrect type provided")
      result[pivotParameter] = edict[pivotParameter][0]
    else:
      result[pivotParameter] = edict[pivotParameter]
    return result


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
    for i, target in enumerate(self.target):
      if len(outcome.shape) == 3:
        prediction[target] = self._invertScaleToNormal(outcome[0, :, i], target)
      else:
        prediction[target] = self._invertScaleToNormal(outcome[0, i], target)
    return prediction
