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
  module for recurrent neural network using short-term model network (LSTM)
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
######
#Internal Modules------------------------------------------------------------------------------------
from .KerasClassifier import KerasClassifier
#Internal Modules End--------------------------------------------------------------------------------

class KerasLSTMClassifier(KerasClassifier):
  """
    recurrent neural network using short-term model network (LSTM) classifier
    constructed using Keras API in TensorFlow
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
    specs.description = r"""Long Short Term Memory networks (LSTM) are a special kind of recurrent neural network, capable
        of learning long-term dependencies. They work tremendously well on a large variety of problems, and
        are now widely used. LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering
        information for long periods of time is practically their default behavior, not something that they
        struggle to learn.
        \zNormalizationPerformed{KerasLSTMClassifier}
        """
    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'KerasLSTMClassifier'
    self.allowedLayers = self.basicLayers + self.kerasDict['kerasRcurrentLayersList']

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  def _checkLayers(self):
    """
      Method used to check layers setups for KERAS model
      @ In, None
      @ Out, None
    """
    for index, layerName in enumerate(self.layerLayout[:-1]):
      layerType = self.initOptionDict[layerName].get('type').lower()
      nextLayerName = self.layerLayout[index+1]
      nextLayerType = self.initOptionDict[nextLayerName].get('type').lower()
      if layerType in ['lstm'] and nextLayerType in ['lstm']:
        if not self.initOptionDict[layerName].get('return_sequences'):
          self.initOptionDict[layerName]['return_sequences'] = True
          self.raiseAWarning('return_sequences is resetted to True for layer',layerName)

  def _preprocessInputs(self,featureVals):
    """
      Perform input feature values before sending to ROM prediction
      @ In, featureVals, numpy.array, i.e. [shapeFeatureValue,numFeatures], values of features
      @ Out, featureVals, numpy.array, predicted values
    """
    #NOTE This is the same as the _preprocessInputs in KerasLSTMRegression
    shape = featureVals.shape
    if len(shape) == 2:
      featureVals = np.reshape(featureVals,(1, shape[0], shape[1]))
    return featureVals
