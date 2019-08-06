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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------
#External Modules------------------------------------------------------------------------------------
import numpy as np
######
#Internal Modules------------------------------------------------------------------------------------
from .KerasClassifier import isTensorflowAvailable
if isTensorflowAvailable():
  from .KerasClassifier import KerasClassifier
#Internal Modules End--------------------------------------------------------------------------------

if isTensorflowAvailable():
  class KerasLSTMClassifier(KerasClassifier):
    """
      recurrent neural network using short-term model network (LSTM) classifier
      constructed using Keras API in TensorFlow
    """

    def __init__(self,messageHandler,**kwargs):
      """
        A constructor that will appropriately intialize a supervised learning object
        @ In, messageHandler, MessageHandler, a MessageHandler object in charge of raising errors, and printing messages
        @ In, kwargs, dict, an arbitrary dictionary of keywords and values
        @ Out, None
      """
      KerasClassifier.__init__(self,messageHandler,**kwargs)
      self.printTag = 'KerasLSTMClassifier'
      self.allowedLayers = self.basicLayers + self.kerasROMDict['kerasRcurrentLayersList']

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
      shape = featureVals.shape
      if len(shape) == 2:
        featureVals = np.reshape(featureVals,(1, shape[0], shape[1]))
      return featureVals
