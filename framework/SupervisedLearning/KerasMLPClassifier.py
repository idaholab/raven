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
  module for Multi-layer perceptron classifier
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import tensorflow as tf
# test if we can reproduce th results
#from tensorflow import set_random_seed
#set_random_seed(2017)
######
import tensorflow.keras as Keras
from tensorflow.keras import models as KerasModels
from tensorflow.keras import layers as KerasLayers
from tensorflow.keras import optimizers as KerasOptimizers
from tensorflow.keras import utils as KerasUtils
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .KerasClassifier import KerasClassifier
#Internal Modules End--------------------------------------------------------------------------------

class KerasMLPClassifier(KerasClassifier):
  """
    Multi-layer perceptron classifier constructed using Keras API in TensorFlow
  """

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler, a MessageHandler object in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    KerasClassifier.__init__(self,messageHandler,**kwargs)
    self._dynamicHandling            = True                                 # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self.printTag = 'KerasMLPClassifier'
    # activation functions for all hidden layers of deep neural network
    self.hiddenLayerActivation = self.initOptionDict.pop('hidden_layer_activations', ['relu'])
    # always required, dimensionalities of hidden layers of deep neural network
    self.hiddenLayerSize = self.initOptionDict.pop('hidden_layer_sizes',[20])
    # Broadcast hidden layer activation function to all hidden layers
    if len(self.hiddenLayerActivation) == 1 and len(self.hiddenLayerActivation) < len(self.hiddenLayerSize):
      self.hiddenLayerActivation = self.hiddenLayerActivation * len(self.hiddenLayerSize)
    elif len(self.hiddenLayerActivation) != len(self.hiddenLayerSize):
      self.raiseAnError(IOError, "The number of activation functions for the hidden layer should be equal the number of hidden layers!")
    # fraction of the input units to drop, default 0
    self.dropoutRate = self.initOptionDict.pop('hidden_layer_dropouts',['0'])
    if len(self.dropoutRate) == 1 and len(self.dropoutRate) < len(self.hiddenLayerSize):
      self.dropoutRate = self.dropoutRate * len(self.hiddenLayerSize)
    elif len(self.dropoutRate) != len(self.hiddenLayerSize):
      self.raiseAnError(IOError, "The number of dropout rates should be equal the number of hidden layers!")

  def __addLayers__(self):
    """
      Method used to add layers for KERAS model
      @ In, None
      @ Out, None
    """
    # start to build the ROM
    # hidden layers
    self.ROM = KerasModels.Sequential()
    for index, layerSize in enumerate(self.hiddenLayerSize):
      activation = self.hiddenLayerActivation[index]
      rate = self.dropoutRate[index]
      if index == 0:
        self.ROM.add(KerasLayers.Dense(layerSize, activation=activation, input_shape=(len(self.features),)))
      else:
        self.ROM.add(KerasLayers.Dense(layerSize, activation=activation))
      self.ROM.add(KerasLayers.Dropout(rate))
    # output layer
    self.ROM.add(KerasLayers.Dense(self.numClasses, activation=self.outputLayerActivation))
