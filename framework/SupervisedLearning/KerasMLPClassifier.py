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
  modules for tensorflow and keras used for deep neural network
  i.e. Multi-layer perceptron classifier
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
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import models as KerasModels
from tensorflow.contrib.keras import layers as KerasLayers
from tensorflow.contrib.keras import optimizers as KerasOptimizers
from tensorflow.contrib.keras import utils as KerasUtils
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
    self.printTag = 'KerasMLPClassifier'
    self.__initLocal__()

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
    self.ROM.add(KerasLayers.Dense(len(self.target), activation=self.outputLayerActivation))
