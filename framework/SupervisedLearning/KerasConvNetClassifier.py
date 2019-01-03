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
  module for Convolutional neural network (CNN)
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
import tensorflow.contrib.keras as Keras
from tensorflow.contrib.keras import models as KerasModels
from tensorflow.contrib.keras import layers as KerasLayers
from tensorflow.contrib.keras import optimizers as KerasOptimizers
from tensorflow.contrib.keras import utils as KerasUtils
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .KerasClassifier import KerasClassifier
#Internal Modules End--------------------------------------------------------------------------------

class KerasConvNetClassifier(KerasClassifier):
  """
    Convolutional neural network (CNN) classifier constructed using Keras API in TensorFlow
  """

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler, a MessageHandler object in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    KerasClassifier.__init__(self,messageHandler,**kwargs)
    self.printTag = 'KerasConvNetClassifier'
    self.layerLayout = self.initOptionDict.pop('layer_layout',None)
    if self.layerLayout is None:
      self.raiseAnError(IOError,"XML node 'layer_layout' is required for ROM class", self.printTag)
    elif not set(self.layerLayout).issubset(list(self.initOptionDict.keys())):
      self.raiseAnError(IOError, "The following layers are not defined '{}'.".format(', '.join(set(self.layerLayout)-set(list(self.initOptionDict.keys())))))

  def __addLayers__(self):
    """
      Method used to add layers for KERAS model
      The structure for Convolutional Neural Network is:
      inputLayer -> [(ConvolutionLayer) * Mi -> ... ->MaxPoolingLayer] * Ni -> ...
      -> Flatten -> ReLU layer/MLP layer -> OutputLayer
      @ In, None
      @ Out, None
    """
    # start to build the ROM
    self.ROM = KerasModels.Sequential()
    # loop over layers
    for index, layerName in enumerate(self.layerLayout[:-1]):
      layerDict = copy.deepcopy(self.initOptionDict[layerName])
      layerType = layerDict.pop('type')
      layerSize = layerDict.pop('dim_out',None)
      layerInstant = self.__class__.availLayer[layerType]
      dropoutRate = layerDict.pop('rate',0.25)
      if layerSize is not None:
        if index == 0:
          self.ROM.add(layerInstant(layerSize,input_shape=(len(self.features),), **layerDict))
        else:
          self.ROM.add(layerInstant(layerSize,**layerDict))
      else:
        if layerType == 'dropout':
          self.ROM.add(layerInstant(dropoutRate))
        else:
          self.ROM.add(layerInstant(**layerDict))
    #output layer
    layerName = self.layerLayout[-1]
    layerDict = self.initOptionDict.pop(layerName)
    layerType = layerDict.pop('type')
    layerInstant = self.__class__.availLayer[layerType]
    self.ROM.add(layerInstant(self.numClasses,**layerDict))
