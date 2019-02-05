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
import copy
import tensorflow as tf
# test if we can reproduce th results
#from tensorflow import set_random_seed
#set_random_seed(2017)
#numpy.random.seed(2017)
######
import tensorflow.keras as keras
from tensorflow.keras import models as KerasModels
from tensorflow.keras import layers as KerasLayers
from tensorflow.keras import optimizers as KerasOptimizers
from tensorflow.keras import utils as KerasUtils
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .KerasClassifier import KerasClassifier
#Internal Modules End--------------------------------------------------------------------------------

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
    self.allowedLayers = self.basicLayers + self.__class__.kerasRcurrentLayersList

  def __addHiddenLayers__(self):
    """
      Method used to add layers for KERAS model
      @ In, None
      @ Out, None
    """
    # start to build the ROM
    self.ROM = KerasModels.Sequential()
    # loop over layers
    for index, layerName in enumerate(self.layerLayout[:-1]):
      layerDict = copy.deepcopy(self.initOptionDict[layerName])
      layerType = layerDict.pop('type').lower()
      if layerType not in self.allowedLayers:
        self.raiseAnError(IOError,'Layers',layerName,'with type',layerType,'is not allowed in',self.printTag)
      layerSize = layerDict.pop('dim_out',None)
      layerInstant = self.__class__.availLayer[layerType]
      nextLayerName = self.layerLayout[index+1]
      nextLayerType = self.initOptionDict[nextLayerName].get('type').lower()
      if layerType in ['lstm'] and nextLayerType in ['lstm']:
        if not layerDict.get('return_sequences'):
          layerDict['return_sequences'] = True
          self.raiseAWarning('return_sequences is resetted to True for layer',layerName)
      dropoutRate = layerDict.pop('rate',0)
      if layerSize is not None:
        if index == 0:
          self.ROM.add(layerInstant(layerSize,input_shape=self.featv.shape[1:], **layerDict))
        else:
          self.ROM.add(layerInstant(layerSize,**layerDict))
      else:
        if layerType == 'dropout':
          self.ROM.add(layerInstant(dropoutRate))
        else:
          self.ROM.add(layerInstant(**layerDict))
