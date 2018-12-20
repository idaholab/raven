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
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class KerasMLPClassifier(supervisedLearning):
  """
    Multi-layer perceptron classifier constructed using Keras API in TensorFlow
  """
  # available optimizers in Keras
  ROMType = 'KerasMLPClassifier'
  availOptimizer = {}
  availOptimizer['SGD'] = KerasOptimizers.SGD
  availOptimizer['RMSprop'] = KerasOptimizers.RMSprop
  availOptimizer['Adagrad'] = KerasOptimizers.Adagrad
  availOptimizer['Adadelta'] = KerasOptimizers.Adadelta
  availOptimizer['Adam'] = KerasOptimizers.Adam
  availOptimizer['Adamax'] = KerasOptimizers.Adamax
  availOptimizer['Nadam'] = KerasOptimizers.Nadam

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler, a MessageHandler object in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    supervisedLearning.__init__(self,messageHandler,**kwargs)
    self.printTag = 'KerasMLPClassifier'

    self.__initLocal__()

  def __initLocal__(self):
    """
      Method used to add additional initialization features used by pickling
      @ In, None
      @ Out, None
    """
    self.externalNorm = True
    self.featv = None
    self.targv = None
    name = self.initOptionDict.pop('name','')
    outputLayerActivation = self.initOptionDict.pop('output_layer_activation', 'softmax')
    hiddenLayerActivation = [elem.strip() for elem in self.initOptionDict.pop('hidden_layer_activations', 'relu').split(',')]
    hiddenLayerSize = [int(elem) for elem in self.initOptionDict.pop('hidden_layer_sizes').split(',')]
    # Broadcast hidden layer activation function to all hidden layers
    if len(hiddenLayerActivation) == 1 and len(hiddenLayerActivation) < len(hiddenLayerSize):
      hiddenLayerActivation = hiddenLayerActivation * len(hiddenLayerSize)
    elif len(hiddenLayerActivation) != len(hiddenLayerSize):
      self.raiseAnError(IOError, "The number of activation functions for the hidden layer should be equal the number of hidden layers!")
    lossFunction = [elem.strip() for elem in self.initOptionDict.pop('loss','mean_squared_error').split(',')]
    metrics = [elem.strip() for elem in self.initOptionDict.pop('metrics','accuracy').split(',')]
    self.batchSize = int(self.initOptionDict.pop('batch_size',20))
    self.epochs = int(self.initOptionDict.pop('epochs', 20))
    dropoutRate = [float(elem) for elem in self.initOptionDict.pop('dropout','0').split(',')]
    if len(dropoutRate) == 1 and len(dropoutRate) < len(hiddenLayerSize):
      dropoutRate = dropoutRate * len(hiddenLayerSize)
    elif len(dropoutRate) != len(hiddenLayerSize):
      self.raiseAnError(IOError, "The number of dropout rates should be equal the number of hidden layers!")
    self.ROM = KerasModels.Sequential()
    # hidden layers
    for index, layerSize in enumerate(hiddenLayerSize):
      activation = hiddenLayerActivation[index]
      rate = dropoutRate[index]
      if index == 0:
        self.ROM.add(KerasLayers.Dense(layerSize, activation=activation, input_shape=(len(self.features),)))
      else:
        self.ROM.add(KerasLayers.Dense(layerSize, activation=activation))
      self.ROM.add(KerasLayers.Dropout(rate))
    # output layer
    self.ROM.add(KerasLayers.Dense(len(self.target), activation=outputLayerActivation))
    # extract settings for optimizer
    optimizerSetting = self.initOptionDict.pop('optimizerSetting', {'optimizer':'adam'})
    optimizerName = optimizerSetting.pop('optimizer')

    for key,value in optimizerSetting.items():
      try:
        optimizerSetting[key] = ast.literal_eval(value)
      except:
        pass
    # set up optimizer
    optimizer = self.__class__.availOptimizer[optimizerName](**optimizerSetting)
    # compile model
    self.ROM.compile(loss=lossFunction, optimizer=optimizer, metrics=metrics)
    self.ROM._make_predict_function() # have to initialize before threading
    # This is needed to solve the thread issue in self.ROM.predict()
    # https://github.com/fchollet/keras/issues/2397#issuecomment-306687500
    self.graph = tf.get_default_graph()

  def __getstate__(self):
    """
      Overwrite state for pickling
      @ In, None
      @ Out, state, dict, namespace dictionary
    """
    # capture what is normally pickled
    state = self.__dict__.copy()

    return state

  def __setstate__(self, newState):
    """
      Initialize the ROM with the data contained in newState
      @ In, newState, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(newState)

    #only train if the original copy was trained
    if self.amITrained:
      self.__trainLocal__(self.featv,self.targv)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    self.featv = featureVals
    self.targv = targetVals
    self.ROM.fit(featureVals, targetVals, epochs=self.epochs, batch_size=self.batchSize)

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError,'KerasMLPClassifier   : __confidenceLocal__ method must be implemented!')

  def __evaluateLocal__(self,featureVals):
    """
      Perform regression on samples in featureVals.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, numpy.array 2-D, features
      @ Out, prediction, dict, predicted values
    """
    prediction = {}
    with self.graph.as_default():
      outcome = self.ROM.predict(featureVals)
    for index, target in enumerate(self.target):
      prediction[target] = outcome[:,index]
    return prediction

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.__initLocal__()

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = {}
    return params

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    self.raiseAnError(NotImplementedError,'KerasMLPClassifier   : __returnCurrentSettingLocal__ method must be implemented!')

  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, list(float), unused
      @ In, names, list(string), unused
      @ In, feat, string, feature to (not) normalize
      @ Out, None
    """
    if not self.externalNorm:
      self.muAndSigmaFeatures[feat] = (0.0,1.0)
    else:
      super(KerasMLPClassifier, self)._localNormalizeData(values,names,feat)
