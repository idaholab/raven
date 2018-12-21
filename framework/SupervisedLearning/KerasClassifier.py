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
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
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
import matplotlib.pyplot as plt
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class KerasClassifier(supervisedLearning):
  """
    Multi-layer perceptron classifier constructed using Keras API in TensorFlow
  """
  # available optimizers in Keras
  ROMType = 'KerasClassifier'
  # An optimizer is required for compiling a Keras model
  availOptimizer = {}
  # stochastic gradient descent optimizer, includes support for momentum,learning rate decay, and Nesterov momentum
  availOptimizer['SGD'] = KerasOptimizers.SGD
  # RMSprop optimizer, usually a good choice for recurrent neural network
  availOptimizer['RMSprop'] = KerasOptimizers.RMSprop
  # Adagrad is an optimzer with parameter-specific learning rates, which are adapted relative to
  # how frequently a parameter gets updated during training. The more updates  a parameter receives,
  # the smaller the updates.
  availOptimizer['Adagrad'] = KerasOptimizers.Adagrad
  # Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving
  # window of gradient updates, instead of accumulating all past gradients. This way, Adadelta
  # continues learning even when many updates have been done.
  availOptimizer['Adadelta'] = KerasOptimizers.Adadelta
  # Adam optimzer
  availOptimizer['Adam'] = KerasOptimizers.Adam
  # Adamax optimizer from Adam paper's section 7
  availOptimizer['Adamax'] = KerasOptimizers.Adamax
  # Nesterov Adam optimizer
  availOptimizer['Nadam'] = KerasOptimizers.Nadam

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a keras deep neural network object
      @ In, messageHandler, MessageHandler, a MessageHandler object in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    supervisedLearning.__init__(self,messageHandler,**kwargs)
    # parameter dictionary at the initial stage
    self.initDict = copy.deepcopy(self.initOptionDict)
    self.printTag = 'KerasClassifier'
    # perform z-score normalization if True
    self.externalNorm = True
    # variale to store feature values, shape=[n_samples, n_features]
    self.featv = None
    # variable to store target values, shape = [n_samples]
    self.targv = None
    # instance of KERAS deep neural network model
    self.ROM = None
    modelName = self.initOptionDict.pop('name','')
    # number of classes for classifier
    self.numClasses = self.initOptionDict.pop('num_classes',2)
    # validation split, default to 0.25
    self.validationSplit = self.initOptionDict.pop('validation_split',0.25)
    # options to plot deep neural network model, default False
    self.plotModel = self.initOptionDict.pop('plot_model',False)
    self.plotModelFilename = self.printTag + "_model.png" if not modelName else modelName + "_model.png"
    # activation function for output layer of deep neural network
    self.outputLayerActivation = self.initOptionDict.pop('output_layer_activation', 'softmax')
    # A loss function that is always required to compile a KERAS model
    self.lossFunction = [elem.strip() for elem in self.initOptionDict.pop('loss','mean_squared_error').split(',')]
    # a metric is a function that is used to judge the performance of KERAS model
    self.metrics = [elem.strip() for elem in self.initOptionDict.pop('metrics','accuracy').split(',')]
    # number of samples per gradient update, default 20
    self.batchSize = int(self.initOptionDict.pop('batch_size',20))
    # number of epochs to train the model. An epoch is an iteration over the entire training data, (default 20)
    self.epochs = int(self.initOptionDict.pop('epochs', 20))
    # extract settings for optimizer
    optimizerSetting = self.initOptionDict.pop('optimizerSetting', {'optimizer':'adam'})
    optimizerName = optimizerSetting.pop('optimizer')

    for key,value in optimizerSetting.items():
      try:
        optimizerSetting[key] = ast.literal_eval(value)
      except:
        pass
    # set up optimizer
    self.optimizer = self.__class__.availOptimizer[optimizerName](**optimizerSetting)
    self.__initLocal__()

  def __initLocal__(self):
    """
      Method used to add additional initialization features
      Such as complile KERAS model
      @ In, None
      @ Out, None
    """
    # This is needed to solve the thread issue in self.ROM.predict()
    # https://github.com/fchollet/keras/issues/2397#issuecomment-306687500
    self.graph = tf.get_default_graph()

  def __addLayers__(self):
    """
      Method used to add layers for KERAS model
      @ In, None
      @ Out, None
    """
    pass

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
    self.__addLayers__()
    self.ROM.compile(loss=self.lossFunction, optimizer=self.optimizer, metrics=self.metrics)
    self.ROM._make_predict_function() # have to initialize before threading
    history = self.ROM.fit(featureVals, targetVals, epochs=self.epochs, batch_size=self.batchSize, validation_split=self.validationSplit)
    # The following requires pydot-ng and graphviz to be installed
    # https://github.com/keras-team/keras/issues/3210
    if self.plotModel:
      KerasUtils.plot_model(self.ROM,to_file=self.plotModelFilename,show_shapes=True)
      self.__plotHistory__(history)

  def __plotHistory__(self, history):
    """
      Plot training & validation accuracy and loss values
      @ In, history, History object of Keras
      @ Out, None
    """
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError,'KerasClassifier   : __confidenceLocal__ method must be implemented!')

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
    self.ROM = None
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
      self.ROM = KerasModels.Sequential.from_config(config)
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    params = self.ROM.get_config()
    return params

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
      super(KerasClassifier, self)._localNormalizeData(values,names,feat)
