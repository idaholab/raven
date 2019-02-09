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
import abc
import numpy as np
import random as rn
import tensorflow as tf
from tensorflow import set_random_seed

######
import tensorflow.keras as Keras
from tensorflow.keras import models as KerasModels
from tensorflow.keras import layers as KerasLayers
from tensorflow.keras import optimizers as KerasOptimizers
from tensorflow.keras import utils as KerasUtils
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

# This is needed when using conda to build tensorflow 1.12 with python 2.7
# Check issue: https://github.com/tensorflow/tensorflow/issues/23999
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class KerasClassifier(supervisedLearning):
  """
    Multi-layer perceptron classifier constructed using Keras API in TensorFlow
  """
  # some modules are commented out since they are not available in TensorFlow 1.1
  # We need to install TensorFlow in a different way
  # available optimizers in Keras
  ROMType = 'KerasClassifier'
  # An optimizer is required for compiling a Keras model
  availOptimizer = {}
  # stochastic gradient descent optimizer, includes support for momentum,learning rate decay, and Nesterov momentum
  availOptimizer['sgd'] = KerasOptimizers.SGD
  # RMSprop optimizer, usually a good choice for recurrent neural network
  availOptimizer['rmsprop'] = KerasOptimizers.RMSprop
  # Adagrad is an optimzer with parameter-specific learning rates, which are adapted relative to
  # how frequently a parameter gets updated during training. The more updates  a parameter receives,
  # the smaller the updates.
  availOptimizer['adagrad'] = KerasOptimizers.Adagrad
  # Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving
  # window of gradient updates, instead of accumulating all past gradients. This way, Adadelta
  # continues learning even when many updates have been done.
  availOptimizer['adadelta'] = KerasOptimizers.Adadelta
  # Adam optimzer
  availOptimizer['adam'] = KerasOptimizers.Adam
  # Adamax optimizer from Adam paper's section 7
  availOptimizer['adamax'] = KerasOptimizers.Adamax
  # Nesterov Adam optimizer
  availOptimizer['nadam'] = KerasOptimizers.Nadam

  # available convolutional layers
  availLayer = {}
  # dense layer
  availLayer['dense'] = KerasLayers.Dense
  # apply dropout to the input
  availLayer['dropout'] = KerasLayers.Dropout
  # Flatten layer
  availLayer['flatten'] = KerasLayers.Flatten
  # 1D convolution layer (e.g. temporal convolution).
  availLayer['conv1d'] = KerasLayers.Conv1D
  # 2D convolution layer (e.g. spatial convolution over images).
  availLayer['conv2d'] = KerasLayers.Conv2D
  # Depthwise separable 1D convolution.
  #availConvNet['separableconv1d'] = KerasLayers.SeparableConv1D
  # Depthwise separable 2D convolution.
  availLayer['separableconv2d'] = KerasLayers.SeparableConv2D
  # Depthwise separable 2D convolution.
  #availConvNet['depthwiseconv2d'] = KerasLayers.DepthwiseConv2D
  # Transposed convolution layer (sometimes called Deconvolution).
  availLayer['conv2dtranspose'] = KerasLayers.Conv2DTranspose
  # 3D convolution layer (e.g. spatial convolution over volumes).
  availLayer['conv3d'] = KerasLayers.Conv3D
  # ransposed convolution layer (sometimes called Deconvolution).
  #availConvNet['conv3dtranspose'] = KerasLayers.Conv3DTranspose
  # Cropping layer for 1D input (e.g. temporal sequence). It crops along the time dimension (axis 1).
  availLayer['cropping1d'] = KerasLayers.Cropping1D
  # Cropping layer for 2D input (e.g. picture). It crops along spatial dimensions, i.e. height and width.
  availLayer['cropping2d'] = KerasLayers.Cropping2D
  # Cropping layer for 3D data (e.g. spatial or spatio-temporal).
  availLayer['cropping3d'] = KerasLayers.Cropping3D
  # Upsampling layer for 1D inputs
  availLayer['upsampling1d'] = KerasLayers.UpSampling1D
  # Upsampling layer for 2D inputs.
  availLayer['upsampling2d'] = KerasLayers.UpSampling2D
  # Upsampling layer for 3D inputs.
  availLayer['upsampling3d'] = KerasLayers.UpSampling3D
  # Zero-padding layer for 1D input (e.g. temporal sequence).
  availLayer['zeropadding1d'] = KerasLayers.ZeroPadding1D
  # Zero-padding layer for 2D input (e.g. picture).
  # This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
  availLayer['zeropadding2d'] = KerasLayers.ZeroPadding2D
  # Zero-padding layer for 3D data (spatial or spatio-tempral)
  availLayer['zeropadding3d'] = KerasLayers.ZeroPadding3D
  # Locally-connected layer for 1D inputs.
  # The LocallyConnected1D layer works similarly to the Conv1D layer, except that weights are unshared,
  # that is, a different set of filters is applied at each different patch of the input.
  availLayer['locallyconnected1d'] = KerasLayers.LocallyConnected1D
  # Locally-connected layer for 2D inputs.
  # The LocallyConnected1D layer works similarly to the Conv2D layer, except that weights are unshared,
  # that is, a different set of filters is applied at each different patch of the input.
  availLayer['locallyconnected2d'] = KerasLayers.LocallyConnected2D

  # available pooling layers
  # Max pooling operation for temporal data.
  availLayer['maxpooling1d'] = KerasLayers.MaxPooling1D
  # Max pooling operation for spatial data.
  availLayer['maxpooling2d'] = KerasLayers.MaxPooling2D
  # Max pooling operation for 3D data (spatial or spatio-temporal).
  availLayer['maxpooling3d'] = KerasLayers.MaxPooling3D
  # Average pooling for temporal data.
  availLayer['averagepooling1d'] = KerasLayers.AveragePooling1D
  # Average pooling for spatial data.
  availLayer['averagepooling2d'] = KerasLayers.AveragePooling2D
  # Average pooling operation for 3D data (spatial or spatio-temporal).
  availLayer['averagepooling3d'] = KerasLayers.AveragePooling3D
  # Global max pooling operation for temporal data.
  availLayer['globalmaxpooling1d'] = KerasLayers.GlobalMaxPooling1D
  # Global average pooling operation for temporal data.
  availLayer['globalaveragepooling1d'] = KerasLayers.GlobalAveragePooling1D
  # Global max pooling operation for spatial data.
  availLayer['globalmaxpooling2d'] = KerasLayers.GlobalMaxPooling2D
  # Global average pooling operation for spatial data.
  availLayer['globalaveragepooling2d'] = KerasLayers.GlobalAveragePooling2D
  # Global Max pooling operation for 3D data.
  availLayer['globalmaxpooling3d'] = KerasLayers.GlobalMaxPooling3D
  # Global Average pooling operation for 3D data.
  availLayer['globalaveragepooling3d'] = KerasLayers.GlobalAveragePooling3D

  # available embedding layers
  # turns positive integers (indexes) into dense vectors of fixed size
  # This layer can only be used as the first layer in a model.
  availLayer['embedding'] = KerasLayers.Embedding

  # available recurrent layers
  # Fully-connected RNN where the output is to be fed back to input.
  availLayer['simplernn'] = KerasLayers.SimpleRNN
  # Gated Recurrent Unit - Cho et al. 2014.
  availLayer['gru'] = KerasLayers.GRU
  # Long Short-Term Memory layer - Hochreiter 1997.
  availLayer['lstm'] = KerasLayers.LSTM
  # Convolutional LSTM.
  # It is similar to an LSTM layer, but the input transformations and recurrent transformations are both convolutional.
  availLayer['convlstm2d'] = KerasLayers.ConvLSTM2D
  # Fast GRU implementation backed by CuDNN.
  #availRecurrent['cudnngru'] = KerasLayers.CuDNNGRU
  # Fast LSTM implementation with CuDNN.
 # availRecurrent['cudnnlstm'] = KerasLayers.CuDNNLSTM

  # available normalization layers
  availNormalization = {}
  availNormalization['batchnormalization'] = KerasLayers.BatchNormalization

  # available noise layers
  availNoise = {}
  # Apply additive zero-centered Gaussian noise.
  # This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
  # Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.
  availNoise['gaussiannoise'] = KerasLayers.GaussianNoise
  # Apply multiplicative 1-centered Gaussian noise. As it is a regularization layer, it is only active at training time.
  availNoise['gaussiandropout'] = KerasLayers.GaussianDropout
  # Applies Alpha Dropout to the input.
  # Alpha Dropout is a Dropout that keeps mean and variance of inputs to their original values, in order to ensure
  # the self-normalizing property even after this dropout. Alpha Dropout fits well to Scaled Exponential Linear Units
  #  by randomly setting activations to the negative saturation value.
  #availNoise['alphadropout'] = KerasLayers.AlphaDropout
  # list of Keras Neural Network Core layers
  kerasCoreLayersList = ['dense',
                        'activation',
                        'dropout',
                        'flatten',
                        'input',
                        'reshape',
                        'permute',
                        'repeatvector',
                        'lambda',
                        'activityregularization',
                        'masking',
                        'spatialdropout1d',
                        'spatialdropout2d',
                        'spatialdropout3d']
  # list of Keras Neural Network Convolutional layers
  kerasConvNetLayersList =    ['conv1d',
                               'conv2d',
                               'conv3d',
                               'separableconv1d',
                               'separableconv2d',
                               'depthwiseconv2d',
                               'conv2dtranspose',
                               'conv3dtranspose',
                               'cropping1d',
                               'cropping2d',
                               'cropping3d',
                               'upsampling1d',
                               'upsampling2d',
                               'upsampling3d',
                               'zeropadding1d',
                               'zeropadding2d',
                               'zeropadding3d']
  # list of Keras Neural Network Pooling layers
  kerasPoolingLayersList =    ['maxpooling1d',
                               'maxpooling2d',
                               'maxpooling3d',
                               'averagepooling1d',
                               'averagepooling2d',
                               'averagepooling3d',
                               'globalmaxpooling1d',
                               'globalmaxpooling2d',
                               'globalmaxpooling3d',
                               'globalaveragepooling1d',
                               'globalaveragepooling2d',
                               'globalaveragepooling3d']
  # list of Keras Neural Network Recurrent layers
  kerasRcurrentLayersList =    ['rnn',
                                'simplernn',
                                'gru',
                                'lstm',
                                'convlstm2d',
                                'simplernncell',
                                'grucell',
                                'lstmcell',
                                'cudnngru',
                                'cudnnlstm']
  # list of Keras Neural Network Locally-connected layers
  kerasLocallyConnectedLayersList =    ['locallyconnected1d',
                                        'locallyconnected2d']
  # list of Keras Neural Network Embedding layers
  kerasEmbeddingLayersList = ['embedding']
  # list of Keras Neural Network Advanced Activation layers
  kerasAdvancedActivationLayersList =    ['leakyrelu',
                                          'prelu',
                                          'elu',
                                          'thresholdedrelu',
                                          'softmax',
                                          'relu']
  # list of Keras Neural Network Normalization layers
  kerasNormalizationLayersList = ['batchnormalization']
  # list of Keras Neural Network Noise layers
  kerasNoiseLayersList =    ['gaussiannoise',
                             'gaussiandropout',
                             'alphadropout']


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
    # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self._dynamicHandling            = True
    # Basic Layers
    self.basicLayers = self.__class__.kerasCoreLayersList + self.__class__.kerasEmbeddingLayersList + \
                       self.__class__.kerasAdvancedActivationLayersList + self.__class__.kerasNormalizationLayersList + \
                       self.__class__.kerasNoiseLayersList
    # LabelEncoder can be used to normalize labels
    self.labelEncoder = preprocessing.LabelEncoder()
    # perform z-score normalization if True
    self.externalNorm = True
    # variale to store feature values, shape=[n_samples, n_features]
    self.featv = None
    # variable to store target values, shape = [n_samples]
    self.targv = None
    # instance of KERAS deep neural network model
    self.ROM = None
    randomSeed = self.initOptionDict.pop('random_seed',None)
    # Set the seed for random number generation to obtain reproducible results
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    if randomSeed is not None:
      # The below is necessary for starting Numpy generated random numbers
      # in a well-defined initial state.
      np.random.seed(randomSeed)
      # The below is necessary for starting core Python generated random numbers
      # in a well-defined state.
      rn.seed(randomSeed)
      # The below tf.set_random_seed() will make random number generation
      # in the TensorFlow backend have a well-defined initial state.
      # For further details, see:
      # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
      set_random_seed(randomSeed)
    modelName = self.initOptionDict.pop('name','')
    # number of classes for classifier
    self.numClasses = self.initOptionDict.pop('num_classes',1)
    # validation split, default to 0.25
    self.validationSplit = self.initOptionDict.pop('validation_split',0.25)
    # options to plot deep neural network model, default False
    self.plotModel = self.initOptionDict.pop('plot_model',False)
    self.plotModelFilename = self.printTag + "_model.png" if not modelName else modelName + "_model.png"
    # activation function for output layer of deep neural network
    self.outputLayerActivation = self.initOptionDict.pop('output_layer_activation', 'softmax')
    # A loss function that is always required to compile a KERAS model
    self.lossFunction = self.initOptionDict.pop('loss','categorical_crossentropy')
    # a metric is a function that is used to judge the performance of KERAS model
    self.metrics = self.initOptionDict.pop('metrics',['accuracy'])
    # number of samples per gradient update, default 20
    self.batchSize = int(self.initOptionDict.pop('batch_size',20))
    # number of epochs to train the model. An epoch is an iteration over the entire training data, (default 20)
    self.epochs = int(self.initOptionDict.pop('epochs', 20))
    # extract settings for optimizer
    optimizerSetting = self.initOptionDict.pop('optimizerSetting', {'optimizer':'adam'})
    optimizerName = optimizerSetting.pop('optimizer').lower()
    for key,value in optimizerSetting.items():
      try:
        optimizerSetting[key] = ast.literal_eval(value)
      except:
        pass
    # set up optimizer
    self.optimizer = self.__class__.availOptimizer[optimizerName](**optimizerSetting)
    # check layer layout, this is always required node, used to build the DNNs
    self.layerLayout = self.initOptionDict.pop('layer_layout',None)
    if self.layerLayout is None:
      self.raiseAnError(IOError,"XML node 'layer_layout' is required for ROM class", self.printTag)
    elif not set(self.layerLayout).issubset(list(self.initOptionDict.keys())):
      self.raiseAnError(IOError, "The following layers are not defined '{}'.".format(', '.join(set(self.layerLayout)
                        -set(list(self.initOptionDict.keys())))))

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

  def __checkLayers__(self):
    """
      Method used to check layers setups for KERAS model
      @ In, None
      @ Out, None
    """
    pass

  def __addHiddenLayers__(self):
    """
      Method used to add hidden layers for KERAS model
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
      dropoutRate = layerDict.pop('rate',0.0)
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

  def __addOutputLayers__(self):
    """
      Method used to add last output layers for KERAS model
      @ In, None
      @ Out, None
    """
    layerName = self.layerLayout[-1]
    layerDict = self.initOptionDict.pop(layerName)
    layerType = layerDict.pop('type').lower()
    layerSize = layerDict.pop('dim_out',None)
    if layerSize is not None and layerSize != self.numClasses:
      self.raiseAWarning('The "dim_out" of last output layer: ', layerName, 'will be resetted to values provided in "num_classes", i.e.', self.numClasses)
    if layerType not in ['dense']:
      self.raiseAnError(IOError,'The last layer should always be Dense layer, but',layerType,'is provided!')
    layerInstant = self.__class__.availLayer[layerType]
    self.ROM.add(layerInstant(self.numClasses,**layerDict))

  def train(self,tdict):
    """
      Method to perform the training of the deep neural network algorithm
      NB.the KerasClassifier object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, dict, training dictionary
      @ Out, None
    """
    if type(tdict) != dict:
      self.raiseAnError(TypeError,'In method "train", the training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = list(tdict.keys()), list(tdict.values())
    # Currently, deep neural networks (DNNs) are only used for classification.
    # Targets for deep neural network should be labels only (i.e. integers only)
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
    targetValues = stats.mode(tval,axis=-1)[0] if len(tval.shape) > 1 else tval
    #targetValues = list(val[-1] if len(tval.shape) > 1 else val for val in tval)

    # We need to 'one-hot-encode' our target variable if multi-classes are requested
    # This means that a column will be created for each output category and a binary variable is inputted for
    # each category.
    if self.numClasses > 1 and 'categorical_crossentropy' in self.lossFunction:
      # Transform the labels (i.e. numerical or non-numerical) to normalized numerical labels
      targetValues = self.labelEncoder.fit_transform(targetValues.ravel())
      targetValues = KerasUtils.to_categorical(targetValues)
      if self.numClasses != targetValues.shape[-1]:
        self.raiseAWarning('The num_classes:',self.numClasses, 'specified by the user is not equal number of classes',
                           targetValues.shape[-1], ' in the provided data!')
        self.raiseAWarning('Reset the num_classes to be consistent with the data!')
        self.numClasses = targetValues.shape[-1]

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
        self._localNormalizeData(values,names,feat)
        fval = (fval - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
        featureValues.append(fval)
      else:
        self.raiseAnError(IOError,'The feature ',feat,' is not in the training set')

    #FIXME: when we do not support anymore numpy <1.10, remove this IF STATEMENT
    if int(np.__version__.split('.')[1]) >= 10:
      featureValues = np.stack(featureValues, axis=-1)
    else:
      sl = (slice(None),) * np.asarray(featureValues[0]).ndim + (np.newaxis,)
      featureValues = np.concatenate([np.asarray(arr)[sl] for arr in featureValues], axis=np.asarray(featureValues[0]).ndim)

    self.__trainLocal__(featureValues,targetValues)
    self.amITrained = True

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature or shape=[numSamples, numTimeSteps, numFeatures]
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    self.featv = featureVals
    self.targv = targetVals
    # check layers
    self.__checkLayers__()
    # hidden layers
    self.__addHiddenLayers__()
    #output layer
    self.__addOutputLayers__()
    self.ROM.compile(loss=self.lossFunction, optimizer=self.optimizer, metrics=self.metrics)
    self.ROM._make_predict_function() # have to initialize before threading
    history = self.ROM.fit(featureVals, targetVals, epochs=self.epochs, batch_size=self.batchSize, validation_split=self.validationSplit)
    # The following requires pydot-ng and graphviz to be installed (See the manual)
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
    with self.graph.as_default():
      outcome = self.ROM.predict(featureVals)
    if self.numClasses > 1 and self.lossFunction in ['categorical_crossentropy']:
      outcome = np.argmax(outcome,axis=1)
      # Transform labels back to original encoding
      outcome = self.labelEncoder.inverse_transform(outcome)
      # TODO, extend to multi-targets, currently we only accept one target
      prediction[self.target[0]] = outcome
    else:
      prediction[self.target[0]] = [round(val[0]) for val in outcome]
    return prediction

  def _preprocessInputs(self,featureVals):
    """
      Perform input feature values before sending to ROM prediction
      @ In, featureVals, numpy.array, 2-D for static case and 3D for time-dependent case, values of features
      @ Out, featureVals, numpy.array, predicted values
    """
    return featureVals

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
