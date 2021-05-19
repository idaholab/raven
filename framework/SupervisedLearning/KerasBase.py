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
  @author: wangc and cogljj
  base class for tensorflow and keras used for deep neural network
  i.e. Multi-layer perceptron classifier, CNN, LSTM
"""

#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
import random as rn
import matplotlib
import platform
from scipy import stats
import os
import utils.importerUtils
tf = utils.importerUtils.importModuleLazyRenamed("tf", globals(), "tensorflow")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class KerasBase(supervisedLearning):
  """
    Multi-layer perceptron base class constructed using Keras API in TensorFlow
  """
  # some modules are commented out since they are not available in TensorFlow 1.1
  # We need to install TensorFlow in a different way
  # available optimizers in Keras
  ROMType = 'KerasBase'
  # An optimizer is required for compiling a Keras model
  availOptimizer = {}

  # available convolutional layers
  availLayer = {}

  # available normalization layers
  availNormalization = {}

  # available noise layers
  availNoise = {}
  # Applies Alpha Dropout to the input.
  # Alpha Dropout is a Dropout that keeps mean and variance of inputs to their original values, in order to ensure
  # the self-normalizing property even after this dropout. Alpha Dropout fits well to Scaled Exponential Linear Units
  #  by randomly setting activations to the negative saturation value.
  #availNoise['alphadropout'] = tf.keras.layers.AlphaDropout
  # Temp Model File that used to dump and load Keras Model
  tempModelFile = "a_temporary_file_for_storing_a_keras_model.h5"
  modelAttr = "the_model_all_serialized_and_turned_into_an_hdf5_file_and_stuff"

  def __init__(self, **kwargs):
    """
      A constructor that will appropriately intialize a keras deep neural network object
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    if len(self.availOptimizer) == 0:
      # stochastic gradient descent optimizer, includes support for momentum,learning rate decay, and Nesterov momentum
      self.availOptimizer['sgd'] = tf.keras.optimizers.SGD
      # RMSprop optimizer, usually a good choice for recurrent neural network
      self.availOptimizer['rmsprop'] = tf.keras.optimizers.RMSprop
      # Adagrad is an optimzer with parameter-specific learning rates, which are adapted relative to
      # how frequently a parameter gets updated during training. The more updates  a parameter receives,
      # the smaller the updates.
      self.availOptimizer['adagrad'] = tf.keras.optimizers.Adagrad
      # Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving
      # window of gradient updates, instead of accumulating all past gradients. This way, Adadelta
      # continues learning even when many updates have been done.
      self.availOptimizer['adadelta'] = tf.keras.optimizers.Adadelta
      # Adam optimzer
      self.availOptimizer['adam'] = tf.keras.optimizers.Adam
      # Adamax optimizer from Adam paper's section 7
      self.availOptimizer['adamax'] = tf.keras.optimizers.Adamax
      # Nesterov Adam optimizer
      self.availOptimizer['nadam'] = tf.keras.optimizers.Nadam

    if len(self.availLayer) == 0:
      # dense layer
      self.availLayer['dense'] = tf.keras.layers.Dense
      # apply dropout to the input
      self.availLayer['dropout'] = tf.keras.layers.Dropout
      # Flatten layer
      self.availLayer['flatten'] = tf.keras.layers.Flatten
      # 1D convolution layer (e.g. temporal convolution).
      self.availLayer['conv1d'] = tf.keras.layers.Conv1D
      # 2D convolution layer (e.g. spatial convolution over images).
      self.availLayer['conv2d'] = tf.keras.layers.Conv2D
      # Depthwise separable 1D convolution.
      #availConvNet['separableconv1d'] = tf.keras.layers.SeparableConv1D
      # Depthwise separable 2D convolution.
      self.availLayer['separableconv2d'] = tf.keras.layers.SeparableConv2D
      # Depthwise separable 2D convolution.
      #availConvNet['depthwiseconv2d'] = tf.keras.layers.DepthwiseConv2D
      # Transposed convolution layer (sometimes called Deconvolution).
      self.availLayer['conv2dtranspose'] = tf.keras.layers.Conv2DTranspose
      # 3D convolution layer (e.g. spatial convolution over volumes).
      self.availLayer['conv3d'] = tf.keras.layers.Conv3D
      # ransposed convolution layer (sometimes called Deconvolution).
      #availConvNet['conv3dtranspose'] = tf.keras.layers.Conv3DTranspose
      # Cropping layer for 1D input (e.g. temporal sequence). It crops along the time dimension (axis 1).
      self.availLayer['cropping1d'] = tf.keras.layers.Cropping1D
      # Cropping layer for 2D input (e.g. picture). It crops along spatial dimensions, i.e. height and width.
      self.availLayer['cropping2d'] = tf.keras.layers.Cropping2D
      # Cropping layer for 3D data (e.g. spatial or spatio-temporal).
      self.availLayer['cropping3d'] = tf.keras.layers.Cropping3D
      # Upsampling layer for 1D inputs
      self.availLayer['upsampling1d'] = tf.keras.layers.UpSampling1D
      # Upsampling layer for 2D inputs.
      self.availLayer['upsampling2d'] = tf.keras.layers.UpSampling2D
      # Upsampling layer for 3D inputs.
      self.availLayer['upsampling3d'] = tf.keras.layers.UpSampling3D
      # Zero-padding layer for 1D input (e.g. temporal sequence).
      self.availLayer['zeropadding1d'] = tf.keras.layers.ZeroPadding1D
      # Zero-padding layer for 2D input (e.g. picture).
      # This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
      self.availLayer['zeropadding2d'] = tf.keras.layers.ZeroPadding2D
      # Zero-padding layer for 3D data (spatial or spatio-tempral)
      self.availLayer['zeropadding3d'] = tf.keras.layers.ZeroPadding3D
      # Locally-connected layer for 1D inputs.
      # The LocallyConnected1D layer works similarly to the Conv1D layer, except that weights are unshared,
      # that is, a different set of filters is applied at each different patch of the input.
      self.availLayer['locallyconnected1d'] = tf.keras.layers.LocallyConnected1D
      # Locally-connected layer for 2D inputs.
      # The LocallyConnected1D layer works similarly to the Conv2D layer, except that weights are unshared,
      # that is, a different set of filters is applied at each different patch of the input.
      self.availLayer['locallyconnected2d'] = tf.keras.layers.LocallyConnected2D

      # available pooling layers
      # Max pooling operation for temporal data.
      self.availLayer['maxpooling1d'] = tf.keras.layers.MaxPooling1D
      # Max pooling operation for spatial data.
      self.availLayer['maxpooling2d'] = tf.keras.layers.MaxPooling2D
      # Max pooling operation for 3D data (spatial or spatio-temporal).
      self.availLayer['maxpooling3d'] = tf.keras.layers.MaxPooling3D
      # Average pooling for temporal data.
      self.availLayer['averagepooling1d'] = tf.keras.layers.AveragePooling1D
      # Average pooling for spatial data.
      self.availLayer['averagepooling2d'] = tf.keras.layers.AveragePooling2D
      # Average pooling operation for 3D data (spatial or spatio-temporal).
      self.availLayer['averagepooling3d'] = tf.keras.layers.AveragePooling3D
      # Global max pooling operation for temporal data.
      self.availLayer['globalmaxpooling1d'] = tf.keras.layers.GlobalMaxPooling1D
      # Global average pooling operation for temporal data.
      self.availLayer['globalaveragepooling1d'] = tf.keras.layers.GlobalAveragePooling1D
      # Global max pooling operation for spatial data.
      self.availLayer['globalmaxpooling2d'] = tf.keras.layers.GlobalMaxPooling2D
      # Global average pooling operation for spatial data.
      self.availLayer['globalaveragepooling2d'] = tf.keras.layers.GlobalAveragePooling2D
      # Global Max pooling operation for 3D data.
      self.availLayer['globalmaxpooling3d'] = tf.keras.layers.GlobalMaxPooling3D
      # Global Average pooling operation for 3D data.
      self.availLayer['globalaveragepooling3d'] = tf.keras.layers.GlobalAveragePooling3D

      # available embedding layers
      # turns positive integers (indexes) into dense vectors of fixed size
      # This layer can only be used as the first layer in a model.
      self.availLayer['embedding'] = tf.keras.layers.Embedding

      # available recurrent layers
      # Fully-connected RNN where the output is to be fed back to input.
      self.availLayer['simplernn'] = tf.keras.layers.SimpleRNN
      # Gated Recurrent Unit - Cho et al. 2014.
      self.availLayer['gru'] = tf.keras.layers.GRU
      # Long Short-Term Memory layer - Hochreiter 1997.
      self.availLayer['lstm'] = tf.keras.layers.LSTM
      # Convolutional LSTM.
      # It is similar to an LSTM layer, but the input transformations and recurrent transformations are both convolutional.
      self.availLayer['convlstm2d'] = tf.keras.layers.ConvLSTM2D
      # Fast GRU implementation backed by CuDNN.
      #availRecurrent['cudnngru'] = tf.keras.layers.CuDNNGRU
      # Fast LSTM implementation with CuDNN.
      # availRecurrent['cudnnlstm'] = tf.keras.layers.CuDNNLSTM

    if len(self.availNormalization) == 0:
      self.availNormalization['batchnormalization'] = tf.keras.layers.BatchNormalization

    if len(self.availNoise) == 0:
      # Apply additive zero-centered Gaussian noise.
      # This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
      # Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.
      self.availNoise['gaussiannoise'] = tf.keras.layers.GaussianNoise
      # Apply multiplicative 1-centered Gaussian noise. As it is a regularization layer, it is only active at training time.
      self.availNoise['gaussiandropout'] = tf.keras.layers.GaussianDropout

    super().__init__(**kwargs)
    self.printTag = 'KerasBase'

  def readInitDict(self, initDict):
    """
      Reads in the initialization dict to initialize this instance
      @ In, initDict, dict, keywords passed to constructor
      @ Out, None
    """
    super().readInitDict(initDict)
    # parameter dictionary at the initial stage
    self.initDict = copy.deepcopy(self.initOptionDict)
    # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self._dynamicHandling = True
    # Basic Layers
    self.basicLayers = self.kerasROMDict['kerasCoreLayersList'] + self.kerasROMDict['kerasEmbeddingLayersList'] + \
                       self.kerasROMDict['kerasAdvancedActivationLayersList'] + self.kerasROMDict['kerasNormalizationLayersList'] + \
                       self.kerasROMDict['kerasNoiseLayersList']
    # LabelEncoder can be used to normalize labels
    from sklearn import preprocessing
    self.labelEncoder = preprocessing.LabelEncoder()
    # perform z-score normalization if True
    self.externalNorm = True
    # variale to store feature values, shape=[n_samples, n_features]
    self.featv = None
    # variable to store target values, shape = [n_samples]
    self.targv = None
    # instance of KERAS deep neural network model
    self._ROM = None
    # the training/testing history of ROM
    self._romHistory = None
    self._sessionConf = None
    randomSeed = self.initOptionDict.pop('random_seed',None)
    # Force TensorFlow to use single thread when reproducible results are requested
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/
    self._sessionConf = tf.ConfigProto(intra_op_parallelism_threads=self.numThreads,
                                  inter_op_parallelism_threads=self.numThreads)
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
      tf.set_random_seed(randomSeed)
    self._session = tf.Session(graph=tf.get_default_graph(), config=self._sessionConf)
    # Base on issue https://github.com/tensorflow/tensorflow/issues/28287
    # The problem is that tensorflow graphs and sessions are not thread safe. So by default
    # a new session (which) does not contain any previously loaded weights, models, and so on)
    # is created for each thread, i.e. for each request. By saving the session that contains all
    # the models and setting it to be used by keras in each thread.
    tf.keras.backend.set_session(self._session)

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
    self._initGraph()

  def _initGraph(self):
    """
      Method used to add additional initialization features
      Such as complile KERAS model
      @ In, None
      @ Out, None
    """
    # This is needed to solve the thread issue in self._ROM.predict()
    # https://github.com/fchollet/keras/issues/2397#issuecomment-306687500
    self.graph = tf.get_default_graph()

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    state = supervisedLearning.__getstate__(self)
    tf.keras.models.save_model(self._ROM, KerasBase.tempModelFile)
    # another method to save the TensorFlow model
    # self._ROM.save(KerasBase.tempModelFile)
    with open(KerasBase.tempModelFile, "rb") as f:
      serialModelData = f.read()
    state[KerasBase.modelAttr] = serialModelData
    os.remove(KerasBase.tempModelFile)
    del state["_ROM"]
    state['initOptionDict'].pop('paramInput',None)
    return state

  def __setstate__(self, d):
    """
      Initialize the ROM with the data contained in newstate
      @ In, d, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    with open(KerasBase.tempModelFile, "wb") as f:
      f.write(d[KerasBase.modelAttr])
    del d[KerasBase.modelAttr]
    tf.keras.backend.set_session(self._session)
    self._ROM = tf.keras.models.load_model(KerasBase.tempModelFile)
    os.remove(KerasBase.tempModelFile)
    self.__dict__.update(d)

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, StaticXmlElement to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError, 'ROM is not yet trained! Cannot write to DataObject.')
    root = writeTo.getRoot()
    writeTo.addScalar('Accuracy',"Training",' '.join([str(elm) for elm in self._romHistory.history['acc']]))
    writeTo.addScalar('Accuracy',"Testing",' '.join([str(elm) for elm in self._romHistory.history['val_acc']]))
    writeTo.addScalar('Loss',"Training",' '.join([str(elm) for elm in self._romHistory.history['loss']]))
    writeTo.addScalar('Loss',"Testing",' '.join([str(elm) for elm in self._romHistory.history['val_loss']]))

  def train(self,tdict):
    """
      Method to perform the training of the deep neural network algorithm
      NB.the KerasBase object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, dict, training dictionary
      @ Out, None
    """
    if type(tdict) != dict:
      self.raiseAnError(TypeError,'In method "train", the training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = zip(*tdict.items())
    targetValues = self._getTrainingTargetValues(names, values)

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
        fval = self._scaleToNormal(fval, feat)
        featureValues.append(fval)
      else:
        self.raiseAnError(IOError,'The feature ',feat,' is not in the training set')
    featureValues = np.stack(featureValues, axis=-1)

    self.__trainLocal__(featureValues,targetValues)
    self.amITrained = True


  def _checkLayers(self):
    """
      Method used to check layers setups for KERAS model
      @ In, None
      @ Out, None
    """
    pass

  def _addHiddenLayers(self):
    """
      Method used to add hidden layers for KERAS model
      @ In, None
      @ Out, None
    """
    # start to build the ROM
    self._ROM = tf.keras.models.Sequential()
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
          self._ROM.add(self._getFirstHiddenLayer(layerInstant, layerSize, layerDict))
        else:
          self._ROM.add(layerInstant(layerSize,**layerDict))
      else:
        if layerType == 'dropout':
          self._ROM.add(layerInstant(dropoutRate))
        else:
          self._ROM.add(layerInstant(**layerDict))

  def _addOutputLayers(self):
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
    self._ROM.add(self._getLastLayer(layerInstant, layerDict))

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature or shape=[numSamples, numTimeSteps, numFeatures]
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    #Need featureVals to be a numpy array with shape:
    # (batches, data per batch, input_features)
    #Need targetVals to be a numpy array with shape for Regressions:
    # (batches, data per batch, output_features)
    self.featv = featureVals
    self.targv = targetVals
    # check layers
    self._checkLayers()
    # hidden layers
    self._addHiddenLayers()
    #output layer
    self._addOutputLayers()
    self._ROM.compile(loss=self.lossFunction, optimizer=self.optimizer, metrics=self.metrics)
    self._ROM._make_predict_function() # have to initialize before threading
    self._romHistory = self._ROM.fit(featureVals, targetVals, epochs=self.epochs, batch_size=self.batchSize, validation_split=self.validationSplit)
    # The following requires pydot-ng and graphviz to be installed (See the manual)
    # https://github.com/keras-team/keras/issues/3210
    if self.plotModel:
      tf.keras.utils.plot_model(self._ROM,to_file=self.plotModelFilename,show_shapes=True)

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals,numpy.array, 2-D or 3-D numpy array, [n_samples,n_features]
        or shape=[numSamples, numTimeSteps, numFeatures]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError,'KerasBase   : __confidenceLocal__ method must be implemented!')

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
      tf.keras.backend.set_session(self._session)
      outcome = self._ROM.predict(featureVals)
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
    self._initGraph()
    self._ROM = None
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
      self._ROM = tf.keras.models.Sequential.from_config(config)
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    params = self._ROM.get_config()
    return params

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self._initGraph()
    self._ROM = None
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
      self._ROM = tf.keras.models.Sequential.from_config(config)
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    params = self._ROM.get_config()
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
      super()._localNormalizeData(values,names,feat)

  def _scaleToNormal(self, values, feat):
    """
      Method to normalize based on previously calculated values
      @ In, values, np.array, array to be normalized
      @ In, feat, string, feature name
      @ Out, scaled, np.array, normalized array
    """
    mu,sigma = self.muAndSigmaFeatures[feat]
    return (values - mu)/sigma

  def _invertScaleToNormal(self, values, feat):
    """
      Method to unnormalize based on previously calculated values
      @ In, values, np.array, array to be normalized
      @ In, feat, string, feature name
      @ Out, scaled, np.array, normalized array
    """
    mu,sigma = self.muAndSigmaFeatures[feat]
    return values*sigma + mu
