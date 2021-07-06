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
from .SupervisedLearning import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class KerasBase(SupervisedLearning):
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
    inputSpecification = super().getInputSpecification()
    inputSpecification.description = r"""TensorFlow-Keras Deep Neural Networks."""
    # for deep learning neural network
    #inputSpecification.addSub(InputData.parameterInputFactory("DNN", InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("hidden_layer_sizes", contentType=InputTypes.IntegerTupleType)) # list of integer
    inputSpecification.addSub(InputData.parameterInputFactory("metrics", contentType=InputTypes.StringListType, default='accuracy')) #list of metrics
    inputSpecification.addSub(InputData.parameterInputFactory("batch_size", contentType=InputTypes.IntegerType, default=20))
    inputSpecification.addSub(InputData.parameterInputFactory("epochs", contentType=InputTypes.IntegerType, default=20))
    inputSpecification.addSub(InputData.parameterInputFactory("random_seed", contentType=InputTypes.IntegerType, default=None))
    inputSpecification.addSub(InputData.parameterInputFactory("plot_model", contentType=InputTypes.BoolType, default=False))
    inputSpecification.addSub(InputData.parameterInputFactory("num_classes",contentType= InputTypes.IntegerType, default=1))
    inputSpecification.addSub(InputData.parameterInputFactory("validation_split", contentType=InputTypes.FloatType, default=0.25))
    inputSpecification.addSub(InputData.parameterInputFactory("output_layer_activation", contentType=InputTypes.StringType, default='softmax'))
    inputSpecification.addSub(InputData.parameterInputFactory("loss", contentType=InputTypes.StringType, default='categorical_crossentropy'))
    # Keras optimizer parameters
    OptimizerSettingInput = InputData.parameterInputFactory('optimizerSetting', contentType=InputTypes.StringType, default=None)
    Beta1Input = InputData.parameterInputFactory('beta_1', contentType=InputTypes.FloatType, default=0.9)
    Beta2Input = InputData.parameterInputFactory('beta_2', contentType=InputTypes.FloatType, default=0.999)
    DecayInput = InputData.parameterInputFactory('decay', contentType=InputTypes.FloatType, default=0.0)
    LRInput = InputData.parameterInputFactory('lr', contentType=InputTypes.FloatType, default=0.001)
    OptimizerInput = InputData.parameterInputFactory('optimizer', contentType=InputTypes.StringType, default='Adam')
    EpsilonInput = InputData.parameterInputFactory('epsilon', contentType=InputTypes.FloatType, default=None)
    MomentumInput = InputData.parameterInputFactory('momentum', contentType=InputTypes.FloatType, default=0.0)
    NesterovInput = InputData.parameterInputFactory('nesterov', contentType=InputTypes.BoolType, default=False)
    RhoInput = InputData.parameterInputFactory('rho', contentType=InputTypes.FloatType, default=0.95)
    OptimizerSettingInput.addSub(Beta1Input)
    OptimizerSettingInput.addSub(Beta2Input)
    OptimizerSettingInput.addSub(DecayInput)
    OptimizerSettingInput.addSub(LRInput)
    OptimizerSettingInput.addSub(OptimizerInput)
    OptimizerSettingInput.addSub(EpsilonInput)
    OptimizerSettingInput.addSub(MomentumInput)
    OptimizerSettingInput.addSub(NesterovInput)
    OptimizerSettingInput.addSub(RhoInput)
    inputSpecification.addSub(OptimizerSettingInput)

    # Keras Layers parameters
    dataFormatEnumType = InputTypes.makeEnumType('dataFormat','dataFormatType',['channels_last', 'channels_first'])
    paddingEnumType = InputTypes.makeEnumType('padding','paddingType',['valid', 'same'])
    interpolationEnumType = InputTypes.makeEnumType('interpolation','interpolationType',['nearest', 'bilinear'])
    ###########################
    #  Dense Layers: regular densely-connected neural network layer
    ###########################
    layerInput = InputData.parameterInputFactory('Dense',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Activation Layers: applies an activation function to an output
    ###########################
    layerInput = InputData.parameterInputFactory('Activation',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'activation' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Dropout Layers: Applies Dropout to the input
    ###########################
    layerInput = InputData.parameterInputFactory('Dropout',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('noise_shape',contentType=InputTypes.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('seed',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Flatten Layers: Flattens the input
    ###########################
    layerInput = InputData.parameterInputFactory('Flatten',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Input Layers: Input() is used to instantiate a Keras tensor
    ###########################
    layerInput = InputData.parameterInputFactory('Input',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Reshape Layers: Reshapes an output to a certain shape
    ###########################
    layerInput = InputData.parameterInputFactory('Reshape',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'target_shape' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('target_shape',contentType=InputTypes.IntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Permute Layers: permutes the dimensions of the input according to a given pattern
    ###########################
    layerInput = InputData.parameterInputFactory('Permute',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'permute_pattern' need to pop out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('permute_pattern',contentType=InputTypes.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('input_shape',contentType=InputTypes.IntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  RepeatVector Layers: repeats the input n times
    ###########################
    layerInput = InputData.parameterInputFactory('RepeatVector',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'repetition_factor' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('repetition_factor',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Lambda Layers: Wraps arbitrary expression as a Layer object
    ###########################
    layerInput = InputData.parameterInputFactory('Lambda',contentType=InputTypes.StringType,strictMode=False)
    layerInput.addParam('name', InputTypes.StringType, True)
    # A function object need to be created and passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('function',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ActivityRegularization Layers: applies an update to the cost function based input activity
    ###########################
    layerInput = InputData.parameterInputFactory('ActivityRegularization',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('l1',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('l2',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Masking Layers: Masks a sequence by using a mask value to skip timesteps
    ###########################
    layerInput = InputData.parameterInputFactory('Masking',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('mask_value',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout1D Layers: Spatial 1D version of Dropout
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###################################
    # Convolutional Layers
    ###################################
    ###########################
    #  Conv1D Layers: 1D convolutioanl layer (e.g. temporal convolutional)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv2D Layers: 2D convolutioanl layer (e.g. spatial convolution over images)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv3D Layers: 3D convolutioanl layer (e.g. spatial convolution over volumes)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SeparableConv1D Layers: Depthwise separable 1D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('SeparableConv1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SeparableConv2D Layers: Depthwise separable 2D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('SeparableConv2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  DepthwiseConv2D Layers: Depthwise separable 2D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('DepthwiseConv2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv2DTranspose Layers: Transposed convolution layer (sometimes called Deconvolution)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv2DTranspose',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('output_padding',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv3DTranspose Layers: Transposed convolution layer (sometimes called Deconvolution)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv3DTranspose',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('output_padding',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    #  Cropping1D  Layers: cropping layer for 1D input (e.g. temporal sequence)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputTypes.IntegerOrIntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Cropping2D  Layers: cropping layer for 2D input (e.g. picutures)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Cropping3D  Layers: cropping layer for 2D input (e.g. picutures)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Upsampling1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('UpSampling2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('interpolation',contentType=interpolationEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('UpSampling3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputTypes.IntegerOrIntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ############################################
    #   Pooling Layers
    ############################################
    ###########################
    #  MaxPooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  MaxPooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  MaxPooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling3D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)

    ######################################
    #   Locally-connected Layers
    ######################################
    ###########################
    #  LocallyConnected1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LocallyConnected1D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LocallyConnected2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LocallyConnected2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ######################################
    #  Recurrent Layers
    ######################################
    ###########################
    #  RNN Layers
    ###########################
    layerInput = InputData.parameterInputFactory('RNN',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SimpleRNN Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SimpleRNN',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GRU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GRU',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('reset_after',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LSTM Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LSTM',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ConvLSTM2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ConvLSTM2D',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SimpleRNNCell Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SimpleRNNCell',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GRUCell Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GRUCell',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('reset_after',contentType=InputTypes.BoolType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LSTMCell Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LSTMCell',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Embedding Layers
    ##########################################
    ###########################
    #  Embdedding Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Embdedding',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('input_dim',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('output_dim',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('embeddings_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('embeddings_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('embdeddings_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('mask_zero',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('input_length',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Advanced Activation Layers
    ##########################################
    ###########################
    #  LeakyRelU Layers: Leaky version of a Rectified Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('LeakyRelU',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('alpha',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  PReLU Layers: Parametric Rectified Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('PReLU',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('alpha_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('alpha_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('alpha_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('shared_axes',contentType=InputTypes.FloatListType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ELU Layers: Exponential Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('ELU',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('alpha',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ThresholdedReLU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ThresholdedReLU',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('theta',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Softmax Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Softmax',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('axis',contentType=InputTypes.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ReLU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ReLU',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('max_value',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('negative_slope',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('threshold',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Normalization Layers
    ##########################################
    ###########################
    #  BatchNormalization Layers
    ###########################
    layerInput = InputData.parameterInputFactory('BatchNormalization',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('axis',contentType=InputTypes.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('momentum',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('epsilon',contentType=InputTypes.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('center',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('scale',contentType=InputTypes.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('beta_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('gamma_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('moving_mean_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('moving_variance_initializer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('beta_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('gamma_regularizer',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('beta_constraint',contentType=InputTypes.StringType))
    layerInput.addSub(InputData.parameterInputFactory('gamma_constraint',contentType=InputTypes.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Noise Layers
    ##########################################
    ###########################
    #  GausianNoise Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GaussianNoise',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'stddev' need to be popped out and the value will be passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('stddev',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GausianDropout Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GaussianDropout',contentType=InputTypes.StringType)
    layerInput.addParam('name', InputTypes.StringType, True)
    # 'stddev' need to be popped out and the value will be passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    #################################################

    layerLayoutInput = InputData.parameterInputFactory('layer_layout',contentType=InputTypes.StringListType, default='no-default')
    inputSpecification.addSub(layerLayoutInput)

    #Estimators can include ROMs, and so because baseNode does a copy, this
    #needs to be after the rest of ROMInput is defined.
    EstimatorInput = InputData.parameterInputFactory('estimator', contentType=InputTypes.StringType, baseNode=inputSpecification)
    EstimatorInput.addParam("estimatorType", InputTypes.StringType, False)
    #The next lines are to make subType and name not required.
    EstimatorInput.addParam("subType", InputTypes.StringType, False)
    EstimatorInput.addParam("name", InputTypes.StringType, False)
    inputSpecification.addSub(EstimatorInput)

    # inputs for cross validations
    cvInput = InputData.parameterInputFactory("CV", contentType=InputTypes.StringType)
    cvInput.addParam("class", InputTypes.StringType)
    cvInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(cvInput)

    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a keras deep neural network object
      @ In, None
      @ Out, None
    """
    super().__init__()
    # Dictionary of Keras Neural Network Core layers
    self.kerasDict = {}

    self.kerasDict['kerasCoreLayersList'] = ['dense',
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
    self.kerasDict['kerasConvNetLayersList'] = ['conv1d',
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
    self.kerasDict['kerasPoolingLayersList'] = ['maxpooling1d',
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
    self.kerasDict['kerasRcurrentLayersList'] = ['rnn',
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
    self.kerasDict['kerasLocallyConnectedLayersList'] = ['locallyconnected1d',
                                            'locallyconnected2d']
    # list of Keras Neural Network Embedding layers
    self.kerasDict['kerasEmbeddingLayersList'] = ['embedding']
    # list of Keras Neural Network Advanced Activation layers
    self.kerasDict['kerasAdvancedActivationLayersList'] = ['leakyrelu',
                                              'prelu',
                                              'elu',
                                              'thresholdedrelu',
                                              'softmax',
                                              'relu']
    # list of Keras Neural Network Normalization layers
    self.kerasDict['kerasNormalizationLayersList'] = ['batchnormalization']
    # list of Keras Neural Network Noise layers
    self.kerasDict['kerasNoiseLayersList'] = ['gaussiannoise',
                                 'gaussiandropout',
                                 'alphadropout']
    self.initializationOptionDict['KerasROMDict'] = self.kerasDict

    self.kerasLayersList = functools.reduce(lambda x,y: x+y, list(self.kerasDict.values()))

    self.kerasROMsList = ['KerasMLPClassifier', 'KerasConvNetClassifier', 'KerasLSTMClassifier', 'KerasLSTMRegression']

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

    self.printTag = 'KerasBase'
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
    self.initOptionDict = {}


  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    nodes, notFound = paramInput.findNodesAndExtractValues(['random_seed', 'num_classes', 'validation_split', 'plot_model',
                                                            'output_layer_activation', 'loss', 'metrics', 'batch_size',
                                                            'layer_layout'])
    assert(not notFound)
    randomSeed = nodes.get('random_seed')
    # Set the seed for random number generation to obtain reproducible results
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    if randomSeed is not None:
      # The below is necessary for starting Numpy generated random numbers
      # in a well-defined initial state.
      np.random.seed(randomSeed)
      # The below is necessary for starting core Python generated random numbers
      # in a well-defined state.
      rn.seed(randomSeed)
      # The below tf.random.set_seed() will make random number generation
      # in the TensorFlow backend have a well-defined initial state.
      # For further details, see:
      # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
      tf.random.set_seed(randomSeed)

    modelName = paramInput.parameterValues['name']
    # number of classes for classifier
    self.numClasses = nodes.get('num_classes')
    # validation split, default to 0.25
    self.validationSplit = nodes.get('validation_split')
    # options to plot deep neural network model, default False
    self.plotModel = nodes.get('plot_model')
    self.plotModelFilename = self.printTag + "_model.png" if not modelName else modelName + "_model.png"
    # activation function for output layer of deep neural network
    self.outputLayerActivation = nodes.get('output_layer_activation')
    # A loss function that is always required to compile a KERAS model
    self.lossFunction = nodes.get('loss')
    # a metric is a function that is used to judge the performance of KERAS model
    self.metrics = nodes.get('metrics')
    # number of samples per gradient update, default 20
    self.batchSize = nodes.get('batch_size')
    # number of epochs to train the model. An epoch is an iteration over the entire training data, (default 20)
    self.epochs = nodes.get('epochs')
    # extract settings for optimizer
    optNode = paramInput.findFirst('optimizerSetting')
    optimizerSetting = {}
    if optNode is None:
      optimizerSetting = {'optimizer':'adam'}
    else:
      for sub in optNode:
        optimizerSetting[sub.getName()] = sub.values
    optimizerName = optimizerSetting.pop('optimizer').lower()
    # set up optimizer
    self.optimizer = self.__class__.availOptimizer[optimizerName](**optimizerSetting)
    # check layer layout, this is always required node, used to build the DNNs
    self.layerLayout = nodes.get('layer_layout')
    for sub in paramInput:
      if sub.getName().lower() in self.kerasLayersList:
        layerName = sub.parameterValues['name']
        self.initOptionDict[layerName] = {}
        self.initOptionDict[layerName]['type'] = sub.getName().lower()
        for node in sub.subparts:
          self.initOptionDict[layerName][node.getName()] = node.value

    if not set(self.layerLayout).issubset(list(self.initOptionDict.keys())):
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
    pass

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    state = supervisedLearning.__getstate__(self)
    tf.keras.models.save_model(self._ROM, KerasBase.tempModelFile, save_format='h5')
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
    writeTo.addScalar('Accuracy',"Training",' '.join([str(elm) for elm in self._romHistory.history['accuracy']]))
    writeTo.addScalar('Accuracy',"Testing",' '.join([str(elm) for elm in self._romHistory.history['val_accuracy']]))
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
    params = copy.deepcopy(self.__dict__)
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
