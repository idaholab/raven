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
from sklearn import preprocessing
import random as rn
import matplotlib
import platform
import functools
from scipy import stats
import os
from ..utils import importerUtils
from ..utils import InputData, InputTypes
tf = importerUtils.importModuleLazyRenamed("tf", globals(), "tensorflow")
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
    # general xml nodes
    inputSpecification.addSub(InputData.parameterInputFactory("metrics", contentType=InputTypes.StringListType,
        descr=r"""list of metrics to be evaluated by
        the model during training and testing. available metrics include
        \textit{binary\_accuracy}, \textit{categorical\_accuracy}, \textit{sparse\_categorical\_accuracy},
        \textit{top\_k\_categorical\_accuracy}, \textit{sparse\_top\_k\_categorical\_accuracy}.""", default=['accuracy']))
    inputSpecification.addSub(InputData.parameterInputFactory("batch_size", contentType=InputTypes.IntegerType,
        descr=r"""number of samples per gradient update.""", default=20))
    inputSpecification.addSub(InputData.parameterInputFactory("epochs", contentType=InputTypes.IntegerType,
        descr=r"""number of epochs to train the model. An epoch
        is an iteration over the entire training data.""", default=20))
    inputSpecification.addSub(InputData.parameterInputFactory("random_seed", contentType=InputTypes.IntegerType,
        descr=r"""a integer to use as random seed.""",  default=None))
    inputSpecification.addSub(InputData.parameterInputFactory("plot_model", contentType=InputTypes.BoolType,
        descr=r"""if true the DNN model constructed by RAVEN will be
        plotted and stored in the working directory. The file name will be \textit{"ROM name" + "\_" + "model.png"}.
        \nb This capability requires the following libraries, i.e. pydot-ng and graphviz to be installed.""", default=False))
    inputSpecification.addSub(InputData.parameterInputFactory("num_classes",contentType= InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of given classifier.""", default=1))
    inputSpecification.addSub(InputData.parameterInputFactory("validation_split", contentType=InputTypes.FloatType,
        descr=r"""float between 0 and 1, the fraction of the training data to
        be used as validation data.""", default=0.25))
    inputSpecification.addSub(InputData.parameterInputFactory("output_layer_activation", contentType=InputTypes.StringType,
        descr=r"""activation function for output layer of deep neural network""", default='softmax'))
    inputSpecification.addSub(InputData.parameterInputFactory("loss", contentType=InputTypes.StringType,
        descr=r"""if the model has multiple outputs, you can use a different
        loss metric on each output by passing a list of loss metrics. The value that will be minimized by the model will then
        be the sum of all individual value from each loss metric. Available loss functions include \textit{mean\_squared\_error},
        \textit{mean\_absolute\_error}, \textit{mean\_absolute\_percentage\_error}, \textit{mean\_squared\_logarithmic\_error},
        \textit{squared\_hinge}, \textit{hinge}, \textit{categorical\_hinge}, \textit{logcosh}, \textit{categorical\_crossentropy},
        \textit{sparse\_categorical\_crossentropy}, \textit{binary\_crossentropy}, \textit{kullback\_leibler\_divergence},
        \textit{poisson}, \textit{cosine\_proximity}.""", default='categorical_crossentropy'))
    # Keras optimizer parameters
    OptimizerSettingInput = InputData.parameterInputFactory('optimizerSetting', contentType=InputTypes.StringType,
        descr=r"""The settings for optimizer""", default=None)
    Beta1Input = InputData.parameterInputFactory('beta_1', contentType=InputTypes.FloatType,
        descr=r"""$0 < beta < 1$. Generally close to 1.""", default=0.9)
    Beta2Input = InputData.parameterInputFactory('beta_2', contentType=InputTypes.FloatType,
        descr=r"""$0 < beta < 1$. Generally close to 1.""", default=0.999)
    DecayInput = InputData.parameterInputFactory('decay', contentType=InputTypes.FloatType,
        descr=r"""learning rate decay over each update.""", default=0.0)
    LRInput = InputData.parameterInputFactory('lr', contentType=InputTypes.FloatType,
        descr=r"""learning rate.""", default=0.001)
    OptimizerInput = InputData.parameterInputFactory('optimizer', contentType=InputTypes.StringType,
        descr=r"""name of optimizer.""", default='Adam')
    EpsilonInput = InputData.parameterInputFactory('epsilon', contentType=InputTypes.FloatType,
        descr=r"""fuzz factor.""", default=None)
    MomentumInput = InputData.parameterInputFactory('momentum', contentType=InputTypes.FloatType,
        descr=r"""Parameter that accelerates SGD in
        the relevant direction and dampens oscillations.""", default=0.0)
    NesterovInput = InputData.parameterInputFactory('nesterov', contentType=InputTypes.BoolType,
        descr=r"""whether to apply Nesterov momentum""", default=False)
    RhoInput = InputData.parameterInputFactory('rho', contentType=InputTypes.FloatType,
        descr=r"""$> 0$""", default=0.95)
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
    layerInput = InputData.parameterInputFactory('Dense',contentType=InputTypes.StringType,
        descr=r"""regular densely-connected neural network layer.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""name of this layer. The value will be
        used in \xmlNode{layer\_layout} to construct the fully connected neural network.""")
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""including {`relu', `tanh', `elu', `selu', `softplus', `softsign', `sigmoid', `hard\_sigmoid', `linear', `softmax'}.
        (see~\ref{activationsDNN})"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector.""", default=True))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the kernel weights
        matrix (see~\ref{constraintsDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the bias vector
        (see ~\ref{constraintsDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.StringType,
        descr=r"""dimensionality of the output space of this layer, required except if this layer is used as the last output layer"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Activation Layers: applies an activation function to an output
    ###########################
    layerInput = InputData.parameterInputFactory('Activation',contentType=InputTypes.StringType,
        descr=r"""activation layer""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""name of the layer""")
    # 'activation' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""including {`relu', `tanh', `elu', `selu', `softplus', `softsign', `sigmoid', `hard\_sigmoid', `linear', `softmax'}.
        (see~\ref{activationsDNN}).""", default='linear'))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Dropout Layers: Applies Dropout to the input
    ###########################
    layerInput = InputData.parameterInputFactory('Dropout',contentType=InputTypes.StringType,
        descr=r"""The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during
        training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by $1/(1 - rate)$
        such that the sum over all inputs is unchanged.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the input units to drop."""))
    layerInput.addSub(InputData.parameterInputFactory('noise_shape',contentType=InputTypes.IntegerTupleType,
        descr=r"""1D integer tensor representing the shape of the binary dropout mask that will be multiplied with
        the input. For instance, if your inputs have shape (batch_size, timesteps, features) and you want the
        dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features)."""))
    layerInput.addSub(InputData.parameterInputFactory('seed',contentType=InputTypes.IntegerType,
        descr=r"""A Python integer to use as random seed."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Flatten Layers: Flattens the input
    ###########################
    layerInput = InputData.parameterInputFactory('Flatten',contentType=InputTypes.StringType,
        descr=r"""Flattens the input. Does not affect the batch size.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=InputTypes.StringType,
        descr=r"""The ordering of the dimensions in the inputs. channels\_last corresponds to inputs with shape
        (batch, ..., channels) while channels_first corresponds to inputs with shape (batch, channels, ...).""",
        default='channels_last'))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Input Layers: Input() is used to instantiate a Keras tensor
    ###########################
    layerInput = InputData.parameterInputFactory('Input',contentType=InputTypes.StringType,
        descr=r"""is used to instantiate a Keras tensor.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Reshape Layers: Reshapes an output to a certain shape
    ###########################
    layerInput = InputData.parameterInputFactory('Reshape',contentType=InputTypes.StringType,
        descr=r"""Layer that reshapes inputs into the given shape.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'target_shape' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('target_shape',contentType=InputTypes.IntegerTupleType,
        descr=r"""the target shape after reshapping"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Permute Layers: permutes the dimensions of the input according to a given pattern
    ###########################
    layerInput = InputData.parameterInputFactory('Permute',contentType=InputTypes.StringType,
        descr=r""" """)
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'permute_pattern' need to pop out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('permute_pattern',contentType=InputTypes.IntegerTupleType,
        descr=r"""Permutes the dimensions of the input according to a given pattern."""))
    layerInput.addSub(InputData.parameterInputFactory('input_shape',contentType=InputTypes.IntegerTupleType,
        descr=r"""the input shape"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  RepeatVector Layers: repeats the input n times
    ###########################
    layerInput = InputData.parameterInputFactory('RepeatVector',contentType=InputTypes.StringType,
        descr=r"""Repeats the input n times.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'repetition_factor' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('repetition_factor',contentType=InputTypes.IntegerType,
        descr=r"""the number of times to repeat the input"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Lambda Layers: Wraps arbitrary expression as a Layer object
    ###########################
    layerInput = InputData.parameterInputFactory('Lambda',contentType=InputTypes.StringType,strictMode=False,
        descr=r"""Wraps arbitrary expressions as a Layer object.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # A function object need to be created and passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('function',contentType=InputTypes.StringType,
        descr=r"""The function to be evaluated. Takes input tensor as first argument."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ActivityRegularization Layers: applies an update to the cost function based input activity
    ###########################
    layerInput = InputData.parameterInputFactory('ActivityRegularization',contentType=InputTypes.StringType,
        descr=r"""Layer that applies an update to the cost function based input activity.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('l1',contentType=InputTypes.FloatType,
        descr=r"""L1 regularization factor (positive float)."""))
    layerInput.addSub(InputData.parameterInputFactory('l2',contentType=InputTypes.FloatType,
        descr=r"""L2 regularization factor (positive float)."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Masking Layers: Masks a sequence by using a mask value to skip timesteps
    ###########################
    layerInput = InputData.parameterInputFactory('Masking',contentType=InputTypes.StringType,
        descr=r"""Masks a sequence by using a mask value to skip timesteps.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('mask_value',contentType=InputTypes.FloatType,
        descr=r"""For each timestep in the input tensor (dimension 1 in the tensor), if all values in the input
        tensor at that timestep are equal to mask\_value, then the timestep will be masked (skipped) in all
        downstream layers (as long as they support masking)."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout1D Layers: Spatial 1D version of Dropout
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout1D',contentType=InputTypes.StringType,
        descr=r"""Spatial 1D version of Dropout. This version performs the same function as Dropout, however,
        it drops entire 1D feature maps instead of individual elements. If adjacent frames within feature
        maps are strongly correlated (as is normally the case in early convolution layers) then regular
        dropout will not regularize the activations and will otherwise just result in an effective learning
        rate decrease. In this case, SpatialDropout1D will help promote independence between feature
        maps and should be used instead.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the input units to drop."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout2D',contentType=InputTypes.StringType,
        descr=r"""Spatial 2D version of Dropout. This version performs the same function as Dropout, however,
        it drops entire 2D feature maps instead of individual elements. If adjacent pixels within feature maps
        are strongly correlated (as is normally the case in early convolution layers) then regular dropout will
        not regularize the activations and will otherwise just result in an effective learning rate decrease.
        In this case, SpatialDropout2D will help promote independence between feature maps and should
        be used instead.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the input units to drop."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""`channels\_first' or `channels\_last'. In `channels_first' mode, the channels dimension
        (the depth) is at index 1, in `channels\_last' mode is it at index 3. """, default='channels_last'))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout3D',contentType=InputTypes.StringType,
        descr=r"""Spatial 3D version of Dropout. This version performs the same function as Dropout, however,
        it drops entire 3D feature maps instead of individual elements. If adjacent voxels within feature maps
        are strongly correlated (as is normally the case in early convolution layers) then regular dropout will
        not regularize the activations and will otherwise just result in an effective learning rate decrease.
        In this case, SpatialDropout3D will help promote independence between feature maps and should
        be used instead.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the input units to drop."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""`channels\_first' or `channels\_last'. In `channels\_first' mode, the channels dimension (the depth)
        is at index 1, in `channels\_last' mode is it at index 4.""", default='channels_last'))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###################################
    # Convolutional Layers
    ###################################
    ###########################
    #  Conv1D Layers: 1D convolutioanl layer (e.g. temporal convolutional)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv1D',contentType=InputTypes.StringType,
        descr=r"""1D convolution layer (e.g. temporal convolution). This layer creates a convolution kernel that
        is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs.
        If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None,
        it is applied to the outputs as well.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""the dimensionality of the output space (i.e. the number of output filters in the convolution)."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of a single integer, specifying the length of the 1D convolution window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of a single integer, specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""One of ``valid'', ``same'' or ``causal'' (case-insensitive). ``valid'' means no padding.
        ``same'' results in padding with zeros evenly to the left/right or up/down of the input such that
        output has the same height/width dimension as the input. ``causal'' results in causal (dilated) convolutions,
        e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should
        not violate the temporal order."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first"""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated
        convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any
        strides value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""egularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the kernel matrix"""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv2D Layers: 2D convolutioanl layer (e.g. spatial convolution over images)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv2D',contentType=InputTypes.StringType,
        descr=r"""2D convolution layer (e.g. spatial convolution over images). This layer creates a convolution kernel
        that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector
        is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs
        as well.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""the dimensionality of the output space (i.e. the number of output filters in the convolution)."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
        and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels_first""", default='channels_last'))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any
        dilation_rate value != 1 is incompatible with specifying any stride value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""Regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the kernel matrix"""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv3D Layers: 3D convolutioanl layer (e.g. spatial convolution over volumes)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv3D',contentType=InputTypes.StringType,
        descr=r"""3D convolution layer (e.g. spatial convolution over volumes). This layer creates a convolution
        kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a
        bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to
        the outputs as well.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""the dimensionality of the output space (i.e. the number of output filters in the convolution)."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 3 integers, specifying the height and width of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height
        and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels_first""", default='channels_last'))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any
        dilation_rate value != 1 is incompatible with specifying any stride value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the kernel matrix"""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SeparableConv1D Layers: Depthwise separable 1D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('SeparableConv1D',contentType=InputTypes.StringType,
        descr=r"""Depthwise separable 1D convolution. This layer performs a depthwise convolution that
        acts separately on channels, followed by a pointwise convolution that mixes channels. If use_bias
        is True and a bias initializer is provided, it adds a bias vector to the output. It then optionally
        applies an activation function to produce the final output.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""the dimensionality of the output space (i.e. the number of output filters in the convolution)."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of a single integer, specifying the length of the 1D convolution window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of a single integer, specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""One of ``valid'', ``same'' or ``causal'' (case-insensitive). ``valid'' means no padding.
        ``same'' results in padding with zeros evenly to the left/right or up/down of the input such that
        output has the same height/width dimension as the input. ``causal'' results in causal (dilated) convolutions,
        e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should
        not violate the temporal order."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels_first"""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated
        convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any
        strides value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the kernel matrix"""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputTypes.IntegerType,
        descr=r"""The number of depthwise convolution output channels for each input channel. The total number of
        depthwise convolution output channels will be equal to num_filters_in * depth_multiplier."""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputTypes.StringType,
        descr=r"""An initializer for the depthwise convolution kernel""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_initializer',contentType=InputTypes.StringType,
        descr=r"""An initializer for the pointwise convolution kernel.""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_regularizer',contentType=InputTypes.StringType,
        descr=r"""Optional regularizer for the pointwise convolution kernel"""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputTypes.StringType,
        descr=r"""Optional regularizer for the depthwise convolution kernel"""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputTypes.StringType,
        descr=r"""Optional projection function to be applied to the depthwise kernel after being updated by an Optimizer
        (e.g. used for norm constraints or value constraints for layer weights). The function must take as input the
        unprojected variable and must return the projected variable (which must have the same shape).
        Constraints are not safe to use when doing asynchronous distributed training"""))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_constraint',contentType=InputTypes.StringType,
        descr=r"""Optional projection function to be applied to the pointwise kernel after being updated by an Optimizer"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SeparableConv2D Layers: Depthwise separable 2D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('SeparableConv2D',contentType=InputTypes.StringType,
        descr=r"""Depthwise separable 2D convolution. Separable convolutions consist of first performing a depthwise
        spatial convolution (which acts on each input channel separately) followed by a pointwise convolution
        which mixes the resulting output channels. The depth_multiplier argument controls how many output channels
        are generated per input channel in the depthwise step. Intuitively, separable convolutions can be understood
        as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an
        Inception block.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""the dimensionality of the output space (i.e. the number of output filters in the convolution)."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integer, specifying the length of the 1D convolution window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integer, specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels_first"""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of 2 integer, specifying the dilation rate to use for dilated
        convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any
        strides value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the kernel matrix"""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputTypes.IntegerType,
        descr=r"""The number of depthwise convolution output channels for each input channel. The total number of
        depthwise convolution output channels will be equal to num_filters_in * depth_multiplier."""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputTypes.StringType,
        descr=r"""An initializer for the depthwise convolution kernel""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_initializer',contentType=InputTypes.StringType,
        descr=r"""An initializer for the pointwise convolution kernel.""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_regularizer',contentType=InputTypes.StringType,
        descr=r"""Optional regularizer for the pointwise convolution kernel"""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputTypes.StringType,
        descr=r"""Optional regularizer for the depthwise convolution kernel"""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputTypes.StringType,
        descr=r"""Optional projection function to be applied to the depthwise kernel after being updated by an Optimizer
        (e.g. used for norm constraints or value constraints for layer weights). The function must take as input the
        unprojected variable and must return the projected variable (which must have the same shape).
        Constraints are not safe to use when doing asynchronous distributed training"""))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_constraint',contentType=InputTypes.StringType,
        descr=r"""Optional projection function to be applied to the pointwise kernel after being updated by an Optimizer"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  DepthwiseConv2D Layers: Depthwise separable 2D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('DepthwiseConv2D',contentType=InputTypes.StringType,
        descr=r"""Depthwise 2D convolution. Depthwise convolution is a type of convolution in which a single
        convolutional filter is apply to each input channel (i.e. in a depthwise way). You can understand depthwise
        convolution as being the first step in a depthwise separable convolution.
        It is implemented via the following steps:
        \begin{itemize}
          \item Split the input into individual channels.
          \item Convolve each input with the layer's kernel (called a depthwise kernel).
          \item Stack the convolved outputs together (along the channels axis).
        \end{itemize}
        Unlike a regular 2D convolution, depthwise convolution does not mix information across different input channels.
        The depth_multiplier argument controls how many output channels are generated per input channel in the
        depthwise step.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
        and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first"""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of 2 integer, specifying the dilation rate to use for dilated
        convolution. Currently, specifying any dilation\_rate value != 1 is incompatible with specifying any
        strides value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputTypes.IntegerType,
        descr=r"""The number of depthwise convolution output channels for each input channel. The total number of
        depthwise convolution output channels will be equal to num_filters_in * depth_multiplier."""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputTypes.StringType,
        descr=r"""An initializer for the depthwise convolution kernel""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputTypes.StringType,
        descr=r"""Optional regularizer for the depthwise convolution kernel"""))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputTypes.StringType,
        descr=r"""Optional projection function to be applied to the depthwise kernel after being updated by an Optimizer
        (e.g. used for norm constraints or value constraints for layer weights). The function must take as input the
        unprojected variable and must return the projected variable (which must have the same shape).
        Constraints are not safe to use when doing asynchronous distributed training"""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
        Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv2DTranspose Layers: Transposed convolution layer (sometimes called Deconvolution)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv2DTranspose',contentType=InputTypes.StringType,
        descr=r"""Transposed convolution layer (sometimes called Deconvolution). The need for transposed convolutions
        generally arises from the desire to use a transformation going in the opposite direction of a normal
        convolution, i.e., from something that has the shape of the output of some convolution to something that
        has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""the dimensionality of the output space (i.e. the number of output filters in the convolution)."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integer, specifying the length of the 1D convolution window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integer, specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first"""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of 2 integer, specifying the dilation rate to use for dilated
        convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any
        strides value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the kernel matrix"""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    layerInput.addSub(InputData.parameterInputFactory('output_padding',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the amount of padding along the height and
        width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions.
        The amount of output padding along a given dimension must be lower than the stride along that same dimension."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv3DTranspose Layers: Transposed convolution layer (sometimes called Deconvolution)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv3DTranspose',contentType=InputTypes.StringType,
        descr=r""" """)
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""the dimensionality of the output space (i.e. the number of output filters in the convolution)."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 3 integer, specifying the length of the 1D convolution window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 3 integer, specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first"""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""an integer or tuple/list of 3 integer, specifying the dilation rate to use for dilated
        convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any
        strides value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied"""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the kernel matrix"""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the bias vector"""))
    layerInput.addSub(InputData.parameterInputFactory('output_padding',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the amount of padding along the height and
        width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions.
        The amount of output padding along a given dimension must be lower than the stride along that same dimension."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    #  Cropping1D  Layers: cropping layer for 1D input (e.g. temporal sequence)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping1D',contentType=InputTypes.StringType,
        descr=r"""Cropping layer for 1D input (e.g. temporal sequence).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Int or tuple of int (length 2) How many units should be trimmed off at the beginning and end of the
        cropping dimension (axis 1). If a single int is provided, the same value will be used for both."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Cropping2D  Layers: cropping layer for 2D input (e.g. picutures)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping2D',contentType=InputTypes.StringType,
        descr=r"""Cropping layer for 2D input (e.g. picture).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        \begin{itemize}
          \item If int: the same symmetric cropping is applied to height and width.
          \item If tuple of 2 ints: interpreted as two different symmetric cropping values for height and width:
            (symmetric\_height\_crop, symmetric\_width\_crop).
          \item If tuple of 2 tuples of 2 ints: interpreted as ((top\_crop, bottom\_crop), (left\_crop, right\_crop))
        \end{itemize}
        """))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Cropping3D  Layers: cropping layer for 2D input (e.g. picutures)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping3D',contentType=InputTypes.StringType,
        descr=r"""Cropping layer for 3D data (e.g. spatial or spatio-temporal).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Int, or tuple of 3 ints, or tuple of 3 tuples of 3 ints.
        \begin{itemize}
          \item If int: the same symmetric cropping is applied to height and width.
          \item If tuple of 3 ints: interpreted as two different symmetric cropping values for depth, height and width:
            (symmetric\_dim1\_crop, symmetric\_dim2\_crop, symmetric\_dim3\_crop).
          \item If tuple of 3 tuples of 2 ints: interpreted as ((top\_crop, bottom\_crop), (left\_crop, right\_crop))
        \end{itemize}"""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Upsampling1D',contentType=InputTypes.StringType,
        descr=r"""Upsampling layer for 1D inputs.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputTypes.IntegerType,
        descr=r"""Upsampling factor."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('UpSampling2D',contentType=InputTypes.StringType,
        descr=r"""Upsampling layer for 2D inputs.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Upsampling factor."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    layerInput.addSub(InputData.parameterInputFactory('interpolation',contentType=interpolationEnumType,
        descr=r"""A string, one of nearest or bilinear."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('UpSampling3D',contentType=InputTypes.StringType,
        descr=r"""Upsampling layer for 3D inputs.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r""" Int, or tuple of 3 integers. The upsampling factors for dim1, dim2 and dim3."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding1D',contentType=InputTypes.StringType,
        descr=r"""Zero-padding layer for 1D input (e.g. temporal sequence).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Int, or tuple of int (length 2), or dictionary. - If int: How many zeros to add at the beginning
        and end of the padding dimension (axis 1). - If tuple of int (length 2): How many zeros to add at the beginning
        and the end of the padding dimension"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding2D',contentType=InputTypes.StringType,
        descr=r"""Zero-padding layer for 2D input (e.g. picture).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        \begin{itemize}
          \item If int: the same symmetric padding is applied to height and width.
          \item If tuple of 2 ints: interpreted as two different symmetric padding values for height and width:
          (symmetric\_height\_pad, symmetric\_width\_pad).
          \item If tuple of 2 tuples of 2 ints: interpreted as ((top\_pad, bottom\_pad), (left\_pad, right\_pad))
        \end{itemize}"""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding3D',contentType=InputTypes.StringType,
        descr=r"""Zero-padding layer for 3D data (spatial or spatio-temporal).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
        \begin{itemize}
          \item If int: the same symmetric padding is applied to height and width.
          \item If tuple of 3 ints: interpreted as two different symmetric padding values for height and width:
          (symmetric\_dim1\_pad, symmetric\_dim2\_pad, symmetric\_dim3\_pad).
          \item If tuple of 3 tuples of 2 ints: interpreted as ((left\_dim1\_pad, right\_dim1\_pad),
          (left\_dim2\_pad, right\_dim2\_pad), (left\_dim3\_pad, right\_dim3\_pad))
        \end{itemize}"""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ############################################
    #   Pooling Layers
    ############################################
    ###########################
    #  MaxPooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling1D',contentType=InputTypes.StringType,
        descr=r"""Max pooling operation for 1D temporal data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerType,
        descr=r"""size of the max pooling window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerType,
        descr=r"""Specifies how much the pooling window moves for each pooling step."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  MaxPooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling2D',contentType=InputTypes.StringType,
        descr=r"""Max pooling operation for 2D spatial data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max
        value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for
        both dimensions."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Integer, tuple of 2 integers, or None. Strides values. Specifies how far the pooling window moves
        for each pooling step. If None, it will default to pool\_size."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  MaxPooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling3D',contentType=InputTypes.StringType,
        descr=r"""Max pooling operation for 3D data (spatial or spatio-temporal).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerTupleType,
        descr=r"""Tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the
        size of the 3D input in each dimension."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerTupleType,
        descr=r"""tuple of 3 integers, or None. Strides values."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling1D',contentType=InputTypes.StringType,
        descr=r"""Average pooling for temporal data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerType,
        descr=r"""size of the average pooling windows."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerType,
        descr=r"""Factor by which to downscale."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling2D',contentType=InputTypes.StringType,
        descr=r"""Average pooling operation for spatial data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will
        halve the input in both spatial dimension. If only one integer is specified, the same window length will
        be used for both dimensions."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool\_size."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling3D',contentType=InputTypes.StringType,
        descr=r"""Average pooling operation for 3D data (spatial or spatio-temporal).""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""tuple of 3 integers, factors by which to downscale (dim1, dim2, dim3). (2, 2, 2) will halve the size
        of the 3D input in each dimension."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""tuple of 3 integers, or None. Strides values."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""one of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        with zeros evenly to the left/right or up/down of the input such that output has the same height/width
        dimension as the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling1D',contentType=InputTypes.StringType,
        descr=r"""Global max pooling operation for 1D temporal data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling1D',contentType=InputTypes.StringType,
        descr=r"""Global average pooling operation for temporal data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling2D',contentType=InputTypes.StringType,
        descr=r"""Global max pooling operation for spatial data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling2D',contentType=InputTypes.StringType,
        descr=r"""Global average pooling operation for spatial data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling3D',contentType=InputTypes.StringType,
        descr=r"""Global Max pooling operation for 3D data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling3D',contentType=InputTypes.StringType,
        descr=r"""Global Average pooling operation for 3D data.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)

    ######################################
    #   Locally-connected Layers
    ######################################
    ###########################
    #  LocallyConnected1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LocallyConnected1D',contentType=InputTypes.StringType,
        descr=r"""Locally-connected layer for 1D inputs.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of this layer"""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of a single integer, specifying the length of the 1D convolution window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of a single integer, specifying the stride length of the convolution."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""Currently only supports ``valid'' (case-insensitive). ``same'' may be supported in the future.
        ``valid'' means no padding."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
        activation: $a(x) = x)$."""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LocallyConnected2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LocallyConnected2D',contentType=InputTypes.StringType,
        descr=r"""Locally-connected layer for 2D inputs.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of this layer"""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of 2 integers, specifying the strides of the convolution along the width and
        height. Can be a single integer to specify the same value for all spatial dimensions."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r"""Currently only support ``valid'' (case-insensitive)."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
        activation: $a(x) = x)$."""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ######################################
    #  Recurrent Layers
    ######################################
    ###########################
    #  RNN Layers
    ###########################
    layerInput = InputData.parameterInputFactory('RNN',contentType=InputTypes.StringType,
        descr=r"""Fully-connected RNN where the output is to be fed back to input.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of this layer"""))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last output in the output sequence, or the full sequence.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last state in addition to the output. """, default=False))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType,
        descr=r"""If True, process the input sequence backwards and return the reversed sequence."""))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType,
        descr=r"""If True, the last state for each sample at index i in a batch will be used as initial state for the
        sample of index i in the following batch.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType,
        descr=r"""If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a
        RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.""",
        default=False))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SimpleRNN Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SimpleRNN',contentType=InputTypes.StringType,
        descr=r"""Fully-connected RNN where the output is to be fed back to input.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of this layer"""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
        activation: $a(x) = x)$."""))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the recurrent\_kernel weights matrix, used for the linear transformation of the
        recurrent state (see~\ref{initializersDNN}).""", default='orthogonal'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType,
        descr=r"""Regularizer function applied to the recurrent\_kernel weights matrix(see ~\ref{regularizersDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the recurrent\_kernel weights matrix(see~\ref{constraintsDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.""",
        default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
        recurrent state.""", default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last output in the output sequence, or the full sequence.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last state in addition to the output. """, default=False))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType,
        descr=r"""If True, process the input sequence backwards and return the reversed sequence."""))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType,
        descr=r"""If True, the last state for each sample at index i in a batch will be used as initial state for the
        sample of index i in the following batch.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType,
        descr=r"""If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a
        RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.""",
        default=False))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GRU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GRU',contentType=InputTypes.StringType,
        descr=r"""Gated Recurrent Unit.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of this layer"""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
        activation: $a(x) = x)$."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use for the recurrent step.""", default='sigmoid'))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the recurrent\_kernel weights matrix, used for the linear transformation of the
        recurrent state (see~\ref{initializersDNN}).""", default='orthogonal'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType,
        descr=r"""Regularizer function applied to the recurrent\_kernel weights matrix(see ~\ref{regularizersDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the recurrent\_kernel weights matrix(see~\ref{constraintsDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.""",
        default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
        recurrent state.""", default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last output in the output sequence, or the full sequence.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last state in addition to the output. """, default=False))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType,
        descr=r"""If True, process the input sequence backwards and return the reversed sequence."""))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType,
        descr=r"""If True, the last state for each sample at index i in a batch will be used as initial state for the
        sample of index i in the following batch.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType,
        descr=r"""If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a
        RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.""",
        default=False))
    layerInput.addSub(InputData.parameterInputFactory('reset_after',contentType=InputTypes.BoolType,
        descr=r"""GRU convention (whether to apply reset gate after or before matrix multiplication).
        False = ``before'', True = ``after'' (default and CuDNN compatible)."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LSTM Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LSTM',contentType=InputTypes.StringType,
        descr=r"""Long Short-Term Memory layer""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of this layer"""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
        activation: $a(x) = x)$."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use for the recurrent step.""", default='sigmoid'))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the recurrent\_kernel weights matrix, used for the linear transformation of the
        recurrent state (see~\ref{initializersDNN}).""", default='orthogonal'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType,
        descr=r"""Regularizer function applied to the recurrent\_kernel weights matrix(see ~\ref{regularizersDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the recurrent\_kernel weights matrix(see~\ref{constraintsDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.""",
        default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
        recurrent state.""", default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last output in the output sequence, or the full sequence.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last state in addition to the output. """, default=False))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType,
        descr=r"""If True, process the input sequence backwards and return the reversed sequence."""))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType,
        descr=r"""If True, the last state for each sample at index i in a batch will be used as initial state for the
        sample of index i in the following batch.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType,
        descr=r"""If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a
        RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.""",
        default=False))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputTypes.BoolType,
        descr=r"""If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also
        force bias\_initializer=``zeros''.""", default=True))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ConvLSTM2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ConvLSTM2D',contentType=InputTypes.StringType,
        descr=r"""2D Convolutional LSTM. Similar to an LSTM layer, but the input transformations and recurrent
        transformations are both convolutional.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
        descr=r"""dimensionality of the output space of this layer"""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of n integers, specifying the dimensions of the convolution window."""))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of n integers, specifying the strides of the convolution. Specifying any
        stride value != 1 is incompatible with specifying any dilation\_rate value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType,
        descr=r""" One of ``valid'' or ``same'' (case-insensitive). ``valid'' means no padding. ``same'' results in padding
        evenly to the left/right or up/down of the input such that output has the same height/width dimension as
        the input."""))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType,
        descr=r"""A string, one of channels\_last (default) or channels\_first."""))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputTypes.IntegerOrIntegerTupleType,
        descr=r"""An integer or tuple/list of n integers, specifying the dilation rate to use for dilated convolution.
        Currently, specifying any dilation\_rate value != 1 is incompatible with specifying any strides value != 1."""))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
        activation: $a(x) = x)$."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType,
        descr=r"""Activation function to use for the recurrent step.""", default='sigmoid'))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
        descr=r"""whether the layer uses a bias vector."""))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the recurrent\_kernel weights matrix, used for the linear transformation of the
        recurrent state (see~\ref{initializersDNN}).""", default='orthogonal'))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
        descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType,
        descr=r"""Regularizer function applied to the recurrent\_kernel weights matrix(see ~\ref{regularizersDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
        descr=r"""regularizer function applied to the output
        of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the recurrent\_kernel weights matrix(see~\ref{constraintsDNN}).""",
        default=None))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
        descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.""",
        default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType,
        descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
        recurrent state.""", default=0.0))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last output in the output sequence, or the full sequence.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputTypes.BoolType,
        descr=r"""Whether to return the last state in addition to the output. """, default=False))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputTypes.BoolType,
        descr=r"""If True, process the input sequence backwards and return the reversed sequence."""))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputTypes.BoolType,
        descr=r"""If True, the last state for each sample at index i in a batch will be used as initial state for the
        sample of index i in the following batch.""", default=False))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputTypes.BoolType,
        descr=r"""If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a
        RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.""",
        default=False))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputTypes.BoolType,
        descr=r"""If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also
        force bias\_initializer=``zeros''.""", default=True))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)

    # The following layers only available in certain versions of TensorFlow.Keras
    # ###########################
    # #  SimpleRNNCell Layers
    # ###########################
    # layerInput = InputData.parameterInputFactory('SimpleRNNCell',contentType=InputTypes.StringType,
    #     descr=r""" """)
    # layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
    #     descr=r"""the name of the layer""")
    # layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
    #     descr=r"""dimensionality of the output space of this layer"""))
    # layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
    #     descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
    #     activation: $a(x) = x)$."""))
    # layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
    #     descr=r"""whether the layer uses a bias vector."""))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
    #     descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType,
    #     descr=r"""Initializer for the recurrent\_kernel weights matrix, used for the linear transformation of the
    #     recurrent state (see~\ref{initializersDNN}).""", default='orthogonal'))
    # layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
    #     descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""Regularizer function applied to the recurrent\_kernel weights matrix(see ~\ref{regularizersDNN}).""",
    #     default=None))
    # layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    # layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the output
    #     of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
    #     descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType,
    #     descr=r"""Constraint function applied to the recurrent\_kernel weights matrix(see~\ref{constraintsDNN}).""",
    #     default=None))
    # layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
    #     descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    # layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType,
    #     descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.""",
    #     default=0.0))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType,
    #     descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
    #     recurrent state.""", default=0.0))
    # inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    # ###########################
    # #  GRUCell Layers
    # ###########################
    # layerInput = InputData.parameterInputFactory('GRUCell',contentType=InputTypes.StringType,
    #     descr=r""" """)
    # layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
    #     descr=r"""the name of the layer""")
    # layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
    #     descr=r"""dimensionality of the output space of this layer"""))
    # layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
    #     descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
    #     activation: $a(x) = x)$."""))
    # layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
    #     descr=r"""whether the layer uses a bias vector."""))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
    #     descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType,
    #     descr=r"""Initializer for the recurrent\_kernel weights matrix, used for the linear transformation of the
    #     recurrent state (see~\ref{initializersDNN}).""", default='orthogonal'))
    # layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
    #     descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""Regularizer function applied to the recurrent\_kernel weights matrix(see ~\ref{regularizersDNN}).""",
    #     default=None))
    # layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    # layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the output
    #     of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
    #     descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType,
    #     descr=r"""Constraint function applied to the recurrent\_kernel weights matrix(see~\ref{constraintsDNN}).""",
    #     default=None))
    # layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
    #     descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    # layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType,
    #     descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.""",
    #     default=0.0))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType,
    #     descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
    #     recurrent state.""", default=0.0))
    # layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputTypes.IntegerType,
    #     descr=r""" """))
    # layerInput.addSub(InputData.parameterInputFactory('reset_after',contentType=InputTypes.BoolType,
    #     descr=r"""GRU convention (whether to apply reset gate after or before matrix multiplication).
    #     False = ``before'', True = ``after'' (default and CuDNN compatible)."""))
    # inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    # ###########################
    # #  LSTMCell Layers
    # ###########################
    # layerInput = InputData.parameterInputFactory('LSTMCell',contentType=InputTypes.StringType,
    #     descr=r""" """)
    # layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
    #     descr=r"""the name of the layer""")
    # layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputTypes.IntegerType,
    #     descr=r"""dimensionality of the output space of this layer"""))
    # layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputTypes.StringType,
    #     descr=r"""Activation function to use. If you don't specify anything, no activation is applied (ie. ``linear''
    #     activation: $a(x) = x)$."""))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputTypes.StringType,
    #     descr=r"""Activation function to use for the recurrent step.""", default='sigmoid'))
    # layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputTypes.BoolType,
    #     descr=r"""whether the layer uses a bias vector."""))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputTypes.StringType,
    #     descr=r"""initializer for the kernel weights matrix (see~\ref{initializersDNN}).""", default='glorot_uniform'))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputTypes.StringType,
    #     descr=r"""Initializer for the recurrent\_kernel weights matrix, used for the linear transformation of the
    #     recurrent state (see~\ref{initializersDNN}).""", default='orthogonal'))
    # layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputTypes.StringType,
    #     descr=r"""initializer for the bias vector (see ~\ref{initializersDNN}).""", default='zeros'))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the kernel weights matrix (see ~\ref{regularizersDNN})."""))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""Regularizer function applied to the recurrent\_kernel weights matrix(see ~\ref{regularizersDNN}).""",
    #     default=None))
    # layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the bias vector (see~\ref{regularizersDNN}).""", default=None))
    # layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputTypes.StringType,
    #     descr=r"""regularizer function applied to the output
    #     of the layer (its ``activation''). (see~\ref{regularizersDNN})""", default=None))
    # layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputTypes.StringType,
    #     descr=r"""constraint function applied to the kernel weights matrix (see~\ref{constraintsDNN})."""))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputTypes.StringType,
    #     descr=r"""Constraint function applied to the recurrent\_kernel weights matrix(see~\ref{constraintsDNN}).""",
    #     default=None))
    # layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputTypes.StringType,
    #     descr=r"""constraint function applied to the bias vector (see ~\ref{constraintsDNN})"""))
    # layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputTypes.FloatType,
    #     descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.""",
    #     default=0.0))
    # layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputTypes.FloatType,
    #     descr=r"""Float between 0 and 1. Fraction of the units to drop for the linear transformation of the
    #     recurrent state.""", default=0.0))
    # layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputTypes.BoolType,
    #     descr=r"""If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also
    #     force bias\_initializer=``zeros''.""", default=True))
    # layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputTypes.IntegerType,
    #     descr=r""" """))
    # inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Embedding Layers
    ##########################################
    ###########################
    #  Embdedding Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Embdedding',contentType=InputTypes.StringType,
        descr=r"""Turns positive integers (indexes) into dense vectors of fixed size.
        e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        This layer can only be used as the first layer in a model.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('input_dim',contentType=InputTypes.IntegerType,
        descr=r"""Size of the vocabulary."""))
    layerInput.addSub(InputData.parameterInputFactory('output_dim',contentType=InputTypes.StringType,
        descr=r"""Dimension of the dense embedding."""))
    layerInput.addSub(InputData.parameterInputFactory('embeddings_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the embeddings matrix (see~\ref{initializersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('embeddings_regularizer',contentType=InputTypes.StringType,
        descr=r"""Regularizer function applied to the embeddings matrix (see ~\ref{regularizersDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('embdeddings_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint function applied to the embeddings matrix (see~\ref{constraintsDNN})."""))
    layerInput.addSub(InputData.parameterInputFactory('mask_zero',contentType=InputTypes.BoolType,
        descr=r"""whether or not the input value 0 is a special ``padding'' value that should be masked out.
        This is useful when using recurrent layers which may take variable length input. If this is True, then
        all subsequent layers in the model need to support masking or an exception will be raised. If mask\_zero
        is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size
        of vocabulary + 1)."""))
    layerInput.addSub(InputData.parameterInputFactory('input_length',contentType=InputTypes.IntegerType,
        descr=r"""Length of input sequences, when it is constant. This argument is required if you are going to
        connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed)."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Advanced Activation Layers
    ##########################################
    ###########################
    #  LeakyRelU Layers: Leaky version of a Rectified Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('LeakyReLU',contentType=InputTypes.StringType,
        descr=r"""Leaky version of a Rectified Linear Unit.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('alpha',contentType=InputTypes.FloatType,
        descr=r"""Negative slope coefficient.""", default=0.3))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  PReLU Layers: Parametric Rectified Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('PReLU',contentType=InputTypes.StringType,
        descr=r"""Parametric Rectified Linear Unit.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('alpha_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer function for the weights."""))
    layerInput.addSub(InputData.parameterInputFactory('alpha_regularizer',contentType=InputTypes.StringType,
        descr=r"""Regularizer for the weights."""))
    layerInput.addSub(InputData.parameterInputFactory('alpha_constraint',contentType=InputTypes.StringType,
        descr=r"""Constraint for the weights."""))
    layerInput.addSub(InputData.parameterInputFactory('shared_axes',contentType=InputTypes.FloatListType,
        descr=r"""The axes along which to share learnable parameters for the activation function. """))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ELU Layers: Exponential Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('ELU',contentType=InputTypes.StringType,
        descr=r"""Exponential Linear Unit.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('alpha',contentType=InputTypes.FloatType,
        descr=r"""Scale for the negative factor."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ThresholdedReLU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ThresholdedReLU',contentType=InputTypes.StringType,
        descr=r"""Thresholded Rectified Linear Unit.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('theta',contentType=InputTypes.FloatType,
        descr=r""" Float >= 0. Threshold location of activation."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Softmax Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Softmax',contentType=InputTypes.StringType,
        descr=r"""Softmax activation function.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('axis',contentType=InputTypes.IntegerListType,
        descr=r"""Integer, or list of Integers, axis along which the softmax normalization is applied."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ReLU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ReLU',contentType=InputTypes.StringType,
        descr=r"""Rectified Linear Unit activation function.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('max_value',contentType=InputTypes.FloatType,
        descr=r""" Float >= 0. Maximum activation value. Default to None, which means unlimited."""))
    layerInput.addSub(InputData.parameterInputFactory('negative_slope',contentType=InputTypes.FloatType,
        descr=r"""Float >= 0. Negative slope coefficient.""", default=0))
    layerInput.addSub(InputData.parameterInputFactory('threshold',contentType=InputTypes.FloatType,
        descr=r"""Float >= 0. Threshold value for thresholded activation.""", default=0.0))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Normalization Layers
    ##########################################
    ###########################
    #  BatchNormalization Layers
    ###########################
    layerInput = InputData.parameterInputFactory('BatchNormalization',contentType=InputTypes.StringType,
        descr=r"""Layer that normalizes its inputs. Batch normalization applies a transformation that maintains
        the mean output close to 0 and the output standard deviation close to 1.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    layerInput.addSub(InputData.parameterInputFactory('axis',contentType=InputTypes.IntegerType,
        descr=r"""the axis that should be normalized (typically the features axis)."""))
    layerInput.addSub(InputData.parameterInputFactory('momentum',contentType=InputTypes.FloatType,
        descr=r"""Momentum for the moving average."""))
    layerInput.addSub(InputData.parameterInputFactory('epsilon',contentType=InputTypes.FloatType,
        descr=r"""Small float added to variance to avoid dividing by zero."""))
    layerInput.addSub(InputData.parameterInputFactory('center',contentType=InputTypes.BoolType,
        descr=r"""If True, add offset of beta to normalized tensor. If False, beta is ignored."""))
    layerInput.addSub(InputData.parameterInputFactory('scale',contentType=InputTypes.BoolType,
        descr=r""" If True, multiply by gamma. If False, gamma is not used. """))
    layerInput.addSub(InputData.parameterInputFactory('beta_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the beta weight."""))
    layerInput.addSub(InputData.parameterInputFactory('gamma_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the gamma weight."""))
    layerInput.addSub(InputData.parameterInputFactory('moving_mean_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the moving mean."""))
    layerInput.addSub(InputData.parameterInputFactory('moving_variance_initializer',contentType=InputTypes.StringType,
        descr=r"""Initializer for the moving variance."""))
    layerInput.addSub(InputData.parameterInputFactory('beta_regularizer',contentType=InputTypes.StringType,
        descr=r"""Optional regularizer for the beta weight."""))
    layerInput.addSub(InputData.parameterInputFactory('gamma_regularizer',contentType=InputTypes.StringType,
        descr=r"""Optional regularizer for the gamma weight."""))
    layerInput.addSub(InputData.parameterInputFactory('beta_constraint',contentType=InputTypes.StringType,
        descr=r"""Optional constraint for the beta weight."""))
    layerInput.addSub(InputData.parameterInputFactory('gamma_constraint',contentType=InputTypes.StringType,
        descr=r"""Optional constraint for the gamma weight."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Noise Layers
    ##########################################
    ###########################
    #  GausianNoise Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GaussianNoise',contentType=InputTypes.StringType,
        descr=r"""Apply additive zero-centered Gaussian noise. This is useful to mitigate overfitting (you could see
        it as a form of random data augmentation). Gaussian Noise (GS) is a natural choice as corruption process
        for real valued inputs. As it is a regularization layer, it is only active at training time.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'stddev' need to be popped out and the value will be passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('stddev',contentType=InputTypes.FloatType,
        descr=r"""standard deviation of the noise distribution."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GausianDropout Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GaussianDropout',contentType=InputTypes.StringType,
        descr=r"""Apply multiplicative 1-centered Gaussian noise. As it is a regularization layer, it is only active
        at training time.""")
    layerInput.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name of the layer""")
    # 'stddev' need to be popped out and the value will be passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputTypes.FloatType,
        descr=r"""drop probability (as with Dropout). The multiplicative noise will have standard deviation
        $sqrt(rate / (1 - rate))$."""))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    #################################################
    layerLayoutInput = InputData.parameterInputFactory('layer_layout',contentType=InputTypes.StringListType,
        descr=r"""The layout of the neural network layers, i.e., the list of names of neural network layers""",
        default='no-default')
    inputSpecification.addSub(layerLayoutInput)

    return inputSpecification

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

    self.kerasLayersList = functools.reduce(lambda x,y: x+y, list(self.kerasDict.values()))

    self.kerasROMsList = ['KerasMLPClassifier', 'KerasMLPRegression', 'KerasConvNetClassifier', 'KerasLSTMClassifier', 'KerasLSTMRegression']

    if len(self.availOptimizer) == 0:
      # stochastic gradient descent optimizer, includes support for momentum,learning rate decay, and Nesterov momentum
      import tensorflow.keras as keras #Needed if lazily loading tensorflow 2.6
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
    self.basicLayers = self.kerasDict['kerasCoreLayersList'] + self.kerasDict['kerasEmbeddingLayersList'] + \
                       self.kerasDict['kerasAdvancedActivationLayersList'] + self.kerasDict['kerasNormalizationLayersList'] + \
                       self.kerasDict['kerasNoiseLayersList']
    # LabelEncoder can be used to normalize labels
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
                                                            'layer_layout', 'epochs'])
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
      for sub in optNode.subparts:
        optimizerSetting[sub.getName()] = sub.value
    optimizerName = optimizerSetting.pop('optimizer').lower()
    # set up optimizer
    self.optimizer = self.__class__.availOptimizer[optimizerName](**optimizerSetting)
    # check layer layout, this is always required node, used to build the DNNs
    self.layerLayout = nodes.get('layer_layout')
    for sub in paramInput.subparts:
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
    ## Potential Pickling issue: https://github.com/tensorflow/tensorflow/issues/33283
    state = SupervisedLearning.__getstate__(self)
    tf.keras.models.save_model(self._ROM, KerasBase.tempModelFile, save_format='h5')
    # another method to save the TensorFlow model
    # self._ROM.save(KerasBase.tempModelFile)
    with open(KerasBase.tempModelFile, "rb") as f:
      serialModelData = f.read()
    state[KerasBase.modelAttr] = serialModelData
    os.remove(KerasBase.tempModelFile)
    del state["_ROM"]
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

    self._train(featureValues,targetValues)
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

  def _train(self,featureVals,targetVals):
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
