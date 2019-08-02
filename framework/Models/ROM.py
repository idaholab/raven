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
Module where the base class and the specialization of different type of Model are
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import inspect
import itertools
import numpy as np
import functools
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Dummy import Dummy
import SupervisedLearning
from utils import utils
from utils import xmlUtils
from utils import InputData
import Files
import LearningGate
#Internal Modules End--------------------------------------------------------------------------------


class ROM(Dummy):
  """
    ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls. This one seems a bit excessive, are all of these for this class?
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(ROM, cls).getInputSpecification()

    IndexSetInputType = InputData.makeEnumType("indexSet","indexSetType",["TensorProduct","TotalDegree","HyperbolicCross","Custom"])
    CriterionInputType = InputData.makeEnumType("criterion", "criterionType", ["bic","aic","gini","entropy","mse"])

    # general
    inputSpecification.addSub(InputData.parameterInputFactory('Features',contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory('Target',contentType=InputData.StringType))
    # segmenting and clustering
    segment = InputData.parameterInputFactory("Segment", strictMode=True)
    segmentGroups = InputData.makeEnumType('segmentGroup', 'sesgmentGroupType', ['segment', 'cluster'])
    segment.addParam('grouping', segmentGroups)
    subspace = InputData.parameterInputFactory('subspace', contentType=InputData.StringType)
    subspace.addParam('divisions', InputData.IntegerType, False)
    subspace.addParam('pivotLength', InputData.FloatType, False)
    subspace.addParam('shift', InputData.StringType, False)
    segment.addSub(subspace)
    clsfr = InputData.parameterInputFactory('Classifier', strictMode=True, contentType=InputData.StringType)
    clsfr.addParam('class', InputData.StringType, True)
    clsfr.addParam('type', InputData.StringType, True)
    segment.addSub(clsfr)
    metric = InputData.parameterInputFactory('Metric', strictMode=True, contentType=InputData.StringType)
    metric.addParam('class', InputData.StringType, True)
    metric.addParam('type', InputData.StringType, True)
    segment.addSub(metric)
    feature = InputData.parameterInputFactory('feature', strictMode=True, contentType=InputData.StringType)
    feature.addParam('weight', InputData.FloatType)
    segment.addSub(feature)
    inputSpecification.addSub(segment)
    # unsorted
    inputSpecification.addSub(InputData.parameterInputFactory("persistence", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("gradient", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("simplification", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("graph", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("knn", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("partitionPredictor", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("smooth", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("kernel", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("bandwidth", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("p", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SKLtype", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_iter", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_iter_no_change", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("tol", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha_1", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha_2", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda_1", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda_2", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("compute_score", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("threshold_lambda", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputData.StringType))  #bool
    inputSpecification.addSub(InputData.parameterInputFactory("normalize", contentType=InputData.StringType))  #bool
    inputSpecification.addSub(InputData.parameterInputFactory("verbose", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("alpha", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("l1_ratio", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_iter", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("warm_start", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("positive", contentType=InputData.StringType)) #bool?
    inputSpecification.addSub(InputData.parameterInputFactory("eps", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_alphas", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("precompute", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_nonzero_coefs", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_path", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("max_n_alphas", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("criterion", contentType=CriterionInputType))
    inputSpecification.addSub(InputData.parameterInputFactory("penalty", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("dual", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("C", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("intercept_scaling", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("class_weight", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("random_state", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("cv", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("shuffle", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("loss", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("epsilon", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("eta0", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("solver", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("alphas", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("scoring", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("gcv_mode", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("store_cv_values", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("learning_rate", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("power_t", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("multi_class", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("kernel", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("degree", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("gamma", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("coef0", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("probability", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("shrinking", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("cache_size", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("nu", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("code_size", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_prior", contentType=InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("class_prior", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("binarize", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_neighbors", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("weights", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("algorithm", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("leaf_size", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("metric", contentType=InputData.StringType)) #enum?
    inputSpecification.addSub(InputData.parameterInputFactory("radius", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("outlier_label", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("shrink_threshold", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("priors", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("reg_param", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("splitter", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("max_features", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_depth", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("min_samples_split", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("min_samples_leaf", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_leaf_nodes", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("regr", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("corr", contentType=InputData.StringType)) #enum?
    inputSpecification.addSub(InputData.parameterInputFactory("beta0", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("storage_mode", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("theta0", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("thetaL", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("thetaU", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("nugget", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("optimizer", contentType=InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("random_start", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_restarts_optimizer", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("normalize_y", contentType=InputData.StringType))
    # GaussPolynomialROM and HDMRRom
    inputSpecification.addSub(InputData.parameterInputFactory("IndexPoints", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("IndexSet",contentType=IndexSetInputType))
    inputSpecification.addSub(InputData.parameterInputFactory('pivotParameter',contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("PolynomialOrder", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SobolOrder", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SparseGrid", contentType=InputData.StringType))
    InterpolationInput = InputData.parameterInputFactory('Interpolation', contentType=InputData.StringType)
    InterpolationInput.addParam("quad", InputData.StringType, False)
    InterpolationInput.addParam("poly", InputData.StringType, False)
    InterpolationInput.addParam("weight", InputData.FloatType, False)
    inputSpecification.addSub(InterpolationInput)
    # ARMA
    inputSpecification.addSub(InputData.parameterInputFactory('correlate', contentType=InputData.StringListType))
    inputSpecification.addSub(InputData.parameterInputFactory("P", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("Q", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("seed", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("reseedCopies", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("Fourier", contentType=InputData.FloatListType))
    inputSpecification.addSub(InputData.parameterInputFactory("preserveInputCDF", contentType=InputData.BoolType))
    ### ARMA zero filter
    zeroFilt = InputData.parameterInputFactory('ZeroFilter', contentType=InputData.StringType)
    zeroFilt.addParam('tol', InputData.FloatType)
    inputSpecification.addSub(zeroFilt)
    ### ARMA out truncation
    outTrunc = InputData.parameterInputFactory('outTruncation', contentType=InputData.StringListType)
    domainEnumType = InputData.makeEnumType('domain', 'truncateDomainType', ['positive', 'negative'])
    outTrunc.addParam('domain', domainEnumType, True)
    inputSpecification.addSub(outTrunc)
    ### ARMA specific fourier
    specFourier = InputData.parameterInputFactory('SpecificFourier', strictMode=True)
    specFourier.addParam("variables", InputData.StringListType, True)
    specFourier.addSub(InputData.parameterInputFactory('periods', contentType=InputData.FloatListType))
    inputSpecification.addSub(specFourier)
    ### ARMA peaks
    peaks = InputData.parameterInputFactory('Peaks')
    window = InputData.parameterInputFactory('window',contentType=InputData.FloatListType)
    window.addParam('width', InputData.FloatType, True)
    peaks.addSub(window)
    peaks.addParam('threshold', InputData.FloatType)
    peaks.addParam('target', InputData.StringType)
    peaks.addParam('period', InputData.FloatType)
    inputSpecification.addSub(peaks)
    # inputs for neural_network
    inputSpecification.addSub(InputData.parameterInputFactory("activation", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("batch_size", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("learning_rate_init", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("momentum", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("nesterovs_momentum", contentType=InputData.StringType)) # bool
    inputSpecification.addSub(InputData.parameterInputFactory("early_stopping", contentType=InputData.StringType)) # bool
    inputSpecification.addSub(InputData.parameterInputFactory("validation_fraction", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta_1", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta_2", contentType=InputData.FloatType))
    # PolyExp
    inputSpecification.addSub(InputData.parameterInputFactory("maxNumberExpTerms", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("numberExpTerms", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("maxPolyOrder", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("polyOrder", contentType=InputData.IntegerType))
    coeffRegressorEnumType = InputData.makeEnumType("coeffRegressor","coeffRegressorType",["poly","spline","nearest"])
    inputSpecification.addSub(InputData.parameterInputFactory("coeffRegressor", contentType=coeffRegressorEnumType))
    # DMD
    inputSpecification.addSub(InputData.parameterInputFactory("rankSVD", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("energyRankSVD", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("rankTLSQ", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("exactModes", contentType=InputData.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("optimized", contentType=InputData.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("dmdType", contentType=InputData.StringType))

    # for deep learning neural network
    #inputSpecification.addSub(InputData.parameterInputFactory("DNN", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("hidden_layer_sizes", contentType=InputData.IntegerTupleType)) # list of integer
    inputSpecification.addSub(InputData.parameterInputFactory("metrics", contentType=InputData.StringListType)) #list of metrics
    inputSpecification.addSub(InputData.parameterInputFactory("batch_size", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("epochs", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("random_seed", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("plot_model", contentType=InputData.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("num_classes",contentType= InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("validation_split", contentType=InputData.FloatType))

    # Keras optimizer parameters
    OptimizerSettingInput = InputData.parameterInputFactory('optimizerSetting', contentType=InputData.StringType)
    Beta1Input = InputData.parameterInputFactory('beta_1', contentType=InputData.FloatType)
    Beta2Input = InputData.parameterInputFactory('beta_2', contentType=InputData.FloatType)
    DecayInput = InputData.parameterInputFactory('decay', contentType=InputData.FloatType)
    LRInput = InputData.parameterInputFactory('lr', contentType=InputData.FloatType)
    OptimizerInput = InputData.parameterInputFactory('optimizer', contentType=InputData.StringType)
    EpsilonInput = InputData.parameterInputFactory('epsilon', contentType=InputData.FloatType)
    MomentumInput = InputData.parameterInputFactory('momentum', contentType=InputData.FloatType)
    NesterovInput = InputData.parameterInputFactory('nesterov', contentType=InputData.StringType)
    RhoInput = InputData.parameterInputFactory('rho', contentType=InputData.FloatType)
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
    dataFormatEnumType = InputData.makeEnumType('dataFormat','dataFormatType',['channels_last', 'channels_first'])
    paddingEnumType = InputData.makeEnumType('padding','paddingType',['valid', 'same'])
    interpolationEnumType = InputData.makeEnumType('interpolation','interpolationType',['nearest', 'bilinear'])
    ###########################
    #  Dense Layers: regular densely-connected neural network layer
    ###########################
    layerInput = InputData.parameterInputFactory('Dense',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Activation Layers: applies an activation function to an output
    ###########################
    layerInput = InputData.parameterInputFactory('Activation',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'activation' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Dropout Layers: Applies Dropout to the input
    ###########################
    layerInput = InputData.parameterInputFactory('Dropout',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('noise_shape',contentType=InputData.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('seed',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Flatten Layers: Flattens the input
    ###########################
    layerInput = InputData.parameterInputFactory('Flatten',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Input Layers: Input() is used to instantiate a Keras tensor
    ###########################
    layerInput = InputData.parameterInputFactory('Input',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Reshape Layers: Reshapes an output to a certain shape
    ###########################
    layerInput = InputData.parameterInputFactory('Reshape',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'target_shape' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('target_shape',contentType=InputData.IntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Permute Layers: permutes the dimensions of the input according to a given pattern
    ###########################
    layerInput = InputData.parameterInputFactory('Permute',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'permute_pattern' need to pop out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('permute_pattern',contentType=InputData.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('input_shape',contentType=InputData.IntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  RepeatVector Layers: repeats the input n times
    ###########################
    layerInput = InputData.parameterInputFactory('RepeatVector',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'repetition_factor' need to be popped out and only the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('repetition_factor',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Lambda Layers: Wraps arbitrary expression as a Layer object
    ###########################
    layerInput = InputData.parameterInputFactory('Lambda',contentType=InputData.StringType,strictMode=False)
    layerInput.addParam('name', InputData.StringType, True)
    # A function object need to be created and passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('function',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ActivityRegularization Layers: applies an update to the cost function based input activity
    ###########################
    layerInput = InputData.parameterInputFactory('ActivityRegularization',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('l1',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('l2',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Masking Layers: Masks a sequence by using a mask value to skip timesteps
    ###########################
    layerInput = InputData.parameterInputFactory('Masking',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('mask_value',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout1D Layers: Spatial 1D version of Dropout
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SpatialDropout3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SpatialDropout3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'rate' need to be popped out and the value will be passed to the given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###################################
    # Convolutional Layers
    ###################################
    ###########################
    #  Conv1D Layers: 1D convolutioanl layer (e.g. temporal convolutional)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv2D Layers: 2D convolutioanl layer (e.g. spatial convolution over images)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv3D Layers: 3D convolutioanl layer (e.g. spatial convolution over volumes)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SeparableConv1D Layers: Depthwise separable 1D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('SeparableConv1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SeparableConv2D Layers: Depthwise separable 2D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('SeparableConv2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('pointwise_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  DepthwiseConv2D Layers: Depthwise separable 2D convolutioanl layer
    ###########################
    layerInput = InputData.parameterInputFactory('DepthwiseConv2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('depth_multiplier',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('depthwise_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv2DTranspose Layers: Transposed convolution layer (sometimes called Deconvolution)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv2DTranspose',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('output_padding',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Conv3DTranspose Layers: Transposed convolution layer (sometimes called Deconvolution)
    ###########################
    layerInput = InputData.parameterInputFactory('Conv3DTranspose',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('output_padding',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    #  Cropping1D  Layers: cropping layer for 1D input (e.g. temporal sequence)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputData.IntegerOrIntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Cropping2D  Layers: cropping layer for 2D input (e.g. picutures)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Cropping3D  Layers: cropping layer for 2D input (e.g. picutures)
    ###########################
    layerInput = InputData.parameterInputFactory('Cropping3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('cropping',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Upsampling1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('UpSampling2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('interpolation',contentType=interpolationEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    # Upsampling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('UpSampling3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputData.IntegerOrIntegerTupleType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ZeroPadding3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ZeroPadding3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ############################################
    #   Pooling Layers
    ############################################
    ###########################
    #  MaxPooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  MaxPooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  MaxPooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('MaxPooling3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputData.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  AveragePooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('AveragePooling3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('pool_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalMaxPooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalMaxPooling3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GlobalAveragePooling3D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GlobalAveragePooling3D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)

    ######################################
    #   Locally-connected Layers
    ######################################
    ###########################
    #  LocallyConnected1D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LocallyConnected1D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LocallyConnected2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LocallyConnected2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ######################################
    #  Recurrent Layers
    ######################################
    ###########################
    #  RNN Layers
    ###########################
    layerInput = InputData.parameterInputFactory('RNN',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputData.BoolType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SimpleRNN Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SimpleRNN',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputData.BoolType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GRU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GRU',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('reset_after',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LSTM Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LSTM',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ConvLSTM2D Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ConvLSTM2D',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_size',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('strides',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('padding',contentType=paddingEnumType))
    layerInput.addSub(InputData.parameterInputFactory('data_format',contentType=dataFormatEnumType))
    layerInput.addSub(InputData.parameterInputFactory('dilation_rate',contentType=InputData.IntegerOrIntegerTupleType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('return_sequences',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('return_state',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('go_backwards',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('stateful',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unroll',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  SimpleRNNCell Layers
    ###########################
    layerInput = InputData.parameterInputFactory('SimpleRNNCell',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GRUCell Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GRUCell',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('reset_after',contentType=InputData.BoolType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  LSTMCell Layers
    ###########################
    layerInput = InputData.parameterInputFactory('LSTMCell',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('dim_out',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_activation',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('use_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('activity_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('kernel_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('bias_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('recurrent_dropout',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('unit_forget_bias',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('implementation',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Embedding Layers
    ##########################################
    ###########################
    #  Embdedding Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Embdedding',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('input_dim',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('output_dim',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('embeddings_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('embeddings_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('embdeddings_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('mask_zero',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('input_length',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Advanced Activation Layers
    ##########################################
    ###########################
    #  LeakyRelU Layers: Leaky version of a Rectified Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('LeakyRelU',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('alpha',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  PReLU Layers: Parametric Rectified Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('PReLU',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('alpha_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('alpha_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('alpha_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('shared_axes',contentType=InputData.FloatListType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ELU Layers: Exponential Linear Unit
    ###########################
    layerInput = InputData.parameterInputFactory('ELU',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('alpha',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ThresholdedReLU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ThresholdedReLU',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('theta',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  Softmax Layers
    ###########################
    layerInput = InputData.parameterInputFactory('Softmax',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('axis',contentType=InputData.IntegerType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  ReLU Layers
    ###########################
    layerInput = InputData.parameterInputFactory('ReLU',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('max_value',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('negative_slope',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('threshold',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Normalization Layers
    ##########################################
    ###########################
    #  BatchNormalization Layers
    ###########################
    layerInput = InputData.parameterInputFactory('BatchNormalization',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    layerInput.addSub(InputData.parameterInputFactory('axis',contentType=InputData.IntegerType))
    layerInput.addSub(InputData.parameterInputFactory('momentum',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('epsilon',contentType=InputData.FloatType))
    layerInput.addSub(InputData.parameterInputFactory('center',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('scale',contentType=InputData.BoolType))
    layerInput.addSub(InputData.parameterInputFactory('beta_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('gamma_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('moving_mean_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('moving_variance_initializer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('beta_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('gamma_regularizer',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('beta_constraint',contentType=InputData.StringType))
    layerInput.addSub(InputData.parameterInputFactory('gamma_constraint',contentType=InputData.StringType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ##########################################
    #  Noise Layers
    ##########################################
    ###########################
    #  GausianNoise Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GaussianNoise',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'stddev' need to be popped out and the value will be passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('stddev',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    ###########################
    #  GausianDropout Layers
    ###########################
    layerInput = InputData.parameterInputFactory('GaussianDropout',contentType=InputData.StringType)
    layerInput.addParam('name', InputData.StringType, True)
    # 'stddev' need to be popped out and the value will be passed to given layer
    layerInput.addSub(InputData.parameterInputFactory('rate',contentType=InputData.FloatType))
    inputSpecification.addSub(layerInput,InputData.Quantity.zero_to_infinity)
    #################################################

    layerLayoutInput = InputData.parameterInputFactory('layer_layout',contentType=InputData.StringListType)
    inputSpecification.addSub(layerLayoutInput)

    #Estimators can include ROMs, and so because baseNode does a copy, this
    #needs to be after the rest of ROMInput is defined.
    EstimatorInput = InputData.parameterInputFactory('estimator', contentType=InputData.StringType, baseNode=inputSpecification)
    EstimatorInput.addParam("estimatorType", InputData.StringType, False)
    #The next lines are to make subType and name not required.
    EstimatorInput.addParam("subType", InputData.StringType, False)
    EstimatorInput.addParam("name", InputData.StringType, False)
    inputSpecification.addSub(EstimatorInput)

    # inputs for cross validations
    cvInput = InputData.parameterInputFactory("CV", contentType=InputData.StringType)
    cvInput.addParam("class", InputData.StringType)
    cvInput.addParam("type", InputData.StringType)
    inputSpecification.addSub(cvInput)

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['PointSet','HistorySet']

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self,runInfoDict)
    self.initializationOptionDict = {}          # ROM initialization options
    self.amITrained                = False      # boolean flag, is the ROM trained?
    self.supervisedEngine          = None       # dict of ROM instances (== number of targets => keys are the targets)
    self.printTag = 'ROM MODEL'
    self.cvInstance               = None             # Instance of provided cross validation
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

    self.kerasROMsList = ['KerasMLPClassifier', 'KerasConvNetClassifier', 'KerasLSTMClassifier']
    # for Clustered ROM
    self.addAssemblerObject('Classifier','-1',True)
    self.addAssemblerObject('Metric','-n',True)
    self.addAssemblerObject('CV','-1',True)

  def __getstate__(self):
    """
      Method for choosing what gets serialized in this class
      @ In, None
      @ Out, d, dict, things to serialize
    """
    d = copy.copy(self.__dict__)
    # NOTE assemblerDict isn't needed if ROM already trained, but it can create an infinite recursion
    ## for the ROMCollection if left in, so remove it on getstate.
    del d['assemblerDict']
    return d

  def __setstate__(self, d):
    """
      Method for unserializing.
      @ In, d, dict, things to unserialize
      @ Out, None
    """
    # default setstate behavior
    self.__dict__ = d
    # since we pop this out during saving state, initialize it here
    self.assemblerDict = {}

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy._readMoreXML(self, xmlNode)
    self.initializationOptionDict['name'] = self.name
    paramInput = ROM.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    for child in paramInput.subparts:
      if child.getName() == 'CV':
        self.cvInstance = child.value.strip()
        continue
      if len(child.parameterValues) > 0 and child.getName().lower() not in self.kerasLayersList:
        if child.getName() == 'alias':
          continue
        if child.getName() not in self.initializationOptionDict.keys():
          self.initializationOptionDict[child.getName()]={}
        # "tuple" here allows values to be listed, probably not great but works
        key = child.value if not isinstance(child.value,list) else tuple(child.value)
        self.initializationOptionDict[child.getName()][key]=child.parameterValues
      else:
        if child.getName() in ['estimator', 'optimizerSetting']:
          self.initializationOptionDict[child.getName()] = {}
          for node in child.subparts:
            self.initializationOptionDict[child.getName()][node.getName()] = node.value
        elif child.getName().lower() in self.kerasLayersList and self.subType in self.kerasROMsList:
          layerName = child.parameterValues['name']
          self.initializationOptionDict[layerName] = {}
          self.initializationOptionDict[layerName]['type'] = child.getName().lower()
          for node in child.subparts:
            self.initializationOptionDict[layerName][node.getName()] = node.value
        else:
          self.initializationOptionDict[child.getName()] = child.value
    # if working with a pickled ROM, send along that information
    if self.subType == 'pickledROM':
      self.initializationOptionDict['pickled'] = True
    self._initializeSupervisedGate(paramInput=paramInput, **self.initializationOptionDict)
    #the ROM is instanced and initialized
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(SupervisedLearning),True)) - set(self.mods))
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(LearningGate),True)) - set(self.mods))

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize this class
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    # retrieve cross validation object
    if self.cvInstance is not None:
      self.cvInstance = self.retrieveObjectFromAssemblerDict('CV', self.cvInstance)
      self.cvInstance.initialize(runInfo, inputs, initDict)

  def _initializeSupervisedGate(self,**initializationOptions):
    """
      Method to initialize the supervisedGate class
      @ In, initializationOptions, dict, the initialization options
      @ Out, None
    """
    self.supervisedEngine = LearningGate.returnInstance('SupervisedGate', self.subType, self, **initializationOptions)

  def reset(self):
    """
      Reset the ROM
      @ In,  None
      @ Out, None
    """
    self.supervisedEngine.reset()
    self.amITrained   = False

  def reseed(self,seed):
    """
      Used to reset the seed of the underlying ROM.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    self.supervisedEngine.reseed(seed)

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = self.supervisedEngine.getInitParams()
    return paramDict

  def train(self,trainingSet):
    """
      This function train the ROM
      @ In, trainingSet, dict or PointSet or HistorySet, data used to train the ROM; if an HistorySet is provided the a list of ROM is created in order to create a temporal-ROM
      @ Out, None
    """
    if type(trainingSet).__name__ == 'ROM':
      self.initializationOptionDict = copy.deepcopy(trainingSet.initializationOptionDict)
      self.trainingSet              = copy.copy(trainingSet.trainingSet)
      self.amITrained               = copy.deepcopy(trainingSet.amITrained)
      self.supervisedEngine         = copy.deepcopy(trainingSet.supervisedEngine)
    else:
      # TODO: The following check may need to be moved to Dummy Class -- wangc 7/30/2018
      if type(trainingSet).__name__ != 'dict' and trainingSet.type == 'HistorySet':
        pivotParameterId = self.supervisedEngine.pivotParameterId
        if not trainingSet.checkIndexAlignment(indexesToCheck=pivotParameterId):
          self.raiseAnError(IOError, "The data provided by the data object", trainingSet.name, "is not synchonized!",
                  "The time-dependent ROM requires all the histories are synchonized!")
      self.trainingSet = copy.copy(self._inputToInternal(trainingSet))
      self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
      # grab assembled stuff and pass it through
      ## TODO this should be changed when the SupervisedLearning objects themselves can use the Assembler
      self.supervisedEngine.train(self.trainingSet, self.assemblerDict)
      self.amITrained = self.supervisedEngine.amITrained

  def confidence(self,request,target = None):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, datatype, feature coordinates (request)
      @ Out, confidenceDict, dict, the dict containing the confidence on each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    inputToROM = self._inputToInternal(request)
    confidenceDict = self.supervisedEngine.confidence(inputToROM)
    return confidenceDict

  def evaluate(self,request):
    """
      When the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
      @ In, request, datatype, feature coordinates (request)
      @ Out, outputEvaluation, dict, the dict containing the outputs for each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    inputToROM       = self._inputToInternal(request)
    outputEvaluation = self.supervisedEngine.evaluate(inputToROM)
    # assure numpy array formatting # TODO can this be done in the supervised engine instead?
    for k,v in outputEvaluation.items():
      outputEvaluation[k] = np.atleast_1d(v)
    return outputEvaluation

  def _externalRun(self,inRun):
    """
      Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
      @ In, inRun, datatype, feature coordinates
      @ Out, returnDict, dict, the return dictionary containing the results
    """
    returnDict = self.evaluate(inRun)
    self._replaceVariablesNamesWithAliasSystem(returnDict, 'output', True)
    self._replaceVariablesNamesWithAliasSystem(inRun, 'input', True)
    return returnDict

  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, rlz, dict, This will hold two pieces of information,
          the first will be the input data used to generate this sample,
          the second will be the output of this model given the specified
          inputs
    """
    Input = self.createNewInput(myInput, samplerType, **kwargs)
    inRun = self._manipulateInput(Input[0])
    # collect results from model run
    result = self._externalRun(inRun)
    # build realization
    # assure rlz has all metadata
    self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'] ,'input',True)
    rlz = dict((var,np.atleast_1d(kwargs[var])) for var in kwargs.keys())
    # update rlz with input space from inRun and output space from result
    rlz.update(dict((var,np.atleast_1d(inRun[var] if var in kwargs['SampledVars'] else result[var])) for var in set(itertools.chain(result.keys(),inRun.keys()))))
    return rlz

  def convergence(self,trainingSet):
    """
      This is to get the cross validation score of ROM
      @ In, trainingSize, int, the size of current training size
      @ Out, cvScore, dict, the dict containing the score of cross validation
    """
    if self.subType.lower() != 'scikitlearn':
      self.raiseAnError(IOError, 'convergence calculation is not Implemented for ROM', self.name, 'with type', self.subType)
    cvScore = self._crossValidationScore(trainingSet)
    return cvScore

  def _crossValidationScore(self, trainingSet):
    """
      The function calculates the cross validation score on ROMs
      @ In, trainingSize, int, the size of current training size
      @ Out, cvMetrics, dict, the calculated cross validation metrics
    """
    if len(self.supervisedEngine.supervisedContainer) > 1:
      self.raiseAnError(IOError, "Cross Validation Method is not implemented for Clustered ROMs")
    cvMetrics = None
    if self._checkCV(len(trainingSet)):
      # reset the ROM before perform cross validation
      cvMetrics = {}
      self.reset()
      outputMetrics = self.cvInstance.interface.run([self, trainingSet])
      exploredTargets = []
      for cvKey, metricValues in outputMetrics.items():
        info = self.cvInstance.interface._returnCharacteristicsOfCvGivenOutputName(cvKey)
        if info['targetName'] in exploredTargets:
          self.raiseAnError(IOError, "Multiple metrics are used in cross validation '", self.cvInstance.name, "' for ROM '", rom.name,  "'!")
        exploredTargets.append(info['targetName'])
        cvMetrics[self.name] = (info['metricType'], metricValues)
    return cvMetrics

  def _checkCV(self, trainingSize):
    """
      The function will check whether we can use Cross Validation or not
      @ In, trainingSize, int, the size of current training size
      @ Out, None
    """
    useCV = True
    initDict =  self.cvInstance.interface.initializationOptionDict
    if 'SciKitLearn' in initDict.keys() and 'n_splits' in initDict['SciKitLearn'].keys():
      if trainingSize < utils.intConversion(initDict['SciKitLearn']['n_splits']):
        useCV = False
    else:
      useCV = False
    return useCV

  def writePointwiseData(self, writeTo):
    """
      Called by the OutStreamPrint object to cause the ROM to print information about itself
      @ In, writeTo, DataObject, data structure to add data to
      @ Out, None
    """
    # TODO handle statepoint ROMs (dynamic, but rom doesn't handle intrinsically)
    ## should probably let the LearningGate handle this! It knows how to stitch together pieces, sort of.
    engines = self.supervisedEngine.supervisedContainer
    for engine in engines:
      engine.writePointwiseData(writeTo)

  def writeXML(self, what='all'):
    """
      Called by the OutStreamPrint object to cause the ROM to print itself
      @ In, what, string, optional, keyword requesting what should be printed
      @ Out, xml, xmlUtils.StaticXmlElement, written meta
    """
    #determine dynamic or static
    dynamic = self.supervisedEngine.isADynamicModel
    # determine if it can handle dynamic data
    handleDynamicData = self.supervisedEngine.canHandleDynamicData
    # get pivot parameter
    pivotParameterId = self.supervisedEngine.pivotParameterId
    # find some general settings needed for either dynamic or static handling
    ## get all the targets the ROMs have
    ROMtargets = self.supervisedEngine.initializationOptions['Target'].split(",")
    ## establish requested targets
    targets = ROMtargets if what=='all' else what.split(',')
    ## establish sets of engines to work from
    engines = self.supervisedEngine.supervisedContainer
    # if the ROM is "dynamic" (e.g. time-dependent targets), then how we print depends
    #    on whether the engine is naturally dynamic or whether we need to handle that part.
    if dynamic and not handleDynamicData:
      # time-dependent, but we manage the output (chopped)
      xml = xmlUtils.DynamicXmlElement('ROM', pivotParam = pivotParameterId)
      ## pre-print printing
      engines[0].writeXMLPreamble(xml) #let the first engine write the preamble
      for s,rom in enumerate(engines):
        pivotValue = self.supervisedEngine.historySteps[s]
        #for target in targets: # should be handled by SVL engine or here??
        #  #skip the pivot param
        #  if target == pivotParameterId:
        #    continue
        #otherwise, call engine's print method
        self.raiseAMessage('Printing time-like',pivotValue,'ROM XML')
        subXML = xmlUtils.StaticXmlElement(self.supervisedEngine.supervisedContainer[0].printTag)
        rom.writeXML(subXML, skip = [pivotParameterId])
        for element in subXML.getRoot():
          xml.addScalarNode(element, pivotValue)
        #xml.addScalarNode(subXML.getRoot(), pivotValue)
    else:
      # directly accept the results from the engine
      xml = xmlUtils.StaticXmlElement(self.name)
      ## pre-print printing
      engines[0].writeXMLPreamble(xml)
      engines[0].writeXML(xml)
    return xml
