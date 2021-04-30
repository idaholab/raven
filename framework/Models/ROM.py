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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import inspect
import itertools
import numpy as np
import functools
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Dummy import Dummy
import Decorators
import SupervisedLearning
from utils import utils
from utils import xmlUtils
from utils import InputData, InputTypes
from Decorators.Parallelization import Parallel
import Files
import LearningGate
#Internal Modules End--------------------------------------------------------------------------------

# set enviroment variable to avoid parallelim degradation in some surrogate models
os.environ["MKL_NUM_THREADS"]="1"

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

    IndexSetInputType = InputTypes.makeEnumType("indexSet","indexSetType",["TensorProduct","TotalDegree","HyperbolicCross","Custom"])
    CriterionInputType = InputTypes.makeEnumType("criterion", "criterionType", ["bic","aic","gini","entropy","mse"])
    ###########
    # general #
    ###########
    inputSpecification.addSub(InputData.parameterInputFactory('Features',contentType=InputTypes.StringListType))
    inputSpecification.addSub(InputData.parameterInputFactory('Target',contentType=InputTypes.StringListType))

    ######################
    # dynamically loaded #
    ######################
    for typ in SupervisedLearning.factory.knownTypes():
      obj = SupervisedLearning.factory.returnClass(typ)
      if hasattr(obj, 'getInputSpecifications'):
        subspecs = obj.getInputSpecifications()
        print('Known:', typ)
        print(subspecs)
        inputSpecification.mergeSub(subspecs)

    ####################
    # manually entered #
    ####################
    # segmenting and clustering
    segment = InputData.parameterInputFactory("Segment", strictMode=True)
    segmentGroups = InputTypes.makeEnumType('segmentGroup', 'sesgmentGroupType', ['segment', 'cluster', 'interpolate'])
    segment.addParam('grouping', segmentGroups)
    subspace = InputData.parameterInputFactory('subspace', contentType=InputTypes.StringType)
    subspace.addParam('divisions', InputTypes.IntegerType, False)
    subspace.addParam('pivotLength', InputTypes.FloatType, False)
    subspace.addParam('shift', InputTypes.StringType, False)
    segment.addSub(subspace)
    clusterEvalModeEnum = InputTypes.makeEnumType('clusterEvalModeEnum', 'clusterEvalModeType', ['clustered', 'truncated', 'full'])
    segment.addSub(InputData.parameterInputFactory('evalMode', strictMode=True, contentType=clusterEvalModeEnum))
    segment.addSub(InputData.parameterInputFactory('evaluationClusterChoice', strictMode=True, contentType=InputTypes.makeEnumType('choiceGroup', 'choiceGroupType', ['first', 'random', 'centroid'])))
    ## clusterFeatures
    segment.addSub(InputData.parameterInputFactory('clusterFeatures', contentType=InputTypes.StringListType))
    ## max cycles (for Interpolated ROMCollection)
    segment.addSub(InputData.parameterInputFactory('maxCycles', contentType=InputTypes.IntegerType))
    ## classifier
    clsfr = InputData.parameterInputFactory('Classifier', strictMode=True, contentType=InputTypes.StringType)
    clsfr.addParam('class', InputTypes.StringType, True)
    clsfr.addParam('type', InputTypes.StringType, True)
    segment.addSub(clsfr)
    ## metric
    metric = InputData.parameterInputFactory('Metric', strictMode=True, contentType=InputTypes.StringType)
    metric.addParam('class', InputTypes.StringType, True)
    metric.addParam('type', InputTypes.StringType, True)
    segment.addSub(metric)
    segment.addSub(InputData.parameterInputFactory('macroParameter', contentType=InputTypes.StringType))
    inputSpecification.addSub(segment)
    ##### END ROMCollection
    # pickledROM
    inputSpecification.addSub(InputData.parameterInputFactory('clusterEvalMode', contentType=clusterEvalModeEnum))
    inputSpecification.addSub(InputData.parameterInputFactory('maxCycles', contentType=InputTypes.IntegerType)) # for Interpolated ROMCollection
    # unsorted
    inputSpecification.addSub(InputData.parameterInputFactory("persistence", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("gradient", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("simplification", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("graph", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("knn", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("partitionPredictor", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("smooth", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("kernel", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("bandwidth", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("p", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SKLtype", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_iter", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_iter_no_change", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("tol", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha_1", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha_2", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda_1", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda_2", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("compute_score", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("threshold_lambda", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.StringType))  #bool
    inputSpecification.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.StringType))  #bool
    inputSpecification.addSub(InputData.parameterInputFactory("verbose", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("l1_ratio", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_iter", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("warm_start", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("positive", contentType=InputTypes.StringType)) #bool?
    inputSpecification.addSub(InputData.parameterInputFactory("eps", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_alphas", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("precompute", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_nonzero_coefs", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_path", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("max_n_alphas", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("criterion", contentType=CriterionInputType))
    inputSpecification.addSub(InputData.parameterInputFactory("penalty", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("dual", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("C", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("intercept_scaling", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("class_weight", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("cv", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("shuffle", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("loss", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("epsilon", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("eta0", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("solver", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("alphas", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("scoring", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("gcv_mode", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("store_cv_values", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("learning_rate", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("power_t", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("multi_class", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("kernel", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("degree", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("gamma", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("coef0", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("probability", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("shrinking", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("cache_size", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("nu", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("code_size", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_prior", contentType=InputTypes.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("class_prior", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("binarize", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_neighbors", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("weights", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("algorithm", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("leaf_size", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("metric", contentType=InputTypes.StringType)) #enum?
    inputSpecification.addSub(InputData.parameterInputFactory("radius", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("outlier_label", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("shrink_threshold", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("priors", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("reg_param", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("splitter", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("max_features", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_depth", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("min_samples_split", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("min_samples_leaf", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_leaf_nodes", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("regr", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("corr", contentType=InputTypes.StringType)) #enum?
    inputSpecification.addSub(InputData.parameterInputFactory("beta0", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("storage_mode", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("theta0", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("thetaL", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("thetaU", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("nugget", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("optimizer", contentType=InputTypes.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("random_start", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_restarts_optimizer", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("normalize_y", contentType=InputTypes.StringType))
    # GaussPolynomialROM and HDMRRom
    inputSpecification.addSub(InputData.parameterInputFactory("IndexPoints", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("IndexSet",contentType=IndexSetInputType))
    inputSpecification.addSub(InputData.parameterInputFactory('pivotParameter',contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("PolynomialOrder", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SobolOrder", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SparseGrid", contentType=InputTypes.StringType))
    InterpolationInput = InputData.parameterInputFactory('Interpolation', contentType=InputTypes.StringType)
    InterpolationInput.addParam("quad", InputTypes.StringType, False)
    InterpolationInput.addParam("poly", InputTypes.StringType, False)
    InterpolationInput.addParam("weight", InputTypes.FloatType, False)
    inputSpecification.addSub(InterpolationInput)
    # ARMA
    inputSpecification.addSub(InputData.parameterInputFactory('correlate', contentType=InputTypes.StringListType))
    inputSpecification.addSub(InputData.parameterInputFactory("P", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("Q", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("seed", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("reseedCopies", contentType=InputTypes.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("Fourier", contentType=InputTypes.FloatListType))
    inputSpecification.addSub(InputData.parameterInputFactory("nyquistScalar", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("preserveInputCDF", contentType=InputTypes.BoolType))
    ### ARMA zero filter
    zeroFilt = InputData.parameterInputFactory('ZeroFilter', contentType=InputTypes.StringType)
    zeroFilt.addParam('tol', InputTypes.FloatType)
    inputSpecification.addSub(zeroFilt)
    ### ARMA out truncation
    outTrunc = InputData.parameterInputFactory('outTruncation', contentType=InputTypes.StringListType)
    domainEnumType = InputTypes.makeEnumType('domain', 'truncateDomainType', ['positive', 'negative'])
    outTrunc.addParam('domain', domainEnumType, True)
    inputSpecification.addSub(outTrunc)
    ### ARMA specific fourier
    specFourier = InputData.parameterInputFactory('SpecificFourier', strictMode=True)
    specFourier.addParam("variables", InputTypes.StringListType, True)
    specFourier.addSub(InputData.parameterInputFactory('periods', contentType=InputTypes.FloatListType))
    inputSpecification.addSub(specFourier)
    ### ARMA multicycle
    multiYear = InputData.parameterInputFactory('Multicycle')
    multiYear.addSub(InputData.parameterInputFactory('cycles', contentType=InputTypes.IntegerType))
    growth = InputData.parameterInputFactory('growth', contentType=InputTypes.FloatType)
    growth.addParam('targets', InputTypes.StringListType, True)
    growth.addParam('start_index', InputTypes.IntegerType)
    growth.addParam('end_index', InputTypes.IntegerType)
    growthEnumType = InputTypes.makeEnumType('growth', 'armaGrowthType', ['exponential', 'linear'])
    growth.addParam('mode', growthEnumType, True)
    multiYear.addSub(growth)
    inputSpecification.addSub(multiYear)
    ### ARMA peaks
    peaks = InputData.parameterInputFactory('Peaks')
    nbin= InputData.parameterInputFactory('nbin',contentType=InputTypes.IntegerType)
    window = InputData.parameterInputFactory('window',contentType=InputTypes.FloatListType)
    window.addParam('width', InputTypes.FloatType, True)
    peaks.addSub(window)
    peaks.addSub(nbin)
    peaks.addParam('threshold', InputTypes.FloatType)
    peaks.addParam('target', InputTypes.StringType)
    peaks.addParam('period', InputTypes.FloatType)
    inputSpecification.addSub(peaks)
    # inputs for neural_network
    inputSpecification.addSub(InputData.parameterInputFactory("activation", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("batch_size", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("learning_rate_init", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("momentum", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("nesterovs_momentum", contentType=InputTypes.StringType)) # bool
    inputSpecification.addSub(InputData.parameterInputFactory("early_stopping", contentType=InputTypes.StringType)) # bool
    inputSpecification.addSub(InputData.parameterInputFactory("validation_fraction", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta_1", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta_2", contentType=InputTypes.FloatType))
    # PolyExp
    inputSpecification.addSub(InputData.parameterInputFactory("maxNumberExpTerms", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("numberExpTerms", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("maxPolyOrder", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("polyOrder", contentType=InputTypes.IntegerType))
    coeffRegressorEnumType = InputTypes.makeEnumType("coeffRegressor","coeffRegressorType",["poly","spline","nearest"])
    inputSpecification.addSub(InputData.parameterInputFactory("coeffRegressor", contentType=coeffRegressorEnumType))
    # DMD
    inputSpecification.addSub(InputData.parameterInputFactory("rankSVD", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("energyRankSVD", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("rankTLSQ", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("exactModes", contentType=InputTypes.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("optimized", contentType=InputTypes.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("dmdType", contentType=InputTypes.StringType))

    # for deep learning neural network
    #inputSpecification.addSub(InputData.parameterInputFactory("DNN", InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("hidden_layer_sizes", contentType=InputTypes.IntegerTupleType)) # list of integer
    inputSpecification.addSub(InputData.parameterInputFactory("metrics", contentType=InputTypes.StringListType)) #list of metrics
    inputSpecification.addSub(InputData.parameterInputFactory("batch_size", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("epochs", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("random_seed", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("plot_model", contentType=InputTypes.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("num_classes",contentType= InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("validation_split", contentType=InputTypes.FloatType))

    # Keras optimizer parameters
    OptimizerSettingInput = InputData.parameterInputFactory('optimizerSetting', contentType=InputTypes.StringType)
    Beta1Input = InputData.parameterInputFactory('beta_1', contentType=InputTypes.FloatType)
    Beta2Input = InputData.parameterInputFactory('beta_2', contentType=InputTypes.FloatType)
    DecayInput = InputData.parameterInputFactory('decay', contentType=InputTypes.FloatType)
    LRInput = InputData.parameterInputFactory('lr', contentType=InputTypes.FloatType)
    OptimizerInput = InputData.parameterInputFactory('optimizer', contentType=InputTypes.StringType)
    EpsilonInput = InputData.parameterInputFactory('epsilon', contentType=InputTypes.FloatType)
    MomentumInput = InputData.parameterInputFactory('momentum', contentType=InputTypes.FloatType)
    NesterovInput = InputData.parameterInputFactory('nesterov', contentType=InputTypes.StringType)
    RhoInput = InputData.parameterInputFactory('rho', contentType=InputTypes.FloatType)
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

    layerLayoutInput = InputData.parameterInputFactory('layer_layout',contentType=InputTypes.StringListType)
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
    cls.validateDict['Output'][0]['type'        ] = ['PointSet', 'HistorySet', 'DataSet']

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.initializationOptionDict = {}    # ROM initialization options
    self.amITrained = False               # boolean flag, is the ROM trained?
    self.supervisedEngine = None          # dict of ROM instances (== number of targets => keys are the targets)
    self.printTag = 'ROM MODEL'           # label
    self.cvInstance = None                # Instance of provided cross validation
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
    # for Clustered ROM
    self.addAssemblerObject('Classifier', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('Metric', InputData.Quantity.zero_to_infinity)
    self.addAssemblerObject('CV', InputData.Quantity.zero_to_one)

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
    # input params isn't picklable (right now)
    d['initializationOptionDict'].pop('paramInput', None)
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

  def applyRunInfo(self, runInfo):
    """
      Take information from the RunInfo
      @ In, runInfo, dict, RunInfo info
      @ Out, None
    """
    self.initializationOptionDict['NumThreads'] = runInfo.get('NumThreads', 1)

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
    self.initializationOptionDict['paramInput'] = paramInput
    self._initializeSupervisedGate(**self.initializationOptionDict)
    #the ROM is instanced and initialized

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
    self.supervisedEngine = LearningGate.factory.returnInstance('SupervisedGate', self.subType, **initializationOptions)

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

  def provideExpectedMetaKeys(self):
    """
      Overrides the base class method to assure child engine is also polled for its keys.
      @ In, None
      @ Out, metaKeys, set(str), names of meta variables being provided
      @ Out, metaParams, dict, the independent indexes related to expected keys
    """
    # load own keys and params
    metaKeys, metaParams = Dummy.provideExpectedMetaKeys(self)
    # add from engine
    keys, params = self.supervisedEngine.provideExpectedMetaKeys()
    metaKeys = metaKeys.union(keys)
    metaParams.update(params)
    return metaKeys, metaParams

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

  @Decorators.timingProfile
  def evaluate(self, request):
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

  @Parallel()
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

  def setAdditionalParams(self, params):
    """
      Used to set parameters at a time other than initialization (such as deserializing).
      @ In, params, dict, new params to set (internals depend on ROM)
      @ Out, None
    """
    self.supervisedEngine.setAdditionalParams(params)

  def convergence(self,trainingSet):
    """
      This is to get the cross validation score of ROM
      @ In, trainingSize, int, the size of current training size
      @ Out, cvScore, dict, the dict containing the score of cross validation
    """
    if self.subType.lower() not in ['scikitlearn','ndinvdistweight']:
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
      outputMetrics = self.cvInstance._pp.run([self, trainingSet])
      exploredTargets = []
      for cvKey, metricValues in outputMetrics.items():
        info = self.cvInstance._pp._returnCharacteristicsOfCvGivenOutputName(cvKey)
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
    initDict =  self.cvInstance._pp.initializationOptionDict
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
    ROMtargets = self.supervisedEngine.initializationOptions['Target']
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
