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
  Module containing interface with SciKit-Learn clustering
  Created on Feb 13, 2015

  @author: senrs

  TODO:
  For Clustering:
  1) paralleization: n_jobs parameter to some of the algorithms
"""

#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------

#External Modules---------------------------------------------------------------
import scipy.cluster as hier
import numpy as np
import abc
import ast
import copy
import platform
#External Modules End-----------------------------------------------------------
#Internal Modules---------------------------------------------------------------
from .utils import utils
from .utils import mathUtils
from .BaseClasses import MessageUser
from .EntityFactoryBase import EntityFactory
#Internal Modules End-----------------------------------------------------------

# FIXME: temporarily force to use Agg backend for now, otherwise it will cause segmental fault for test:
# test_dataMiningHierarchical.xml in tests/framework/PostProcessors/DataMiningPostProcessor/Clustering
# For the record, when using dendrogram, we have to force matplotlib.use('Agg')
# In the future, I think all the plots should moved to OutStreamPlots -- wangc
#display = utils.displayAvailable()
#if not display:
#  matplotlib.use('Agg')


if utils.displayAvailable() and platform.system() != 'Windows':
  import matplotlib
  matplotlib.use('TkAgg')
import matplotlib.pylab as plt

class unSupervisedLearning(utils.metaclass_insert(abc.ABCMeta), MessageUser):
  """
    This is the general interface to any unSuperisedLearning learning method.
    Essentially it contains a train, and evaluate methods
  """
  returnType = ''         ## this describe the type of information generated the
                          ## possibility are 'boolean', 'integer', 'float'

  modelType = ''          ## the broad class of the interpolator

  @staticmethod
  def checkArrayConsistency(arrayIn):
    """
      This method checks the consistency of the in-array
      @ In, arrayIn, a 1D numpy array, the array to validate
      @ Out, (consistent, errorMsg), tuple,
        consistent is a boolean where false means the input array is not a
        1D numpy array.
        errorMsg, string, the error message if the input array is inconsistent.
    """
    if type(arrayIn) != np.ndarray:
      return (False, ' The object is not a numpy array')

    ## The input data matrix kind is different for different clustering
    ## algorithms, e.g.:
    ##  [n_samples, n_features] for MeanShift and KMeans
    ##  [n_samples,n_samples]   for AffinityPropogation and SpectralCLustering

    ## In other words, MeanShift and KMeans work with points in a vector space,
    ## whereas AffinityPropagation and SpectralClustering can work with
    ## arbitrary objects, as long as a similarity measure exists for such
    ## objects. The input matrix supplied to unSupervisedLearning models as 1-D
    ## arrays of size [n_samples], (either n_features of or n_samples of them)

    if len(arrayIn.shape) != 1:
      return(False, ' The array must be 1-d')

    return (True, '')

  def __init__(self, **kwargs):
    """
      constructor for unSupervisedLearning class.
      @ In, kwargs, dict, arguments for the unsupervised learning algorithm
    """
    super().__init__()
    self.printTag = 'unSupervised'

    ## booleanFlag that controls the normalization procedure. If true, the
    ## normalization is performed. Default = True
    if kwargs != None:
      self.initOptionDict = kwargs
    else:
      self.initOptionDict = {}

    ## Labels are passed, if known a priori (optional), they used in quality
    ## estimate
    if 'Labels' in self.initOptionDict.keys():
      self.labelFeature = self.initOptionDict['Labels']
      self.initOptionDict.pop('Labels')
    else:
      self.labelFeature = None

    if 'Features' in self.initOptionDict.keys():
      self.features = self.initOptionDict['Features'].split(',')
      self.initOptionDict.pop('Features')
    else:
      self.features = None

    if 'verbosity' in self.initOptionDict:
      self.verbosity = self.initOptionDict['verbosity']
      self.initOptionDict.pop('verbosity')
    else:
      self.verbosity = None

    # average value and sigma are used for normalization of the feature data
    # a dictionary where for each feature a tuple (average value, sigma)
    self.muAndSigmaFeatures = {}
    #these need to be declared in the child classes!!!!
    self.amITrained = False

    ## The normalized training data
    self.normValues = None

  def updateFeatures(self, features):
    """
      Change the Features that this classifier targets. If this ROM is trained already, raises an error.
      @ In, features, list(str), list of new features
      @ Out, None
    """
    self.raiseAWarning('Features for learning engine type "{}" have been reset, so ROM is untrained!'.format(self.printTag))
    self.amITrained = False
    self.features = features

  def train(self, tdict, metric=None):
    """
      Method to perform the training of the unSuperVisedLearning algorithm
      NB. The unSuperVisedLearning object is committed to convert the dictionary
      that is passed (in), into the local format the interface with the kernels
      requires. So far the base class will do the translation into numpy.
      @ In, tdict, dict, training dictionary
      @ Out, None
    """
    self.metric = metric
    if not isinstance(tdict, dict):
      self.raiseAnError(IOError, ' method "train". The training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))

    featureCount = len(self.features)
    if not isinstance(tdict[utils.first(tdict.keys())],dict):
      realizationCount = utils.first(tdict.values()).size

    ############################################################################
    ## Error-handling

    ## Do all of our error handling upfront to make the meat of the code more
    ## readable:

    ## Check if the user requested something that is not available
    unidentifiedFeatures = set(self.features) - set(tdict.keys())
    if len(unidentifiedFeatures) > 0:
      ## Me write English good!
      if len(unidentifiedFeatures) == 1:
        msg = 'The requested feature: %s does not exist in the training set.' % list(unidentifiedFeatures)[0]
      else:
        msg = 'The requested features: %s do not exist in the training set.' % str(list(unidentifiedFeatures))
      self.raiseAnError(IOError, msg)

    ## Check that all of the values have the same length
    if not isinstance(utils.first(tdict.values()), dict):
      for name, val in tdict.items():
        if name in self.features and realizationCount != val.size:
          self.raiseAnError(IOError, ' In training set, the number of realizations are inconsistent among the requested features.')

    ## Check if a label feature is provided by the user and in the training data
    if self.labelFeature in tdict:
      self.labelValues = tidct[self.labelFeature]
      resp = self.checkArrayConsistency(self.labelValues)
      if not resp[0]:
        self.raiseAnError(IOError, 'In training set for ground truth labels ' + self.labelFeature + ':' + resp[1])
    else:
      self.raiseAWarning(' The ground truth labels are not known a priori')
      self.labelValues = None

    ## Not sure when this would ever happen, but check that the data you are
    ## given is a 1D array?
    # for name,val in tdict.items():
    #   if name in self.features:
    #     resp = self.checkArrayConsistency(val)
    #     if not resp[0]:
    #       self.raiseAnError(IOError, ' In training set for feature ' + name + ':' + resp[1])

    ## End Error-handling
    ############################################################################
    if metric is None:
      self.normValues = np.zeros(shape = (realizationCount, featureCount))
      for cnt, feat in enumerate(self.features):
        featureValues = tdict[feat]
        (mu,sigma) = mathUtils.normalizationFactors(featureValues)
        ## Store the normalized training data, and the normalization factors for
        ## later use
        self.normValues[:, cnt] = (featureValues - mu) / sigma
        self.muAndSigmaFeatures[feat] = (mu,sigma)
    else:
      # metric != None
      ## The dictionary represents a HistorySet
      if isinstance(utils.first(tdict.values()),dict):
        ## normalize data

        ## But why this way? This should be one of the options, this looks like
        ## a form of shape matching, however what if I don't want similar
        ## shapes, I want similar valued curves in space? sigma and mu should
        ## not be forced to be computed within a curve.
        tdictNorm={}
        for key in tdict:
          tdictNorm[key]={}
          for var in tdict[key]:
            (mu,sigma) = mathUtils.normalizationFactors(tdict[key][var])
            tdictNorm[key][var] = (tdict[key][var]-mu)/sigma

        cardinality = len(tdictNorm.keys())
        self.normValues = np.zeros((cardinality,cardinality))
        keys = list(tdictNorm.keys())
        for i in range(cardinality):
          for j in range(i,cardinality):
            # process the input data for the metric, numpy.array is required
            assert(list(tdictNorm[keys[i]].keys()) == list(tdictNorm[keys[j]].keys()))
            numParamsI = len(tdictNorm[keys[i]].keys())
            numStepsI = len(utils.first(tdictNorm[keys[i]].values()))
            numStepsJ = len(utils.first(tdictNorm[keys[j]].values()))

            inputI = np.empty((numParamsI, numStepsI))
            inputJ = np.empty((numParamsI, numStepsJ))
            for ind, params in enumerate(tdictNorm[keys[i]].keys()):
              valueI = tdictNorm[keys[i]][params]
              valueJ = tdictNorm[keys[j]][params]
              inputI[ind] = valueI
              inputJ[ind] = valueJ
            pairedData = ((inputI,None), (inputJ,None))
            # FIXME: Using loops can be very slow for large number of realizations
            self.normValues[i][j] = metric.evaluate(pairedData)
            if i != j:
              self.normValues[j][i] = self.normValues[i][j]
      else:
        ## PointSet
        normValues = np.zeros(shape = (realizationCount, featureCount))
        self.normValues = np.zeros(shape = (realizationCount, realizationCount))
        for cnt, feat in enumerate(self.features):
          featureValues = tdict[feat]
          (mu,sigma) = mathUtils.normalizationFactors(featureValues)
          normValues[:, cnt] = (featureValues - mu) / sigma
        # compute the pairwised distance for given matrix
        self.normValues = metric.evaluatePairwise((normValues,None))

    self._train()
    self.amITrained = True

  ## I'd be willing to bet this never gets called, and if it did it would crash
  ## under specific settings, namely using a history set. - unknown (maybe Dan?)
  ## -> for the record, I call it to get the labels in the ROMCollection.Clusters - talbpaul
  def evaluate(self, edict):
    """
      Method to perform the evaluation of a point or a set of points through
      the previous trained unSuperVisedLearning algorithm
      NB. The superVisedLearning object is committed to convert the dictionary
      that is passed (in), into the local format the interface with the kernels
      requires.
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluation, numpy.array, array of evaluated points
    """
    if not self.amITrained:
      self.raiseAnError('ROM must be trained before evaluating!')
    if not isinstance(edict, dict):
      self.raiseAnError(IOError, ' Method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))

    names = edict.keys()

    realizationCount = utils.first(edict.values()).size
    featureCount = len(self.features)

    ############################################################################
    ## Error-handling

    ## Do all of our error handling upfront to make the meat of the code more
    ## readable:

    ## Check if the user requested something that is not available
    unidentifiedFeatures = set(self.features) - set(edict.keys())
    if len(unidentifiedFeatures) > 0:
      ## Me write English good!
      if len(unidentifiedFeatures) == 1:
        msg = 'The requested feature: %s does not exist in the evaluate set.' % list(unidentifiedFeatures)[0]
      else:
        msg = 'The requested features: %s do not exist in the evaluate set.' % str(list(unidentifiedFeatures))
      self.raiseAnError(IOError, msg)

    for name,values in edict.items():
      resp = self.checkArrayConsistency(values)
      if not resp[0]:
        self.raiseAnError(IOError, ' In evaluate request for feature ' + name + ':' + resp[1])

    ## End Error-handling
    ############################################################################

    ## I don't think this is necessary?
    # if self.labelFeature in edict.keys():
    #   self.labelValues = edict[self.labelFeature]

    # construct the evaluation matrix
    normedValues = np.zeros(shape = (realizationCount, featureCount))
    for cnt, feature in enumerate(self.features):
      featureValues = edict[feature]
      (mu,sigma) = self.muAndSigmaFeatures[feature]
      normedValues[:, cnt] = (featureValues - mu) / sigma
    evaluation = self.__evaluateLocal__(normedValues)

    return evaluation

  def confidence(self):
    """
      This call is used to get an estimate of the confidence in the prediction
      of the clusters. The base class self.confidence checks if the clusters are
      already evaluated (trained) then calls the local confidence
      @ In, None
      @ Out, confidence, float, the confidence
    """
    if self.amITrained:
      return self.__confidenceLocal__()
    else:
      self.raiseAnError(IOError, ' The confidence check is performed before training.')

  def getDataMiningType(self):
    """
      This method is used to return the type of data mining algorithm to be employed
      @ In, none
      @ Out, none
    """
    pass


  @abc.abstractmethod
  def _train(self):
    """
      Perform training...
      @ In, none
      @ Out, none
    """

  @abc.abstractmethod
  def __evaluateLocal__(self, featureVals):
    """
      @ In,  featureVals, 2-D numpy.array, [n_samples,n_features]
      @ Out, targetVals , 1-D numpy.array, [n_samples]
    """

  @abc.abstractmethod
  def __confidenceLocal__(self):
    """
      This should return an estimation of the quality of the prediction.
      @ In, none
      @ Out, none
     """
#
#

class SciKitLearn(unSupervisedLearning):
  """
    SciKitLearn interface for unsupervised Learning
  """
  modelType = 'SciKitLearn'
  availImpl = {}

  def __init__(self, **kwargs):
    """
     constructor for SciKitLearn class.
     @ In, kwargs, dict, arguments for the SciKitLearn algorithm
     @ Out, None
    """
    unSupervisedLearning.__init__(self, **kwargs)
    if len(self.availImpl) == 0:
      import sklearn.cluster
      import sklearn.mixture
      import sklearn.manifold
      import sklearn.decomposition
      self.availImpl['cluster'] = {}  # Generalized Cluster
      self.availImpl['cluster']['AffinityPropogation'    ] = (sklearn.cluster.AffinityPropagation    , 'float')  # Perform Affinity Propagation Clustering of data.
      self.availImpl['cluster']['DBSCAN'                 ] = (sklearn.cluster.DBSCAN                 , 'float')  # Perform DBSCAN clustering from vector array or distance matrix.
      self.availImpl['cluster']['KMeans'                 ] = (sklearn.cluster.KMeans                 , 'float')  # K-Means Clustering
      self.availImpl['cluster']['MiniBatchKMeans'        ] = (sklearn.cluster.MiniBatchKMeans        , 'float')  # Mini-Batch K-Means Clustering
      self.availImpl['cluster']['MeanShift'              ] = (sklearn.cluster.MeanShift              , 'float')  # Mean Shift Clustering
      self.availImpl['cluster']['SpectralClustering'     ] = (sklearn.cluster.SpectralClustering     , 'float')  # Apply clustering to a projection to the normalized laplacian.
      self.availImpl['cluster']['Agglomerative'          ] = (sklearn.cluster.AgglomerativeClustering, 'float')  # Agglomerative Clustering - Feature of SciKit-Learn version 0.15
      #  self.availImpl['cluster']['FeatureAgglomeration'   ] = (cluster.FeatureAgglomeration   , 'float')  # - Feature of SciKit-Learn version 0.15
      #  self.availImpl['cluster']['Ward'                   ] = (cluster.Ward                   , 'float')  # Ward hierarchical clustering: constructs a tree and cuts it.

      #  self.availImpl['bicluster'] = {}
      #  self.availImpl['bicluster']['SpectralBiclustering'] = (cluster.bicluster.SpectralBiclustering, 'float')  # Spectral biclustering (Kluger, 2003).
      #  self.availImpl['bicluster']['SpectralCoclustering'] = (cluster.bicluster.SpectralCoclustering, 'float')  # Spectral Co-Clustering algorithm (Dhillon, 2001).

      self.availImpl['mixture'] = {}  # Generalized Gaussion Mixture Models (Classification)
      self.availImpl['mixture']['GMM'  ] = (sklearn.mixture.GaussianMixture  , 'float')  # Gaussian Mixture Model
      ## Comment is not even right on it, but the DPGMM is being deprecated by SKL who
      ## admits that it is not working correctly which also explains why it is buried in
      ## their documentation.
      # self.availImpl['mixture']['DPGMM'] = (sklearn.mixture.DPGMM, 'float')  # Variational Inference for the Infinite Gaussian Mixture Model.
      self.availImpl['mixture']['VBGMM'] = (sklearn.mixture.BayesianGaussianMixture, 'float')  # Variational Inference for the Gaussian Mixture Model

      self.availImpl['manifold'] = {}  # Manifold Learning (Embedding techniques)
      self.availImpl['manifold']['LocallyLinearEmbedding'  ] = (sklearn.manifold.LocallyLinearEmbedding  , 'float')  # Locally Linear Embedding
      self.availImpl['manifold']['Isomap'                  ] = (sklearn.manifold.Isomap                  , 'float')  # Isomap
      self.availImpl['manifold']['MDS'                     ] = (sklearn.manifold.MDS                     , 'float')  # MultiDimensional Scaling
      self.availImpl['manifold']['SpectralEmbedding'       ] = (sklearn.manifold.SpectralEmbedding       , 'float')  # Spectral Embedding for Non-linear Dimensionality Reduction
      #  self.availImpl['manifold']['locally_linear_embedding'] = (sklearn.manifold.locally_linear_embedding, 'float')  # Perform a Locally Linear Embedding analysis on the data.
      #  self.availImpl['manifold']['spectral_embedding'      ] = (sklearn.manifold.spectral_embedding      , 'float')  # Project the sample on the first eigen vectors of the graph Laplacian.

      self.availImpl['decomposition'] = {}  # Matrix Decomposition
      self.availImpl['decomposition']['PCA'                 ] = (sklearn.decomposition.PCA                 , 'float')  # Principal component analysis (PCA)
     # self.availImpl['decomposition']['ProbabilisticPCA'    ] = (sklearn.decomposition.ProbabilisticPCA    , 'float')  # Additional layer on top of PCA that adds a probabilistic evaluationPrincipal component analysis (PCA)
      self.availImpl['decomposition']['RandomizedPCA'       ] = (sklearn.decomposition.PCA       , 'float')  # Principal component analysis (PCA) using randomized SVD
      self.availImpl['decomposition']['KernelPCA'           ] = (sklearn.decomposition.KernelPCA           , 'float')  # Kernel Principal component analysis (KPCA)
      self.availImpl['decomposition']['FastICA'             ] = (sklearn.decomposition.FastICA             , 'float')  # FastICA: a fast algorithm for Independent Component Analysis.
      self.availImpl['decomposition']['TruncatedSVD'        ] = (sklearn.decomposition.TruncatedSVD        , 'float')  # Dimensionality reduction using truncated SVD (aka LSA).
      self.availImpl['decomposition']['SparsePCA'           ] = (sklearn.decomposition.SparsePCA           , 'float')  # Sparse Principal Components Analysis (SparsePCA)
      self.availImpl['decomposition']['MiniBatchSparsePCA'  ] = (sklearn.decomposition.MiniBatchSparsePCA  , 'float')  # Mini-batch Sparse Principal Components Analysis
      #  self.availImpl['decomposition']['ProjectedGradientNMF'] = (sklearn.decomposition.ProjectedGradientNMF, 'float')  # Non-Negative matrix factorization by Projected Gradient (NMF)
      #  self.availImpl['decomposition']['FactorAnalysis'      ] = (sklearn.decomposition.FactorAnalysis      , 'float')  # Factor Analysis (FA)
      #  self.availImpl['decomposition']['NMF'                 ] = (sklearn.decomposition.NMF                 , 'float')  # Non-Negative matrix factorization by Projected Gradient (NMF)
      #  self.availImpl['decomposition']['SparseCoder'         ] = (sklearn.decomposition.SparseCoder         , 'float')  # Sparse coding
      #  self.availImpl['decomposition']['DictionaryLearning'  ] = (sklearn.decomposition.DictionaryLearning  , 'float')  # Dictionary Learning
      #  self.availImpl['decomposition']['MiniBatchDictionaryLearning'] = (sklearn.decomposition.MiniBatchDictionaryLearning, 'float')  # Mini-batch dictionary learning
      #  self.availImpl['decomposition']['fastica'                    ] = (sklearn.decomposition.fastica                    , 'float')  # Perform Fast Independent Component Analysis.
      #  self.availImpl['decomposition']['dict_learning'              ] = (sklearn.decomposition.dict_learning              , 'float')  # Solves a dictionary learning matrix factorization problem.

      #  self.availImpl['covariance'] = {}  # Covariance Estimators
      #  self.availImpl['covariance']['EmpiricalCovariance'] = (sklearn.covariance.EmpiricalCovariance, 'float')  # Maximum likelihood covariance estimator
      #  self.availImpl['covariance']['EllipticEnvelope'   ] = (sklearn.covariance.EllipticEnvelope   , 'float')  # An object for detecting outliers in a Gaussian distributed dataset.
      #  self.availImpl['covariance']['GraphLasso'         ] = (sklearn.covariance.GraphLasso         , 'float')  # Sparse inverse covariance estimation with an l1-penalized estimator.
      #  self.availImpl['covariance']['GraphLassoCV'       ] = (sklearn.covariance.GraphLassoCV       , 'float')  # Sparse inverse covariance w/ cross-validated choice of the l1 penalty
      #  self.availImpl['covariance']['LedoitWolf'         ] = (sklearn.covariance.LedoitWolf         , 'float')  # LedoitWolf Estimator
      #  self.availImpl['covariance']['MinCovDet'          ] = (sklearn.covariance.MinCovDet          , 'float')  # Minimum Covariance Determinant (MCD): robust estimator of covariance
      #  self.availImpl['covariance']['OAS'                ] = (sklearn.covariance.OAS                , 'float')  # Oracle Approximating Shrinkage Estimator
      #  self.availImpl['covariance']['ShrunkCovariance'   ] = (sklearn.covariance.ShrunkCovariance   , 'float')  # Covariance estimator with shrinkage

      #  self.availImpl['neuralNetwork'] = {}  # Covariance Estimators
      #  self.availImpl['neuralNetwork']['BernoulliRBM'] = (neural_network.BernoulliRBM, 'float')  # Bernoulli Restricted Boltzmann Machine (RBM).

    self.printTag = 'SCIKITLEARN'

    if 'SKLtype' not in self.initOptionDict.keys():
      self.raiseAnError(IOError, ' to define a scikit learn unSupervisedLearning Method the SKLtype keyword is needed (from KDD ' + self.name + ')')

    SKLtype, SKLsubType = self.initOptionDict['SKLtype'].split('|')
    self.initOptionDict.pop('SKLtype')

    if not SKLtype in self.__class__.availImpl.keys():
      self.raiseAnError(IOError, ' Unknown SKLtype ' + SKLtype + '(from KDD ' + self.name + ')')

    if not SKLsubType in self.__class__.availImpl[SKLtype].keys():
      self.raiseAnError(IOError, ' Unknown SKLsubType ' + SKLsubType + '(from KDD ' + self.name + ')')

    self.SKLtype = SKLtype
    self.SKLsubType = SKLsubType

    self.__class__.returnType = self.__class__.availImpl[SKLtype][SKLsubType][1]
    self.Method = self.__class__.availImpl[SKLtype][SKLsubType][0]()

    paramsDict = self.Method.get_params()

    ## Let's only keep the parameters that the Method understands, throw
    ## everything else away, maybe with a warning message?
    tempDict = {}

    for key, value in self.initOptionDict.items():
      if key in paramsDict:
        try:
          tempDict[key] = ast.literal_eval(value)
        except:
          tempDict[key] = value
      else:
        self.raiseAWarning('Ignoring unknown parameter %s to the method of type %s' % (key, SKLsubType))
    self.initOptionDict = tempDict

    self.Method.set_params(**self.initOptionDict)
    self.normValues = None
    self.outputDict = {}

  def _train(self):
    """
      Perform training on samples in self.normValues: array,
      shape = [n_samples, n_features] or [n_samples, n_samples]
      @ In, None
      @ Out, None
    """
    import sklearn.cluster
    import sklearn.neighbors
    ## set bandwidth for MeanShift clustering
    if hasattr(self.Method, 'bandwidth'):
      if 'bandwidth' not in self.initOptionDict.keys():
        self.initOptionDict['bandwidth'] = sklearn.cluster.estimate_bandwidth(self.normValues,quantile=0.3)
      self.Method.set_params(**self.initOptionDict)

    ## We need this connectivity if we want to use structured ward
    if hasattr(self.Method, 'connectivity'):
      ## we should find a smart way to define the number of neighbors instead of
      ## default constant integer value(10)
      connectivity = sklearn.neighbors.kneighbors_graph(self.normValues, n_neighbors = min(10,len(self.normValues[:,0])-1))
      connectivity = 0.5 * (connectivity + connectivity.T)
      self.initOptionDict['connectivity'] = connectivity
      self.Method.set_params(**self.initOptionDict)

    self.outputDict['outputs'] = {}
    self.outputDict['inputs' ] = self.normValues

    ## This is the stuff that will go into the solution export or just float
    ## around and maybe never be used
    self.metaDict = {}

    ## What are you doing here? Calling half of these methods does nothing
    ## unless you store the data somewhere. If you are going to blindly call
    ## whatever methods that exist in the class, then at least store them for
    ## later. Why is this done again on the PostProcessor side? I am struggling
    ## to understand what this code's purpose is except to obfuscate our
    ## interaction with skl.

    # if   hasattr(self.Method, 'fit_predict'):
    #   self.Method.fit_predict(self.normValues)
    # elif hasattr(self.Method, 'predict'):
    #   self.Method.fit(self.normValues)
    #   self.Method.predict(self.normValues)
    # elif hasattr(self.Method, 'fit_transform'):
    #   self.Method.fit_transform(self.normValues)
    # elif hasattr(self.Method, 'transform'):
    #   self.Method.fit(self.normValues)
    #   self.Method.transform(self.normValues)
    self.Method.fit(self.normValues)

    ## I don't care what algorithm you ran, these are the only things I care
    ## about, if I find one of them, then I am going to save it in our defined
    ## variable names
    variableMap = {'labels_': 'labels',
                   'embedding_': 'embeddingVectors',
                   'embedding_vectors_': 'embeddingVectors'}

    ## This will store stuff that should go into the solution export, but
    ## these each need some massaging so we will not handle this automatically.
    # metaMap = {'cluster_centers_': 'clusterCenters',
    #            'means_': 'means',
    #            'covars_': 'covars'}

    ## Not used right now, but maybe someone will want it?
    # otherMap = {'n_clusters': 'noClusters',
    #             'weights_': 'weights',
    #             'cluster_centers_indices_': 'clusterCentersIndices',
    #             'precs_': 'precs',
    #             'noComponents_': 'noComponents',
    #             'reconstructionError': 'reconstruction_error_',
    #             'explained_variance_': 'explainedVariance',
    #             'explained_variance_ratio_': 'explainedVarianceRatio'}

    for key,val in self.Method.__dict__.items():
      if key in variableMap:
        ## Translate the skl name to our naming convention
        self.outputDict['outputs'][variableMap[key]] = copy.deepcopy(val)
      ## The meta information needs special handling otherwise, we could just
      ## do this here and be done in two lines
      # if key in metaMap:
      #   self.metaDict[metaMap[key]] = copy.deepcopy(val)

    ## Below generates the output Dictionary from the trained algorithm, can be
    ## defined in a new method....
    if 'cluster' == self.SKLtype:
      if hasattr(self.Method, 'cluster_centers_') :
        centers = self.Method.cluster_centers_
      elif self.metric is None:
        ## This methods is used by any other clustering algorithm that does
        ## not generatecluster_centers_ to generate the cluster centers as the
        ## average location of all points in the cluster.
        if hasattr(self.Method,'n_clusters'):
          numClusters = self.Method.n_clusters
        else:
          numClusters = len(set(self.Method.labels_))


        centers = np.zeros([numClusters,len(self.features)])
        counter = np.zeros(numClusters)
        for val,index in enumerate(self.Method.labels_):
          centers[index] += self.normValues[val]
          counter[index] += 1

        for index,val in enumerate(centers):
          if counter[index] == 0.:
            self.raiseAnError(RuntimeError, 'The data-mining clustering method '
                                            + str(self.Method)
                                            + ' has generated a 0-size cluster')
          centers[index] = centers[index] / float(counter[index])
      else:
        centers = None

      if centers is not None:
        ## I hope these arrays are consistently ordered...
        ## We are mixing our internal storage of muAndSigma with SKLs
        ## representation of our data, I believe it is fair to say that we
        ## hand the data to SKL in the same order that we have it stored.
        for cnt, feature in enumerate(self.features):
          (mu,sigma) = self.muAndSigmaFeatures[feature]
          for center in centers:
            center[cnt] = center[cnt] * sigma + mu

        self.metaDict['clusterCenters'] = centers
    elif 'mixture' == self.SKLtype:
      # labels = self.Method.fit_predict(self.normValues)
      ## The fit_predict is not available in all versions of sklearn for GMMs
      ## besides the data should already be fit above
      labels = self.Method.predict(self.normValues)

      self.outputDict['outputs']['labels'] = labels

      if hasattr(self.Method, 'converged_'):
        if not self.Method.converged_:
          self.raiseAnError(RuntimeError, self.SKLtype + '|' + self.SKLsubType
                                          + ' did not converged. (from KDD->'
                                          + self.SKLsubType + ')')

      ## For both means and covars below:
      ##   We are mixing our internal storage of muAndSigma with SKLs
      ##   representation of our data, I believe it is fair to say that we
      ##   hand the data to SKL in the same order that we have it stored.
      if hasattr(self.Method, 'means_'):
        means = copy.deepcopy(self.Method.means_)

        for cnt, feature in enumerate(self.features):
          (mu,sigma) = self.muAndSigmaFeatures[feature]
          for center in means:
            center[cnt] = center[cnt] * sigma + mu
        self.metaDict['means'] = means

      if hasattr(self.Method, 'covariances_') :
        covariance = copy.deepcopy(self.Method.covariances_)

        for row, rowFeature in enumerate(self.features):
          rowSigma = self.muAndSigmaFeatures[rowFeature][1]
          for col, colFeature in enumerate(self.features):
            colSigma = self.muAndSigmaFeatures[colFeature][1]
            #if covariance type == full, the shape is (n_components, n_features, n_features)
            if len(covariance.shape) == 3:
              covariance[:,row,col] = covariance[:,row,col] * rowSigma * colSigma
            else:
              #XXX if covariance type == diag, this will be wrong.
              covariance[row,col] = covariance[row,col] * rowSigma * colSigma
        self.metaDict['covars'] = covariance
    elif 'decomposition' == self.SKLtype:
      if 'embeddingVectors' not in self.outputDict['outputs']:
        if hasattr(self.Method, 'transform'):
          embeddingVectors = self.Method.transform(self.normValues)
          self.outputDict['outputs']['embeddingVectors'] = embeddingVectors
        elif hasattr(self.Method, 'fit_transform'):
          embeddingVectors = self.Method.fit_transform(self.normValues)
          self.outputDict['outputs']['embeddingVectors'] = embeddingVectors
        else:
          self.raiseAWarning('The embedding vectors could not be computed.')

      if hasattr(self.Method, 'components_'):
        self.metaDict['components'] = self.Method.components_

      if hasattr(self.Method, 'means_'):
        self.metaDict['means'] = self.Method.means_

      if hasattr(self.Method, 'explained_variance_'):
        self.explainedVariance_ = copy.deepcopy(self.Method.explained_variance_)
        self.metaDict['explainedVariance'] = self.explainedVariance_

      if hasattr(self.Method, 'explained_variance_ratio_'):
        self.metaDict['explainedVarianceRatio'] = self.Method.explained_variance_ratio_

  def __evaluateLocal__(self, featureVals):
    """
      Method to return labels of an already trained unSuperVised algorithm.
      @ In, featureVals, numpy.array, feature values
      @ Out, labels, numpy.array, labels
    """
    if hasattr(self.Method, 'predict'):
      labels = self.Method.predict(featureVals)
    else:
      labels = self.Method.fit_predict(featureVals)


    return labels

  def __confidenceLocal__(self):
    """
      This should return an estimation dictionary of the quality of the
      prediction.
      @ In, None
      @ Out, self.outputdict['confidence'], dict, dictionary of the confidence
      metrics of the algorithms
    """
    import sklearn.metrics
    self.outputDict['confidence'] = {}

    ## I believe you should always have labels populated when dealing with a
    ## clustering algorithm, this second condition may be redundant
    if 'cluster' == self.SKLtype and 'labels' in self.outputDict['outputs']:
      labels = self.outputDict['outputs']['labels']

      if np.unique(labels).size > 1:
        self.outputDict['confidence']['silhouetteCoefficient'] = sklearn.metrics.silhouette_score(self.normValues , labels)

      if hasattr(self.Method, 'inertia_'):
        self.outputDict['confidence']['inertia'] = self.Method.inertia_

      ## If we have ground truth labels, then compute some additional confidence
      ## metrics
      if self.labelValues is not None:
        self.outputDict['confidence']['homogeneity'              ] =          sklearn.metrics.homogeneity_score(self.labelValues, labels)
        self.outputDict['confidence']['completenes'              ] =         sklearn.metrics.completeness_score(self.labelValues, labels)
        self.outputDict['confidence']['vMeasure'                 ] =            sklearn.metrics.v_measure_score(self.labelValues, labels)
        self.outputDict['confidence']['adjustedRandIndex'        ] =        sklearn.metrics.adjusted_rand_score(self.labelValues, labels)
        self.outputDict['confidence']['adjustedMutualInformation'] = sklearn.metrics.adjusted_mutual_info_score(self.labelValues, labels)
    elif 'mixture' == self.SKLtype:
      if hasattr(self.Method, 'aic'):
        self.outputDict['confidence']['aic'  ] = self.Method.aic(self.normValues)   ## Akaike Information Criterion
        self.outputDict['confidence']['bic'  ] = self.Method.bic(self.normValues)   ## Bayesian Information Criterion
        self.outputDict['confidence']['score'] = self.Method.score(self.normValues) ## log probabilities of each data point

    return self.outputDict['confidence']

  def getDataMiningType(self):
    """
      This method is used to return the type of data mining algorithm to be employed
      @ In, none
      @ Out, self.SKLtype, string, type of data mining algorithm
    """
    return self.SKLtype

#
#

class temporalSciKitLearn(unSupervisedLearning):
  """
    Data mining library to perform SciKitLearn algorithms along temporal data
  """

  def __init__(self, **kwargs):
    """
      constructor for temporalSciKitLearn class.
      @ In, kwargs, arguments for the SciKitLearn algorithm
      @ Out, None
    """
    unSupervisedLearning.__init__(self, **kwargs)
    self.printTag = 'TEMPORALSCIKITLEARN'

    if 'SKLtype' not in self.initOptionDict.keys():
      self.raiseAnError(IOError, ' to define a scikit learn unSupervisedLearning Method the SKLtype keyword is needed (from KDD ' + self.name + ')')

    self.SKLtype, self.SKLsubType = self.initOptionDict['SKLtype'].split('|')
    self.pivotParameter = self.initOptionDict.get('pivotParameter', 'Time')

    #Pop necessary to keep from confusing SciKitLearn with extra option
    self.reOrderStep = int(self.initOptionDict.pop('reOrderStep', 5))

    # return a SciKitLearn instance as engine for SKL data mining
    self.SKLEngine = factory.returnInstance('SciKitLearn', **self.initOptionDict)

    self.normValues = None
    self.outputDict = {}

  @staticmethod
  def checkArrayConsistency(arrayin, shape):
    """
      This method checks the consistency of the in-array
      @ In, object... It should be an array
      @ Out, tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    """
    if type(arrayin) != np.ndarray:
      return (False, ' The object is not a numpy array')

    if arrayin.shape[0] != shape[0] or arrayin.shape[1] != shape[1]:
      return (False, ' The object shape is not correct')

    ## The input data matrix kind is different for different clustering methods
    ## e.g. [n_samples, n_features] for MeanShift and KMeans
    ##     [n_samples,n_samples]  for AffinityPropogation and SpectralClustering
    ## In other words, MeanShift and KMeans work with points in a vector space,
    ## whereas AffinityPropagation and SpectralClustering can work with
    ## arbitrary objects, as long as a similarity measure exists for such
    ## objects
    ## The input matrix supplied to unSupervisedLearning models as 1-D arrays o
    ##  size [n_samples], (either n_features of or n_samples of them)
    # if len(arrayin.shape) != 1: return(False, ' The array must be 1-d')
    return (True, '')

  def __deNormalizeData__(self,feat,t,data):
    """
      Method to denormalize data based on the mean and standard deviation stored
      in self.
      @In, feat, string, the feature for which the input is to be denormalized
      @In, t, float, time step identifier
      @In, data, list, input values to be denormalized
      @Out, deNormData, list, output values after denormalization
    """
    N = data.shape[0]
    deNormData = np.zeros(shape=data.shape)
    mu, sig = self.muAndSigmaFeatures[feat][0,t], self.muAndSigmaFeatures[feat][1,t]
    for n in range(N):
      deNormData[n] = data[n]*sig+mu
    return deNormData

  def train(self, tdict):
    """
      Method to train this class.
      @ In, tdict, dictionary, training dictionary
      @ Out, None
    """
    ## need to overwrite train method because time dependent data mining
    ## requires different treatment of input
    if type(tdict) != dict:
      self.raiseAnError(IOError, ' method "train". The training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))

    names = list(tdict.keys())
    values = list(tdict.values())

    self.numberOfSample = values[0].shape[0]
    self.numberOfHistoryStep = values[0].shape[1]

    ############################################################################
    ## Error-handling

    ## Do all of our error handling upfront to make the meat of the code more
    ## readable:

    ## Check if the user requested something that is not available
    unidentifiedFeatures = set(self.features) - set(names)
    if len(unidentifiedFeatures) > 0:
      ## Me write English good!
      if len(unidentifiedFeatures) == 1:
        msg = 'The requested feature: %s does not exist in the training set.' % list(unidentifiedFeatures)[0]
      else:
        msg = 'The requested features: %s do not exist in the training set.' % str(list(unidentifiedFeatures))
      self.raiseAnError(IOError, msg)

    ## Check that all of the values have the same length

    ## Check if a label feature is provided by the user and in the training data
    if self.labelFeature in names:
      self.labelValues = tidct[self.labelFeature]
      resp = self.checkArrayConsistency(self.labelValues,[self.numberOfSample, self.numberOfHistoryStep])
      if not resp[0]:
        self.raiseAnError(IOError, 'In training set for ground truth labels ' + self.labelFeature + ':' + resp[1])
    else:
      self.raiseAWarning(' The ground truth labels are not known a priori')
      self.labelValues = None

    ## End Error-handling
    ############################################################################

    self.normValues = {}

    for cnt,feature in enumerate(self.features):
      resp = self.checkArrayConsistency(tdict[feature], [self.numberOfSample, self.numberOfHistoryStep])
      if not resp[0]:
        self.raiseAnError(IOError, ' In training set for feature ' + feature + ':' + resp[1])

      self.normValues[feature] = np.zeros(shape = tdict[feature].shape)
      self.muAndSigmaFeatures[feature] = np.zeros(shape=(2,self.numberOfHistoryStep))

      for t in range(self.numberOfHistoryStep):
        featureValues = tdict[feature][:,t]
        (mu,sigma) = mathUtils.normalizationFactors(featureValues)

        ## Store the normalized training data, and the normalization factors for
        ## later use
        self.normValues[feature][:,t] = (featureValues - mu) / sigma

        self.muAndSigmaFeatures[feature][0,t] = mu
        self.muAndSigmaFeatures[feature][1,t] = sigma

    self.inputDict = tdict
    self._train()
    self.amITrained = True

  def _train(self):
    """
      Method to train this class.
    """

    self.outputDict['outputs'] = {}
    self.outputDict['inputs' ] = self.normValues

    ## This is the stuff that will go into the solution export or just float
    ## around and maybe never be used
    self.metaDict = {}

    for t in range(self.numberOfHistoryStep):
      sklInput = {}
      for feat in self.features:
        sklInput[feat] = self.inputDict[feat][:,t]

      self.SKLEngine.features = sklInput
      self.SKLEngine.train(sklInput)
      self.SKLEngine.confidence()

      ## Store everything from the specific timestep's SKLEngine into a running
      ## list
      for key,val in self.SKLEngine.outputDict['outputs'].items():
        if key not in self.outputDict['outputs']:
          self.outputDict['outputs'][key] = {} # [None]*self.numberOfHistoryStep
        self.outputDict['outputs'][key][t] = val

      for key,val in self.SKLEngine.metaDict.items():
        if key not in self.metaDict:
          self.metaDict[key] = {} # [None]*self.numberOfHistoryStep
        self.metaDict[key][t] = val

      if self.SKLtype in ['cluster']:

        if 'clusterCenters' not in self.metaDict.keys():
          self.metaDict['clusterCenters'] = {}

        if 'clusterCentersIndices' not in self.metaDict.keys():
          self.metaDict['clusterCentersIndices'] = {}

        # # collect labels
        # if hasattr(self.SKLEngine.Method, 'labels_'):
        #   self.outputDict['labels'][t] = self.SKLEngine.Method.labels_

        # # collect cluster centers
        if hasattr(self.SKLEngine.Method, 'cluster_centers_'):
          self.metaDict['clusterCenters'][t] = np.zeros(shape=self.SKLEngine.metaDict['clusterCenters'].shape)
          for cnt, feat in enumerate(self.features):
            self.metaDict['clusterCenters'][t][:,cnt] = self.SKLEngine.metaDict['clusterCenters'][:,cnt]
        else:
          self.metaDict['clusterCenters'][t] = self.__computeCenter__(sklInput, self.outputDict['outputs']['labels'][t])

        # collect number of clusters
        if hasattr(self.SKLEngine.Method, 'n_clusters'):
          noClusters = self.SKLEngine.Method.n_clusters
        else:
          noClusters = self.metaDict['clusterCenters'][t].shape[0]

        # collect cluster indices
        # if hasattr(self.SKLEngine.Method, 'cluster_centers_indices_'):
        #   self.metaDict['clusterCentersIndices'][t] = self.SKLEngine.Method.cluster_centers_indices_
        #   self.metaDict['clusterCentersIndices'][t] = range(noClusters)
        # else:
        #   self.metaDict['clusterCentersIndices'][t] = range(noClusters)  # use list(set(self.SKLEngine.Method.labels_)) to collect outliers
        self.metaDict['clusterCentersIndices'][t] = list(range(noClusters))

        # # collect optional output
        # if hasattr(self.SKLEngine.Method, 'inertia_'):
        #   if 'inertia' not in self.outputDict.keys(): self.outputDict['inertia'] = {}
        #   self.outputDict['inertia'][t] = self.SKLEngine.Method.inertia_

        # re-order clusters
        if t > 0:
          remap = self.__reMapCluster__(t, self.metaDict['clusterCenters'], self.metaDict['clusterCentersIndices'])
          for n in range(len(self.metaDict['clusterCentersIndices'][t])):
            self.metaDict['clusterCentersIndices'][t][n] = remap[self.metaDict['clusterCentersIndices'][t][n]]
          for n in range(len(self.outputDict['outputs']['labels'][t])):
            if self.outputDict['outputs']['labels'][t][n] >=0:
              self.outputDict['outputs']['labels'][t][n] = remap[self.SKLEngine.Method.labels_[n]]
          ## TODO: Remap the cluster centers now...
      elif self.SKLtype in ['mixture']:
        if 'means' not in self.metaDict.keys():
          self.metaDict['means'] = {}
        if 'componentMeanIndices' not in self.metaDict.keys():
          self.metaDict['componentMeanIndices'] = {}

        # # collect component membership
        if 'labels' not in self.outputDict['outputs']:
          self.outputDict['outputs']['labels'] = {}
        self.outputDict['outputs']['labels'][t] = self.SKLEngine.evaluate(sklInput)

        # # collect component means
        if hasattr(self.SKLEngine.Method, 'means_'):
          self.metaDict['means'][t] = np.zeros(shape=self.SKLEngine.Method.means_.shape)
          for cnt, feat in enumerate(self.features):
            self.metaDict['means'][t][:,cnt] = self.__deNormalizeData__(feat,t,self.SKLEngine.Method.means_[:,cnt])
        else:
          self.metaDict['means'][t] = self.__computeCenter__(Input['Features'], self.outputDict['labels'][t])

        # # collect number of components
        if hasattr(self.SKLEngine.Method, 'n_components'):
          numComponents = self.SKLEngine.Method.n_components
        else:
          numComponents = self.metaDict['means'][t].shape[0]

        # # collect component indices
        self.metaDict['componentMeanIndices'][t] = list(range(numComponents))

        # # collect optional output
        if hasattr(self.SKLEngine.Method, 'weights_'):
          if 'weights' not in self.metaDict.keys():
            self.metaDict['weights'] = {}
          self.metaDict['weights'][t] = self.SKLEngine.Method.weights_

        if 'covars' in self.SKLEngine.metaDict:
          if 'covars' not in self.metaDict.keys():
            self.metaDict['covars'] = {}
          self.metaDict['covars'][t] = self.SKLEngine.metaDict['covars']

        if hasattr(self.SKLEngine.Method, 'precs_'):
          if 'precs' not in self.metaDict.keys():
            self.metaDict['precs'] = {}
          self.metaDict['precs'][t] = self.SKLEngine.Method.precs_

        # if hasattr(self.SKLEngine.Method, 'converged_'):
        #   if 'converged' not in self.outputDict.keys():
        #     self.outputDict['converged'] = {}
        #   self.outputDict['converged'][t] = self.SKLEngine.Method.converged_

        # re-order components
        if t > 0:
          remap = self.__reMapCluster__(t, self.metaDict['means'], self.metaDict['componentMeanIndices'])
          for n in range(len(self.metaDict['componentMeanIndices'][t])):
            self.metaDict['componentMeanIndices'][t][n] = remap[self.metaDict['componentMeanIndices'][t][n]]

          for n in range(len(self.outputDict['outputs']['labels'][t])):
            if self.outputDict['outputs']['labels'][t][n] >=0:
              self.outputDict['outputs']['labels'][t][n] = remap[self.outputDict['outputs']['labels'][t][n]]
      elif 'manifold' == self.SKLtype:
        # if 'noComponents' not in self.outputDict.keys():
        #   self.outputDict['noComponents'] = {}

        if 'embeddingVectors' not in self.outputDict['outputs']:
          self.outputDict['outputs']['embeddingVectors'] = {}

        if hasattr(self.SKLEngine.Method, 'embedding_'):
          self.outputDict['outputs']['embeddingVectors'][t] = self.SKLEngine.Method.embedding_

        if 'transform' in dir(self.SKLEngine.Method):
          self.outputDict['outputs']['embeddingVectors'][t] = self.SKLEngine.Method.transform(self.SKLEngine.normValues)
        elif 'fit_transform' in dir(self.SKLEngine.Method):
          self.outputDict['outputs']['embeddingVectors'][t] = self.SKLEngine.Method.fit_transform(self.SKLEngine.normValues)

        # if hasattr(self.SKLEngine.Method, 'reconstruction_error_'):
        #   if 'reconstructionError_' not in self.outputDict.keys():
        #     self.outputDict['reconstructionError_'] = {}
        #   self.outputDict['reconstructionError_'][t] = self.SKLEngine.Method.reconstruction_error_
      elif 'decomposition' == self.SKLtype:
        for var in ['explainedVarianceRatio','means','explainedVariance',
                    'components']:
          if var not in self.metaDict:
            self.metaDict[var] = {}

        if hasattr(self.SKLEngine.Method, 'components_'):
          self.metaDict['components'][t] = self.SKLEngine.Method.components_

        ## This is not the same thing as the components above! This is the
        ## transformed data, the other composes the transformation matrix to get
        ## this. Whoever designed this, you are causing me no end of headaches
        ## with this code... I am pretty sure this can all be handled within the
        ## post-processor rather than adding this frankenstein of code just to
        ## gain access to the skl techniques.
        if 'embeddingVectors' not in self.outputDict['outputs']:
          if 'transform' in dir(self.SKLEngine.Method):
            embeddingVectors = self.SKLEngine.Method.transform(self.SKLEngine.normValues)
          elif 'fit_transform' in dir(self.SKLEngine.Method):
            embeddingVectors = self.SKLEngine.Method.fit_transform(self.SKLEngine.normValues)
          self.outputDict['outputs']['embeddingVectors'][t] = embeddingVectors

        if hasattr(self.SKLEngine.Method, 'means_'):
          self.metaDict['means'][t] = self.SKLEngine.Method.means_
        if hasattr(self.SKLEngine.Method, 'explained_variance_'):
          self.metaDict['explainedVariance'][t] = self.SKLEngine.Method.explained_variance_
        if hasattr(self.SKLEngine.Method, 'explained_variance_ratio_'):
          self.metaDict['explainedVarianceRatio'][t] = self.SKLEngine.Method.explained_variance_ratio_

      else:
        self.raiseAnError(IOError, 'Unknown type: ' + str(self.SKLtype))

  def __computeCenter__(self, data, labels):
    """
      Method to compute cluster center for clustering algorithms that do not return such information.
      This is needed to re-order cluster number
      @In, data, dict, each value of the dict is a 1-d array of data
      @In, labels, list, list of label for each sample
      @Out, clusterCenter, array, shape = [no_clusters, no_features], center coordinate
    """
    point = {}
    for cnt, l in enumerate(labels):
      if l >= 0 and l not in point.keys():
        point[l] = []
      if l >= 0:
        point[l].append(cnt)
    noCluster = len(point.keys())

    if noCluster == 0:
      self.raiseAnError(ValueError, 'number of cluster is 0!!!')

    clusterCenter = np.zeros(shape=(noCluster,len(self.features)))

    for cnt, feat in enumerate(self.features):
      for ind, l in enumerate(point.keys()):
        clusterCenter[ind,cnt] = np.average(data[feat][point[l]])

    return clusterCenter

  def __computeDist__(self,t,n1,n2,dataCenter,opt):
    """
      Computes the distance between two cluster centers.
      Four different distance metrics are implemented, which can be specified by input opt
      @In, t, float, current time
      @In, n1, integer, center index 1
      @In, n2, integer, center index 2
      @In, dataCenter, dict, each value contains the center coordinate at each time step
      @In, opt, string, specifies which distance metric to use
      @Out, dist, float, distance between center n1 and center n2
    """
    x1 = dataCenter[t-1][n1,:]
    x2 = dataCenter[t][n2,:]

    if opt in ['Distance']:
      dist = np.sqrt(np.dot(x1-x2,x1-x2))
      return dist

    if opt in ['Overlap']:
      l1 = self.outputDict['outputs']['labels'][t-1]
      l2 = self.SKLEngine.Method.labels_

      point1 = []
      point2 = []

      for n in range(len(l1)):
        if l1[n] == n1:
          point1.append(n)
      for n in range(len(l2)):
        if l2[n] == n2:
          point2.append(n)
      dist = - len(set(point1).intersection(point2))

      return dist

    if opt in ['DistVariance']:
      l1 = self.outputDict['outputs']['labels'][t-1]
      l2 = self.SKLEngine.Method.labels_

      dist = np.sqrt(np.dot(x1-x2,x1-x2))
      v1 = v2 = N1 = N2 = 0
      noFeat = len(self.features)
      for n in range(len(l1)):
        # compute variance of points with label l1
        if l1[n] == n1:
          x = np.zeros(shape=(noFeat,))
          for cnt, feat in enumerate(self.features):
            x[cnt] = self.inputDict[feat][n,t-1]
          v1 += np.sqrt(np.dot(x-x1,x-x1))**2
          N1 += 1
      for n in range(len(l2)):
        # compute variance of points with label l2
        if l2[n] == n2:
          x = np.zeros(shape=(noFeat,))
          for cnt, feat in enumerate(self.features):
            x[cnt] = self.inputDict[feat][n,t]
          v2 += np.sqrt(np.dot(x-x2,x-x2))**2
          N2 += 1
      dist += np.abs(np.sqrt(v1/(N1-1)*1.0) - np.sqrt(v2/(N2-1)*1.0))
      return dist

    if opt in ['DistanceWithDecay']:
      K = self.reOrderStep
      decR = 1
      dist = 0
      for k in range(1,K+1):
        if t-k >= 0:
          if n1 < dataCenter[t-k].shape[0]:
            x1 = dataCenter[t-k][n1,:]
            dist += np.sqrt(np.dot(x1-x2,x1-x2))*np.exp(-(k-1)*decR)

      return dist

  def __reMapCluster__(self,t,dataCenter,dataCenterIndex):
    """
      Computes the remapping relationship between the current time step cluster and the previous time step
      @In, t, float, current time
      @In, dataCenter, dict, each value contains the center coordinate at each time step
      @In, dataCenterIndex, dict, each value contains the center index at each time step
      @Out, remap, list, remapping relation between the current time step cluster and the previous time step
    """
    indices1 = dataCenterIndex[t-1]
    indices2 = dataCenterIndex[t]

    N1 = dataCenter[t-1].shape[0]
    N2 = dataCenter[t].shape[0]

    dMatrix = np.zeros(shape=(N1,N2))
    for n1 in range(N1):
      for n2 in range(N2):
        dMatrix[n1,n2] = self.__computeDist__(t,n1,n2,dataCenter,'DistanceWithDecay')
    _, mapping = self.__localReMap__(dMatrix, (list(range(N1)), list(range(N2))))

    remap = {}
    f1, f2 = [False]*N1, [False]*N2
    for mp in mapping:
      i1, i2 = mp[0], mp[1]
      if f1[i1] or f2[i2]:
        self.raiseAnError(ValueError, 'Mapping is overlapped. ')

      remap[indices2[i2]] = indices1[i1]
      f1[i1], f2[i2] = True, True

    if N2 > N1:
      # for the case the new cluster comes up
      tmp = 1
      for n2 in range(N2):
        if indices2[n2] not in remap.keys():
          remap[indices2[n2]] = max(indices1)+tmp
          # remap[indices2[n2]] = self.maxNoClusters + 1 # every discontinuity would introduce a new cluster index.

    return remap

  def __localReMap__(self, dMatrix,loc):
    """
      Method to return the mapping based on distance stored in dMatrix, the returned mapping shall minimize the global sum of distance
      This function is recursively called to find the global minimum, so is computationally expensive --- FIXME
      @In, dMatrix, array, shape = (no_clusterAtPreviousTimeStep, no_clusterAtCurrentTimeStep)
      @In, loc, tuple, the first element is the cluster indeces for previous time step and the second one is for the current time step
      @Out, sumDist, float, global sum of distance
      @Out, localReMap, list, remapping relation between the row and column identifier of dMatrix
    """
    if len(loc[0]) == 1:
      sumDist, localReMap = np.inf, -1
      n1 = loc[0][0]
      for n2 in loc[1]:
        if dMatrix[n1,n2] < sumDist:
          sumDist = dMatrix[n1,n2]
          localReMap = n2
      return sumDist, [(n1,localReMap)]
    elif len(loc[1]) == 1:
      sumDist, localReMap = np.inf, -1
      n2 = loc[1][0]
      for n1 in loc[0]:
        if dMatrix[n1,n2] < sumDist:
          sumDist = dMatrix[n1,n2]
          localReMap = n1
      return sumDist, [(localReMap,n2)]
    else:
      sumDist, i1, i2, localReMap = np.inf, -1, -1, []
      n1 = loc[0][0]
      temp1 = copy.deepcopy(loc[0])
      temp1.remove(n1)
      for n2 in loc[1]:
        temp2 = copy.deepcopy(loc[1])
        temp2.remove(n2)
        d_temp, l = self.__localReMap__(dMatrix, (temp1,temp2))
        if dMatrix[n1,n2] + d_temp < sumDist:
          sumDist = dMatrix[n1,n2] + d_temp
          i1, i2, localReMap = n1, n2, l
      localReMap.append((i1,i2))
      return sumDist, localReMap

  def __evaluateLocal__(self, featureVals):
    """
      Not implemented for this class
    """
    pass

  def __confidenceLocal__(self):
    """
      Not implemented for this class
    """
    pass

  def getDataMiningType(self):
    """
      This method is used to return the type of data mining algorithm to be employed
      @ In, none
      @ Out, self.SKLtype, string, type of data mining algorithm
    """
    return self.SKLtype


class Scipy(unSupervisedLearning):
  """
    Scipy interface for hierarchical Learning
  """
  modelType = 'Scipy'
  availImpl = {}
  availImpl['cluster'] = {}
  availImpl['cluster']['Hierarchical'] = (hier.hierarchy, 'float')  # Perform Hierarchical Clustering of data.

  def __init__(self, **kwargs):
    """
     constructor for Scipy class.
     @ In, kwargs, dict, arguments for the Scipy algorithm
     @ Out, None
    """
    unSupervisedLearning.__init__(self, **kwargs)
    self.printTag = 'SCIPY'
    if 'SCIPYtype' not in self.initOptionDict.keys():
      self.raiseAnError(IOError, ' to define a Scipy unSupervisedLearning Method the SCIPYtype keyword is needed (from KDD ' + self.name + ')')

    SCIPYtype, SCIPYsubType = self.initOptionDict['SCIPYtype'].split('|')
    self.initOptionDict.pop('SCIPYtype')

    if not SCIPYtype in self.__class__.availImpl.keys():
      self.raiseAnError(IOError, ' Unknown SCIPYtype ' + SCIPYtype)
    if not SCIPYsubType in self.__class__.availImpl[SCIPYtype].keys():
      self.raiseAnError(IOError, ' Unknown SCIPYsubType ' + SCIPYsubType)

    self.__class__.returnType = self.__class__.availImpl[SCIPYtype][SCIPYsubType][1]
    self.Method = self.__class__.availImpl[SCIPYtype][SCIPYsubType][0]
    self.SCIPYtype = SCIPYtype
    self.SCIPYsubType = SCIPYsubType
    self.normValues = None
    self.outputDict = {}

  def _train(self):
    """
      Perform training on samples in self.normValues: array, shape = [n_samples, n_features] or [n_samples, n_samples]
      @ In, None
      @ Out, None
    """
    self.outputDict['outputs'] = {}
    self.outputDict['inputs' ] = self.normValues
    if hasattr(self.Method, 'linkage'):
      self.linkage = self.Method.linkage(self.normValues,self.initOptionDict['method'],self.initOptionDict['metric'])

      if 'dendrogram' in self.initOptionDict and self.initOptionDict['dendrogram'] == 'true':
        self.advDendrogram(self.linkage,
                           p                = float(self.initOptionDict['p']),
                           leaf_rotation    = 90.,
                           leaf_font_size   = 12.,
                           truncate_mode    = self.initOptionDict['truncationMode'],
                           show_leaf_counts = self.initOptionDict['leafCounts'],
                           show_contracted  = self.initOptionDict['showContracted'],
                           annotate_above   = self.initOptionDict['annotatedAbove'],
                           #orientation      = self.initOptionDict['orientation'],
                           max_d            = self.initOptionDict['level'])

      self.labels_ = hier.hierarchy.fcluster(self.linkage, self.initOptionDict['level'],self.initOptionDict['criterion'])
      self.outputDict['outputs']['labels'] = self.labels_
    return self.labels_

  def advDendrogram(self,*args, **kwargs):
    """
      This methods actually creates the dendrogram
      @ In, None
      @ Out, None
    """
    plt.figure()
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
      kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    ddata = hier.hierarchy.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
      plt.title('Hierarchical Clustering Dendrogram')
      plt.xlabel('sample index')
      plt.ylabel('distance')
      for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        if y > annotate_above:
          plt.plot(x, y, 'o', c=c)
          plt.annotate("%.3g" % y, (x, y), xytext=(15, 11),
                       textcoords='offset points',
                       va='top', ha='center')
      if max_d:
        plt.axhline(y=max_d, c='0.1')
    if 'dendFileID' in self.initOptionDict:
      title = self.initOptionDict['dendFileID'] + '.pdf'
    else:
      title = 'dendrogram.pdf'
    plt.savefig(title)
    plt.close()

  def __evaluateLocal__(self,*args, **kwargs):
    """
      Method to return output of an already trained scipy algorithm.
      @ In, featureVals, numpy.array, feature values
      @ Out, self.dData, numpy.array, dendrogram
    """
    pass

  def __confidenceLocal__(self):
    pass

  def getDataMiningType(self):
    """
      This method is used to return the type of data mining algorithm to be employed
      @ In, none
      @ Out, self.SCIPYtype, string, type of data mining algorithm
    """
    return self.SCIPYtype

factory = EntityFactory('unSuperVisedLearning')
factory.registerType('SciKitLearn', SciKitLearn)
factory.registerType('temporalSciKitLearn', temporalSciKitLearn)
factory.registerType('Scipy', Scipy)
