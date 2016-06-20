"""
  Module containing interface with SciKit-Learn clustering
  Created on Feb 13, 2015

  @author: senrs

  TODO:
  For Clustering:
  1) paralleization: n_jobs parameter to some of the algorithms
  2) find a smarter way to choose the parameters that are used as default, eg:
     number of clusters, init, leaf_size, etc.
  3) dimensionality reduction: maybe using PCA

  unSuperVisedLearning: Include other algorithms, such as:
  1) Gaussian Mixture models
  2) Manifold Learning,
  3) BiClustering, etc...
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from sklearn import cluster, mixture, manifold, decomposition, covariance, neural_network
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
import numpy as np
import abc
import ast
import copy
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
import utils
import MessageHandler
import PostProcessors #import returnFilterInterface
import DataObjects
#Internal Modules End--------------------------------------------------------------------------------

class unSupervisedLearning(utils.metaclass_insert(abc.ABCMeta), MessageHandler.MessageUser):
  """
    This is the general interface to any unSuperisedLearning learning method.
    Essentially it contains a train, and evaluate methods
  """
  returnType = ''  # this describe the type of information generated the possibility are 'boolean', 'integer', 'float'
  modelType = ''  # the broad class of the interpolator

  @staticmethod
  def checkArrayConsistency(arrayIn):
    """
      This method checks the consistency of the in-array
      @ In, arrayIn, object,  It should be an array
      @ Out, (consistent, 'error msg'), tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    """
    if type(arrayIn) != np.ndarray: return (False, ' The object is not a numpy array')
    # The input data matrix kind is different for different clustering algorithms
    # e.g. [n_samples, n_features] for MeanShift and KMeans
    #     [n_samples,n_samples]   for AffinityPropogation and SpectralCLustering
    # In other words, MeanShift and KMeans work with points in a vector space,
    # whereas AffinityPropagation and SpectralClustering can work with arbitrary objects, as long as a similarity measure exists for such objects
    # The input matrix supplied to unSupervisedLearning models as 1-D arrays of size [n_samples], (either n_features of or n_samples of them)
    if len(arrayIn.shape) != 1: return(False, ' The array must be 1-d')
    return (True, '')

  def __init__(self, messageHandler, **kwargs):
    """
      constructor for unSupervisedLearning class.
      @ In, messageHandler, object, Message handler object
      @ In, kwargs, dict, arguments for the unsupervised learning algorithm
    """
    self.printTag = 'unSupervised'
    self.messageHandler = messageHandler
    # booleanFlag that controls the normalization procedure. If true, the normalization is performed. Default = True
    if kwargs != None: self.initOptionDict = kwargs
    else             : self.initOptionDict = {}
    if 'Labels'       in self.initOptionDict.keys():  # Labels are passed, if known appriori (optional), they used in quality estimate
      self.labels = self.initOptionDict['Labels'  ]
      self.initOptionDict.pop('Labels')
    else: self.labels = None
    if 'Features'     in self.initOptionDict.keys():
      self.features = self.initOptionDict['Features'].split(',')
      self.initOptionDict.pop('Features')
    else: self.features = None
    self.verbosity = self.initOptionDict['verbosity'] if 'verbosity' in self.initOptionDict else None
    # average value and sigma are used for normalization of the feature data
    # a dictionary where for each feature a tuple (average value, sigma)
    self.muAndSigmaFeatures = {}
    #these need to be declared in the child classes!!!!
    self.amITrained = False

  def train(self, tdict):
    """
      Method to perform the training of the unSuperVisedLearning algorithm
      NB.the unSuperVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, dict, training dictionary
      @ Out, None
    """
    if type(tdict) != dict: self.raiseAnError(IOError, ' method "train". The training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values = list(tdict.keys()), list(tdict.values())
    if self.labels in names:
      self.labelValues = values[names.index(self.labels)]
      resp = self.checkArrayConsistency(self.labelValues)
      if not resp[0]: self.raiseAnError(IOError, 'In training set for ground truth labels ' + self.labels + ':' + resp[1])
    else            : self.raiseAWarning(' The ground truth labels are not known a priori')
    for cnt, feat in enumerate(self.features):
      if feat not in names: self.raiseAnError(IOError, ' The feature sought ' + feat + ' is not in the training set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: self.raiseAnError(IOError, ' In training set for feature ' + feat + ':' + resp[1])
        if self.normValues is None: self.normValues = np.zeros(shape = (values[names.index(feat)].size, len(self.features)))
        if values[names.index(feat)].size != self.normValues[:, 0].size: self.raiseAnError(IOError, ' In training set, the number of values provided for feature ' + feat + ' are != number of target outcomes!')
        self._localNormalizeData(values, names, feat)
        if self.muAndSigmaFeatures[feat][1] == 0: self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0], np.max(np.absolute(values[names.index(feat)])))
        if self.muAndSigmaFeatures[feat][1] == 0: self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0], 1.0)
        self.normValues[:, cnt] = (values[names.index(feat)] - self.muAndSigmaFeatures[feat][0]) / self.muAndSigmaFeatures[feat][1]
    self.__trainLocal__()
    self.amITrained = True

  def _localNormalizeData(self, values, names, feat):
    """
      Method to normalize data based on the mean and standard deviation.  If undesired for a particular algorithm,
      this method can be overloaded to simply pass.
      @ In, values, list,  list of feature values (from tdict)
      @ In, names,  list,  names of features (from tdict)
      @ In, feat, list, list of features (from Model)
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (np.average(values[names.index(feat)]), np.std(values[names.index(feat)]))

  def evaluate(self, edict):
    """
      Method to perform the evaluation of a point or a set of points through the previous trained unSuperVisedLearning algorithm
      NB.the superVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluation, numpy.array, array of evaluated points
    """
    if type(edict) != dict: self.raiseAnError(IOError, ' Method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: self.raiseAnError(IOError, ' In evaluate request for feature ' + names[index] + ':' + resp[1])
    if self.labels in names:
      self.labelValues = values[names.index(self.labels)]
    # construct the evaluation matrix
    featureValues = np.zeros(shape = (values[0].size, len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: self.raiseAnError(IOError, ' The feature sought ' + feat + ' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: self.raiseAnError(IOError, ' In training set for feature ' + feat + ':' + resp[1])
        featureValues[:, cnt] = ((values[names.index(feat)] - self.muAndSigmaFeatures[feat][0])) / self.muAndSigmaFeatures[feat][1]
    evaluation = self.__evaluateLocal__(featureValues)
    return evaluation

  def confidence(self):
    """
      This call is used to get an estimate of the confidence in the prediction of the clusters.
      The base class self.confidence checks if the clusters are already evaluated (trained) then calls the local confidence
      @ In, none
      @ Out, confidence, float, the confidence
    """
    if self.amITrained: return self.__confidenceLocal__()
    else:               self.raiseAnError(IOError, ' The confidence check is performed before evaluating the clusters.')


  @abc.abstractmethod
  def __trainLocal__(self):
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
  availImpl['cluster'] = {}  # Generalized Cluster
  availImpl['cluster']['AffinityPropogation'    ] = (cluster.AffinityPropagation    , 'float')  # Perform Affinity Propagation Clustering of data.
  availImpl['cluster']['DBSCAN'                 ] = (cluster.DBSCAN                 , 'float')  # Perform DBSCAN clustering from vector array or distance matrix.
  availImpl['cluster']['KMeans'                 ] = (cluster.KMeans                 , 'float')  # K-Means Clustering
  availImpl['cluster']['MiniBatchKMeans'        ] = (cluster.MiniBatchKMeans        , 'float')  # Mini-Batch K-Means Clustering
  availImpl['cluster']['MeanShift'              ] = (cluster.MeanShift              , 'float')  # Mean Shift Clustering
  availImpl['cluster']['SpectralClustering'     ] = (cluster.SpectralClustering     , 'float')  # Apply clustering to a projection to the normalized laplacian.
  #  availImpl['cluster']['AgglomerativeClustering'] = (cluster.AgglomerativeClustering, 'float')  # Agglomerative Clustering - Feature of SciKit-Learn version 0.15
  #  availImpl['cluster']['FeatureAgglomeration'   ] = (cluster.FeatureAgglomeration   , 'float')  # - Feature of SciKit-Learn version 0.15
  #  availImpl['cluster']['Ward'                   ] = (cluster.Ward                   , 'float')  # Ward hierarchical clustering: constructs a tree and cuts it.

  #  availImpl['bicluster'] = {}
  #  availImpl['bicluster']['SpectralBiclustering'] = (cluster.bicluster.SpectralBiclustering, 'float')  # Spectral biclustering (Kluger, 2003).
  #  availImpl['bicluster']['SpectralCoclustering'] = (cluster.bicluster.SpectralCoclustering, 'float')  # Spectral Co-Clustering algorithm (Dhillon, 2001).

  availImpl['mixture'] = {}  # Generalized Gaussion Mixture Models (Classification)
  availImpl['mixture']['GMM'  ] = (mixture.GMM  , 'float')  # Gaussian Mixture Model
  availImpl['mixture']['DPGMM'] = (mixture.DPGMM, 'float')  # Variational Inference for the Infinite Gaussian Mixture Model.
  availImpl['mixture']['VBGMM'] = (mixture.VBGMM, 'float')  # Variational Inference for the Gaussian Mixture Model

  availImpl['manifold'] = {}  # Manifold Learning (Embedding techniques)
  availImpl['manifold']['LocallyLinearEmbedding'  ] = (manifold.LocallyLinearEmbedding  , 'float')  # Locally Linear Embedding
  availImpl['manifold']['Isomap'                  ] = (manifold.Isomap                  , 'float')  # Isomap
  availImpl['manifold']['MDS'                     ] = (manifold.MDS                     , 'float')  # MultiDimensional Scaling
  availImpl['manifold']['SpectralEmbedding'       ] = (manifold.SpectralEmbedding       , 'float')  # Spectral Embedding for Non-linear Dimensionality Reduction
  #  availImpl['manifold']['locally_linear_embedding'] = (manifold.locally_linear_embedding, 'float')  # Perform a Locally Linear Embedding analysis on the data.
  #  availImpl['manifold']['spectral_embedding'      ] = (manifold.spectral_embedding      , 'float')  # Project the sample on the first eigen vectors of the graph Laplacian.

  availImpl['decomposition'] = {}  # Matrix Decomposition
  availImpl['decomposition']['PCA'                 ] = (decomposition.PCA                 , 'float')  # Principal component analysis (PCA)
#  availImpl['decomposition']['ProbabilisticPCA'    ] = (decomposition.ProbabilisticPCA    , 'float')  # Additional layer on top of PCA that adds a probabilistic evaluationPrincipal component analysis (PCA)
  availImpl['decomposition']['RandomizedPCA'       ] = (decomposition.RandomizedPCA       , 'float')  # Principal component analysis (PCA) using randomized SVD
  availImpl['decomposition']['KernelPCA'           ] = (decomposition.KernelPCA           , 'float')  # Kernel Principal component analysis (KPCA)
  availImpl['decomposition']['FastICA'             ] = (decomposition.FastICA             , 'float')  # FastICA: a fast algorithm for Independent Component Analysis.
  availImpl['decomposition']['TruncatedSVD'        ] = (decomposition.TruncatedSVD        , 'float')  # Dimensionality reduction using truncated SVD (aka LSA).
  availImpl['decomposition']['SparsePCA'           ] = (decomposition.SparsePCA           , 'float')  # Sparse Principal Components Analysis (SparsePCA)
  availImpl['decomposition']['MiniBatchSparsePCA'  ] = (decomposition.MiniBatchSparsePCA  , 'float')  # Mini-batch Sparse Principal Components Analysis
  #  availImpl['decomposition']['ProjectedGradientNMF'] = (decomposition.ProjectedGradientNMF, 'float')  # Non-Negative matrix factorization by Projected Gradient (NMF)
  #  availImpl['decomposition']['FactorAnalysis'      ] = (decomposition.FactorAnalysis      , 'float')  # Factor Analysis (FA)
  #  availImpl['decomposition']['NMF'                 ] = (decomposition.NMF                 , 'float')  # Non-Negative matrix factorization by Projected Gradient (NMF)
  #  availImpl['decomposition']['SparseCoder'         ] = (decomposition.SparseCoder         , 'float')  # Sparse coding
  #  availImpl['decomposition']['DictionaryLearning'  ] = (decomposition.DictionaryLearning  , 'float')  # Dictionary Learning
  #  availImpl['decomposition']['MiniBatchDictionaryLearning'] = (decomposition.MiniBatchDictionaryLearning, 'float')  # Mini-batch dictionary learning
  #  availImpl['decomposition']['fastica'                    ] = (decomposition.fastica                    , 'float')  # Perform Fast Independent Component Analysis.
  #  availImpl['decomposition']['dict_learning'              ] = (decomposition.dict_learning              , 'float')  # Solves a dictionary learning matrix factorization problem.

  #  availImpl['covariance'] = {}  # Covariance Estimators
  #  availImpl['covariance']['EmpiricalCovariance'] = (covariance.EmpiricalCovariance, 'float')  # Maximum likelihood covariance estimator
  #  availImpl['covariance']['EllipticEnvelope'   ] = (covariance.EllipticEnvelope   , 'float')  # An object for detecting outliers in a Gaussian distributed dataset.
  #  availImpl['covariance']['GraphLasso'         ] = (covariance.GraphLasso         , 'float')  # Sparse inverse covariance estimation with an l1-penalized estimator.
  #  availImpl['covariance']['GraphLassoCV'       ] = (covariance.GraphLassoCV       , 'float')  # Sparse inverse covariance w/ cross-validated choice of the l1 penalty
  #  availImpl['covariance']['LedoitWolf'         ] = (covariance.LedoitWolf         , 'float')  # LedoitWolf Estimator
  #  availImpl['covariance']['MinCovDet'          ] = (covariance.MinCovDet          , 'float')  # Minimum Covariance Determinant (MCD): robust estimator of covariance
  #  availImpl['covariance']['OAS'                ] = (covariance.OAS                , 'float')  # Oracle Approximating Shrinkage Estimator
  #  availImpl['covariance']['ShrunkCovariance'   ] = (covariance.ShrunkCovariance   , 'float')  # Covariance estimator with shrinkage

  #  availImpl['neuralNetwork'] = {}  # Covariance Estimators
  #  availImpl['neuralNetwork']['BernoulliRBM'] = (neural_network.BernoulliRBM, 'float')  # Bernoulli Restricted Boltzmann Machine (RBM).

  def __init__(self, messageHandler, **kwargs):
    """
     constructor for SciKitLearn class.
     @ In, messageHandler, MessageHandler, Message handler object
     @ In, kwargs, dict, arguments for the SciKitLearn algorithm
     @ Out, None
    """
    unSupervisedLearning.__init__(self, messageHandler, **kwargs)
    self.printTag = 'SCIKITLEARN'
    if 'SKLtype' not in self.initOptionDict.keys(): self.raiseAnError(IOError, ' to define a scikit learn unSupervisedLearning Method the SKLtype keyword is needed (from KDD ' + self.name + ')')
    SKLtype, SKLsubType = self.initOptionDict['SKLtype'].split('|')
    self.initOptionDict.pop('SKLtype')
    if not SKLtype in self.__class__.availImpl.keys(): self.raiseAnError(IOError, ' Unknown SKLtype ' + SKLtype + '(from KDD ' + self.name + ')')
    if not SKLsubType in self.__class__.availImpl[SKLtype].keys(): self.raiseAnError(IOError, ' Unknown SKLsubType ' + SKLsubType + '(from KDD ' + self.name + ')')
    self.__class__.returnType = self.__class__.availImpl[SKLtype][SKLsubType][1]
    self.Method = self.__class__.availImpl[SKLtype][SKLsubType][0]()
    self.SKLtype = SKLtype
    self.SKLsubType = SKLsubType
    paramsDict = self.Method.get_params()
    if 'cluster' == SKLtype:
      if 'n_clusters' in paramsDict.keys():
        if 'n_clusters' not in self.initOptionDict.keys(): self.initOptionDict['n_clusters'] = 8
      else:
        if 'n_clusters' in self.initOptionDict.keys(): self.initOptionDict.pop('n_clusters')
      if 'preference'   in paramsDict.keys(): self.initOptionDict['preference'  ] = None  # AffinityPropogation
      if 'leaf_size'    in paramsDict.keys(): self.initOptionDict['leaf_size'   ] = 30  # DBSCAN
      if 'eps'          in paramsDict.keys(): self.initOptionDict['eps'         ] = 0.2  # DBSCAN
      if 'random_state' in paramsDict.keys(): self.initOptionDict['random_state'] = 0
    # These parameters above and some more effects the quality of the Cardinality Reduction algorithms...Look at TODO list at the top of the source

    if 'mixture' == SKLtype:
      if 'n_components' in paramsDict.keys():
        if 'n_components' not in self.initOptionDict.keys(): self.initOptionDict['n_components'] = 5
      else :
        if 'n_components' in self.initOptionDict.keys(): self.initOptionDict.pop('n_components')
      self.noComponents_ = self.initOptionDict['n_components']
    if 'decomposition' == SKLtype or 'manifold' == SKLtype: self.noComponents_ = self.initOptionDict['n_components']
    for key, value in self.initOptionDict.items():
      try   : self.initOptionDict[key] = ast.literal_eval(value)
      except: pass
    self.Method.set_params(**self.initOptionDict)
    self.normValues = None
    self.outputDict = {}

  def __trainLocal__(self):
    """
      Perform training on samples in self.normValues: array, shape = [n_samples, n_features] or [n_samples, n_samples]
      @ In, None
      @ Out, None
    """
    if hasattr(self.Method, 'bandwidth'):  # set bandwidth for MeanShift clustering
      self.initOptionDict['bandwidth'] = cluster.estimate_bandwidth(self.normValues,quantile=0.3)
#       self.Method.set_params(**self.initOptionDict)
    if hasattr(self.Method, 'connectivity'):  # We need this connectivity if we want to use structured ward
      connectivity = kneighbors_graph(self.normValues, n_neighbors = 10)  # we should find a smart way to define the number of neighbors instead of default constant integer value(10)
      connectivity = 0.5 * (connectivity + connectivity.T)
      self.initOptionDict['connectivity'] = connectivity
      self.Method.set_params(**self.initOptionDict)
    self.outputDict['outputs'] = {}
    self.outputDict['inputs' ] = self.normValues
    ## What are you doing here? Calling half of these methods does nothing
    ## unless you store the data somewhere. If you are going to blindly call
    ## whatever methods that exist in the class, then at least store them for
    ## later. Why is this done again on the PostProcessor side? I am struggling
    ## to understand what this code's purpose is except to obfuscate our
    ## interaction with skl.
    if   hasattr(self.Method, 'fit_predict'):
        self.Method.fit_predict(self.normValues)
    elif hasattr(self.Method, 'predict'):
        self.Method.fit(self.normValues)
        self.Method.predict(self.normValues)
    elif hasattr(self.Method, 'fit_transform'):
        self.Method.fit_transform(self.normValues)
    elif hasattr(self.Method, 'transform'):
        self.Method.fit(self.normValues)
        self.Method.transform(self.normValues)
# Below generates the output Dictionary from the trained algorithm, can be defined in a new method....
    if 'cluster' == self.SKLtype:
        if hasattr(self.Method, 'n_clusters') :
            self.noClusters = self.Method.n_clusters
            self.outputDict['outputs']['noClusters'           ] = copy.deepcopy(self.noClusters)
        if hasattr(self.Method, 'labels_') :
            self.labels_ = self.Method.labels_
            self.outputDict['outputs']['labels'               ] = copy.deepcopy(self.labels_)
        if hasattr(self.Method, 'cluster_centers_') :
            self.clusterCenters_ = copy.deepcopy(self.Method.cluster_centers_)
            ## I hope these arrays are consistently ordered...
            ## We are mixing our internal storage of muAndSigma with SKLs
            ## representation of our data, I believe it is fair to say that we
            ## hand the data to SKL in the same order that we have it stored.
            for cnt, feat in enumerate(self.features):
              for center in self.clusterCenters_:
                center[cnt] = center[cnt] * self.muAndSigmaFeatures[feat][1] + self.muAndSigmaFeatures[feat][0]
            self.outputDict['outputs']['clusterCenters'       ] = self.clusterCenters_
        if hasattr(self.Method, 'cluster_centers_indices_') :
            self.clusterCentersIndices_ = copy.deepcopy(self.Method.cluster_centers_indices_)
            self.outputDict['outputs']['clusterCentersIndices'] = self.clusterCentersIndices_
        if hasattr(self.Method, 'inertia_') :
            self.inertia_ = self.Method.inertia_
            self.outputDict['outputs']['inertia'              ] = self.inertia_
    elif 'mixture' == self.SKLtype:
        if hasattr(self.Method, 'weights_') :
            self.weights_ = copy.deepcopy(self.Method.weights_)
            self.outputDict['outputs']['weights'  ] = copy.deepcopy(self.weights_)
        if hasattr(self.Method, 'means_') :
            self.means_ = copy.deepcopy(self.Method.means_)

            ## I hope these arrays are consistently ordered...
            ## We are mixing our internal storage of muAndSigma with SKLs
            ## representation of our data, I believe it is fair to say that we
            ## hand the data to SKL in the same order that we have it stored.
            for cnt, feat in enumerate(self.features):
              for center in self.means_:
                center[cnt] = center[cnt] * self.muAndSigmaFeatures[feat][1] + self.muAndSigmaFeatures[feat][0]

            self.outputDict['outputs']['means'    ] = self.means_
        if hasattr(self.Method, 'covars_') :
            self.covars_ = copy.deepcopy(self.Method.covars_)

            ## I hope these arrays are consistently ordered...
            ## We are mixing our internal storage of muAndSigma with SKLs
            ## representation of our data, I believe it is fair to say that we
            ## hand the data to SKL in the same order that we have it stored.
            for row, rowFeat in enumerate(self.features):
              for col, colFeat in enumerate(self.features):
                self.covars_[row,col] = self.covars_[row,col] * self.muAndSigmaFeatures[rowFeat][1] * self.muAndSigmaFeatures[colFeat][1]

            self.outputDict['outputs']['covars'   ] = self.covars_
        if hasattr(self.Method, 'precs_') :
            self.precs_ = copy.deepcopy(self.Method.precs_)
            self.outputDict['outputs']['precs_'   ] = self.precs_
        if hasattr(self.Method, 'converged_') :
            if not self.Method.converged_ : self.raiseAnError(RuntimeError, self.SKLtype + '|' + self.SKLsubType + ' did not converged. (from KDD->' + self.SKLsubType + ')')
            self.converged_ = copy.deepcopy(self.Method.converged_)
            self.outputDict['outputs']['converged'] = self.converged_
    elif 'manifold' == self.SKLtype:
        self.outputDict['outputs']['noComponents'] = copy.deepcopy(self.noComponents_)
        if hasattr(self.Method, 'embedding_'):
            self.embeddingVectors_ = copy.deepcopy(self.Method.embedding_)
            self.outputDict['outputs']['embeddingVectors_'] = copy.deepcopy(self.embeddingVectors_)
        if hasattr(self.Method, 'reconstruction_error_'):
            self.reconstructionError_ = copy.deepcopy(self.Method.reconstruction_error_)
            self.outputDict['outputs']['reconstructionError_'] = copy.deepcopy(self.reconstructionError_)
    elif 'decomposition' == self.SKLtype:
        self.outputDict['outputs']['noComponents'] = copy.deepcopy(self.noComponents_)
        if hasattr(self.Method, 'components_'):
            self.components_ = copy.deepcopy(self.Method.components_)
            self.outputDict['outputs']['components'] = self.components_
        if hasattr(self.Method, 'means_'):
            self.means_ = copy.deepcopy(self.Method.means_)
            self.outputDict['outputs']['means'] = self.means_
        if hasattr(self.Method, 'explained_variance_'):
            self.explainedVariance_ = copy.deepcopy(self.Method.explained_variance_)
            self.outputDict['outputs']['explainedVariance'] = self.explainedVariance_
        if hasattr(self.Method, 'explained_variance_ratio_'):
            self.explainedVarianceRatio_ = copy.deepcopy(self.Method.explained_variance_ratio_)
            self.outputDict['outputs']['explainedVarianceRatio'] = self.explainedVarianceRatio_
    else: print ('Not Implemented yet!...', self.SKLtype)

#     elif 'bicluster' == self.SKLtype:
#         if hasattr(self.Method, 'n_clusters') :
#             self.noClusters = self.Method.n_clusters
#             self.outputDict['outputs']['noClusters'           ] = self.noClusters
#         if hasattr(self.Method, 'row_labels_') :
#             self.rowLabels_ = self.Method.row_labels_
#             self.outputDict['outputs']['rowLabels'            ] = self.rowLabels_
#         if hasattr(self.Method, 'column_labels_') :
#             self.columnLabels_ = self.Method.column_labels_
#             self.outputDict['outputs']['columnLabels'         ] = self.columnLabels_
#         if hasattr(self.Method, 'bicluster_') :
#             self.biClusters_ = self.Method.biClusters_
#             self.outputDict['outputs']['biClusters'           ] = self.biClusters_


  def __evaluateLocal__(self, featureVals):
    """
      Method to return labels of an already trained unSuperVised algorithm.
      @ In, featureVals, numpy.array, feature values
      @ Out, self.labels_, numpy.array, labels
    """
    self.normValues = featureVals
    if hasattr(self.Method, 'predict'): self.labels_ = self.Method.predict(featureVals)
    else                              : self.labels_ = self.Method.fit_predict(featureVals)
    self.outputDict['outputs']['labels'] = self.labels_

    if hasattr(self.Method, 'inertia_') :
        self.inertia_ = self.Method.inertia_
        self.outputDict['outputs']['inertia'] = self.inertia_
    return self.labels_

  def __confidenceLocal__(self):
    """
      This should return an estimation dictionary of the quality of the prediction.
      @ In, None
      @ Out, self.outputdict['confidence'], dict, dictionary of the confidence metrics of the algorithms
    """
    self.outputDict['confidence'] = {}
    if 'cluster' == self.SKLtype:
        if hasattr(self.Method, 'labels_'):
            if np.unique(self.labels_).size > 1:
              self.outputDict['confidence']['silhouetteCoefficient'  ] = metrics.silhouette_score          (self.normValues , self.labels_)
        if hasattr(self.Method, 'inertia_'):
            self.outputDict['confidence']['inertia'                  ] = self.inertia_
        if  self.labels:
            self.outputDict['confidence']['homogeneity'              ] = metrics.homogeneity_score         (self.labelValues, self.labels_)
            self.outputDict['confidence']['completenes'              ] = metrics.completeness_score        (self.labelValues, self.labels_)
            self.outputDict['confidence']['vMeasure'                 ] = metrics.v_measure_score           (self.labelValues, self.labels_)
            self.outputDict['confidence']['adjustedRandIndex'        ] = metrics.adjusted_rand_score       (self.labelValues, self.labels_)
            self.outputDict['confidence']['adjustedMutualInformation'] = metrics.adjusted_mutual_info_score(self.labelValues, self.labels_)
    elif 'mixture' == self.SKLtype:
        self.outputDict['confidence']['aic'  ] = self.Method.aic(self.normValues)  # Akaike Information Criterion
        self.outputDict['confidence']['bic'  ] = self.Method.bic(self.normValues)  # Bayesian Information Criterion
        self.outputDict['confidence']['score'] = self.Method.score(self.normValues)  # log probabilities of each data point
    return self.outputDict['confidence']

#
#

class temporalSciKitLearn(unSupervisedLearning):
  """
    Data mining library to perform SciKitLearn algorithms along temporal data
  """

  def __init__(self, messageHandler, **kwargs):
    """
      constructor for temporalSciKitLearn class.
      @ In: messageHandler, Message handler object
      @ In: kwargs, arguments for the SciKitLearn algorithm
    """
    unSupervisedLearning.__init__(self, messageHandler, **kwargs)
    self.printTag = 'TEMPORALSCIKITLEARN'
    if 'SKLtype' not in self.initOptionDict.keys(): self.raiseAnError(IOError, ' to define a scikit learn unSupervisedLearning Method the SKLtype keyword is needed (from KDD ' + self.name + ')')
    self.SKLtype, self.SKLsubType = self.initOptionDict['SKLtype'].split('|')
    self.timeID = self.initOptionDict.get('timeID', 'Time')
    self.reOrderStep = int(self.initOptionDict.get('reOrderStep', 5))
    # return a SciKitLearn instance as engine for SKL data mining
    self.SKLEngine = returnInstance('SciKitLearn',self, **self.initOptionDict)
    self.normValues = None
    self.outputDict = {}
    if 'decomposition' == self.SKLtype or 'manifold' == self.SKLtype: self.noComponents_ = self.initOptionDict['n_components']

  @staticmethod
  def checkArrayConsistency(arrayin, shape):
    """
      This method checks the consistency of the in-array
      @ In, object... It should be an array
      @ Out, tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    """
    if type(arrayin) != np.ndarray: return (False, ' The object is not a numpy array')
    if arrayin.shape[0] != shape[0] or arrayin.shape[1] != shape[1]:
      return (False, ' The object shape is not correct')
    # The input data matrix kind is different for different clustering algorithms
    # e.g. [n_samples, n_features] for MeanShift and KMeans
    #     [n_samples,n_samples]   for AffinityPropogation and SpectralCLustering
    # In other words, MeanShift and KMeans work with points in a vector space,
    # whereas AffinityPropagation and SpectralClustering can work with arbitrary objects, as long as a similarity measure exists for such objects
    # The input matrix supplied to unSupervisedLearning models as 1-D arrays of size [n_samples], (either n_features of or n_samples of them)
#     if len(arrayin.shape) != 1: return(False, ' The array must be 1-d')
    return (True, '')

  def _localNormalizeData(self, values, names, feat):
    """
      Method to normalize data based on the mean and standard deviation.  If undesired for a particular algorithm,
      this method can be overloaded to simply pass.
      @ In, values, dict, each value of values is an array with shape = [no_history, no_timeStep], feature values
      @ In, names,  list,  names of features (from tdict)
      @ In, feat, string, the feature for which value is to be normalized
      @ Out, normV, array, shape = [no_history, no_timeStep], normalized values
    """
    normV = np.zeros(shape = values[names.index(feat)].shape)
    self.muAndSigmaFeatures[feat] = np.zeros(shape=(2,self.noTimeStep))
    for t in range(self.noTimeStep):
      self.muAndSigmaFeatures[feat][0,t] = np.average(values[names.index(feat)][:,t])
      self.muAndSigmaFeatures[feat][1,t] = np.std(values[names.index(feat)][:,t])
      if self.muAndSigmaFeatures[feat][1,t] == 0: self.muAndSigmaFeatures[feat][1,t] = np.max(np.absolute(values[names.index(feat)][:,t]))
      if self.muAndSigmaFeatures[feat][1,t] == 0: self.muAndSigmaFeatures[feat][1,t] = 1.0
      normV[:, t] = 1.0 * (values[names.index(feat)][:,t] - self.muAndSigmaFeatures[feat][0,t]) / self.muAndSigmaFeatures[feat][1,t]
    return normV

  def __deNormalizeData__(self,feat,t,data):
    """
      Method to denormalize data based on the mean and standard deviation stored in self.
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
    # need to overwrite train method because time dependent data mining requires different treatment of input
    if type(tdict) != dict: self.raiseAnError(IOError, ' method "train". The training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values = list(tdict.keys()), list(tdict.values())
    self.noSample, self.noTimeStep = values[0].shape[0], values[0].shape[1]
    if self.labels in names:
      self.labelValues = values[names.index(self.labels)]
      resp = self.checkArrayConsistency(self.labelValues,[self.noSample, self.noTimeStep])
      if not resp[0]: self.raiseAnError(IOError, 'In training set for ground truth labels ' + self.labels + ':' + resp[1])
    else            : self.raiseAWarning(' The ground truth labels are not known appriori')
#     for cnt, feat in enumerate(self.features):
    for feat in self.features:
      if feat not in names: self.raiseAnError(IOError, ' The feature sought ' + feat + ' is not in the training set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)],[self.noSample, self.noTimeStep])
        if not resp[0]: self.raiseAnError(IOError, ' In training set for feature ' + feat + ':' + resp[1])
        if self.normValues is None: self.normValues = {}
        self.normValues[feat] = self._localNormalizeData(values, names, feat)
    self.inputDict = tdict
    self.__trainLocal__()
    self.amITrained = True

  def __trainLocal__(self):
    """
      Method to train this class.
    """
    self.outputDict['outputs'] = {}
    self.outputDict['inputs' ] = self.normValues

    Input = {}
    for t in range(self.noTimeStep):
      Input['Features'] ={}
      for feat in self.features.keys():     Input['Features'][feat] = self.inputDict[feat][:,t]
      self.SKLEngine.features = Input['Features']
      self.SKLEngine.train(Input['Features'])
      self.SKLEngine.confidence()

      if self.SKLtype in ['cluster']:
        if 'labels' not in self.outputDict.keys():                  self.outputDict['labels'] = {}
        if 'clusterCenters' not in self.outputDict.keys():          self.outputDict['clusterCenters'] = {}
        if 'noClusters' not in self.outputDict.keys():              self.outputDict['noClusters'] = {}
        if 'clusterCentersIndices' not in self.outputDict.keys():   self.outputDict['clusterCentersIndices'] = {}
        # collect labels
        if hasattr(self.SKLEngine.Method, 'labels_'):   self.outputDict['labels'][t] = self.SKLEngine.Method.labels_
        # collect cluster centers
        if hasattr(self.SKLEngine.Method, 'cluster_centers_'):
          self.outputDict['clusterCenters'][t] = np.zeros(shape=self.SKLEngine.Method.cluster_centers_.shape)
          for cnt, feat in enumerate(self.features):
            self.outputDict['clusterCenters'][t][:,cnt] = self.__deNormalizeData__(feat,t,self.SKLEngine.Method.cluster_centers_[:,cnt])
        else:
          self.outputDict['clusterCenters'][t] = self.__computeCenter__(Input['Features'], self.outputDict['labels'][t])
        # collect number of clusters
        if hasattr(self.SKLEngine.Method, 'n_clusters'):
          self.outputDict['noClusters'][t] = self.SKLEngine.Method.n_clusters
        else:
          self.outputDict['noClusters'][t] = self.outputDict['clusterCenters'][t].shape[0]
        # collect cluster indices
        if hasattr(self.SKLEngine.Method, 'cluster_centers_indices_'):
          self.outputDict['clusterCentersIndices'][t] = self.SKLEngine.Method.cluster_centers_indices_
          self.outputDict['clusterCentersIndices'][t] = range(self.outputDict['noClusters'][t])
        else:
          self.outputDict['clusterCentersIndices'][t] = range(self.outputDict['noClusters'][t])  # use list(set(self.SKLEngine.Method.labels_)) to collect outliers
        # collect optional output
        if hasattr(self.SKLEngine.Method, 'inertia_'):
          if 'inertia' not in self.outputDict.keys(): self.outputDict['inertia'] = {}
          self.outputDict['inertia'][t] = self.SKLEngine.Method.inertia_
        # re-order clusters
        if t > 0:
          remap = self.__reMapCluster__(t, self.outputDict['clusterCenters'], self.outputDict['clusterCentersIndices'])
          for n in range(len(self.outputDict['clusterCentersIndices'][t])):
            self.outputDict['clusterCentersIndices'][t][n] = remap[self.outputDict['clusterCentersIndices'][t][n]]
          for n in range(len(self.outputDict['labels'][t])):
            if self.outputDict['labels'][t][n] >=0: self.outputDict['labels'][t][n] = remap[self.SKLEngine.Method.labels_[n]]

      elif self.SKLtype in ['mixture']:
        if 'labels' not in self.outputDict.keys():                  self.outputDict['labels'] = {}
        if 'means' not in self.outputDict.keys():                   self.outputDict['means'] = {}
        if 'noComponents' not in self.outputDict.keys():            self.outputDict['noComponents'] = {}
        if 'componentMeanIndices' not in self.outputDict.keys():    self.outputDict['componentMeanIndices'] = {}
        # collect component membership
        self.outputDict['labels'][t] = self.SKLEngine.evaluate(Input['Features'])
        # collect component means
        if hasattr(self.SKLEngine.Method, 'means_'):
          self.outputDict['means'][t] = np.zeros(shape=self.SKLEngine.Method.means_.shape)
          for cnt, feat in enumerate(self.features):
            self.outputDict['means'][t][:,cnt] = self.__deNormalizeData__(feat,t,self.SKLEngine.Method.means_[:,cnt])
        else:
          self.outputDict['means'][t] = self.__computeCenterr__(Input['Features'], self.outputDict['labels'][t])
        # collect number of components
        if hasattr(self.SKLEngine.Method, 'n_components'):
          self.outputDict['noComponents'][t] = self.SKLEngine.Method.n_components
        else:
          self.outputDict['noComponents'][t] = self.outputDict['means'][t].shape[0]
        # collect component indices
        self.outputDict['componentMeanIndices'][t] = range(self.outputDict['noComponents'][t])
        # collect optional output
        if hasattr(self.SKLEngine, 'weights_'):
          if 'weights' not in self.outputDict.keys(): self.outputDict['weights'] = {}
          self.outputDict['weights'][t] = self.SKLEngine.weights_
        if hasattr(self.SKLEngine, 'covars_'):
          if 'covars' not in self.outputDict.keys(): self.outputDict['covars'] = {}
          self.outputDict['covars'][t] = self.SKLEngine.covars_
        if hasattr(self.SKLEngine, 'precs_'):
          if 'precs' not in self.outputDict.keys(): self.outputDict['precs'] = {}
          self.outputDict['precs'][t] = self.SKLEngine.precs_
        if hasattr(self.SKLEngine, 'converged_'):
          if 'converged' not in self.outputDict.keys(): self.outputDict['converged'] = {}
          self.outputDict['converged'][t] = self.SKLEngine.converged_
        # re-order components
        if t > 0:
          remap = self.__reMapCluster__(t, self.outputDict['means'], self.outputDict['componentMeanIndices'])
          for n in range(len(self.outputDict['componentMeanIndices'][t])):
            self.outputDict['componentMeanIndices'][t][n] = remap[self.outputDict['componentMeanIndices'][t][n]]
          for n in range(len(self.outputDict['labels'][t])):
            if self.outputDict['labels'][t][n] >=0: self.outputDict['labels'][t][n] = remap[self.outputDict['labels'][t][n]]

      elif 'manifold' == self.SKLtype:
        if 'noComponents' not in self.outputDict.keys():        self.outputDict['noComponents'] = {}
        if 'embeddingVectors_' not in self.outputDict.keys():   self.outputDict['embeddingVectors_'] = {}

        self.outputDict['noComponents'][t] = self.SKLEngine.noComponents_
        if hasattr(self.SKLEngine.Method, 'embedding_'):
          self.outputDict['embeddingVectors_'][t] = self.SKLEngine.Method.embedding_
        if   'transform'     in dir(self.SKLEngine.Method):
          self.outputDict['embeddingVectors_'][t] = self.SKLEngine.Method.transform(self.SKLEngine.normValues)
        elif 'fit_transform' in dir(self.SKLEngine.Method):
          self.outputDict['embeddingVectors_'][t] = self.SKLEngine.Method.fit_transform(self.SKLEngine.normValues)
        if hasattr(self.SKLEngine.Method, 'reconstruction_error_'):
            if 'reconstructionError_' not in self.outputDict.keys():  self.outputDict['reconstructionError_'] = {}
            self.outputDict['reconstructionError_'][t] = self.SKLEngine.Method.reconstruction_error_

      elif 'decomposition' == self.SKLtype:

        if 'noComponents' not in self.outputDict.keys():      self.outputDict['noComponents'] = {}
        if 'components' not in self.outputDict.keys():        self.outputDict['components'] = {}

        self.outputDict['noComponents'][t] = self.SKLEngine.noComponents_
        if hasattr(self.SKLEngine.Method, 'components_'):
          self.outputDict['components'][t] = self.SKLEngine.Method.components_
        if   'transform'     in dir(self.SKLEngine.Method):
          self.outputDict['components'][t] = self.SKLEngine.Method.transform(self.SKLEngine.normValues)
        elif 'fit_transform' in dir(self.SKLEngine.Method):
          self.outputDict['components'][t] = self.SKLEngine.Method.fit_transform(self.SKLEngine.normValues)
        if hasattr(self.SKLEngine.Method, 'means_'):
            self.outputDict['means'] = self.SKLEngine.Method.means_
        if hasattr(self.SKLEngine.Method, 'explained_variance_'):
            self.outputDict['explainedVariance'] = self.SKLEngine.Method.explained_variance_
        if hasattr(self.SKLEngine.Method, 'explained_variance_ratio_'):
            self.outputDict['explainedVarianceRatio'] = self.SKLEngine.Method.explained_variance_ratio_

      else: print ('Not Implemented yet!...', self.SKLtype)

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
      if l >= 0 and l not in point.keys():    point[l] = []
      if l >= 0:                              point[l].append(cnt)
    noCluster = len(point.keys())
    if noCluster == 0:                        self.raiseAnError(ValueError, 'number of cluster is 0!!!')
    clusterCenter = np.zeros(shape=(noCluster,len(self.features)))
    for cnt, feat in enumerate(self.features):
      for ind, l in enumerate(point.keys()):  clusterCenter[ind,cnt] = np.average(data[feat][point[l]])
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
    x1, x2 = dataCenter[t-1][n1,:], dataCenter[t][n2,:]
    if opt in ['Distance']:
      dist = np.sqrt(np.dot(x1-x2,x1-x2))
      return dist
    if opt in ['Overlap']:
      l1, l2 = self.outputDict['labels'][t-1], self.SKLEngine.Method.labels_
      point1, point2 = [], []
      for n in range(len(l1)):
        if l1[n] == n1: point1.append(n)
      for n in range(len(l2)):
        if l2[n] == n2: point2.append(n)
      dist = - len(set(point1).intersection(point2))
      return dist
    if opt in ['DistVariance']:
      l1, l2 = self.outputDict['labels'][t-1], self.SKLEngine.Method.labels_
      dist = np.sqrt(np.dot(x1-x2,x1-x2))
      v1, v2, N1, N2 = 0, 0, 0, 0
      noFeat = len(self.features)
      for n in range(len(l1)): # compute variance of points with label l1
        if l1[n] == n1:
          x = np.zeros(shape=(noFeat,))
          for cnt, feat in enumerate(self.features):    x[cnt] = self.inputDict[feat][n,t-1]
          v1 += np.sqrt(np.dot(x-x1,x-x1))**2
          N1 += 1
      for n in range(len(l2)): # compute variance of points with label l2
        if l2[n] == n2:
          x = np.zeros(shape=(noFeat,))
          for cnt, feat in enumerate(self.features):    x[cnt] = self.inputDict[feat][n,t]
          v2 += np.sqrt(np.dot(x-x2,x-x2))**2
          N2 += 1
      dist += np.abs(np.sqrt(v1/(N1-1)*1.0) - np.sqrt(v2/(N2-1)*1.0))
      return dist
    if opt in ['DistanceWithDecay']:
      K, decR, dist = self.reOrderStep, 1, 0
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
    indices1, indices2 = dataCenterIndex[t-1], dataCenterIndex[t]
    N1, N2 = dataCenter[t-1].shape[0], dataCenter[t].shape[0]
    dMatrix = np.zeros(shape=(N1,N2))
    for n1 in range(N1):
      for n2 in range(N2):
        dMatrix[n1,n2] = self.__computeDist__(t,n1,n2,dataCenter,'DistanceWithDecay')
    _, mapping = self.__localReMap__(dMatrix, (range(N1), range(N2)))

    remap = {}
    f1, f2 = [False]*N1, [False]*N2
    for mp in mapping:
      i1, i2 = mp[0], mp[1]
      if f1[i1] or f2[i2]:      self.raiseAnError(ValueError, 'Mapping is overlapped. ')
      remap[indices2[i2]] = indices1[i1]
      f1[i1], f2[i2] = True, True

    if N2 > N1: # for the case the new cluster comes up
      tmp = 1
      for n2 in range(N2):
        if indices2[n2] not in remap.keys():    remap[indices2[n2]] = max(indices1)+tmp # remap[indices2[n2]] = self.maxNoClusters + 1 # every discondinuity would introduce a new cluster index.
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

__interfaceDict = {}
__interfaceDict['SciKitLearn'] = SciKitLearn
__interfaceDict['temporalSciKitLearn'] = temporalSciKitLearn
__base = 'unSuperVisedLearning'

def returnInstance(modelClass, caller, **kwargs):
  """
    This function return an instance of the request model type
    @ In, modelClass, string, representing the instance to create
    @ In, caller, object, object that will share its messageHandler instance
    @ In, kwargs, dict, a dictionary specifying the keywords and values needed to create the instance.
    @ Out, object, an instance of a Model
  """
  try: return __interfaceDict[modelClass](caller.messageHandler, **kwargs)
  except KeyError: caller.raiseAnError(NameError, 'unSuperVisedLEarning', 'Not known ' + __base + ' type ' + str(modelClass))

def returnClass(modelClass, caller):
  """
    This function return an instance of the request model type
    @ In, modelClass, string, representing the class to retrieve
    @ In, caller, object, object that will share its messageHandler instance
    @ Out, the class definition of the Model
  """
  try: return __interfaceDict[modelClass]
  except KeyError: caller.raiseanError(NameError, 'unSuperVisedLEarning', 'not known ' + __base + ' type ' + modelClass)
