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
  Created on May 8, 2018

  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for MSR (Morse-Smale Regression) Rom
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import math
import sys
import utils.importerUtils
sklearn = utils.importerUtils.importModuleLazy("sklearn", globals())
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .NDinterpolatorRom import NDinterpolatorRom
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

def _toStr(s):
  """
    Removes unicode from strings in Python 2 so amsc can use it.
    @ In, s, unicode or str, String to convert to plain str
    @ Out, s, str, Converted str
  """
  if sys.version_info.major > 2:
    return s
  return s.encode('ascii')

class MSR(NDinterpolatorRom):
  """
    MSR class - Computes an approximated hierarchical Morse-Smale decomposition
    from an input point cloud consisting of an arbitrary number of input
    parameters and one or more response values per input point
  """
  def __init__(self, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag = 'MSR ROM'
    supervisedLearning.__init__(self, **kwargs)
    self.acceptedGraphParam = ['approximate knn', 'delaunay', 'beta skeleton', \
                               'relaxed beta skeleton']
    self.acceptedPersistenceParam = ['difference','probability','count','area']
    self.acceptedGradientParam = ['steepest', 'maxflow']
    self.acceptedNormalizationParam = ['feature', 'zscore', 'none']
    self.acceptedPredictorParam = ['kde', 'svm']
    self.acceptedKernelParam = ['uniform', 'triangular', 'epanechnikov',
                                'biweight', 'quartic', 'triweight', 'tricube',
                                'gaussian', 'cosine', 'logistic', 'silverman',
                                'exponential']
    self.__amsc = []                      # AMSC object
    # Some sensible default arguments
    self.gradient = 'steepest'            # Gradient estimate methodology
    self.graph = 'beta skeleton'          # Neighborhood graph used
    self.beta = 1                         # beta used in the beta skeleton graph
                                          #  and its relaxed version
    self.knn = -1                         # k-nearest neighbor value for either
                                          #  the approximate knn strategy, or
                                          #  for initially pruning the beta
                                          #  skeleton graphs. (this could also
                                          #  potentially be used for restricting
                                          #  the models influencing a query
                                          #  point to only use those models
                                          #  belonging to a limited
                                          #  neighborhood of training points)
    self.simplification = 0               # Morse-smale simplification amount
                                          #  this should probably be normalized
                                          #  to [0,1], however for now it is not
                                          #  and the scale of it will depend on
                                          #  the type of persistence used
    self.persistence = 'difference'       # Strategy for merging topo partitions
    self.weighted = False                 # Should the linear models be weighted
                                          #  by probability information?
    self.normalization = None             # Should any normalization be
                                          #  performed within the AMSC? No, this
                                          #  data should already be standardized
    self.partitionPredictor = 'kde'       # The method used to predict the label
                                          #  of each query point (can be soft).
    self.blending = False                 # Flag: blend the predictions
                                          #  depending on soft label predictions
                                          #  or use only the most likely local
                                          #  model
    self.kernel = 'gaussian'              # What kernel should be used in the
                                          #  kde approach
    self.bandwidth = 1.                   # The bandwidth for the kde approach

    # Read everything in first, and then do error checking as some parameters
    # will not matter, but we can still throw a warning message that they may
    # want to clean up there input file. In some cases, we will have to do
    # value checking in place since the type cast can fail.
    for key,val in kwargs.items():
      if key.lower() == 'graph':
        self.graph = _toStr(val.strip()).lower()
      elif key.lower() == "gradient":
        self.gradient = _toStr(val.strip()).lower()
      elif key.lower() == "beta":
        try:
          self.beta = float(val)
        except ValueError:
          # If the user has specified a graph, use it, otherwise be sure to use
          #  the default when checking whether this is a warning or an error
          if 'graph' in kwargs:
            graph = _toStr(kwargs['graph'].strip()).lower()
          else:
            graph = self.graph
          if graph.endswith('beta skeleton'):
            self.raiseAnError(IOError, 'Requested invalid beta value:',
                              val, '(Allowable range: (0,2])')
          else:
            self.raiseAWarning('Requested invalid beta value:', self.beta,
                               '(Allowable range: (0,2]), however beta is',
                               'ignored when using the', graph,
                               'graph structure.')
      elif key.lower() == 'knn':
        try:
          self.knn = int(val)
        except ValueError:
          self.raiseAnError(IOError, 'Requested invalid knn value:',
                            val, '(Should be an integer value, knn <= 0 implies'
                            ,'use of the fully connected point set)')
      elif key.lower() == 'simplification':
        try:
          self.simplification = float(val)
        except ValueError:
          self.raiseAnError(IOError, 'Requested invalid simplification level:',
                            val, '(should be a floating point value)')
      elif key.lower() == 'bandwidth':
        if val == 'variable' or val == 'auto':
          self.bandwidth = val
        else:
          try:
            self.bandwidth = float(val)
          except ValueError:
            # If the user has specified a strategy, use it, otherwise be sure to
            #  use the default when checking whether this is a warning or an error
            if 'partitionPredictor' in kwargs:
              partPredictor = _toStr(kwargs['partitionPredictor'].strip()).lower()
            else:
              partPredictor = self.partitionPredictor
            if partPredictor == 'kde':
              self.raiseAnError(IOError, 'Requested invalid bandwidth value:',
                                val,'(should be a positive floating point value)')
            else:
              self.raiseAWarning('Requested invalid bandwidth value:',val,
                                 '(bandwidth > 0 or \"variable\"). However, it is ignored when',
                                 'using the', partPredictor, 'partition',
                                 'predictor')
      elif key.lower() == 'persistence':
        self.persistence = _toStr(val.strip()).lower()
      elif key.lower() == 'partitionpredictor':
        self.partitionPredictor = _toStr(val.strip()).lower()
      elif key.lower() == 'smooth':
        self.blending = True
      elif key.lower() == "kernel":
        self.kernel = val
      else:
        pass

    # Morse-Smale specific error handling
    if self.graph not in self.acceptedGraphParam:
      self.raiseAnError(IOError, 'Requested unknown graph type:',
                        '\"'+self.graph+'\"','(Available options:',
                        self.acceptedGraphParam,')')
    if self.gradient not in self.acceptedGradientParam:
      self.raiseAnError(IOError, 'Requested unknown gradient method:',
                        '\"'+self.gradient+'\"', '(Available options:',
                        self.acceptedGradientParam,')')
    if self.beta <= 0 or self.beta > 2:
      if self.graph.endswith('beta skeleton'):
        self.raiseAnError(IOError, 'Requested invalid beta value:',
                          self.beta, '(Allowable range: (0,2])')
      else:
        self.raiseAWarning('Requested invalid beta value:', self.beta,
                           '(Allowable range: (0,2]), however beta is',
                           'ignored when using the', self.graph,
                           'graph structure.')
    if self.persistence not in self.acceptedPersistenceParam:
      self.raiseAnError(IOError, 'Requested unknown persistence method:',
                        '\"'+self.persistence+'\"', '(Available options:',
                        self.acceptedPersistenceParam,')')
    if self.partitionPredictor not in self.acceptedPredictorParam:
      self.raiseAnError(IOError, 'Requested unknown partition predictor:'
                        '\"'+repr(self.partitionPredictor)+'\"','(Available options:',
                        self.acceptedPredictorParam,')')
    if self.bandwidth <= 0:
      if self.partitionPredictor == 'kde':
        self.raiseAnError(IOError, 'Requested invalid bandwidth value:',
                          self.bandwidth, '(bandwidth > 0)')
      else:
        self.raiseAWarning(IOError, 'Requested invalid bandwidth value:',
                          self.bandwidth, '(bandwidth > 0). However, it is',
                          'ignored when using the', self.partitionPredictor,
                          'partition predictor')

    if self.kernel not in self.acceptedKernelParam:
      if self.partitionPredictor == 'kde':
        self.raiseAnError(IOError, 'Requested unknown kernel:',
                          '\"'+self.kernel+'\"', '(Available options:',
                          self.acceptedKernelParam,')')
      else:
        self.raiseAWarning('Requested unknown kernel:', '\"'+self.kernel+'\"',
                           '(Available options:', self.acceptedKernelParam,
                           '), however the kernel is ignored when using the',
                           self.partitionPredictor,'partition predictor.')
    self.__resetLocal__()

  def __getstate__(self):
    """
      Overwrite state (for pickle-ing)
      we do not pickle the HDF5 (C++) instance
      but only the info to re-load it
      @ In, None
      @ Out, state, dict, namespace dictionary
    """
    state = dict(self.__dict__)
    state.pop('_MSR__amsc')
    state.pop('kdTree')
    return state

  def __setstate__(self,state):
    """
      Initialize the ROM with the data contained in state
      @ In, state, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    for key, value in state.items():
      setattr(self, key, value)
    self.kdTree             = None
    self.__amsc             = []
    self.__trainLocal__(self.X,self.Y)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      @ In, featureVals, np.ndarray or list of list, shape=[n_samples, n_features],
        an array of input feature values
      @ In, targetVals, np.ndarray, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """

    # # Possibly load this here in case people have trouble building it, so it
    # # only errors if they try to use it?
    from AMSC_Object import AMSC_Object

    self.X = featureVals[:][:]
    self.Y = targetVals

    if self.weighted:
      self.raiseAnError(NotImplementedError,
                    ' cannot use weighted data right now.')
    else:
      weights = None

    if self.knn <= 0:
      self.knn = self.X.shape[0]

    names = [_toStr(name) for name in self.features + self.target]
    # Data is already normalized, so ignore this parameter
    ### Comment replicated from the post-processor version, not sure what it
    ### means (DM)
    # FIXME: AMSC_Object employs unsupervised NearestNeighbors algorithm from
    #        scikit learn.
    #        The NearestNeighbor algorithm is implemented in
    #        SupervisedLearning, which requires features and targets by
    #        default, which we don't have here. When the NearestNeighbor is
    #        implemented in unSupervisedLearning switch to it.
    for index in range(len(self.target)):
      self.__amsc.append( AMSC_Object(X=self.X, Y=self.Y[:,index], w=weights, names=names,
                                      graph=self.graph, gradient=self.gradient,
                                      knn=self.knn, beta=self.beta,
                                      normalization=None,
                                      persistence=self.persistence) )
      self.__amsc[index].Persistence(self.simplification)
      self.__amsc[index].BuildLinearModels(self.simplification)

    # We need a KD-Tree for querying neighbors
    self.kdTree = sklearn.neighbors.KDTree(self.X)

    distances,_ = self.kdTree.query(self.X,k=self.knn)
    distances = distances.flatten()

    # The following are a list of common kernels defined centered at zero with
    # either infinite support or a support defined over the interval [1,1].
    # See: https://en.wikipedia.org/wiki/Kernel_(statistics)
    # Thus, the use of this indicator function. When using these kernels, we
    # must be sure to first scale the parameter into this support before calling
    # it. In our case, we want to center our information, such that the maximum
    # value occurs when the two points coincide, and so we will set u to be
    # inversely proportional to the distance between two points, and scaled by
    # a bandwidth parameter (either the user will fix, or we will compute)
    def indicator(u):
      """
        Method to return the indicator (see explaination above)
        @ In, u, float, the value to inquire
        @ Out, indicator, float, the abs of u
      """
      return np.abs(u)<1

    if self.kernel == 'uniform':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Uniform kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return 0.5*indicator(u)
    elif self.kernel == 'triangular':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Triangular kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (1-abs(u))*indicator(u)
    elif self.kernel == 'epanechnikov':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Epanechnikov kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return ( 3./4. )*(1-u**2)*indicator(u)
    elif self.kernel == 'biweight' or self.kernel == 'quartic':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Biweight kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (15./16.)*(1-u**2)**2*indicator(u)
    elif self.kernel == 'triweight':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Triweight kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (35./32.)*(1-u**2)**3*indicator(u)
    elif self.kernel == 'tricube':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Tricube kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (70./81.)*(1-abs(u)**3)**3*indicator(u)
    elif self.kernel == 'gaussian':
      if self.bandwidth == 'auto':
        self.bandwidth = 1.06*distances.std()*len(distances)**(-1./5.)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Gaussian kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return 1./np.sqrt(2*math.pi)*np.exp(-0.5*u**2)
    elif self.kernel == 'cosine':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Cosine kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return math.pi/4.*math.cos(u*math.pi/2.)*indicator(u)
    elif self.kernel == 'logistic':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Logistic kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return 1./(np.exp(u)+2+np.exp(-u))
    elif self.kernel == 'silverman':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Silverman kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        sqrt2 = math.sqrt(2)
        return 0.5 * np.exp(-abs(u)/sqrt2) * np.sin(abs(u)/sqrt2+math.pi/4.)
    elif self.kernel == 'exponential':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Exponential kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return np.exp(-abs(u))
    self.__kernel = kernel

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      Should return distance to nearest neighbor or average prediction error of
      all neighbors?
      @ In, featureVals, 2-D numpy array [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError, '__confidenceLocal__ method must be implemented!')

  def __evaluateLocal__(self,featureVals):
    """
      Perform regression on samples in featureVals.
      This will use the local predictor of each neighboring point weighted by its
      distance to that point.
      @ In, featureVals, numpy.array 2-D, features
      @ Out, returnDict, dict, dict of predicted values for each target ({'target1':numpy.array 1-D,'target2':numpy.array 1-D}
    """
    returnDict = {}
    for index, target in enumerate(self.target):
      if self.partitionPredictor == 'kde':
        partitions = self.__amsc[index].Partitions(self.simplification)
        weights = {}
        dists = np.zeros((featureVals.shape[0],self.X.shape[0]))
        for i,row in enumerate(featureVals):
          dists[i] = np.sqrt(((row-self.X)**2).sum(axis=-1))
        # This is a variable-based bandwidth that will adjust to the density
        # around the given query point
        if self.bandwidth == 'variable':
          h = sorted(dists)[self.knn-1]
        else:
          h = self.bandwidth
        for key,indices in partitions.items():
          #############
          ## Using SciKit Learn, we have a limited number of kernel functions to
          ## choose from.
          # kernel = self.kernel
          # if kernel == 'uniform':
          #   kernel = 'tophat'
          # if kernel == 'triangular':
          #   kernel = 'linear'
          # kde = KernelDensity(kernel=kernel, bandwidth=h).fit(self.X[indices,])
          # weights[key] = np.exp(kde.score_samples(featureVals))
          #############
          ## OR
          #############
          weights[key] = 0
          for idx in indices:
            weights[key] += self.__kernel(dists[:,idx]/h)
          weights[key]
          #############

        if self.blending:
          weightedPredictions = np.zeros(featureVals.shape[0])
          sumW = 0
          for key in partitions.keys():
            fx = self.__amsc[index].Predict(featureVals,key)
            wx = weights[key]
            sumW += wx
            weightedPredictions += fx*wx
          returnDict[target] = weightedPredictions if sumW == 0 else weightedPredictions / sumW
        else:
          predictions = np.zeros(featureVals.shape[0])
          maxWeights = np.zeros(featureVals.shape[0])
          for key in partitions.keys():
            fx = self.__amsc[index].Predict(featureVals,key)
            wx = weights[key]
            predictions[wx > maxWeights] = fx
            maxWeights[wx > maxWeights] = wx
          returnDict[target] = predictions
      elif self.partitionPredictor == 'svm':
        partitions = self.__amsc[index].Partitions(self.simplification)
        labels = np.zeros(self.X.shape[0])
        for idx,(key,indices) in enumerate(partitions.items()):
          labels[np.array(indices)] = idx
        # In order to make this deterministic for testing purposes, let's fix
        # the random state of the SVM object. Maybe, this could be exposed to the
        # user, but it shouldn't matter too much what the seed is for this.
        svc = sklearn.svm.SVC(probability=True,random_state=np.random.RandomState(8),tol=1e-15)
        svc.fit(self.X,labels)
        probabilities = svc.predict_proba(featureVals)

        classIdxs = list(svc.classes_)
        if self.blending:
          weightedPredictions = np.zeros(len(featureVals))
          sumW = 0
          for idx,key in enumerate(partitions.keys()):
            fx = self.__amsc[index].Predict(featureVals,key)
            # It could be that a particular partition consists of only the extrema
            # and they themselves point to cells with different opposing extrema.
            # That is, a maximum points to a different minimum than the minimum in
            # the two point partition. Long story short, we need to be prepared for
            # an empty partition which will thus not show up in the predictions of
            # the SVC, since no point has it as a label.
            if idx not in classIdxs:
              wx = np.zeros(probabilities.shape[0])
            else:
              realIdx = list(svc.classes_).index(idx)
              wx = probabilities[:,realIdx]
            if self.blending:
              weightedPredictions = weightedPredictions + fx*wx
              sumW += wx
          returnDict[target] = weightedPredictions if sumW == 0 else weightedPredictions / sumW
        else:
          predictions = np.zeros(featureVals.shape[0])
          maxWeights = np.zeros(featureVals.shape[0])
          for idx,key in enumerate(partitions.keys()):
            fx = self.__amsc[index].Predict(featureVals,key)
            # It could be that a particular partition consists of only the extrema
            # and they themselves point to cells with different opposing extrema.
            # That is, a maximum points to a different minimum than the minimum in
            # the two point partition. Long story short, we need to be prepared for
            # an empty partition which will thus not show up in the predictions of
            # the SVC, since no point has it as a label.
            if idx not in classIdxs:
              wx = np.zeros(probabilities.shape[0])
            else:
              realIdx = list(svc.classes_).index(idx)
              wx = probabilities[:,realIdx]
            predictions[wx > maxWeights] = fx
            maxWeights[wx > maxWeights] = wx
          returnDict[target] = predictions
      return returnDict


  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.X      = []
    self.Y      = []
    self.__amsc = []
    self.kdTree = None
