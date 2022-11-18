##############################################################################
 # Software License Agreement (BSD License)                                   #
 #                                                                            #
 # Copyright 2014 University of Utah                                          #
 # Scientific Computing and Imaging Institute                                 #
 # 72 S Central Campus Drive, Room 3750                                       #
 # Salt Lake City, UT 84112                                                   #
 #                                                                            #
 # THE BSD LICENSE                                                            #
 #                                                                            #
 # Redistribution and use in source and binary forms, with or without         #
 # modification, are permitted provided that the following conditions         #
 # are met:                                                                   #
 #                                                                            #
 # 1. Redistributions of source code must retain the above copyright          #
 #    notice, this list of conditions and the following disclaimer.           #
 # 2. Redistributions in binary form must reproduce the above copyright       #
 #    notice, this list of conditions and the following disclaimer in the     #
 #    documentation and/or other materials provided with the distribution.    #
 # 3. Neither the name of the copyright holder nor the names of its           #
 #    contributors may be used to endorse or promote products derived         #
 #    from this software without specific prior written permission.           #
 #                                                                            #
 # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       #
 # IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  #
 # OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    #
 # IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           #
 # INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   #
 # NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  #
 # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      #
 # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   #
 # THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          #
 ##############################################################################

import sys
import numpy as np
import time
import os
import itertools
import collections

####################################################
# This is tenuous at best, if the the directory structure of RAVEN changes, this
# will need to be updated, make sure you add this to the beginning of the search
# path, so that you try to grab the locally built one before relying on an
# installed version
myPath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,myPath)
try:
  import amsc
except ImportError as e:
  makeFilePath = os.path.realpath(os.path.join(myPath,'..','..','amsc.mk'))
  sys.stderr.write(str(e)+"\n")
  sys.stderr.write('It appears you do not have the AMSC library. Try '
                   + 'running the following command:' + os.linesep
                   + '\tmake -f ' + makeFilePath + os.linesep)
  sys.exit(1)
################################################################################


import scipy.optimize
import scipy.stats
import scipy

##Let's see what statsmodels weighted linear regression does
#import statsmodels.api as sm

def WeightedLinearModel(X,y,w):
  """ A wrapper for playing with the linear regression used per segment. The
      benefit of having this out here is that we do not have to adjust it in
      several places in the AMSC class, since it can build linear models for
      an arbitrary subset of dimensions, as well.
      @ In, X, a matrix of input samples
      @ In, y, a vector of output responses corresponding to the input samples
      @ In, w, a vector of weights corresponding to the input samples
      @ Out, a tuple consisting of the fits y-intercept and the the list of
        linear coefficients.
  """
  ## Using scipy directly to do weighted linear regression on non-centered data
  Xw = np.ones((X.shape[0],X.shape[1]+1))
  Xw[:,1:] = X
  Xw = Xw * np.sqrt(w)[:, None]
  yw = y * np.sqrt(w)
  results = scipy.linalg.lstsq(Xw, yw)[0]
  yIntercept = results[0]
  betaHat = results[1:]

  return (yIntercept,betaHat)

class AMSC_Object(object):
  """ A wrapper class for the C++ approximate Morse-Smale complex Object that
      also communicates with the UI via Qt's signal interface
  """
  def __init__(self, X, Y, w=None, names=None, graph='beta skeleton',
               gradient='steepest', knn=-1, beta=1.0, normalization=None,
               persistence='difference', edges=None, debug=False):
    """ Initialization method that takes at minimum a set of input points and
        corresponding output responses.
        @ In, X, an m-by-n array of values specifying m n-dimensional samples
        @ In, Y, a m vector of values specifying the output responses
          corresponding to the m samples specified by X
        @ In, w, an optional m vector of values specifying the weights
          associated to each of the m samples used. Default of None means all
          points will be equally weighted
        @ In, names, an optional list of strings that specify the names to
          associate to the n input dimensions and 1 output dimension. Default of
          None means input variables will be x0,x1...,x(n-1) and the output will
          be y
        @ In, graph, an optional string specifying the type of neighborhood
          graph to use. Default is 'beta skeleton,' but other valid types are:
          'delaunay,' 'relaxed beta skeleton,' 'none', or 'approximate knn'
        @ In, gradient, an optional string specifying the type of gradient
          estimator
          to use. Currently the only available option is 'steepest'
        @ In, knn, an optional integer value specifying the maximum number of
          k-nearest neighbors used to begin a neighborhood search. In the case
          of graph='[relaxed] beta skeleton', we will begin with the specified
          approximate knn graph and prune edges that do not satisfy the empty
          region criteria.
        @ In, beta, an optional floating point value between 0 and 2. This
          value is only used when graph='[relaxed] beta skeleton' and specifies
          the radius for the empty region graph computation (1=Gabriel graph,
          2=Relative neighbor graph)
        @ In, normalization, an optional string specifying whether the
          inputs/output should be scaled before computing. Currently, two modes
          are supported 'zscore' and 'feature'. 'zscore' will ensure the data
          has a mean of zero and a standard deviation of 1 by subtracting the
          mean and dividing by the variance. 'feature' scales the data into the
          unit hypercube.
        @ In, persistence, an optional string specifying how we will compute
          the persistence hierarchy. Currently, three modes are supported
          'difference', 'probability' and 'count'. 'difference' will take the
          function value difference of the extrema and its closest function
          valued neighboring saddle, 'probability' will augment this value by
          multiplying the probability of the extremum and its saddle, and count
          will make the larger point counts more persistent.
        @ In, edges, an optional list of custom edges to use as a starting point
          for pruning, or in place of a computed graph.
        @ In, debug, an optional boolean flag for whether debugging output
          should be enabled.
    """
    super(AMSC_Object,self).__init__()
    if X is not None and len(X) > 1:
      self.Reinitialize(X, Y, w, names, graph, gradient, knn, beta,
                        normalization, persistence, edges, debug)
    else:
      # Set some reasonable defaults
      self.SetEmptySettings()

  def SetEmptySettings(self):
    """
       Empties all internal storage containers
    """
    self.partitions = {}
    self.persistence = 0.

    self.segmentFits = {}
    self.extremumFits = {}

    self.segmentFitnesses = {}
    self.extremumFitnesses = {}

    self.mergeSequence = {}

    self.selectedExtrema = []
    self.selectedSegments = []

    self.filters = {}

    self.minIdxs = []
    self.maxIdxs = []

    self.X = []
    self.Y = []
    self.w = []

    self.normalization = None

    self.names = []
    self.Xnorm = []
    self.Ynorm = []

    self.__amsc = None

  def Reinitialize(self, X, Y, w=None, names=None, graph='beta skeleton', gradient='steepest', knn=-1, beta=1.0, normalization=None, persistence='difference', edges=None, debug=False):
    """ Allows the caller to basically start over with a new dataset.
        @ In, X, an m-by-n array of values specifying m n-dimensional samples
        @ In, Y, a m vector of values specifying the output responses
          corresponding to the m samples specified by X
        @ In, w, an optional m vector of values specifying the weights
          associated to each of the m samples used. Default of None means all
          points will be equally weighted
        @ In, names, an optional list of strings that specify the names to
          associate to the n input dimensions and 1 output dimension. Default of
          None means input variables will be x0,x1...,x(n-1) and the output will
          be y
        @ In, graph, an optional string specifying the type of neighborhood
          graph to use. Default is 'beta skeleton,' but other valid types are:
          'delaunay,' 'relaxed beta skeleton,' or 'approximate knn'
        @ In, gradient, an optional string specifying the type of gradient
          estimator
          to use. Currently the only available option is 'steepest'
        @ In, knn, an optional integer value specifying the maximum number of
          k-nearest neighbors used to begin a neighborhood search. In the case
          of graph='[relaxed] beta skeleton', we will begin with the specified
          approximate knn graph and prune edges that do not satisfy the empty
          region criteria.
        @ In, beta, an optional floating point value between 0 and 2. This
          value is only used when graph='[relaxed] beta skeleton' and specifies
          the radius for the empty region graph computation (1=Gabriel graph,
          2=Relative neighbor graph)
        @ In, normalization, an optional string specifying whether the
          inputs/output should be scaled before computing. Currently, two modes
          are supported 'zscore' and 'feature'. 'zscore' will ensure the data
          has a mean of zero and a standard deviation of 1 by subtracting the
          mean and dividing by the variance. 'feature' scales the data into the
          unit hypercube.
        @ In, persistence, an optional string specifying how we will compute
          the persistence hierarchy. Currently, three modes are supported
          'difference', 'probability' and 'count'. 'difference' will take the
          function value difference of the extrema and its closest function
          valued neighboring saddle, 'probability' will augment this value by
          multiplying the probability of the extremum and its saddle, and count
          will make the larger point counts more persistent.
    """
    import sklearn.neighbors
    import sklearn.linear_model
    import sklearn.preprocessing

    self.partitions = {}
    self.persistence = 0.

    self.segmentFits = {}
    self.extremumFits = {}

    self.segmentFitnesses = {}
    self.extremumFitnesses = {}

    self.mergeSequence = {}

    self.selectedExtrema = []
    self.selectedSegments = []

    self.filters = {}

    self.minIdxs = []
    self.maxIdxs = []

    self.partitionColors = {}
    self.colorIdx = 0

    self.X = X
    self.Y = Y
    if w is not None:
      self.w = np.array(w)
    else:
      self.w = np.ones(len(Y))*1.0/float(len(Y))

    self.names = names
    self.normalization = normalization
    self.graph = graph
    self.gradient = gradient
    self.knn = knn
    self.beta = beta

    if self.X is None or self.Y is None:
      print('There is no data to process, what would the Maker have me do?')
      self.SetEmptySettings()
      return

    if self.names is None:
      self.names = []
      for d in range(self.GetDimensionality()):
        self.names.append('x%d' % d)
      self.names.append('y')

    if normalization == 'feature':
      # This doesn't work with one-dimensional arrays on older versions of
      #  sklearn
      min_max_scaler = sklearn.preprocessing.MinMaxScaler()
      self.Xnorm = min_max_scaler.fit_transform(np.atleast_2d(self.X))
      self.Ynorm = min_max_scaler.fit_transform(np.atleast_2d(self.Y))
    elif normalization == 'zscore':
      self.Xnorm = sklearn.preprocessing.scale(self.X, axis=0, with_mean=True,
                                               with_std=True, copy=True)
      self.Ynorm = sklearn.preprocessing.scale(self.Y, axis=0, with_mean=True,
                                               with_std=True, copy=True)
    else:
      self.Xnorm = np.array(self.X)
      self.Ynorm = np.array(self.Y)

    if knn <= 0:
      knn = len(self.Xnorm)-1

    if debug:
      sys.stderr.write('Graph Preparation: ')
      start = time.clock()

    if knn <= 0:
      knn = len(self.Y)-1

    if edges is None:
      knnAlgorithm = sklearn.neighbors.NearestNeighbors(n_neighbors=knn,
                                                        algorithm='kd_tree')
      knnAlgorithm.fit(self.Xnorm)
      edges = knnAlgorithm.kneighbors(self.Xnorm, return_distance=False)
      if debug:
        end = time.clock()
        sys.stderr.write('%f s\n' % (end-start))

      pairs = []                              # prevent duplicates with this guy
      for e1 in range(0,edges.shape[0]):
        for col in range(0,edges.shape[1]):
          e2 = edges.item(e1,col)
          if e1 != e2:
            pairs.append((e1,e2))
    else:
      pairs = edges

    # As seen here:
    #  http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
    seen = set()
    pairs = [ x for x in pairs if not (x in seen or x[::-1] in seen
                                       or seen.add(x))]
    edgesToPrune = []
    for edge in pairs:
      edgesToPrune.append(edge[0])
      edgesToPrune.append(edge[1])

    if debug:
      end = time.clock()
      sys.stderr.write('%f s\n' % (end-start))
      sys.stderr.write('Decomposition: ')
      start = time.clock()

    self.__amsc = amsc.AMSCFloat(amsc.vectorFloat(self.Xnorm.flatten()),
                                 amsc.vectorFloat(self.Y),
                                 amsc.vectorString(self.names), str(self.graph),
                                 str(self.gradient), int(self.knn),
                                 float(self.beta), str(persistence),
                                 amsc.vectorFloat(self.w),
                                 amsc.vectorInt(edgesToPrune), debug)

    if debug:
      end = time.clock()
      sys.stderr.write('%f s\n' % (end-start))

    hierarchy = self.__amsc.PrintHierarchy().strip().split(' ')

    self.persistences = []
    self.mergeSequence = {}
    for line in hierarchy:
      if line.startswith('Maxima') or line.startswith('Minima'):
        tokens = line.split(',')
        p = float(tokens[1])
        dyingIndex = int(tokens[2])
        parentIndex = int(tokens[3])

        self.mergeSequence[dyingIndex] = (parentIndex,p)
        self.persistences.append(p)

    self.persistences = sorted(list(set(self.persistences)))

    partitions = self.Partitions(self.persistences[0])
    cellIdxs = np.array(list(partitions.keys()))
    self.minIdxs = np.unique(cellIdxs[:,0])
    self.maxIdxs = np.unique(cellIdxs[:,1])

  def SetWeights(self, w=None):
    """ Sets the weights associated to the m input samples
        @ In, w, optional m vector specifying the new weights to use for the
          data points. Default is None and resets the weights to be uniform.
    """
    if w is not None:
      self.w = np.array(w)
    elif len(self.Y) > 0:
      self.w = np.ones(len(self.Y))*1.0/float(len(self.Y))

    if self.FitsSynced():
      self.BuildModels()

  def GetMergeSequence(self):
    """ Returns a data structure holding the ordered merge sequence of extrema
        simplification
        @ Out, a dictionary of tuples where the key is the dying extrema and the
          tuple is the parent index and the persistence associated to the dying
          index, in that order.
    """
    return self.mergeSequence

  def Partitions(self,persistence=None):
    """ Returns the partitioned data based on a specified persistence level.
        @ In, persistence, a floating point value specifying the size of the
          smallest feature we want to track. Default = None means consider all
          features.
        @ Out, a dictionary lists where each key is a min-max tuple specifying
          the index of the minimum and maximum, respectively. Each entry will
          hold a list of indices specifying points that are associated to this
          min-max pair.
    """
    if self.__amsc is None:
      return None
    if persistence is None:
      persistence = self.persistence
    if persistence not in self.partitions:
      partitions = self.__amsc.GetPartitions(persistence)
      tupleKeyedPartitions = {}
      minMaxKeys = partitions.keys()
      for strMinMax in minMaxKeys:
        indices = partitions[strMinMax]
        minMax = tuple(map(int,strMinMax.split(',')))
        tupleKeyedPartitions[minMax] = indices
      self.partitions[persistence] = tupleKeyedPartitions
    return self.partitions[persistence]

  def StableManifolds(self,persistence=None):
    """ Returns the partitioned data based on a specified persistence level.
        @ In, persistence, a floating point value specifying the size of the
          smallest feature we want to track. Default = None means consider all
          features.
        @ Out, a dictionary lists where each key is a integer specifying
          the index of the maximum. Each entry will hold a list of indices
          specifying points that are associated to this maximum.
    """
    if persistence is None:
      persistence = self.persistence
    return self.__amsc.GetStableManifolds(persistence)

  def UnstableManifolds(self,persistence=None):
    """ Returns the partitioned data based on a specified persistence level.
        @ In, persistence, a floating point value specifying the size of the
          smallest feature we want to track. Default = None means consider all
          features.
        @ Out, a dictionary lists where each key is a integer specifying
          the index of the minimum. Each entry will hold a list of indices
          specifying points that are associated to this minimum.
    """
    if persistence is None:
      persistence = self.persistence
    return self.__amsc.GetUnstableManifolds(persistence)

  def SegmentFitCoefficients(self):
    """ Returns a dictionary keyed off the min-max index pairs defining
        Morse-Smale segments where the values are the linear coefficients of
        the input dimensions sorted in the same order as the input data.
        @ Out, a dictionary with tuples as keys specifying a pair of integers
          denoting minimum and maximum indices. The values associated to the
          dictionary keys are the linear coefficients fit for each min-max pair.
    """
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    coefficients = {}
    for key,fit in self.segmentFits.items():
      coefficients[key] = fit[1:]
      # coefficients[key] = fit[:]
    return coefficients

  def SegmentFitnesses(self):
    """ Returns a dictionary keyed off the min-max index pairs defining
        Morse-Smale segments where the values are the R^2 metrics of the linear
        fits for each Morse-Smale segment.
        @ Out, a dictionary with tuples as keys specifying a pair of integers
          denoting minimum and maximum indices. The values associated to the
          dictionary keys are the R^2 values for each linear fit of the
          Morse-Smale segments defined by the min-max pair of integers.
    """
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    rSquared = {}
    for key,fitness in self.segmentFitnesses.items():
      rSquared[key] = fitness
    return rSquared

  def SegmentPearsonCoefficients(self):
    """ Returns a dictionary keyed off the min-max index pairs defining
        Morse-Smale segments where the values are the Pearson correlation
        coefficients of the input dimensions sorted in the same order as the
        input data.
        @ Out, a dictionary with tuples as keys specifying a pair of integers
          denoting minimum and maximum indices. The values associated to the
          dictionary keys are the Pearson correlation coefficients associated
          to each subset of the data.
    """
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    pearson = {}
    for key,fit in self.pearson.items():
      pearson[key] = fit[:]
    return pearson

  def SegmentSpearmanCoefficients(self):
    """ Returns a dictionary keyed off the min-max index pairs defining
        Morse-Smale segments where the values are the Spearman rank correlation
        coefficients of the input dimensions sorted in the same order as the
        input data.
        @ Out, a dictionary with tuples as keys specifying a pair of integers
          denoting minimum and maximum indices. The values associated to the
          dictionary keys are the Spearman rank correlation coefficients
          associated to each subset of the data.
    """
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    spearman = {}
    for key,fit in self.spearman.items():
      spearman[key] = fit[:]
    return spearman

  def GetMask(self,indices=None):
    """ Applies all data filters to the input data and returns a list of
        filtered indices that specifies the rows of data that satisfy all
        conditions.
        @ In, indices, an optional integer list of indices to start from, if not
          supplied, then the mask will be applied to all indices of the data.
        @ Out, a 1-dimensional array of integer indices that is a subset of
          the input data row indices specifying rows that satisfy every set
          filter criterion.
    """
    if indices is None:
      indices = list(range(0,self.GetSampleSize()))

    mask = np.ones(len(indices), dtype=bool)
    for header,bounds in self.filters.items():
      if header in self.names:
        idx = self.names.index(header)
        if idx >= 0 and idx < len(self.names)-1:
          vals = self.X[indices,idx]
        elif idx == len(self.names)-1:
          vals = self.Y[indices]
      elif header == 'Predicted from Linear Fit':
        vals = self.PredictY(indices, fit='linear', applyFilters=False)
      elif header == 'Predicted from Maximum Fit':
        vals = self.PredictY(indices, fit='maximum', applyFilters=False)
      elif header == 'Predicted from Minimum Fit':
        vals = self.PredictY(indices, fit='minimum', applyFilters=False)
      elif header == 'Residual from Linear Fit':
        vals = self.Residuals(indices, fit='linear', applyFilters=False)
      elif header == 'Residual from Maximum Fit':
        vals = self.Residuals(indices, fit='maximum', applyFilters=False)
      elif header == 'Residual from Minimum Fit':
        vals = self.Residuals(indices, fit='minimum', applyFilters=False)
      elif header == 'Probability':
        vals = self.w[indices]
      mask = np.logical_and(mask, bounds[0] <= vals)
      mask = np.logical_and(mask, vals < bounds[1])

    indices = np.array(indices)[mask]
    indices = np.array(sorted(list(set(indices))))
    return indices

  def ComputePerDimensionFitErrors(self,key):
    """ Heuristically builds lower-dimensional linear patches for a Morse-Smale
        segment specified by a tuple of integers, key. The heuristic is to sort
        the set of linear coefficients by magnitude and progressively refit the
        data using more and more dimensions and computing R^2 values for each
        lower dimensional fit until we arrive at the full dimensional linear fit
        @ In, key, a tuple of two integers specifying the minimum and maximum
          indices used to key the partition upon which we are retrieving info.
        @ Out, a tuple of three equal sized lists that specify the index order
          of the dimensions added where the indices match the input data's
          order, the R^2 values for each progressively finer fit, and the
          F-statistic for each progressively finer fit. Thus, an index order of
          [2,3,1,0] would imply the first fit uses only dimension 2, and
          the next fit uses dimension 2 and 3, and the next fit uses 2, 3, and
          1, and the final fit uses dimensions 2, 1, 3, and 0.
    """
    partitions = self.Partitions(self.persistence)
    if key not in self.segmentFits or key not in partitions:
      return None

    beta_hat = self.segmentFits[key][1:]
    yIntercept = self.segmentFits[key][0]
    # beta_hat = self.segmentFits[key][:]
    # yIntercept = 0
    items = partitions[key]

    X = self.Xnorm[np.array(items),:]
    y = self.Y[np.array(items)]
    w = self.w[np.array(items)]

    yHat = X.dot(beta_hat) + yIntercept
    RSS2 = np.sum(w*(y-yHat)**2)/np.sum(w)

    RSS1 = 0

    rSquared = []
    ## From here: http://en.wikipedia.org/wiki/F-test
    fStatistic = [] ## the computed F statistic
    indexOrder = list(reversed(np.argsort(np.absolute(beta_hat))))
    for i,nextDim in enumerate(indexOrder):
      B = np.zeros(self.GetDimensionality())
      for activeDim in indexOrder[0:
        (i+1)]:
        B[activeDim] = beta_hat[activeDim]

      X = self.X[np.array(items),:]
      X = X[:,indexOrder[0:(i+1)]]
      ## In the first case, X will be one-dimensional, so we have to enforce a
      ## reshape in order to get it to play nice.
      X = np.reshape(X,(len(items),i+1))
      y = self.Y[np.array(items)]
      w = self.w[np.array(items)]

      (temp_yIntercept,temp_beta_hat) = WeightedLinearModel(X,y,w)

      yHat = X.dot(temp_beta_hat) + temp_yIntercept

      # Get a weighted mean
      yMean = np.average(y,weights=w)

      RSS2 = np.sum(w*(y-yHat)**2)/np.sum(w)
      if RSS1 == 0:
        fStatistic.append(0)
      else:
        fStatistic.append(  (RSS1-RSS2)/(len(indexOrder)-i) \
                          / (RSS2/(len(y)-len(indexOrder)))  )

      SStot = np.sum(w*(y-yMean)**2)/np.sum(w)
      rSquared.append(1-(RSS2/SStot))
      RSS1 = RSS2

    return (indexOrder,rSquared,fStatistic)

  def Persistence(self, p=None):
    """ Sets or returns the persistence simplfication level to be used for
        representing this Morse-Smale complex
        @ In, p, a floating point value that will set the persistence value,
          if this value is set to None, then this function will return the
          current persistence leve.
        @ Out, if no p value is supplied then this function will return the
          current persistence setting. If a p value is supplied, it will be
          returned as it will be the new persistence setting of this object.
    """
    if p is None:
      return self.persistence
    self.persistence = p
    self.segmentFits = {}
    self.extremumFits = {}
    self.segmentFitnesses = {}
    self.extremumFitnesses = {}
    return self.persistence

  def BuildModels(self,persistence=None):
    """ Forces the construction of linear fits per Morse-Smale segment and
        Gaussian fits per stable/unstable manifold for the user-specified
        persistence level.
        @ In, persistence, a floating point value specifying the simplification
          level to use, if this value is None, then we will build models based
          on the internally set persistence level for this Morse-Smale object.
    """
    self.segmentFits = {}
    self.extremumFits = {}
    self.segmentFitnesses = {}
    self.extremumFitnesses = {}
    self.BuildLinearModels(persistence)
    self.ComputeStatisticalSensitivity()

  def BuildLinearModels(self, persistence=None):
    """ Forces the construction of linear fits per Morse-Smale segment.
        @ In, persistence, a floating point value specifying the simplification
          level to use, if this value is None, then we will build models based
          on the internally set persistence level for this Morse-Smale object.
    """
    partitions = self.Partitions(persistence)

    for key,items in partitions.items():
      X = self.Xnorm[np.array(items),:]
      y = np.array(self.Y[np.array(items)])
      w = self.w[np.array(items)]

      (temp_yIntercept,temp_beta_hat) = WeightedLinearModel(X,y,w)
      self.segmentFits[key] = np.hstack((temp_yIntercept,temp_beta_hat))

      yHat = X.dot(self.segmentFits[key][1:]) + self.segmentFits[key][0]

      self.segmentFitnesses[key] = sum(np.sqrt((yHat-y)**2))

  def GetNames(self):
    """ Returns the names of the input and output dimensions in the order they
        appear in the input data.
        @ Out, a list of strings specifying the input + output variable names.
    """
    return self.names

  def GetNormedX(self,rows=None,cols=None,applyFilters=False):
    """ Returns the normalized input data requested by the user
        @ In, rows, a list of non-negative integers specifying the row indices
          to return
        @ In, cols, a list of non-negative integers specifying the column
          indices to return
        @ In, applyFilters, a boolean specifying whether data filters should be
          used to prune the results
        @ Out, a matrix of floating point values specifying the normalized data
          values used in internal computations filtered by the three input
          parameters.
    """
    if rows is None:
      rows = list(range(0,self.GetSampleSize()))
    if cols is None:
      cols = list(range(0,self.GetDimensionality()))

    if applyFilters:
      rows = self.GetMask(rows)
    retValue = self.Xnorm[rows,:]
    return retValue[:,cols]

  def GetX(self,rows=None,cols=None,applyFilters=False):
    """ Returns the input data requested by the user
        @ In, rows, a list of non-negative integers specifying the row indices
          to return
        @ In, cols, a list of non-negative integers specifying the column
          indices to return
        @ In, applyFilters, a boolean specifying whether data filters should be
          used to prune the results
        @ Out, a matrix of floating point values specifying the input data
          values filtered by the three input parameters.
    """
    if rows is None:
      rows = list(range(0,self.GetSampleSize()))
    if cols is None:
      cols = list(range(0,self.GetDimensionality()))

    rows = sorted(list(set(rows)))
    if applyFilters:
      rows = self.GetMask(rows)

    retValue = self.X[rows,:]
    if len(rows) == 0:
      return []
    return retValue[:,cols]

  def GetY(self, indices=None, applyFilters=False):
    """ Returns the output data requested by the user
        @ In, indices, a list of non-negative integers specifying the
          row indices to return
        @ In, applyFilters, a boolean specifying whether data filters should be
          used to prune the results
        @ Out, a list of floating point values specifying the output data
          values filtered by the two input parameters.
    """
    if indices is None:
      indices = list(range(0,self.GetSampleSize()))
    else:
      indices = sorted(list(set(indices)))

    if applyFilters:
      indices = self.GetMask(indices)
    if len(indices) == 0:
      return []
    return self.Y[indices]

  def GetLabel(self, indices=None, applyFilters=False):
    """ Returns the label pair indices requested by the user
        @ In, indices, a list of non-negative integers specifying the
          row indices to return
        @ In, applyFilters, a boolean specifying whether data filters should be
          used to prune the results
        @ Out, a list of integer 2-tuples specifying the minimum and maximum
          index of the specified rows.
    """
    if indices is None:
      indices = list(range(0,self.GetSampleSize()))
    elif isinstance(indices,collections.Iterable):
      indices = sorted(list(set(indices)))
    else:
      indices = [indices]

    if applyFilters:
      indices = self.GetMask(indices)
    if len(indices) == 0:
      return []
    partitions = self.__amsc.GetPartitions(self.persistence)
    labels = self.X.shape[0]*[None]
    for strMinMax in partitions.keys():
      partIndices = partitions[strMinMax]
      label = tuple(map(int,strMinMax.split(',')))
      for idx in np.intersect1d(partIndices,indices):
        labels[idx] = label

    labels = np.array(labels)
    if len(indices) == 1:
      return labels[indices][0]
    return labels[indices]

  def GetWeights(self, indices=None, applyFilters=False):
    """ Returns the weights requested by the user
        @ In, indices, a list of non-negative integers specifying the
          row indices to return
        @ In, applyFilters, a boolean specifying whether data filters should be
          used to prune the results
        @ Out, a list of floating point values specifying the weights associated
          to the input data rows filtered by the two input parameters.
    """
    if indices is None:
      indices = list(range(0,self.GetSampleSize()))
    else:
      indices = sorted(list(set(indices)))

    if applyFilters:
      indices = self.GetMask(indices)

    if len(indices) == 0:
      return []
    return self.w[indices]

  def Predict(self, x, key):
    """ Returns the predicted response of x given a model index
        @ In, x, a list of input values matching the dimensionality of the
          input space
        @ In, key, a 2-tuple specifying a min-max id pair used for determining
          which model is being used for prediction
        @ Out, a predicted response value for the given input point
    """
    partitions = self.Partitions(self.persistence)
    beta_hat = self.segmentFits[key][1:]
    y_intercept = self.segmentFits[key][0]
    if len(x.shape) == 1:
      return x.dot(beta_hat) + y_intercept
    else:
      predictions = []
      for xi in x:
        predictions.append(xi.dot(beta_hat) + y_intercept)
      return predictions

  def PredictY(self,indices=None, fit='linear',applyFilters=False):
    """ Returns the predicted output values requested by the user
        @ In, indices, a list of non-negative integers specifying the
          row indices to predict
        @ In, fit, an optional string specifying which fit should be used to
          predict each location, 'linear' = Morse-Smale segment, 'maxima' =
          descending/stable manifold, 'minima' = ascending/unstable manifold.
          Only 'linear' is available in this version.
        @ In, applyFilters, a boolean specifying whether data filters should be
          used to prune the results
        @ Out, a list of floating point values specifying the predicted output
          values filtered by the three input parameters.
    """
    partitions = self.Partitions(self.persistence)

    predictedY = np.zeros(self.GetSampleSize())
    if fit == 'linear':
      for key,items in partitions.items():
        beta_hat = self.segmentFits[key][1:]
        y_intercept = self.segmentFits[key][0]
        for idx in items:
          predictedY[idx] = self.Xnorm[idx,:].dot(beta_hat) + y_intercept
    ## Possible extension to fit data per stable or unstable manifold would
    ## go here

    if indices is None:
      indices = list(range(0,self.GetSampleSize()))
    if applyFilters:
      indices = self.GetMask(indices)
    indices = np.array(sorted(list(set(indices))))
    return predictedY[indices]

  def Residuals(self,indices=None,fit='linear',signed=False,applyFilters=False):
    """ Returns the residual between the output data and the predicted output
        values requested by the user
        @ In, indices, a list of non-negative integers specifying the
          row indices for which to compute residuals
        @ In, fit, an optional string specifying which fit should be used to
          predict each location, 'linear' = Morse-Smale segment, 'maxima' =
          descending/stable manifold, 'minima' = ascending/unstable manifold
        @ In, applyFilters, a boolean specifying whether data filters should be
          used to prune the results
        @ Out, a list of floating point values specifying the signed difference
          between the predicted output values and the original output data
          filtered by the three input parameters.
    """
    if indices is None:
      indices = list(range(0,self.GetSampleSize()))
    else:
      indices = sorted(list(set(indices)))
    if applyFilters:
      indices = self.GetMask(indices)

    indices = np.array(sorted(list(set(indices))))

    yRange = max(self.Y) - min(self.Y)
    actualY = self.GetY(indices)
    predictedY = self.PredictY(indices,fit)
    if signed:
      residuals = (actualY-predictedY)/yRange
    else:
      residuals = np.absolute(actualY-predictedY)/yRange

    return residuals

  def GetColors(self):
    """ Returns a dictionary of colors where the keys specify Morse-Smale
        segment min-max integer index pairs, unstable/ascending manifold minima
        integer indices, and stable/descending manifold maxima integer indices.
        The values are hex strings specifying unique colors for each different
        type of segment.
        @ Out, a dictionary specifying unique colors for each Morse-Smale
          segment, stable/descending manifold, and unstable/ascending manifold.
    """
    partitions = self.Partitions(self.persistence)
    partColors = {}
    for key in partitions.keys():
      minKey,maxKey = key
      if key not in self.partitionColors:
        self.partitionColors[key] = next(self.colorList)
      if minKey not in self.partitionColors:
        self.partitionColors[minKey] = next(self.colorList)
      if maxKey not in self.partitionColors:
        self.partitionColors[maxKey] = next(self.colorList)

      # Only get the colors we need for this level of the partition
      partColors[key] = self.partitionColors[key]
      partColors[minKey] = self.partitionColors[minKey]
      partColors[maxKey] = self.partitionColors[maxKey]

    return partColors

  def GetSelectedExtrema(self):
    """ Returns the extrema highlighted as being selected in an attached UI
        @ Out, a list of non-negative integer indices specifying the extrema
          selected.
    """
    return self.selectedExtrema

  def GetSelectedSegments(self):
    """ Returns the Morse-Smale segments highlighted as being selected in an
        attached UI
        @ Out, a list of non-negative integer index pairs specifying the min-max
          pairs associated to the selected Morse-Smale segments.
    """
    return self.selectedSegments

  def GetCurrentLabels(self):
    """ Returns a list of tuples that specifies the min-max index labels
        associated to each input sample
        @ Out, a list of tuples that are each a pair of non-negative integers
          specifying the min-flow and max-flow indices associated to each input
          sample at the current level of persistence
    """
    partitions = self.Partitions(self.persistence)
    return partitions.keys()

  def GetSampleSize(self,key = None):
    """ Returns the number of samples in the input data
        @ In, key, an optional 2-tuple specifying a min-max id pair used for
          determining which partition size should be returned. If not specified
          then the size of the entire data set will be returned.
        @ Out, an integer specifying the number of samples.
    """
    if key is None:
      return len(self.Y)
    else:
      return len(self.partitions[self.persistence][key])

  def GetDimensionality(self):
    """ Returns the dimensionality of the input space of the input data
        @ Out, an integer specifying the dimensionality of the input samples.
    """
    return self.X.shape[1]

  def GetClassification(self,idx):
    """ Given an index, this function will report whether that sample is a local
        minimum, a local maximum, or a regular point.
        @ In, idx, a non-negative integer less than the sample size of the input
        data.
        @ Out, a string specifying the classification type of the input sample:
          will be 'maximum,' 'minimum,' or 'regular.'
    """
    if idx in self.minIdxs:
      return 'minimum'
    elif idx in self.maxIdxs:
      return 'maximum'
    return 'regular'

  def ComputeStatisticalSensitivity(self):
    """ Computes the per segment Pearson correlation coefficients and the
        Spearman rank correlation coefficients and stores them internally.
    """
    partitions = self.Partitions()

    self.pearson = {}
    self.spearman = {}
    for key,items in partitions.items():
      X = self.Xnorm[np.array(items),:]
      y = self.Y[np.array(items)]

      self.pearson[key] = []
      self.spearman[key] = []

      for col in range(0,X.shape[1]):
        sigmaXcol = np.std(X[:,col])
        self.pearson[key].append(scipy.stats.pearsonr(X[:,col], y)[0])
        self.spearman[key].append(scipy.stats.spearmanr(X[:,col], y)[0])

  def PrintHierarchy(self):
    """ Writes the complete Morse-Smale merge hierarchy to a string object.
        @ Out, a string object storing the entire merge hierarchy of all minima
          and maxima.
    """
    return self.__amsc.PrintHierarchy()

  def GetNeighbors(self,idx):
    """ Returns a list of neighbors for the specified index
        @ In, an integer specifying the query point
        @ Out, a integer list of neighbors indices
    """
    return self.__amsc.Neighbors(idx)

try:
  import PySide.QtCore as qtc
  __QtAvailable = True
except ImportError as e:
  try:
    import PySide2.QtCore as qtc
    __QtAvailable = True
  except ImportError as e:
    __QtAvailable = False

if __QtAvailable:

  TolColors = ['#88CCEE', '#DDCC77', '#AA4499', '#117733', '#332288', '#999933',
             '#44AA99', '#882255', '#CC6677']

  class QAMSC_Object(AMSC_Object,qtc.QObject):
    ## Paul Tol's colorblind safe colors
    colorList = itertools.cycle(TolColors)

    sigPersistenceChanged = qtc.Signal()
    sigSelectionChanged = qtc.Signal()
    sigFilterChanged = qtc.Signal()
    sigDataChanged = qtc.Signal()
    sigModelsChanged = qtc.Signal()
    sigWeightsChanged = qtc.Signal()

    def Reinitialize(self, X, Y, w=None, names=None, graph='beta skeleton',
                     gradient='steepest', knn=-1, beta=1.0, normalization=None,
                     persistence='difference', edges=None, debug=False):
      """ Allows the caller to basically start over with a new dataset.
          @ In, X, an m-by-n array of values specifying m n-dimensional samples
          @ In, Y, a m vector of values specifying the output responses
            corresponding to the m samples specified by X
          @ In, w, an optional m vector of values specifying the weights
            associated to each of the m samples used. Default of None means all
            points will be equally weighted
          @ In, names, an optional list of strings that specify the names to
            associate to the n input dimensions and 1 output dimension. Default of
            None means input variables will be x0,x1...,x(n-1) and the output will
            be y
          @ In, graph, an optional string specifying the type of neighborhood
            graph to use. Default is 'beta skeleton,' but other valid types are:
            'delaunay,' 'relaxed beta skeleton,' or 'approximate knn'
          @ In, gradient, an optional string specifying the type of gradient
            estimator
            to use. Currently the only available option is 'steepest'
          @ In, knn, an optional integer value specifying the maximum number of
            k-nearest neighbors used to begin a neighborhood search. In the case
            of graph='[relaxed] beta skeleton', we will begin with the specified
            approximate knn graph and prune edges that do not satisfy the empty
            region criteria.
          @ In, beta, an optional floating point value between 0 and 2. This
            value is only used when graph='[relaxed] beta skeleton' and specifies
            the radius for the empty region graph computation (1=Gabriel graph,
            2=Relative neighbor graph)
          @ In, normalization, an optional string specifying whether the
            inputs/output should be scaled before computing. Currently, two modes
            are supported 'zscore' and 'feature'. 'zscore' will ensure the data
            has a mean of zero and a standard deviation of 1 by subtracting the
            mean and dividing by the variance. 'feature' scales the data into the
            unit hypercube.
          @ In, persistence, an optional string specifying how we will compute
            the persistence hierarchy. Currently, three modes are supported
            'difference', 'probability' and 'count'. 'difference' will take the
            function value difference of the extrema and its closest function
            valued neighboring saddle, 'probability' will augment this value by
            multiplying the probability of the extremum and its saddle, and count
            will make the larger point counts more persistent.
      """
      super(QAMSC_Object,self).Reinitialize(X, Y, w, names, graph, gradient,
                                            knn, beta, normalization,
                                            persistence, edges, debug)
      self.sigDataChanged.emit()

    def Persistence(self, p=None):
      """ Sets or returns the persistence simplfication level to be used for
          representing this Morse-Smale complex
          @ In, p, a floating point value that will set the persistence value,
            if this value is set to None, then this function will return the
            current persistence leve.
          @ Out, if no p value is supplied then this function will return the
            current persistence setting. If a p value is supplied, it will be
            returned as it will be the new persistence setting of this object.
      """
      if p is None:
        return self.persistence
      pers = super(QAMSC_Object,self).Persistence(p)
      self.sigPersistenceChanged.emit()
      return pers

    def SetWeights(self, w=None):
      """ Sets the weights associated to the m input samples
          @ In, w, optional m vector specifying the new weights to use for the
            data points. Default is None and resets the weights to be uniform.
      """
      super(QAMSC_Object,self).SetWeights(w)
      self.sigWeightsChanged.emit()

    def BuildModels(self,persistence=None):
      """ Forces the construction of linear fits per Morse-Smale segment and
          Gaussian fits per stable/unstable manifold for the user-specified
          persistence level.
          @ In, persistence, a floating point value specifying the simplification
            level to use, if this value is None, then we will build models based
            on the internally set persistence level for this Morse-Smale object.
      """
      super(QAMSC_Object,self).BuildModels(persistence)
      self.sigModelsChanged.emit()

    def SetSelection(self, selectionList, cross_inclusion=False):
      """ Sets the currently selected items of this instance
          @ In, selectionList, a mixed list of 2-tuples and integers representing
            min-max index pairs and extremum indices, respectively
          @ In, cross_inclusion, a boolean that will ensure if you select all of
            the segments attached to an extermum get selected and vice versa
      """
      partitions = self.Partitions(self.persistence)

      self.selectedSegments = []
      self.selectedExtrema = []

      for idx in selectionList:
        ## Here are a few alternatives to do the same thing, I think I like the
        ## not an int test the best because it is less likely to change than the
        ## representation of the pair
        #if isinstance(label, tuple):
        #if hasattr(label, '__len__'):
        if isinstance(idx,int):
          self.selectedExtrema.append(idx)
          #If you select an extremum, also select all of its attached segments
          if cross_inclusion:
            for minMax in partitions.keys():
              if idx in minMax:
                self.selectedSegments.append(minMax)
        else:
          self.selectedSegments.append(idx)
          #If you select an segment, also select all of its attached extrema
          if cross_inclusion:
            self.selectedExtrema.extend(list(idx))

      self.selectedSegments = list(set(self.selectedSegments))
      self.selectedExtrema = list(set(self.selectedExtrema))

      self.sigSelectionChanged.emit()

    def ClearFilter(self):
      """ Erases all currently set filters on any dimension.
      """
      self.filters = {}
      self.sigSelectionChanged.emit()

    def SetFilter(self,name,bounds):
      """ Sets the bounds of the selected dimension as a filter
          @ In, name, a string denoting the variable to which this filter will be
            applied.
          @ In, bounds, a list of two values specifying a lower and upper bound on
            the dimension specified by name.
      """
      if bounds is None:
        self.filters.pop(name,None)
      else:
        self.filters[name] = bounds
      self.sigSelectionChanged.emit()

    def GetFilter(self,name):
      """ Returns the currently set filter for a particular dimension specified.
          @ In, name, a string denoting the variable for which one wants to
            retrieve filtered information.
          @ Out, a list consisting of two values that specify the filter
            boundaries of the queried dimension.
      """
      if name in self.filters.keys():
        return self.filters[name]
      else:
        return None

    def Select(self, idx):
      """ Add a segment or extremum to the list of currently selected items
          @ In, idx, either an non-negative integer or a 2-tuple of non-negative
            integers specifying the index of an extremum or a min-max index pair.
      """
      if isinstance(idx,int):
        if idx not in self.selectedExtrema:
          self.selectedExtrema.append(idx)
      else:
        if idx not in self.sectedSegments:
          self.selectedSegments.append(idx)

        self.sigSelectionChanged.emit()

    def Deselect(self, idx):
      """ Remove a segment or extremum from the list of currently selected items
          @ In, idx, either an non-negative integer or a 2-tuple of non-negative
            integers specifying the index of an extremum or a min-max index pair.
      """
      if isinstance(idx,int):
        if idx in self.selectedExtrema:
          self.selectedExtrema.remove(idx)
      else:
        if idx in self.sectedSegments:
          self.selectedSegments.remove(idx)

        self.sigSelectionChanged.emit()

    def ClearSelection(self):
      """ Empties the list of selected items.
      """
      self.selectedSegments = []
      self.selectedExtrema = []
      self.sigSelectionChanged.emit()

    def GetSelectedIndices(self,segmentsOnly=True):
      """ Returns a mixed list of extremum indices and min-max index pairs
          specifying all of the segments selected.
          @ In, segmentsOnly, a boolean variable that will filter the results to
            only return min-max index pairs.
          @ Out, a list of non-negative integers and 2-tuples consisting of
            non-negative integers.
      """
      partitions = self.Partitions(self.persistence)
      indices = []
      for extPair,indexSet in partitions.items():
        if extPair in self.selectedSegments \
        or extPair[0] in self.selectedExtrema \
        or extPair[1] in self.selectedExtrema:
          indices.extend(indexSet)

      indices = self.GetMask(indices)
      return list(indices)

    def FitsSynced(self):
      """ Returns whether the segment and extremum fits are built for the
          currently selected level of persistence.
          @ Out, a boolean that reports True if everything is synced and False,
            otherwise.
      """
      fitKeys = self.segmentFits.keys()
      rSquaredKeys = self.segmentFitnesses.keys()

      if sorted(fitKeys) != sorted(rSquaredKeys) \
      or sorted(fitKeys) != sorted(self.GetCurrentLabels()) \
      or self.segmentFits is None or len(self.segmentFits) == 0:
        return False

      return True
  # sys.stderr.write(str(e) +'\n')
  # sys.exit(1)
