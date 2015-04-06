import sys
import numpy as np

import os
import time

####################################################
# There is probably a better way to do this
myPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(myPath+'/../src/postprocessors/')
from amsc import *
####################################################

import PySide.QtCore

import sklearn.neighbors
import sklearn.linear_model
import sklearn.preprocessing

import scipy.optimize
import scipy.stats

##Let's see what statsmodels weighted linear regression does
#import statsmodels.api as sm

""" A wrapper class for the C++ approximate Morse-Smale complex Object that also
    communicates with the UI via Qt's signal interface
"""
class AMSC_Object(PySide.QtCore.QObject):
  ## Paul Tol's colorblind safe colors
  colorList = ['#88CCEE', '#DDCC77', '#AA4499', '#117733', '#332288', '#999933',
               '#44AA99', '#882255', '#CC6677']
  ## Alternative Color Lists from Color Brewer
  colorList2 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
                '#e6ab02', '#a6761d', '#666666']
  colorList3 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
                '#ccebc5', '#ffed6f']
  colorList4 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                '#ffff33', '#a65628', '#f781bf', '#999999']
  colorList5 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
                '#ffff99', '#b15928']
  # colorList = ['#e41a1c', '#88CCEE', '#377eb8', '#DDCC77', '#4daf4a',
  #              '#AA4499', '#984ea3', '#ff7f00', '#ffff33',
  #              '#a65628', '#f781bf', '#999999', '#117733', '#332288',
  #              '#999933', '#44AA99', '#882255']

  sigPersistenceChanged = PySide.QtCore.Signal()
  sigSelectionChanged = PySide.QtCore.Signal()
  sigFilterChanged = PySide.QtCore.Signal()
  sigDataChanged = PySide.QtCore.Signal()
  sigModelsChanged = PySide.QtCore.Signal()
  sigWeightsChanged = PySide.QtCore.Signal()

  """ Initialization method that takes at minimum a set of input points and 
      corresponding output responses.
      @ In, X, an m-by-n array of values specifying m n-dimensional samples
      @ In, Y, a m vector of values specifying the output responses corresponding
        to the m samples specified by X
      @ In, w, an optional m vector of values specifying the weights associated
        to each of the m samples used. Default of None means all points will be
        equally weighted
      @ In, names, an optional list of strings that specify the names to
        associate to the n input dimensions and 1 output dimension. Default of
        None means input variables will be x0,x1...,x(n-1) and the output will
        be y
      @ In, graph, an optional string specifying the type of neighborhood graph
        to use. Default is 'beta skeleton,' but other valid types are:
        'delaunay', 'relaxed beta skeleton', or 'approximate knn'
      @ In, gradient, an optional string specifying the type of gradient
        estimator
        to use. Currently the only available option is 'steepest'
      @ In, knn, an optional integer value specifying the maximum number of
        k-nearest neighbors used to begin a neighborhood search. In the case of
        graph='[relaxed] beta skeleton', we will begin with the specified
        approximate knn graph and prune edges that do not satisfy the empty
        region criteria.
      @ In, beta, an optional floating point value between 0 and 2. This value is
        only used when graph='[relaxed] beta skeleton' and specifies the radius
        for the empty region graph computation (1=Gabriel graph, 2=Relative
        neighbor graph)
      @ In, normalization, an optional string specifying whether the
        inputs/output should be scaled before computing. Currently, two modes
        are supported 'zscore' and 'feature'. 'zscore' will ensure the data has
        a mean of zero and a standard deviation of 1 by subtracting the mean and
        dividing by the variance. 'feature' scales the data into the unit
        hypercube.
  """
  def __init__(self, X, Y, w=None, names=None, graph='beta skeleton',
               gradient='steepest', knn=-1, beta=1.0, normalization=None,
               debug=False):
    super(AMSC_Object,self).__init__()

    self.persistence = 0.

    self.segmentFits = {}
    self.extremumFits = {}

    self.segmentFitnesses = {}
    self.extremumFitnesses = {}

    self.mergeSequence = {}

    self.selectedExtrema = []
    self.selectedSegments = []

    self.filters = {}

    self.X = X
    self.Y = Y
    if w is not None:
      self.w = np.array(w)
    else:
      self.w = np.ones(len(Y))*1.0/float(len(Y))

    self.names = names
    self.normalization = normalization

    if self.X is None or self.Y is None:
      print('There is no data to process, what would the Maker have me do?')
      return

    if self.names is None:
      self.names = []
      for d in xrange(self.GetDimensionality()):
        self.names.append('x%d' % d)
      self.names.append('y')

    self.Xnorm = np.array(self.X)
    self.Ynorm = np.array(self.Y)
    if normalization == 'feature':
      min_max_scaler = sklearn.preprocessing.MinMaxScaler()
      self.Xnorm = min_max_scaler.fit_transform(self.X)
      self.Ynorm = min_max_scaler.fit_transform(self.Y)
    elif normalization == 'zscore':
      self.Xnorm = sklearn.preprocessing.scale(self.X, axis=0, with_mean=True,
                                               with_std=True, copy=True)
      self.Ynorm = sklearn.preprocessing.scale(self.Y, axis=0, with_mean=True,
                                               with_std=True, copy=True)

    if debug:
      sys.stderr.write('Graph Preparation: ')
      start = time.clock()
    knnAlgorithm = sklearn.neighbors.NearestNeighbors(n_neighbors=knn + 1,
                                                      algorithm='kd_tree')
    knnAlgorithm.fit(self.Xnorm)
    distances,edges = knnAlgorithm.kneighbors(self.Xnorm)
    if debug:
      end = time.clock()
      sys.stderr.write('%f s\n' % (end-start))
      print(edges.shape)

    edgesToPrune = []
    pairs = []                                # prevent duplicates with this guy
    for e1 in xrange(0,edges.shape[0]):
      for col in xrange(0,edges.shape[1]):
        e2 = edges[e1,col]
        if e1 != e2:
          pairs.append((e1,e2))

    # As seen here:
    #  http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
    seen = set()
    pairs = [ x for x in pairs if not (x in seen or x[::-1] in seen 
                                       or seen.add(x))]

    for edge in pairs:
      edgesToPrune.append(edge[0])
      edgesToPrune.append(edge[1])

    if debug:
      sys.stderr.write('%f s\n' % (end-start))
      sys.stderr.write('Decomposition: ')
      start = time.clock()

    self.__amsc = AMSCFloat(vectorFloat(self.Xnorm.flatten()),
                            vectorFloat(self.Y), vectorString(self.names),
                            str(graph), str(gradient), int(knn), float(beta),
                            vectorInt(edgesToPrune))

    if debug:
      end = time.clock()
      sys.stderr.write('%f s\n' % (end-start))

    hierarchy = self.__amsc.PrintHierarchy().strip().split(' ')

    self.mergeSequence = {}
    for line in hierarchy:
      if line.startswith('Maxima') or line.startswith('Minima'):
        tokens = line.split(',')
        p = float(tokens[1])
        dyingIndex = int(tokens[2])
        parentIndex = int(tokens[3])

        self.mergeSequence[dyingIndex] = (parentIndex,p)

  """ Sets the weights associated to the m input samples
      @ In, w, optional m vector specifying the new weights to use for the data
        points. Default is None and resets the weights to be uniform.
  """
  def SetWeights(self, w=None):
    if w is not None:
      self.w = np.array(w)
    else:
      self.w = np.ones(len(Y))*1.0/float(len(Y))

    if self.FitsSynced():
      self.BuildModels()
    self.sigWeightsChanged.emit()

  """ Returns a data structure holding the ordered merge sequence of extrema
      simplification
      @ Out, a dictionary of tuples where the key is the dying extrema and the
        tuple is the parent index and the persistence associated to the dying
        index, in that order.
  """
  def GetMergeSequence(self):
    return self.mergeSequence

  """ Returns the partitioned data based on a specified persistence level.
      @ In, persistence, a floating point value specifying the size of the
        smallest feature we want to track. Default = None means consider all
        features.
      @ Out, a dictionary lists where each key is a min-max tuple specifying the
        index of the minimum and maximum, respectively. Each entry will hold a
        list of indices specifying points that are associated to this min-max
        pair.
  """
  def Partitions(self,persistence=None):
    if persistence is None:
      persistence = self.persistence
    partitions = self.__amsc.GetPartitions(persistence)
    tupleKeyedPartitions = {}
    for strMinMax,indices in partitions.iteritems():
      minMax = tuple(map(int,strMinMax.split(',')))
      tupleKeyedPartitions[minMax] = indices
    return tupleKeyedPartitions

  """
  """
  def SegmentFitCoefficients(self):
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    coefficients = {}
    for key,fit in self.segmentFits.iteritems():
      coefficients[key] = fit[1:]
      # coefficients[key] = fit[:]
    return coefficients

  """
  """
  def SegmentFitnesses(self):
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    rSquared = {}
    for key,fitness in self.segmentFitnesses.iteritems():
      rSquared[key] = fitness
    return rSquared

  """
  """
  def SegmentPearsonCoefficients(self):
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    pearson = {}
    for key,fit in self.pearson.iteritems():
      pearson[key] = fit[:]
    return pearson

  """
  """
  def SegmentSpearmanCoefficients(self):
    if self.segmentFits is None or len(self.segmentFits) == 0:
      self.BuildModels(self.persistence)
    spearman = {}
    for key,fit in self.spearman.iteritems():
      spearman[key] = fit[:]
    return spearman

  """
  """
  def GetMask(self,indices=None):
    if indices is None:
      indices = list(xrange(0,self.GetSampleSize()))

    mask = np.ones(len(indices), dtype=bool)
    for header,bounds in self.filters.iteritems():
      if header in self.names:
        idx = self.names.index(header)
        if idx >= 0 and idx < len(self.names)-1:
          vals = self.X[indices,idx]
        elif idx == len(self.names)-1:
          vals = self.Y[indices]
      elif header == 'Predicted from Linear Fit':
        vals = self.PredictY(indices,fit='linear',applyFilters=False)
      elif header == 'Predicted from Maximum Fit':
        vals = self.PredictY(indices,fit='maximum',applyFilters=False)
      elif header == 'Predicted from Minimum Fit':
        vals = self.PredictY(indices,fit='minimum',applyFilters=False)
      elif header == 'Residual from Linear Fit':
        vals = self.Residuals(indices,fit='linear',applyFilters=False)
      elif header == 'Residual from Maximum Fit':
        vals = self.Residuals(indices,fit='maximum',applyFilters=False)
      elif header == 'Residual from Minimum Fit':
        vals = self.Residuals(indices,fit='minimum',applyFilters=False)
      elif header == 'Probability':
        vals = self.w[indices]


      mask = np.logical_and(mask, bounds[0] <= vals)
      mask = np.logical_and(mask, vals < bounds[1])
    indices = np.array(indices)[mask]
    indices = np.array(sorted(list(set(indices))))
    return indices

  """
  """
  def ComputePerDimensionFitErrors(self,key):
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
      for activeDim in indexOrder[0:(i+1)]:
        B[activeDim] = beta_hat[activeDim]

      X = self.X[np.array(items),:]
      X = X[:,indexOrder[0:(i+1)]]
      ## In the first case, X will be one-dimensional, so we have to enforce a
      ## reshape in order to get it to play nice.
      X = np.reshape(X,(len(items),i+1))
      y = self.Y[np.array(items)]
      w = self.w[np.array(items)]

      linearModel = sklearn.linear_model.LinearRegression(fit_intercept=True,
                                                          normalize=False,
                                                          copy_X=True)
      tempFit = linearModel.fit(X,y)
      temp_beta_hat = tempFit.coef_
      temp_yIntercept = tempFit.intercept_

#      smX = sm.add_constant(X)
#      model = sm.WLS(y, smX, w)
#      results = model.fit()
#      temp_beta_hat = results.params[1:]
#      temp_yIntercept = results.params[0]

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

  """
  """
  def Persistence(self, p=None):
    if p is None:
      return self.persistence
    self.persistence = p
    self.segmentFits = {}
    self.extremumFits = {}
    self.segmentFitnesses = {}
    self.extremumFitnesses = {}
    self.sigPersistenceChanged.emit()

  """
  """
  def BuildModels(self,persistence=None):
    self.segmentFits = {}
    self.extremumFits = {}
    self.segmentFitnesses = {}
    self.extremumFitnesses = {}
    self.BuildLinearModels(persistence)
    self.BuildGaussianModels(persistence)
    self.ComputeExtremaShapeDescriptors()
    # self.BuildPolynomialModels(persistence)
    self.ComputeStatisticalSensitivity()
    self.sigModelsChanged.emit()

  """
  """
  def BuildLinearModels(self, persistence=None):
    partitions = self.Partitions(persistence)

    for key,items in partitions.iteritems():
      X = self.Xnorm[np.array(items),:]
      y = np.array(self.Ynorm[np.array(items)])
      w = self.w[np.array(items)]

      linearModel = sklearn.linear_model.LinearRegression(fit_intercept=True,
                                                          normalize=False,
                                                          copy_X=True)
#      linearModel = sklearn.linear_model.LinearRegression(fit_intercept=False,
#                                                          normalize=False,
#                                                          copy_X=True)

      lmFit = linearModel.fit(X,y)
      self.segmentFits[key] = np.hstack((lmFit.intercept_,lmFit.coef_))
#      self.segmentFits[key] = lmFit.coef_
#      print('SKL',self.segmentFits[key])

#      smX = sm.add_constant(X)
#      # smX = X
#      model = sm.WLS(y, smX, w)
#      results = model.fit()
#      self.segmentFits[key] = results.params
#      print('SM',results.params)

      yHat = X.dot(self.segmentFits[key][1:]) + self.segmentFits[key][0]
#      # yHat = X.dot(self.segmentFits[key][:]) + self.segmentFits[key][0]

      self.segmentFitnesses[key] = linearModel.score(X,y)

  """
  """
  def BuildPolynomialModels(self, persistence=None):
    partitions = self.Partitions(persistence)
    for extType in [0,1]:
      count = 0
      extFlowSet = {}
      for key,items in partitions.iteritems():
        extIdx = key[extType]
        if extIdx not in extFlowSet.keys():
          extFlowSet[extIdx] = []
        for idx in items:
          extFlowSet[extIdx].append(idx)

      for extIdx,indices in extFlowSet.iteritems():
        X = self.Xnorm[np.array(indices),:]
        Y = self.Y[np.array(indices)]
        self.extremumFits[extIdx] = np.polyfit(X,Y,2)
        yHat = np.zeros(X.shape[0])
        for i in xrange(X.shape[0]):
          yHat[i] = 0 #FIXME
        self.extremumFitnesses[extIdx] = 1 - np.sum((yHat - Y)**2)/np.sum((Y - np.mean(Y))**2)

  """
  """
  def BuildGaussianModels(self, persistence=None):
    dimCount = len(self.names)-1
    # For now, if we are doing anything more than a moderate amount of
    # dimensions, use a constrained Gaussian
    constrainedGaussian = (dimCount > 10)
#############DEBUG##############################################################
    # constrainedGaussian = True
#############END DEBUG##########################################################

    partitions = self.Partitions(self.persistence)
    for extType in [0,1]:
      count = 0
      extFlowSet = {}
      for key,items in partitions.iteritems():
        extIdx = key[extType]
        if extIdx not in extFlowSet.keys():
          extFlowSet[extIdx] = []
        for idx in items:
          extFlowSet[extIdx].append(idx)

      paramCount = 1
      if constrainedGaussian:
        paramCount += dimCount
      else:
        for d in xrange(dimCount+1):
          paramCount += d

      for extIdx,indices in extFlowSet.iteritems():
        if len(indices) < paramCount:
          print('Too few samples, skipping this segment: %d' % extIdx)
          continue

        X = self.Xnorm[np.array(indices),:]
        Y = self.Y[np.array(indices)]
        W = self.w[np.array(indices)]

        # For fitting a multivariate Gaussian, we will fix the mean to be at the
        # extrema and the y-offset to match the output value at the extrema's
        # location
        mu = self.Xnorm[extIdx,:]
        c = self.Y[extIdx]

        paramGuess = []

        if constrainedGaussian:
          # Define a variable number of inputs for our constrained Gaussian, it
          # is constrained because we are only fitting diagonal components of
          # the covariance matrix
          def residuals(*arg):
            a = arg[0][-1]
            xvec = arg[1]
            yvec = arg[2]
            err = []
            A = np.identity(len(arg[0][:-1]))*np.array(arg[0][:-1])
            Adet = np.linalg.det(np.linalg.inv(A))
            for idx in xrange(0,len(yvec)):
              v = mu-xvec[idx]
              C = a # a*(1/math.sqrt(2*math.pi**dimCount*Adet))
              yPredicted = C*np.exp(-(v.dot(A).dot(v))) + c - C
              err.append((yvec[idx] - yPredicted))
            return err
          # Not sure what is a good starting place for the covariance, so just
          # use the identity matrix
          for d in xrange(0,paramCount-1):
            paramGuess.append(1.0)
        else:
          # Define a variable number of inputs for our Gaussian
          def residuals(*arg):
            a = arg[0][-1]
            xvec = arg[1]
            yvec = arg[2]
            err = []

            A = np.zeros((dimCount,dimCount))
            idx = 0
            for dRow in xrange(dimCount):
              for dCol in xrange(dRow,dimCount):
                A[dRow,dCol] = A[dCol,dRow] = arg[0][idx]
                idx += 1

            # Fastest way to tell if a matrix is positive definite
            try:
              np.linalg.cholesky(A)
            except np.linalg.LinAlgError as msg:
#              if 'Matrix is not positive definite' in msg.args:
                # Return a large value to penalize non-conformant solution.
                return 1e50*np.ones(len(yvec))
#                return sys.float_info.max*np.ones(len(yvec))

            #Adet = np.linalg.det(np.linalg.inv(A))
            for idx in xrange(0,len(yvec)):
              v = mu-xvec[idx]
              C = a # a*(1/math.sqrt(2*math.pi**dimCount*Adet))
              yPredicted = C*np.exp(-(v.dot(A).dot(v))) + c - C
              err.append((yvec[idx] - yPredicted))
            return err


          # Not sure what is a good starting place for the covariance, so just
          # use the identity matrix
          for dRow in xrange(dimCount):
            for dCol in xrange(dRow,dimCount):
              paramGuess.append(1.0*(dRow==dCol))

        if extType:
          # Amplitude estimate (the range of this data, opens down for the maxima
          # thus amplitude should be positive
          paramGuess.append(self.Y[extIdx]-min(self.Y[indices]))
        else:
          # Amplitude estimate (the range of this data, opens up for minima
          # thus the amplitude should be negative
          paramGuess.append(self.Y[extIdx]-max(self.Y[indices]))

        test = scipy.optimize.leastsq(residuals, paramGuess, args=(X,Y),
                                      full_output=True)
#        print(test)

        a = test[0][-1]
        if constrainedGaussian:
          A = np.identity(dimCount)*test[0][0:-1]
        else:
          A = np.zeros((dimCount,dimCount))
          idx = 0
          for dRow in xrange(dimCount):
            for dCol in xrange(dRow,dimCount):
              A[dRow,dCol] = A[dCol,dRow] = test[0][idx]
              idx += 1
        #Adet = np.linalg.det(np.linalg.inv(A))

        def GaussFit(x):
          v = mu - x
          C = a # a*(1/sqrt(2*pi**dimCount*Adet))
          return C*np.exp(-(v.dot(A).dot(v))) + c - C

        yHat = np.zeros(X.shape[0])
        for i in xrange(X.shape[0]):
          yHat[i] = GaussFit(X[i,])
        rSquared = 1 - np.sum((yHat - Y)**2)/np.sum((Y - np.mean(Y))**2)

        self.extremumFits[extIdx] = (mu,c,a,A)
        self.extremumFitnesses[extIdx] = rSquared

  """
  """
  def GetNames(self):
    return self.names

  """
  """
  def GetNormedX(self,rows=None,cols=None,applyFilters=False):
    if rows is None:
      rows = list(xrange(0,self.GetSampleSize()))
    if cols is None:
      cols = list(xrange(0,self.GetDimensionality()))

    if applyFilters:
      rows = self.GetMask(rows)
    retValue = self.Xnorm[rows,:]
    return retValue[:,cols]

  """
  """
  def GetX(self,rows=None,cols=None,applyFilters=False):
    if rows is None:
      rows = list(xrange(0,self.GetSampleSize()))
    if cols is None:
      cols = list(xrange(0,self.GetDimensionality()))

    rows = sorted(list(set(rows)))
    if applyFilters:
      rows = self.GetMask(rows)

    retValue = self.X[rows,:]
    if len(rows) == 0:
      return []
    return retValue[:,cols]
    # return self.X[rows,cols]

  """
  """
  def GetY(self,indices=None, applyFilters=False):
    if indices is None:
      indices = list(xrange(0,self.GetSampleSize()))
    else:
      indices = sorted(list(set(indices)))

    if applyFilters:
      indices = self.GetMask(indices)
    if len(indices) == 0:
      return []
    return self.Y[indices]

  """
  """
  def GetWeights(self,indices=None, applyFilters=False):
    if indices is None:
      indices = list(xrange(0,self.GetSampleSize()))
    else:
      indices = sorted(list(set(indices)))

    if applyFilters:
      indices = self.GetMask(indices)

    if len(indices) == 0:
      return []
    return self.w[indices]

  """
  """
  def PredictY(self,indices=None, fit='linear',applyFilters=False):
    partitions = self.Partitions(self.persistence)
    
    predictedY = np.zeros(self.GetSampleSize())
    if fit == 'linear':
      for key,items in partitions.iteritems():
        beta_hat = self.segmentFits[key][1:]
        y_intercept = self.segmentFits[key][0]
        # beta_hat = self.segmentFits[key][:]
        # y_intercept = 0
        for idx in items:
          predictedY[idx] = self.Xnorm[idx,:].dot(beta_hat) + y_intercept
    else:
      extType = 0
      if fit == 'maximum':
        extType = 1
      for key,items in partitions.iteritems():
        extIdx = key[extType]
        (mu,c,a,A) = self.extremumFits[extIdx]
        def GaussFit(x):
          v = mu - x
          C = a # a*(1/sqrt(2*pi**dimCount*Adet))
          return C*np.exp(-(v.dot(A).dot(v))) + c#+ c - C

        for idx in items:
          predictedY[idx] = GaussFit(self.Xnorm[idx,:])

    if indices is None:
      indices = list(xrange(0,self.GetSampleSize()))
    if applyFilters:
      indices = self.GetMask(indices)
    indices = np.array(sorted(list(set(indices))))
    return predictedY[indices]

  """
  """
  def Residuals(self,indices=None, fit='linear', signed=False, applyFilters=False):
    if indices is None:
      indices = list(xrange(0,self.GetSampleSize()))
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

    # if(fit =='linear'):
    #   w = self.w[indices]
    #   for i in xrange(len(residuals)):
    #     residuals[i] *= w[i]

    return residuals

  """
  """
  def GetColors(self):
    partitions = self.Partitions(self.persistence)
    partColors = {}
    for i,key in enumerate(partitions.keys()):
      partColors[key] = self.colorList[(i % len(self.colorList))]

    for i,key in enumerate(partitions.keys()):
      minKey = key[0]
      maxKey = key[1]
      if minKey not in partColors.keys():
        partColors[minKey] = self.colorList2[(i % len(self.colorList2))]
      if maxKey not in partColors.keys():
        partColors[maxKey] = self.colorList3[(i % len(self.colorList3))]

    return partColors

  """
  """
  def GetSelectedExtrema(self):
    return self.selectedExtrema

  """
  """
  def GetSelectedSegments(self):
    return self.selectedSegments

  """
  """
  def FitsSynced(self):
    return self.SegmentFitsSynced() and self.ExtremumFitsSynced()

  """
  """
  def SegmentFitsSynced(self):
    fitKeys = self.segmentFits.keys()
    rSquaredKeys = self.segmentFitnesses.keys()

    if sorted(fitKeys) != sorted(rSquaredKeys) \
    or sorted(fitKeys) != sorted(self.GetCurrentLabels()) \
    or self.segmentFits is None or len(self.segmentFits) == 0:
      return False

    return True

  """
  """
  def ExtremumFitsSynced(self):
    extIdxs = []
    for extPair in self.GetCurrentLabels():
      extIdxs.extend(list(extPair))

    extIdxs = list(set(extIdxs))
    fitKeys = self.extremumFits.keys()
    rSquaredKeys = self.extremumFitnesses.keys()

    if sorted(fitKeys) != sorted(rSquaredKeys) \
    or sorted(fitKeys) != sorted(extIdxs) \
    or self.extremumFits is None or len(self.extremumFits) == 0:
      return False

    return True

  """
  """
  def GetCurrentLabels(self):
    partitions = self.Partitions(self.persistence)
    return partitions.keys()

  """ Sets the currently selected items of this instance
      selectionList - a mixed list of 2-tuples and integers representing
                      min-max index pairs and extremum indices, respectively
      cross_inclusion - This will ensure if you select all of the segments
                        attached to an extermum get selected and vice versa
  """
  def SetSelection(self, selectionList, cross_inclusion=False):
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

  """
  """
  def ClearFilter(self):
    self.filters = {}
    self.sigSelectionChanged.emit()

  """
  """
  def SetFilter(self,name,bounds):
    if bounds is None:
      self.filters.pop(name,None)
    else:
      self.filters[name] = bounds
      print(name, bounds)
    self.sigSelectionChanged.emit()

  """
  """
  def GetFilter(self,name):
    if name in self.filters.keys():
      return self.filters[name]
    else:
      return None

  """
  """
  def Select(self, idx):
    if isinstance(idx,int):
      if idx not in self.selectedExtrema:
        self.selectedExtrema.append(idx)
    else:
      if idx not in self.sectedSegments:
        self.selectedSegments.append(idx)

      self.sigSelectionChanged.emit()

  """
  """
  def Deselect(self, idx):
    if isinstance(idx,int):
      if idx in self.selectedExtrema:
        self.selectedExtrema.remove(idx)
    else:
      if idx in self.sectedSegments:
        self.selectedSegments.remove(idx)

      self.sigSelectionChanged.emit()

  """
  """
  def ClearSelection(self):
    self.selectedSegments = []
    self.selectedExtrema = []
    self.sigSelectionChanged.emit()

  """
  """
  def GetSelectedIndices(self,segmentsOnly=True):
    partitions = self.Partitions(self.persistence)
    indices = []
    for extPair,indexSet in partitions.iteritems():
      if extPair in self.selectedSegments \
      or extPair[0] in self.selectedExtrema \
      or extPair[1] in self.selectedExtrema:
        indices.extend(indexSet)

    indices = self.GetMask(indices)   
    return list(indices)

  """
  """
  def GetSampleSize(self):
    return len(self.Y)

  """
  """
  def GetDimensionality(self):
    return self.X.shape[1]

  """
  """
  def ComputeExtremaShapeDescriptors(self):
    self.derivatives = {}
    for ext,fit in self.extremumFits.iteritems():
      (mu,c,a,A) = fit
      
      # def dfdu(x):
      #   v = x - mu
      #   C = a
      #   return C*np.exp(-(v.dot(A).dot(v)))

      # def dudxm(x,m):
      #   v = x - mu
      #   mySum = 0
      #   for i in xrange(0,len(x)):
      #     mySum += 2*v[i]*A[m,i]
      #   return -mySum

      # def du2dxmdxn(x,m,n):
      #   return -2*A[m,n]

      x = self.Xnorm[int(ext),:]
      self.derivatives[int(ext)] = np.zeros(shape=(len(x),len(x)))
      for m in xrange(len(x)):
        for n in xrange(len(x)):
          # val1 = dfdu(x)
          # val2 = dudxm(x,m)
          # val3  = dudxm(x,n)
          # val4 = du2dxmdxn(x,m,n)
          # self.derivatives[int(ext)][m,n] = val1*(val2*val3 + val4)
          ## You are a big dum dum, Dan, all you had to do was this:
          self.derivatives[int(ext)][m,n] = -2*a*A[m,n]

  """
  """
  def GetSecondOrderDerivatives(self,key):
    return self.derivatives[key]

  """
  """
  def GetConcentrationMatrix(self,key):
    (mu,c,a,A) = self.extremumFits[key]
    # covariance = np.linalg.inv(A)
    return A

  """
  """
  def GetExtremumFitCoefficients(self,key):
    (mu,c,a,A) = self.extremumFits[key]
    return (mu,c,a,A)

  """
  """
  def GetClassification(self,idx):
    partitions = self.Partitions(0)
    for minMaxPair in partitions.keys():
      if idx == minMaxPair[0]:
        return 'minimum'
      elif idx == minMaxPair[1]:
        return 'maximum'
    return 'regular'

  """
  """
  def ComputeStatisticalSensitivity(self):
    partitions = self.Partitions()

    self.pearson = {}
    self.spearman = {}
    for key,items in partitions.iteritems():
      X = self.Xnorm[np.array(items),:]
      y = self.Y[np.array(items)]
      w = self.w[np.array(items)]

      # betaHat = self.segmentFits[key][1:]
      # betaHat = self.segmentFits[key][:]

      sigmaY = np.std(y)

      self.pearson[key] = []
      self.spearman[key] = []
      for col in xrange(0,X.shape[1]):
        sigmaXcol = np.std(X[:,col])
        rcol = scipy.stats.pearsonr(X[:,col], y)[0]
        rcol2 = np.cov(X[:,col],y)[1,0]/(sigmaY*sigmaXcol)
        ## Alternative formula for determining the Pearson correlation
        # rcol3 = sigmaXcol/sigmaY * betaHat[col]
        # print(rcol,rcol2,rcol3)
        self.pearson[key].append(rcol)
        self.spearman[key].append(scipy.stats.spearmanr(X[:,col], y)[0])
    print('Pearson',self.pearson)
    print('Spearman',self.spearman)

  """
  """
  def XMLFormattedHierarchy(self):
    return self.__amsc.XMLFormattedHierarchy()

  """
  """
  def PrintHierarchy(self):
    return self.__amsc.PrintHierarchy()
