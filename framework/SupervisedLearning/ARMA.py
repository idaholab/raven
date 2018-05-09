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
  Specific ROM implementation for ARMA (Autoregressive Moving Average) ROM
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import itertools
import statsmodels
import numpy as np
from scipy import optimize
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import randomUtils
import Distributions
from .SupervisedLearning import superVisedLearning
from sklearn import linear_model, neighbors
#Internal Modules End--------------------------------------------------------------------------------

class ARMA(superVisedLearning):
  """
    Autoregressive Moving Average model for time series analysis. First train then evaluate.
    Specify a Fourier node in input file if detrending by Fourier series is needed.

    Time series Y: Y = X + \sum_{i}\sum_k [\delta_ki1*sin(2pi*k/basePeriod_i)+\delta_ki2*cos(2pi*k/basePeriod_i)]
    ARMA series X: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
  """
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler: a MessageHandler object in charge of raising errors,
                           and printing messages
      @ In, kwargs: an arbitrary dictionary of keywords and values
    """
    superVisedLearning.__init__(self,messageHandler,**kwargs)
    self.printTag          = 'ARMA'
    self._dynamicHandling  = True # This ROM is able to manage the time-series on its own.
    self.armaPara          = {}
    self.fourierResults    = {} # dictionary of Fourier results, by target
    self.armaNormPara      = {} # dictionary of the Gaussian-normal ARMA characterstics, by target
    self.armaResult        = {} # dictionary of assorted useful arma information, by target
    self.armaPara['Pmax']      = kwargs.get('Pmax', 3)
    self.armaPara['Pmin']      = kwargs.get('Pmin', 0)
    self.armaPara['Qmax']      = kwargs.get('Qmax', 3)
    self.armaPara['Qmin']      = kwargs.get('Qmin', 0)
    self.armaPara['dimension'] = len(self.features)
    self.reseedCopies          = kwargs.get('reseedCopies',True)
    self.outTruncation         = kwargs.get('outTruncation', None)     # Additional parameters to allow user to specify the time series to be all positive or all negative
    self.pivotParameterID      = kwargs.get('pivotParameter', 'Time')
    self.pivotParameterValues  = None                                  # In here we store the values of the pivot parameter (e.g. Time)
    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")
    ##if len(self.target) > 2:
    ##  self.raiseAnError(IOError,"Multi-target ARMA not available yet!")
    # Initialize parameters for Fourier detrending
    if 'Fourier' not in self.initOptionDict.keys():
      self.hasFourierSeries = False
    else:
      self.hasFourierSeries = True
      self.fourierPara = {}
      self.fourierPara['basePeriod'] = [float(temp) for temp in self.initOptionDict['Fourier'].split(',')]
      self.fourierPara['FourierOrder'] = {}
      if 'FourierOrder' not in self.initOptionDict.keys():
        for basePeriod in self.fourierPara['basePeriod']:
          self.fourierPara['FourierOrder'][basePeriod] = 4
      else:
        temps = self.initOptionDict['FourierOrder'].split(',')
        for index, basePeriod in enumerate(self.fourierPara['basePeriod']):
          self.fourierPara['FourierOrder'][basePeriod] = int(temps[index])
      if len(self.fourierPara['basePeriod']) != len(self.fourierPara['FourierOrder']):
        self.raiseAnError(ValueError, 'Length of FourierOrder should be ' + str(len(self.fourierPara['basePeriod'])))

  def __getstate__(self):
    """
      Obtains state of object for pickling.
      @ In, None
      @ Out, d, dict, stateful dictionary
    """
    d = copy.copy(self.__dict__)
    # set up a seed for the next pickled iteration
    if self.reseedCopies:
      rand = randomUtils.randomIntegers(1,int(2**20),self)
      d['random seed'] = rand
    return d

  def __setstate__(self,d):
    """
      Sets state of object from pickling.
      @ In, d, dict, stateful dictionary
      @ Out, None
    """
    seed = d.pop('random seed',None)
    if seed is not None:
      self.reseed(seed)
    self.__dict__ = d

  def _localNormalizeData(self,values,names,feat): # This function is not used in this class and can be removed
    """
      Overwrites default normalization procedure.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.

      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
    """
    self.raiseADebug('Training...')
    # obtain pivot parameter
    self.raiseADebug('... gathering pivot values ...')
    self.pivotParameterValues = targetVals[:,:,self.target.index(self.pivotParameterID)]
    if len(self.pivotParameterValues) > 1:
      self.raiseAnError(Exception,self.printTag +" does not handle multiple histories data yet! # histories: "+str(len(self.pivotParameterValues)))
    self.pivotParameterValues.shape = (self.pivotParameterValues.size,)
    targetVals = np.delete(targetVals,self.target.index(self.pivotParameterID),2)[0]
    # targetVals now has shape (1, # time samples, # targets)
    self.target.pop(self.target.index(self.pivotParameterID))
    # XXX WORKING
    for t,target in enumerate(self.target):
      self.raiseADebug('... training target "{}" ...'.format(target))
      timeSeriesData = copy.deepcopy(targetVals[:,t])
      # set up the Arma parameters dict for this target, including the noisy data
      #self.armaPara[target] = {'rSeries': timeSeriesData}
      # if we're removing Fourier signal, do that now.
      if self.hasFourierSeries:
        self.raiseADebug('... ... analyzing Fourier signal ...')
        self.fourierResults[target] = self._trainFourier(self.pivotParameterValues,
                                                         self.fourierPara['basePeriod'],
                                                         self.fourierPara['FourierOrder'],
                                                         timeSeriesData)
        #self.armaPara[target]['rSeries'] -= self.fourierResults[target]['predict']}
        timeSeriesData -= self.fourierResults[target]['predict']
      # Transform data to obatain normal distrbuted series. See
      # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
      # Applied Energy, 87(2010) 843-855
      self.raiseADebug('... ... analyzing ARMA properties ...')
      self.armaNormPara[target] = self._generateCDF(timeSeriesData)
      self.armaPara[target] = {'rSeries': timeSeriesData,
                               'rSeriesNorm': self._dataConversion(timeSeriesData, target, 'normalize')}
      self._trainARMA(target)
      self.raiseADebug('... ... finished training target "{}"'.format(target))

  def _trainFourier(self, pivotValues, basePeriod, order, values):
    """
      Perform fitting of Fourier series on self.timeSeriesDatabase
      @ In, pivotValues, np.array, list of values for the independent variable (e.g. time)
      @ In, basePeriod, list, list of the base periods
      @ In, order, dict, Fourier orders to extract for each base period
      @ In, values, np.array, list of values for the dependent variable (signal to take fourier from)
      @ Out, fourierResult, dict, results of this training in keys 'residues', 'fOrder', 'predict'
    """
    fourierSeriesAll = self._generateFourierSignal(pivotValues,
                                                   basePeriod,
                                                   order)
    fourierEngine = linear_model.LinearRegression()

    # get the combinations of fourier signal orders to consider
    temp = {}
    for bp in self.fourierPara['FourierOrder'].keys():
      temp[bp] = range(1,order[bp]+1)
    fourOrders = list(itertools.product(*temp.values())) # generate the set of combinations of the Fourier order

    criterionBest = np.inf
    fSeriesBest = []
    fourierResult={}
    fourierResult['residues'] = 0
    fourierResult['fOrder'] = []

    for fOrder in fourOrders:
      fSeries = np.zeros(shape=(pivotValues.size,2*sum(fOrder)))
      indexTemp = 0
      for index,bp in enumerate(order.keys()):
        fSeries[:,indexTemp:indexTemp+fOrder[index]*2] = fourierSeriesAll[bp][:,0:fOrder[index]*2]
        indexTemp += fOrder[index]*2
      fourierEngine.fit(fSeries,values)
      r = (fourierEngine.predict(fSeries)-values)**2
      if r.size > 1:
        r = sum(r)
      r = r/pivotValues.size
      criterionCurrent = copy.copy(r)
      if  criterionCurrent< criterionBest:
        fourierResult['fOrder'] = copy.deepcopy(fOrder)
        fSeriesBest = copy.deepcopy(fSeries)
        fourierResult['residues'] = copy.deepcopy(r)
        criterionBest = copy.deepcopy(criterionCurrent)

    fourierEngine.fit(fSeriesBest,values)
    fourierResult['predict'] = np.asarray(fourierEngine.predict(fSeriesBest))
    return fourierResult

  def _trainARMA(self,target):
    """
      Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
      Data series to this function has been normalized so that it is standard gaussian
      @ In, target, string, name of target to train ARMA for
      @ Out, None, but sets values in self.armaResult[target]
    """
    self.armaResult[target] = {}
    Pmax = self.armaPara['Pmax']
    Pmin = self.armaPara['Pmin']
    Qmax = self.armaPara['Qmax']
    Qmin = self.armaPara['Qmin']
    dim = self.armaPara['dimension']

    criterionBest = np.inf
    self.raiseADebug('... ... searching optimal ARMA parameters ...')
    for p in range(Pmin,Pmax+1):
      for q in range(Qmin,Qmax+1):
        if p is 0 and q is 0:
          continue          # dump case so we pass
        init = [0.0]*(p+q)*dim**2
        init_S = np.identity(dim)
        for n1 in range(dim):
          init.append(init_S[n1,n1])

        rOpt = {}
        rOpt = optimize.fmin(self._computeARMALikelihood,
                             init,
                             args=(target, p, q, self.armaPara[target]['rSeriesNorm']),
                             full_output = True)
        numParam = (p+q)*dim**2/self.pivotParameterValues.size
        criterionCurrent = self._computeAICorBIC(self.armaResult[target]['sigHat'],
                                                 numParam,
                                                 'BIC',
                                                 self.pivotParameterValues.size,
                                                 obj='min')
        if criterionCurrent < criterionBest or 'P' not in self.armaResult[target].keys():
          # to save the first iteration results
          self.armaResult[target]['P'] = p
          self.armaResult[target]['Q'] = q
          self.armaResult[target]['param'] = rOpt[0]
          criterionBest = criterionCurrent

    # saving training results
    Phi, Theta, Cov = self._armaParamAssemb(self.armaResult[target]['param'],
                                            self.armaResult[target]['P'],
                                            self.armaResult[target]['Q'],dim )
    self.armaResult[target]['Phi'] = Phi
    self.armaResult[target]['Theta'] = Theta
    # TODO this for loop shouldn't be necessary to set a vector, does it need a transpose?
    self.armaResult[target]['sig'] = np.zeros(shape=(1, dim ))
    for n in range(dim ):
      self.armaResult[target]['sig'][0,n] = np.sqrt(Cov[n,n])

  def _generateCDF(self, data):
    """
      Generate empirical CDF function of the input data, and save the results in self
      @ In, data, array, shape = [n_timeSteps, n_dimension], data over which the CDF will be generated
      @ Out, armaNormPara, dict, description of normal parameters by pivot value
    """
    armaNormPara = {}
    armaNormPara['resCDF'] = {}

    if len(data.shape) == 1:
      data = np.reshape(data, newshape = (data.shape[0],1))
    num_bins = [0]*data.shape[1] # initialize num_bins, which will be calculated later by Freedman Diacoins rule

    for d in range(data.shape[1]):
      num_bins[d] = self._computeNumberBins(data[:,d])
      counts, binEdges = np.histogram(data[:,d], bins = num_bins[d], normed = True)
      Delta = np.zeros(shape=(num_bins[d],1))
      for n in range(num_bins[d]):
        Delta[n,0] = binEdges[n+1]-binEdges[n]
      temp = np.cumsum(counts)*np.average(Delta)
      cdf = np.insert(temp, 0, temp[0]) # minimum of CDF is set to temp[0] instead of 0 to avoid numerical issues
      armaNormPara['resCDF'][d] = {}
      armaNormPara['resCDF'][d]['bins'] = copy.deepcopy(binEdges)
      armaNormPara['resCDF'][d]['binsMax'] = max(binEdges)
      armaNormPara['resCDF'][d]['binsMin'] = min(binEdges)
      armaNormPara['resCDF'][d]['CDF'] = copy.deepcopy(cdf)
      armaNormPara['resCDF'][d]['CDFMax'] = max(cdf)
      armaNormPara['resCDF'][d]['CDFMin'] = min(cdf)
      armaNormPara['resCDF'][d]['binSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[b] for b in binEdges])
      armaNormPara['resCDF'][d]['cdfSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[c] for c in cdf])
    return armaNormPara

  def _computeNumberBins(self, data):
    """
      Compute number of bins determined by Freedman Diaconis rule
      https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
      @ In, data, array, shape = [n_sample], data over which the number of bins is decided
      @ Out, numBin, int, number of bins determined by Freedman Diaconis rule
    """
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    if IQR <= 0.0:
      self.raiseAnError(RuntimeError,"IQR is <= zero. Percentile 75% and Percentile 25% are the same: "+str(np.percentile(data, 25)))
    binSize = 2.0*IQR*(data.size**(-1.0/3.0))
    numBin = int((max(data)-min(data))/binSize)
    return numBin

  def _getCDF(self,target, d, x):
    """
      Get residue CDF value at point x for d-th dimension
      @ In, target, str, target whose CDF we are getting
      @ In, d, int, dimension id
      @ In, x, float, variable value for which the CDF is computed
      @ Out, y, float, CDF value
    """
    if x <= self.armaNormPara[target]['resCDF'][d]['binsMin']:
      y = self.armaNormPara[target]['resCDF'][d]['CDF'][0]
    elif x >= self.armaNormPara[target]['resCDF'][d]['binsMax']:
      y = self.armaNormPara[target]['resCDF'][d]['CDF'][-1]
    else:
      ind = self.armaNormPara[target]['resCDF'][d]['binSearchEng'].kneighbors(x, return_distance=False)
      X, Y = self.armaNormPara[target]['resCDF'][d]['bins'][ind], self.armaNormPara[target]['resCDF'][d]['CDF'][ind]
      if X[0,0] <= X[0,1]:
        x1, x2, y1, y2 = X[0,0], X[0,1], Y[0,0], Y[0,1]
      else:
        x1, x2, y1, y2 = X[0,1], X[0,0], Y[0,1], Y[0,0]
      if x1 == x2:
        y = (y1+y2)/2.0
      else:
        y = y1 + 1.0*(y2-y1)/(x2-x1)*(x-x1)
    return y

  def _getInvCDF(self,target,d,x):
    """
      Get inverse residue CDF at point x for d-th dimension
      @ In, target, str, target whose inverse CDF we are getting
      @ In, d, int, dimension id
      @ In, x, float, the CDF value for which the inverse value is computed
      @ Out, y, float, variable value
    """
    # XXX needs "target"
    if x < 0 or x > 1:
      self.raiseAnError(ValueError, 'Input to __getRInvCDF__ is not in unit interval' )
    elif x <= self.armaNormPara[target]['resCDF'][d]['CDFMin']:
      y = self.armaNormPara[target]['resCDF'][d]['bins'][0]
    elif x >= self.armaNormPara[target]['resCDF'][d]['CDFMax']:
      y = self.armaNormPara[target]['resCDF'][d]['bins'][-1]
    else:
      ind = self.armaNormPara[target]['resCDF'][d]['cdfSearchEng'].kneighbors(x, return_distance=False)
      X = self.armaNormPara[target]['resCDF'][d]['CDF'][ind]
      Y = self.armaNormPara[target]['resCDF'][d]['bins'][ind]
      if X[0,0] <= X[0,1]:
        x1, x2, y1, y2 = X[0,0], X[0,1], Y[0,0], Y[0,1]
      else:
        x1, x2, y1, y2 = X[0,1], X[0,0], Y[0,1], Y[0,0]
      if x1 == x2:
        y = (y1+y2)/2.0
      else:
        y = y1 + 1.0*(y2-y1)/(x2-x1)*(x-x1)
    return y

  def _dataConversion(self, data, target, obj):
    """
      Transform input data to a Normal/empirical distribution data set.
      @ In, data, array, shape=[n_timeStep, n_dimension], input data to be transformed
      @ In, target, str, name of variable that data represents
      @ In, obj, string, specify whether to normalize or denormalize the data
      @ Out, transformedData, array, shape = [n_timeStep, n_dimension], output transformed data that has normal/empirical distribution
    """
    # Instantiate a normal distribution for data conversion
    normTransEngine = Distributions.returnInstance('Normal',self)
    normTransEngine.mean, normTransEngine.sigma = 0, 1
    normTransEngine.upperBoundUsed, normTransEngine.lowerBoundUsed = False, False
    normTransEngine.initializeDistribution()

    if len(data.shape) == 1:
      data = np.reshape(data, newshape = (data.shape[0],1))
    # Transform data
    transformedData = np.zeros(shape=data.shape)
    for n1 in range(data.shape[0]):
      for n2 in range(data.shape[1]):
        if obj in ['normalize']:
          temp = self._getCDF(target, n2, data[n1,n2])
          # for numerical issues, value less than 1 returned by _getCDF can be greater than 1 when stored in temp
          # This might be a numerical issue of dependent library.
          # It seems gone now. Need further investigation.
          if temp >= 1:
            temp = 1 - np.finfo(float).eps
          elif temp <= 0:
            temp = np.finfo(float).eps
          transformedData[n1,n2] = normTransEngine.ppf(temp)
        elif obj in ['denormalize']:
          temp = normTransEngine.cdf(data[n1, n2])
          transformedData[n1,n2] = self._getInvCDF(target,n2, temp)
        else:
          self.raiseAnError(ValueError, 'Input obj to _dataConversion is not properly set')
    return transformedData

  def _generateFourierSignal(self, Time, basePeriod, fourierOrder):
    """
      Generate fourier signal as specified by the input file
      @ In, basePeriod, list, list of base periods
      @ In, fourierOrder, dict, order for each base period
      @ Out, fourierSeriesAll, array, shape = [n_timeStep, n_basePeriod]
    """
    fourierSeriesAll = {}
    for bp in basePeriod:
      fourierSeriesAll[bp] = np.zeros(shape=(Time.size, 2*fourierOrder[bp]))
      for orderBp in range(fourierOrder[bp]):
        fourierSeriesAll[bp][:, 2*orderBp] = np.sin(2*np.pi*(orderBp+1)/bp*Time)
        fourierSeriesAll[bp][:, 2*orderBp+1] = np.cos(2*np.pi*(orderBp+1)/bp*Time)
    return fourierSeriesAll

  def _armaParamAssemb(self,x,p,q,N):
    """
      Assemble ARMA parameter into matrices
      @ In, x, list, ARMA parameter stored as vector
      @ In, p, int, AR order
      @ In, q, int, MA order
      @ In, N, int, dimensionality of x
      @ Out Phi, list, list of Phi parameters (each as an array) for each AR order
      @ Out Theta, list, list of Theta parameters (each as an array) for each MA order
      @ Out Cov, array, covariance matrix of the noise
    """
    Phi = {}
    Theta = {}
    Cov = np.identity(N)
    # TODO storing these in dictionaries indexed on integers causes slow lookups.
    #   This could be moved to numpy arrays and have the same functionality.
    for i in range(1,p+1):
      Phi[i] = np.zeros(shape=(N,N))
      for n in range(N):
        Phi[i][n,:] = x[N**2*(i-1)+n*N:N**2*(i-1)+(n+1)*N]
    for j in range(1,q+1):
      Theta[j] = np.zeros(shape=(N,N))
      for n in range(N):
        Theta[j][n,:] = x[N**2*(p+j-1)+n*N:N**2*(p+j-1)+(n+1)*N]
    for n in range(N):
      Cov[n,n] = x[N**2*(p+q)+n]
    return Phi, Theta, Cov

  def _computeARMALikelihood(self, x, target, p, q, whiteData):
    """
      Compute the likelihood given an ARMA model
      @ In, x, list, ARMA parameter stored as vector
      @ In, target, str, variable name for target of ARMA
      @ In, p, int, "p" parameter for ARMA, or the number of lags in the autoregressor
      @ In, q, int, "q" parameter for ARMA, or the number of lags in the moving average
      @ In, whiteData, np.array(float), target signal data (normalized to Gaussian)
      @ Out, lkHood, float, output likelihood
    """
    N = self.armaPara['dimension']
    assert(len(x) == N**2*(p+q)+N)
    Phi, Theta, Cov = self._armaParamAssemb(x,p,q,N)
    for n1 in range(N):
      for n2 in range(N):
        if Cov[n1,n2] < 0:
          lkHood = sys.float_info.max
          return lkHood

    CovInv = np.linalg.inv(Cov)
    d = whiteData
    numTimeStep = d.shape[0]
    alpha = np.zeros(shape=d.shape)
    L = -N*numTimeStep/2.0*np.log(2*np.pi) - numTimeStep/2.0*np.log(np.linalg.det(Cov))
    # TODO can this be vectorized further?
    for t in range(numTimeStep):
      alpha[t,:] = d[t,:]
      for i in range(1,min(p,t)+1):
        alpha[t,:] -= np.dot(Phi[i],d[t-i,:])
      for j in range(1,min(q,t)+1):
        alpha[t,:] -= np.dot(Theta[j],alpha[t-j,:])
      L -= 1/2.0*np.dot(np.dot(alpha[t,:].T,CovInv),alpha[t,:])

    sigHat = np.dot(alpha.T,alpha)
    while sigHat.size > 1:
      sigHat = sum(sigHat)
      sigHat = sum(sigHat.T)
    sigHat = sigHat / numTimeStep
    self.armaResult[target]['sigHat'] = sigHat[0,0]
    lkHood = -L
    return lkHood

  def _computeAICorBIC(self,maxL,numParam,cType,histSize,obj='max'):
    """
      Compute the AIC or BIC criteria for model selection.
      @ In, maxL, float, likelihood of given parameters
      @ In, numParam, int, number of parameters
      @ In, cType, string, specify whether AIC or BIC should be returned
      @ In, obj, string, specify the optimization is for maximum or minimum.
      @ In, histSize, int, length of pivot parameters vector
      @ Out, criterionValue, float, value of AIC/BIC
    """
    if obj == 'min':
      flag = -1
    else:
      flag = 1
    if cType == 'BIC':
      criterionValue = -1*flag*np.log(maxL)+numParam*np.log(histSize)
    elif cType == 'AIC':
      criterionValue = -1*flag*np.log(maxL)+numParam*2
    else:
      criterionValue = maxL
    return criterionValue

  def __evaluateLocal__(self,featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    if featureVals.size > 1:
      self.raiseAnError(ValueError, 'The input feature for ARMA for evaluation cannot have size greater than 1. ')

    # Instantiate a normal distribution for time series synthesis (noise part)
    normEvaluateEngine = Distributions.returnInstance('Normal',self)
    normEvaluateEngine.mean, normEvaluateEngine.sigma = 0, 1
    normEvaluateEngine.upperBoundUsed, normEvaluateEngine.lowerBoundUsed = False, False
    normEvaluateEngine.initializeDistribution()

    numTimeStep = len(self.pivotParameterValues)
    # make sure pivot value is in return object
    returnEvaluation = {self.pivotParameterID:self.pivotParameterValues[0:numTimeStep]}

    for tIdx,target in enumerate(self.target):
      # generate normalized noise for data
      ## initialize normalized noise
      tSeriesNoise = np.zeros(shape=self.armaPara[target]['rSeriesNorm'].shape)
      ### TODO This could probably be vectorized for speed gains
      for t in range(numTimeStep):
        for n in range(self.armaPara['dimension']):
          tSeriesNoise[t,n] = normEvaluateEngine.rvs()*self.armaResult[target]['sig'][0,n]
      ## add ARMA data to noise
      ### TODO This too
      tSeriesNorm = np.zeros(shape=(numTimeStep,self.armaPara[target]['rSeriesNorm'].shape[1]))
      tSeriesNorm[0,:] = self.armaPara[target]['rSeriesNorm'][0,:]
      for t in range(numTimeStep):
        for i in range(1,min(self.armaResult[target]['P'], t)+1):
          tSeriesNorm[t,:] += np.dot(tSeriesNorm[t-i,:], self.armaResult[target]['Phi'][i])
        for j in range(1,min(self.armaResult[target]['Q'], t)+1):
          tSeriesNorm[t,:] += np.dot(tSeriesNoise[t-j,:], self.armaResult[target]['Theta'][j])
        tSeriesNorm[t,:] += tSeriesNoise[t,:]

      # Convert data back to empirically distributed (not normalized)
      tSeries = self._dataConversion(tSeriesNorm, target, 'denormalize')

      # Add fourier trends
      if self.hasFourierSeries:
        predict = self.fourierResults[target]['predict']
        self.raiseADebug('Fourier prediciton, series shapes:',predict.shape, tSeries.shape)
        if len(predict.shape) == 1:
          fourierSignal = np.reshape(predict, newshape=(predict.shape[0],1))
        else:
          fourierSignal = predict[0:numTimeStep,:]
        tSeries += fourierSignal

      # Ensure positivity --- FIXME -> what needs to be fixed?
      if self.outTruncation is not None:
        if self.outTruncation == 'positive':
          tSeries = np.absolute(tSeries)
        elif self.outTruncation == 'negative':
          tSeries = -np.absolute(tSeries)

      # store results
      ## XXX assure that tIdx is the right way to find the correct featureVal scalar
      evaluation = (tSeries*featureVals).reshape(returnEvaluation[self.pivotParameterID].shape)
      assert(evaluation.size == returnEvaluation[self.pivotParameterID].size)
      returnEvaluation[target] = evaluation
    # END for target in targets

    return returnEvaluation

  def __confidenceLocal__(self,featureVals):
    """
      This method is currently not needed for ARMA
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      Currently not implemented for ARMA
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      there are no possible default parameters to report
    """
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    """
      override this method to pass the set of parameters of the ROM that can change during simulation
      Currently not implemented for ARMA
    """
    pass

  def reseed(self,seed):
    """
      Used to set the underlying random seed.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    randomUtils.randomSeed(seed)

