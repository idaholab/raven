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
import numpy as np
from scipy import optimize
import copy
import itertools
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
    self._dynamicHandling  = True                                    # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self.armaPara          = {}
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
    if len(self.target) > 2:
      self.raiseAnError(IOError,"Multi-target ARMA not available yet!")
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
    self.pivotParameterValues = targetVals[:,:,self.target.index(self.pivotParameterID)]
    if len(self.pivotParameterValues) > 1:
      self.raiseAnError(Exception,self.printTag +" does not handle multiple histories data yet! # histories: "+str(len(self.pivotParameterValues)))
    self.pivotParameterValues.shape = (self.pivotParameterValues.size,)
    self.timeSeriesDatabase         = copy.deepcopy(np.delete(targetVals,self.target.index(self.pivotParameterID),2))
    self.timeSeriesDatabase.shape   = (self.timeSeriesDatabase.size,)
    self.target.pop(self.target.index(self.pivotParameterID))
    # Fit fourier seires
    if self.hasFourierSeries:
      self.__trainFourier__()
      self.armaPara['rSeries'] = self.timeSeriesDatabase - self.fourierResult['predict']
    else:
      self.armaPara['rSeries'] = self.timeSeriesDatabase

    # Transform data to obatain normal distrbuted series. See
    # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
    # Applied Energy, 87(2010) 843-855
    self.__generateCDF__(self.armaPara['rSeries'])
    self.armaPara['rSeriesNorm'] = self.__dataConversion__(self.armaPara['rSeries'], obj='normalize')

    self.__trainARMA__() # Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}

    del self.timeSeriesDatabase       # Delete to reduce the pickle size, since from now the original data will no longer be used in the evaluation.

  def __trainFourier__(self):
    """
      Perform fitting of Fourier series on self.timeSeriesDatabase
      @ In, none,
      @ Out, none,
    """
    fourierSeriesAll = self.__generateFourierSignal__(self.pivotParameterValues, self.fourierPara['basePeriod'], self.fourierPara['FourierOrder'])
    fourierEngine = linear_model.LinearRegression()
    temp = {}
    for bp in self.fourierPara['FourierOrder'].keys():
      temp[bp] = range(1,self.fourierPara['FourierOrder'][bp]+1)
    fourOrders = list(itertools.product(*temp.values())) # generate the set of combinations of the Fourier order

    criterionBest = np.inf
    fSeriesBest = []
    self.fourierResult={}
    self.fourierResult['residues'] = 0
    self.fourierResult['fOrder'] = []

    for fOrder in fourOrders:
      fSeries = np.zeros(shape=(self.pivotParameterValues.size,2*sum(fOrder)))
      indexTemp = 0
      for index,bp in enumerate(self.fourierPara['FourierOrder'].keys()):
        fSeries[:,indexTemp:indexTemp+fOrder[index]*2] = fourierSeriesAll[bp][:,0:fOrder[index]*2]
        indexTemp += fOrder[index]*2
      fourierEngine.fit(fSeries,self.timeSeriesDatabase)
      r = (fourierEngine.predict(fSeries)-self.timeSeriesDatabase)**2
      if r.size > 1:
        r = sum(r)
      r = r/self.pivotParameterValues.size
      criterionCurrent = copy.copy(r)
      if  criterionCurrent< criterionBest:
        self.fourierResult['fOrder'] = copy.deepcopy(fOrder)
        fSeriesBest = copy.deepcopy(fSeries)
        self.fourierResult['residues'] = copy.deepcopy(r)
        criterionBest = copy.deepcopy(criterionCurrent)

    fourierEngine.fit(fSeriesBest,self.timeSeriesDatabase)
    self.fourierResult['predict'] = np.asarray(fourierEngine.predict(fSeriesBest))

  def __trainARMA__(self):
    """
      Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
      Data series to this function has been normalized so that it is standard gaussian
      @ In, none,
      @ Out, none,
    """
    self.armaResult = {}
    Pmax = self.armaPara['Pmax']
    Pmin = self.armaPara['Pmin']
    Qmax = self.armaPara['Qmax']
    Qmin = self.armaPara['Qmin']

    criterionBest = np.inf
    for p in range(Pmin,Pmax+1):
      for q in range(Qmin,Qmax+1):
        if p is 0 and q is 0:
          continue          # dump case so we pass
        init = [0.0]*(p+q)*self.armaPara['dimension']**2
        init_S = np.identity(self.armaPara['dimension'])
        for n1 in range(self.armaPara['dimension']):
          init.append(init_S[n1,n1])

        rOpt = {}
        rOpt = optimize.fmin(self.__computeARMALikelihood__,init, args=(p,q) ,full_output = True)
        tmp = (p+q)*self.armaPara['dimension']**2/self.pivotParameterValues.size
        criterionCurrent = self.__computeAICorBIC(self.armaResult['sigHat'],noPara=tmp,cType='BIC',obj='min')
        if criterionCurrent < criterionBest or 'P' not in self.armaResult.keys():
          # to save the first iteration results
          self.armaResult['P'] = p
          self.armaResult['Q'] = q
          self.armaResult['param'] = rOpt[0]
          criterionBest = criterionCurrent

    # saving training results
    Phi, Theta, Cov = self.__armaParamAssemb__(self.armaResult['param'],self.armaResult['P'],self.armaResult['Q'],self.armaPara['dimension'] )
    self.armaResult['Phi'] = Phi
    self.armaResult['Theta'] = Theta
    self.armaResult['sig'] = np.zeros(shape=(1, self.armaPara['dimension'] ))
    for n in range(self.armaPara['dimension'] ):
      self.armaResult['sig'][0,n] = np.sqrt(Cov[n,n])

  def __generateCDF__(self, data):
    """
      Generate empirical CDF function of the input data, and save the results in self
      @ In, data, array, shape = [n_timeSteps, n_dimension], data over which the CDF will be generated
      @ Out, none,
    """
    self.armaNormPara = {}
    self.armaNormPara['resCDF'] = {}

    if len(data.shape) == 1:
      data = np.reshape(data, newshape = (data.shape[0],1))
    num_bins = [0]*data.shape[1] # initialize num_bins, which will be calculated later by Freedman Diacoins rule

    for d in range(data.shape[1]):
      num_bins[d] = self.__computeNumberBins__(data[:,d])
      counts, binEdges = np.histogram(data[:,d], bins = num_bins[d], normed = True)
      Delta = np.zeros(shape=(num_bins[d],1))
      for n in range(num_bins[d]):
        Delta[n,0] = binEdges[n+1]-binEdges[n]
      temp = np.cumsum(counts)*np.average(Delta)
      cdf = np.insert(temp, 0, temp[0]) # minimum of CDF is set to temp[0] instead of 0 to avoid numerical issues
      self.armaNormPara['resCDF'][d] = {}
      self.armaNormPara['resCDF'][d]['bins'] = copy.deepcopy(binEdges)
      self.armaNormPara['resCDF'][d]['binsMax'] = max(binEdges)
      self.armaNormPara['resCDF'][d]['binsMin'] = min(binEdges)
      self.armaNormPara['resCDF'][d]['CDF'] = copy.deepcopy(cdf)
      self.armaNormPara['resCDF'][d]['CDFMax'] = max(cdf)
      self.armaNormPara['resCDF'][d]['CDFMin'] = min(cdf)
      self.armaNormPara['resCDF'][d]['binSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[b] for b in binEdges])
      self.armaNormPara['resCDF'][d]['cdfSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[c] for c in cdf])

  def __computeNumberBins__(self, data):
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

  def __getCDF__(self,d,x):
    """
      Get residue CDF value at point x for d-th dimension
      @ In, d, int, dimension id
      @ In, x, float, variable value for which the CDF is computed
      @ Out, y, float, CDF value
    """
    if x <= self.armaNormPara['resCDF'][d]['binsMin']:
      y = self.armaNormPara['resCDF'][d]['CDF'][0]
    elif x >= self.armaNormPara['resCDF'][d]['binsMax']:
      y = self.armaNormPara['resCDF'][d]['CDF'][-1]
    else:
      ind = self.armaNormPara['resCDF'][d]['binSearchEng'].kneighbors(x, return_distance=False)
      X, Y = self.armaNormPara['resCDF'][d]['bins'][ind], self.armaNormPara['resCDF'][d]['CDF'][ind]
      if X[0,0] <= X[0,1]:
        x1, x2, y1, y2 = X[0,0], X[0,1], Y[0,0], Y[0,1]
      else:
        x1, x2, y1, y2 = X[0,1], X[0,0], Y[0,1], Y[0,0]
      if x1 == x2:
        y = (y1+y2)/2.0
      else:
        y = y1 + 1.0*(y2-y1)/(x2-x1)*(x-x1)
    return y

  def __getInvCDF__(self,d,x):
    """
      Get inverse residue CDF at point x for d-th dimension
      @ In, d, int, dimension id
      @ In, x, float, the CDF value for which the inverse value is computed
      @ Out, y, float, variable value
    """
    if x < 0 or x > 1:
      self.raiseAnError(ValueError, 'Input to __getRInvCDF__ is not in unit interval' )
    elif x <= self.armaNormPara['resCDF'][d]['CDFMin']:
      y = self.armaNormPara['resCDF'][d]['bins'][0]
    elif x >= self.armaNormPara['resCDF'][d]['CDFMax']:
      y = self.armaNormPara['resCDF'][d]['bins'][-1]
    else:
      ind = self.armaNormPara['resCDF'][d]['cdfSearchEng'].kneighbors(x, return_distance=False)
      X, Y = self.armaNormPara['resCDF'][d]['CDF'][ind], self.armaNormPara['resCDF'][d]['bins'][ind]
      if X[0,0] <= X[0,1]:
        x1, x2, y1, y2 = X[0,0], X[0,1], Y[0,0], Y[0,1]
      else:
        x1, x2, y1, y2 = X[0,1], X[0,0], Y[0,1], Y[0,0]
      if x1 == x2:
        y = (y1+y2)/2.0
      else:
        y = y1 + 1.0*(y2-y1)/(x2-x1)*(x-x1)
    return y

  def __dataConversion__(self, data, obj):
    """
      Transform input data to a Normal/empirical distribution data set.
      @ In, data, array, shape=[n_timeStep, n_dimension], input data to be transformed
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
          temp = self.__getCDF__(n2, data[n1,n2])
          # for numerical issues, value less than 1 returned by __getCDF__ can be greater than 1 when stored in temp
          # This might be a numerical issue of dependent library.
          # It seems gone now. Need further investigation.
          if temp >= 1:
            temp = 1 - np.finfo(float).eps
          elif temp <= 0:
            temp = np.finfo(float).eps
          transformedData[n1,n2] = normTransEngine.ppf(temp)
        elif obj in ['denormalize']:
          temp = normTransEngine.cdf(data[n1, n2])
          transformedData[n1,n2] = self.__getInvCDF__(n2, temp)
        else:
          self.raiseAnError(ValueError, 'Input obj to __dataConversion__ is not properly set')
    return transformedData

  def __generateFourierSignal__(self, Time, basePeriod, fourierOrder):
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

  def __armaParamAssemb__(self,x,p,q,N):
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
    Phi, Theta, Cov = {}, {}, np.identity(N)
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

  def __computeARMALikelihood__(self,x,*args):
    """
      Compute the likelihood given an ARMA model
      @ In, x, list, ARMA parameter stored as vector
      @ In, args, dict, additional argument
      @ Out, lkHood, float, output likelihood
    """
    if len(args) != 2:
      self.raiseAnError(ValueError, 'args to __computeARMALikelihood__ should have exactly 2 elements')

    p, q, N = args[0], args[1], self.armaPara['dimension']
    if len(x) != N**2*(p+q)+N:
      self.raiseAnError(ValueError, 'input to __computeARMALikelihood__ has wrong dimension')
    Phi, Theta, Cov = self.__armaParamAssemb__(x,p,q,N)
    for n1 in range(N):
      for n2 in range(N):
        if Cov[n1,n2] < 0:
          lkHood = sys.float_info.max
          return lkHood

    CovInv = np.linalg.inv(Cov)
    d = self.armaPara['rSeriesNorm']
    numTimeStep = d.shape[0]
    alpha = np.zeros(shape=d.shape)
    L = -N*numTimeStep/2.0*np.log(2*np.pi) - numTimeStep/2.0*np.log(np.linalg.det(Cov))
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
    self.armaResult['sigHat'] = sigHat[0,0]
    lkHood = -L
    return lkHood

  def __computeAICorBIC(self,maxL,noPara,cType,obj='max'):
    """
      Compute the AIC or BIC criteria for model selection.
      @ In, maxL, float, likelihood of given parameters
      @ In, noPara, int, number of parameters
      @ In, cType, string, specify whether AIC or BIC should be returned
      @ In, obj, string, specify the optimization is for maximum or minimum.
      @ Out, criterionValue, float, value of AIC/BIC
    """
    if obj == 'min':
      flag = -1
    else:
      flag = 1
    if cType == 'BIC':
      criterionValue = -1*flag*np.log(maxL)+noPara*np.log(self.pivotParameterValues.size)
    elif cType == 'AIC':
      criterionValue = -1*flag*np.log(maxL)+noPara*2
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
    tSeriesNoise = np.zeros(shape=self.armaPara['rSeriesNorm'].shape)
    # TODO This could probably be vectorized for speed gains
    for t in range(numTimeStep):
      for n in range(self.armaPara['dimension']):
        tSeriesNoise[t,n] = normEvaluateEngine.rvs()*self.armaResult['sig'][0,n]

    tSeriesNorm = np.zeros(shape=(numTimeStep,self.armaPara['rSeriesNorm'].shape[1]))
    tSeriesNorm[0,:] = self.armaPara['rSeriesNorm'][0,:]
    for t in range(numTimeStep):
      for i in range(1,min(self.armaResult['P'], t)+1):
        tSeriesNorm[t,:] += np.dot(tSeriesNorm[t-i,:], self.armaResult['Phi'][i])
      for j in range(1,min(self.armaResult['Q'], t)+1):
        tSeriesNorm[t,:] += np.dot(tSeriesNoise[t-j,:], self.armaResult['Theta'][j])
      tSeriesNorm[t,:] += tSeriesNoise[t,:]

    # Convert data back to empirically distributed
    tSeries = self.__dataConversion__(tSeriesNorm, obj='denormalize')
    # Add fourier trends
    self.raiseADebug(self.fourierResult['predict'].shape, tSeries.shape)
    if self.hasFourierSeries:
      if len(self.fourierResult['predict'].shape) == 1:
        tempFour = np.reshape(self.fourierResult['predict'], newshape=(self.fourierResult['predict'].shape[0],1))
      else:
        tempFour = self.fourierResult['predict'][0:numTimeStep,:]
      tSeries += tempFour
    # Ensure positivity --- FIXME
    if self.outTruncation is not None:
      if self.outTruncation == 'positive':
        tSeries = np.absolute(tSeries)
      elif self.outTruncation == 'negative':
        tSeries = -np.absolute(tSeries)
    returnEvaluation = {}
    returnEvaluation[self.pivotParameterID] = self.pivotParameterValues[0:numTimeStep]
    evaluation = tSeries*featureVals
    for index, target in enumerate(self.target):
      returnEvaluation[target] = evaluation[:,index]
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

