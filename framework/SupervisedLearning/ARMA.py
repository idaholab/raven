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
import statsmodels.api as sm # VARMAX is in sm.tsa
from statsmodels.tsa.arima_model import ARMA as smARMA
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
    self.cdfData           = {} # dictionary of fitted CDF parameters, by target
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

    # can only handle one scaling input currently
    if len(self.features) != 1:
      self.raiseAnError(IOError,"The ARMA can only currently handle a single feature, which scales the outputs!")

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
      self.armaResult[target] = self._trainARMA(target)
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
      @ Out, results, statsmodels.tsa.arima_model.ARMAResults, trained ARMA
    """
    # XXX WORKING
    # input parameters
    Pmax = self.armaPara['Pmax']
    Pmin = self.armaPara['Pmin']
    Qmax = self.armaPara['Qmax']
    Qmin = self.armaPara['Qmin']

    # scrub ARMA to make it gaussian aka normalized aka white
    ## first, charaacterize the CDF of the data
    self.cdfData[target] = self._generateCDF(self.armaPara[target]['rSeries'])

    # train white model
    model = smARMA(self.armaPara[target]['rSeries'], order=(Pmax,Qmax))
    results = model.fit(disp=False)
    return results

  def _generateCDF(self, data):
    """
      Generate empirical CDF function of the input data, and save the results in self
      @ In, data, array, shape = [n_timeSteps, n_dimension], data over which the CDF will be generated
      @ Out, armaNormPara, dict, description of normal parameters by pivot value
    """
    #armaNormPara = {}
    #armaNormPara['resCDF'] = {}

    if len(data.shape) == 1:
      data = np.reshape(data, newshape = (data.shape[0],1))
    numPivots, numSamples = data.shape

    cdfParams = [{}]*numSamples

    for d in range(numSamples):
      num_bins = self._computeNumberBins(data[:,d])
      counts, binEdges = np.histogram(data[:,d], bins = num_bins, normed = True)
      Delta = np.zeros(shape=(num_bins,1))
      for n in range(num_bins):
        Delta[n,0] = binEdges[n+1]-binEdges[n]
      temp = np.cumsum(counts)*np.average(Delta)
      cdf = np.insert(temp, 0, temp[0]) # minimum of CDF is set to temp[0] instead of 0 to avoid numerical issues
      #cdfParams[d] = {}
      cdfParams[d]['bins'] = copy.deepcopy(binEdges)
      cdfParams[d]['binsMax'] = max(binEdges)
      cdfParams[d]['binsMin'] = min(binEdges)
      cdfParams[d]['CDF'] = copy.deepcopy(cdf)
      cdfParams[d]['CDFMax'] = max(cdf)
      cdfParams[d]['CDFMin'] = min(cdf)
      cdfParams[d]['binSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[b] for b in binEdges])
      cdfParams[d]['cdfSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[c] for c in cdf])
    return cdfParams

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

    # XXX DEBUGG
    debugFile = file('signal_bases.csv','w')
    debugFile.writelines('Time,'+','.join([str(x) for x in self.pivotParameterValues])+'\n')
    for tIdx,target in enumerate(self.target):
      result = self.armaResult[target] # ARMAResults object
      covariance = result.cov_params()    # "sigma" in the Chen-Rabiti paper

      # initialize normalized noise
      armaNoise = np.array(normEvaluateEngine.rvs(len(self.armaPara[target]['rSeries'])))*covariance[0,0]
      signal = armaNoise
      debugFile.writelines('signal_noise,'+','.join([str(x) for x in signal])+'\n')

      # generate baseline ARMA
      signal += result.predict()
      debugFile.writelines('signal_arma,'+','.join([str(x) for x in signal])+'\n')

      # Convert data back to empirically distributed (not normalized)
      #signal = armaSignalNorm # FIXME Remove if this works #self._dataConversion(armaSignalNorm, target, 'denormalize')

      # Add fourier trends
      if self.hasFourierSeries:
        fourierSignal = self.fourierResults[target]['predict']
        #if len(fourierSignal.shape) == 1:
        #  fourierSignal = fourierSignal.reshape((fourierSignal.shape[0],1))
        #else:
        # TODO why is this necessary?
        #fourierSignal = fourierSignal[0:numTimeStep,:]
        signal += fourierSignal
        debugFile.writelines('signal_fourier,'+','.join([str(x) for x in signal])+'\n')

      # Ensure positivity --- FIXME -> what needs to be fixed?
      if self.outTruncation is not None:
        if self.outTruncation == 'positive':
          signal = np.absolute(signal)
        elif self.outTruncation == 'negative':
          signal = -np.absolute(signal)

      # store results
      ## FIXME this is ASSUMING the input to ARMA is only ever a single scaling factor.
      signal *= featureVals[0]
      assert(signal.size == returnEvaluation[self.pivotParameterID].size)
      returnEvaluation[target] = signal
      debugFile.writelines('final,'+','.join([str(x) for x in signal])+'\n')
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

