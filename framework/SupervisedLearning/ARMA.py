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
  ### INHERITED METHODS ###
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
    self.trainingData      = {} # holds normalized ('norm') and original ('raw') training data, by target
    self.cdfData           = {} # dictionary of fitted CDF parameters, by target
    self.fourierResults    = {} # dictionary of Fourier results, by target
    self.armaResult        = {} # dictionary of assorted useful arma information, by target
    self.Pmax              = kwargs.get('Pmax', 3) # bounds for autoregressive lag
    self.Pmin              = kwargs.get('Pmin', 0)
    self.Qmax              = kwargs.get('Qmax', 3) # bounds for moving average lag
    self.Qmin              = kwargs.get('Qmin', 0)
    self.reseedCopies      = kwargs.get('reseedCopies',True)
    self.outTruncation     = kwargs.get('outTruncation', None) # Additional parameters to allow user to specify the time series to be all positive or all negative
    self.pivotParameterID  = kwargs.get('pivotParameter', 'Time')
    self.pivotParameterValues = None  # In here we store the values of the pivot parameter (e.g. Time)

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
      # NOTE: someday, this ARMA could be expanded to take Fourier signals in time on the TypicalHistory,
      #   and then use several realizations of the target to train an ND ARMA that captures not only
      #   the mean and variance in time, but the mean, variance, skewness, and kurtosis over time and realizations.
      #   In this way, outliers in the training data could be captured with significantly more representation.
      timeSeriesData = targetVals[:,t]
      if len(timeSeriesData.shape) != 1:
        self.raiseAnError(IOError,'The ARMA only can be trained on a single history realization!  Was given shape {}.'
                                  .format(len(timeSeriesData.shape)))
      self.raiseADebug('... training target "{}" ...'.format(target))
      #timeSeriesData.reshape(len(self.pivotParameterValues))
      # if we're removing Fourier signal, do that now.
      if self.hasFourierSeries:
        self.raiseADebug('... ... analyzing Fourier signal ...')
        self.fourierResults[target] = self._trainFourier(self.pivotParameterValues,
                                                         self.fourierPara['basePeriod'],
                                                         self.fourierPara['FourierOrder'],
                                                         timeSeriesData)
        timeSeriesData -= self.fourierResults[target]['predict']
      # Transform data to obatain normal distrbuted series. See
      # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
      # Applied Energy, 87(2010) 843-855
      self.raiseADebug('... ... analyzing ARMA properties ...')
      ## store de-Fourier training data
      #self.trainingData[target] = {'raw': timeSeriesData} #,
      #                             #'norm': self._dataConversion(timeSeriesData, target, 'normalize')}
      # finish training the ARMA
      self.armaResult[target] = self._trainARMA(timeSeriesData)
      self.raiseADebug('... ... finished training target "{}"'.format(target))

  def __evaluateLocal__(self,featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    if featureVals.size > 1:
      self.raiseAnError(ValueError, 'The input feature for ARMA for evaluation cannot have size greater than 1. ')

    # Instantiate a normal distribution for time series synthesis (noise part)
    # TODO USE THIS, but first retrofix rvs on norm to take "size=") for number of results
    normEvaluateEngine = None
    #normEvaluateEngine = Distributions.returnInstance('Normal',self)
    #normEvaluateEngine.mean, normEvaluateEngine.sigma = 0, 1
    #normEvaluateEngine.upperBoundUsed, normEvaluateEngine.lowerBoundUsed = False, False
    #normEvaluateEngine.initializeDistribution()

    numTimeStep = len(self.pivotParameterValues)
    # make sure pivot value is in return object
    returnEvaluation = {self.pivotParameterID:self.pivotParameterValues[0:numTimeStep]}

    # TODO when we have output printing for ROMs, the distinct signals here could be outputs!
    for tIdx,target in enumerate(self.target):
      result = self.armaResult[target] # ARMAResults object

      # generate baseline ARMA + noise
      signal = self._generateARMASignal(result, numSamples=numTimeStep, randEngine=normEvaluateEngine)

      # Add fourier trends
      if self.hasFourierSeries:
        signal += self.fourierResults[target]['predict']

      # Ensure positivity
      if self.outTruncation is not None:
        if self.outTruncation == 'positive':
          signal = np.absolute(signal)
        elif self.outTruncation == 'negative':
          signal = -np.absolute(signal)

      # store results
      ## FIXME this is ASSUMING the input to ARMA is only ever a single scaling factor.
      signal *= featureVals[0]
      # sanity check on the signal
      assert(signal.size == returnEvaluation[self.pivotParameterID].size)
      returnEvaluation[target] = signal
    # END for target in targets
    return returnEvaluation

  def reseed(self,seed):
    """
      Used to set the underlying random seed.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    randomUtils.randomSeed(seed)

  ### UTILITY METHODS ###
  def _trainARMA(self,data):
    """
      Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
      @ In, data, np.array(float), data on which to train
      @ Out, results, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA
    """
    # input parameters
    # XXX change input parameters to just p,q
    # TODO option to optimize for best p,q?
    Pmax = self.Pmax
    Qmax = self.Qmax
    return smARMA(data, order=(Pmax,Qmax)).fit(disp=False)

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
    temp = [range(1,order[bp]+1) for bp in order]
    fourOrders = list(itertools.product(*temp)) # generate the set of combinations of the Fourier order

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

  def _generateARMASignal(self, model, numSamples=None, randEngine=None):
    """
      Generates a synthetic history from fitted parameters.
      @ In, model, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA such as otained from _trainARMA
      @ In, numSamples, int, optional, number of samples to take (default to pivotParameters length)
      @ Out, hist, np.array(float), synthetic ARMA signal
    """
    if numSamples is None:
      numSamples =  len(self.pivotParameterValues)
    if randEngine is not None:
      # if in debug mode, check to make sure the provided RVS engine can take the correct arguments
      ## this is a weak check, but better than failing in the statsmodels.tsa.arima_process code
      assert(randEngine.rvs(2))
      assert(randEngine.rvs(size=2))
    hist = sm.tsa.arma_generate_sample(ar = np.append(1., -model.arparams),
                                       ma = np.append(1., model.maparams),
                                       nsample = numSamples,
                                       #distrvs = randEngine,
                                       sigma = np.sqrt(model.sigma2))
    return hist

  def _generateFourierSignal(self, pivots, basePeriod, fourierOrder):
    """
      Generate fourier signal as specified by the input file
      @ In, pivots, np.array, pivot values (e.g. time)
      @ In, basePeriod, list, list of base periods
      @ In, fourierOrder, dict, order for each base period
      @ Out, fourier, array, shape = [n_timeStep, n_basePeriod]
    """
    fourier = {}
    for base in basePeriod:
      fourier[base] = np.zeros((pivots.size, 2*fourierOrder[base]))
      for orderBp in range(fourierOrder[base]):
        fourier[base][:, 2*orderBp] = np.sin(2*np.pi*(orderBp+1)/base*pivots)
        fourier[base][:, 2*orderBp+1] = np.cos(2*np.pi*(orderBp+1)/base*pivots)
    return fourier

  ### ESSENTIALLY UNUSED ###
  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure, since we do not desire normalization in this implementation.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

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

