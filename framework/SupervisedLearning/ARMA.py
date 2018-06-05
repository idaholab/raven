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
from .SupervisedLearning import supervisedLearning
from sklearn import linear_model, neighbors
#Internal Modules End--------------------------------------------------------------------------------

class ARMA(supervisedLearning):
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
    supervisedLearning.__init__(self,messageHandler,**kwargs)
    self.printTag          = 'ARMA'
    self._dynamicHandling  = True # This ROM is able to manage the time-series on its own.
    self.trainingData      = {} # holds normalized ('norm') and original ('raw') training data, by target
    self.cdfParams         = {} # dictionary of fitted CDF parameters, by target
    self.fourierResults    = {} # dictionary of Fourier results, by target
    self.armaResult        = {} # dictionary of assorted useful arma information, by target
    self.correlations      = [] # list of correlated variables
    self.Pmax              = kwargs.get('Pmax', 3) # bounds for autoregressive lag
    self.Pmin              = kwargs.get('Pmin', 0)
    self.Qmax              = kwargs.get('Qmax', 3) # bounds for moving average lag
    self.Qmin              = kwargs.get('Qmin', 0)
    self.reseedCopies      = kwargs.get('reseedCopies',True)
    self.outTruncation     = kwargs.get('outTruncation', None) # Additional parameters to allow user to specify the time series to be all positive or all negative
    self.pivotParameterID  = kwargs.get('pivotParameter', 'Time')
    self.pivotParameterValues = None  # In here we store the values of the pivot parameter (e.g. Time)
    self.seed              = kwargs.get('seed',None)

    # get seed if provided
    ## FIXME only applies to VARMA sampling right now, since it has to be sampled through Numpy!
    ## see note under "check for correlation" below.
    if self.seed is None:
      self.seed = randomUtils.randomIntegers(0,4294967295,self)
    else:
      self.seed = int(self.seed)

    self.normEngine = Distributions.returnInstance('Normal',self)
    self.normEngine.mean = 0.0
    self.normEngine.sigma = 1.0
    self.normEngine.upperBoundUsed = False
    self.normEngine.lowerBoundUsed = False
    self.normEngine.initializeDistribution()

    # check for correlation
    correlated = kwargs.get('correlate',None)
    if correlated is not None:
      # FIXME set the numpy seed
      ## we have to do this because VARMA.simulate does not accept a random number generator,
      ## but instead uses numpy directly.  As a result, for now, we have to seed numpy.
      ## Because we use our RNG to set the seed, though, it should follow the global seed still.
      self.raiseADebug('Setting Numpy seed to',self.seed)
      np.random.seed(self.seed)
      # store correlated targets
      corVars = [x.strip() for x in correlated.split(',')]
      for var in corVars:
        if var not in self.target:
          self.raiseAnError(IOError,'Variable "{}" requested in "correlate" but not found among the targets!'.format(var))
      # NOTE: someday, this could be expanded to include multiple sets of correlated variables.
      self.correlations = corVars

    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")

    # can only handle one scaling input currently
    if len(self.features) != 1:
      self.raiseAnError(IOError,"The ARMA can only currently handle a single feature, which scales the outputs!")

    # we aren't set up to optimize p and q anymore, so if they're different error out
    if self.Pmin != self.Pmax or self.Qmax != self.Qmin:
      self.raiseAnError(IOError,'ARMA temporarily has optimizing P and Q disabled; please set Pmax and Pmin to '+
          'the same value, and similarly for Q.  If optimizing is desired, please contact us so we can expedite '+
          'the fix.')

    # Initialize parameters for Fourier detrending
    if 'Fourier' not in self.initOptionDict.keys():
      self.hasFourierSeries = False
    else:
      self.hasFourierSeries = True
      self.fourierPara = {}
      basePeriods = self.initOptionDict['Fourier']
      if isinstance(basePeriods,basestring):
        basePeriods = [float(s) for s in basePeriods.split(',')]
      else:
        basePeriods = [float(basePeriods)]
      self.fourierPara['basePeriod'] = basePeriods
      if len(set(self.fourierPara['basePeriod'])) != len(self.fourierPara['basePeriod']):
        self.raiseAnError(IOError,'The same Fourier value was listed multiple times!')
      self.fourierPara['FourierOrder'] = {}
      if 'FourierOrder' not in self.initOptionDict.keys():
        self.fourierPara['basePeriod'] = dict((basePeriod, 4) for basePeriod in self.fourierPara['basePeriod'])
      else:
        orders = self.initOptionDict['FourierOrder']
        if isinstance(orders,str):
          orders = [int(x) for x in orders.split(',')]
        else:
          orders = [orders]
        if len(self.fourierPara['basePeriod']) != len(orders):
          self.raiseAnError(ValueError, 'Number of FourierOrder entries should be "{}"'
                                         .format(len(self.fourierPara['basePeriod'])))
        self.fourierPara['FourierOrder'] = dict((basePeriod, orders[i])
                                               for i,basePeriod in enumerate(self.fourierPara['basePeriod']))

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
    # set VARMA numpy seed
    self.raiseADebug('Setting Numpy seed to',self.seed)
    np.random.seed(self.seed)

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
    # NOTE: someday, this ARMA could be expanded to take Fourier signals in time on the TypicalHistory,
    #   and then use several realizations of the target to train an ND ARMA that captures not only
    #   the mean and variance in time, but the mean, variance, skewness, and kurtosis over time and realizations.
    #   In this way, outliers in the training data could be captured with significantly more representation.
    if len(self.pivotParameterValues) > 1:
      self.raiseAnError(Exception,self.printTag +" does not handle multiple histories data yet! # histories: "+str(len(self.pivotParameterValues)))
    self.pivotParameterValues.shape = (self.pivotParameterValues.size,)
    targetVals = np.delete(targetVals,self.target.index(self.pivotParameterID),2)[0]
    # targetVals now has shape (1, # time samples, # targets)
    self.target.pop(self.target.index(self.pivotParameterID))

    # prep the correlation data structure
    correlationData = np.zeros([len(self.pivotParameterValues),len(self.correlations)])
    for t,target in enumerate(self.target):
      timeSeriesData = targetVals[:,t]
      # if we're removing Fourier signal, do that now.
      self.raiseADebug('... scrubbing the signal for target "{}" ...'.format(target))
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
      self.cdfParams[target] = self._trainCDF(timeSeriesData)
      # normalize data
      normed = self._normalizeThroughCDF(timeSeriesData, self.cdfParams[target])
      # check if this target is part of a correlation set, or standing alone
      if target in self.correlations:
        # store the data and train it separately in a moment
        ## keep data in order of self.correlations
        correlationData[:,self.correlations.index(target)] = normed
      else:
        # go ahead and train it now
        self.raiseADebug('... ... training ...')
        self.armaResult[target] = self._trainARMA(normed)
        self.raiseADebug('... ... finished training target "{}"'.format(target))

    # now handle the training of the correlated armas
    if len(self.correlations):
      self.varmaResult = self._trainVARMA(correlationData)

  def __evaluateLocal__(self,featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    if featureVals.size > 1:
      self.raiseAnError(ValueError, 'The input feature for ARMA for evaluation cannot have size greater than 1. ')

    # Instantiate a normal distribution for time series synthesis (noise part)
    # TODO USE THIS, but first retrofix rvs on norm to take "size=") for number of results

    # make sure pivot value is in return object
    returnEvaluation = {self.pivotParameterID:self.pivotParameterValues}

    # TODO when we have output printing for ROMs, the distinct signals here could be outputs!
    # leaving "debuggFile" as examples of this, in comments
    #debuggFile = open('signal_bases.csv','w')
    #debuggFile.writelines('Time,'+','.join(str(x) for x in self.pivotParameterValues)+'\n')
    correlatedSample = None
    for tIdx,target in enumerate(self.target):
      # random signal
      if target in self.correlations:
        if correlatedSample is None:
          correlatedSample = self._generateVARMASignal(self.varmaResult,
                                                       numSamples = len(self.pivotParameterValues),
                                                       randEngine = self.normEngine.rvs)
        signal = correlatedSample[:,self.correlations.index(target)]
      else:
        result = self.armaResult[target] # ARMAResults object
        # generate baseline ARMA + noise
        signal = self._generateARMASignal(result,
                                          numSamples = len(self.pivotParameterValues),
                                          randEngine = self.normEngine.rvs)

      # denoise
      signal = self._denormalizeThroughCDF(signal,self.cdfParams[target])
      #debuggFile.writelines('signal_arma,'+','.join(str(x) for x in signal)+'\n')

      # Add fourier trends
      if self.hasFourierSeries:
        signal += self.fourierResults[target]['predict']
        #debuggFile.writelines('signal_fourier,'+','.join(str(x) for x in self.fourierResults[target]['predict'])+'\n')

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
      #debuggFile.writelines('final,'+','.join(str(x) for x in signal)+'\n')
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
  def _computeNumberOfBins(self,data):
    """
      Uses the Freedman-Diaconis rule for histogram binning
      @ In, data, np.array, data to bin
      @ Out, n, integer, number of bins
    """
    iqr = np.percentile(data,75) - np.percentile(data,25)
    if iqr <= 0.0:
      self.raiseAnError(ValueError,'While computing CDF, 25 and 75 percentile are the same number!')
    size = 2.0 * iqr / np.cbrt(data.size)
    # tend towards too many bins, not too few
    return int(np.ceil((max(data) - min(data))/size))

  def _denormalizeThroughCDF(self, data, params):
    """
      Normalizes "data" using a Gaussian normal plus CDF of data
      @ In, data, np.array, data to normalize with
      @ In, params, dict, CDF parameters (as obtained by "generateCDF")
      @ Out, normed, np.array, normalized data
    """
    denormed = self.normEngine.cdf(data)
    denormed = self._sampleICDF(denormed, params)
    return denormed

  def _generateARMASignal(self, model, numSamples=None, randEngine=None):
    """
      Generates a synthetic history from fitted parameters.
      @ In, model, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA such as otained from _trainARMA
      @ In, numSamples, int, optional, number of samples to take (default to pivotParameters length)
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ Out, hist, np.array(float), synthetic ARMA signal
    """
    if numSamples is None:
      numSamples =  len(self.pivotParameterValues)
    hist = sm.tsa.arma_generate_sample(ar = np.append(1., -model.arparams),
                                       ma = np.append(1., model.maparams),
                                       nsample = numSamples,
                                       distrvs = randEngine,
                                       sigma = np.sqrt(model.sigma2),
                                       burnin = 2*max(self.Pmax,self.Qmax)) # @epinas, 2018
    return hist

  def _generateVARMASignal(self, model, numSamples=None, randEngine=None):
    """
      Generates a set of correlated synthetic histories from fitted parameters.
      @ In, model, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA such as otained from _trainARMA
      @ In, numSamples, int, optional, number of samples to take (default to pivotParameters length)
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ Out, hist, np.array(float), synthetic ARMA signal
    """
    if numSamples is None:
      numSamples =  len(self.pivotParameterValues)
    # pick an intial by sampling multinormal distribution?
    # FIXME pick a random state; for now starts at 0
    # FIXME using numpy rng!  Figure out how to use ours!  (already queried Cameron and Ross in HPC dept)
    # FIXME for now bypassing initial state, rng, and params update
    obs, states = model.ssm.simulate(numSamples)
    return obs

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

  def _interpolateDist(self,x,y,Xlow,Xhigh,Ylow,Yhigh,inMask):
    """
      Interplotes values for samples "x" to get dependent values "y" given bins
      @ In, x, np.array, sampled points (independent var)
      @ In, y, np.array, sampled points (dependent var)
      @ In, Xlow, np.array, left-nearest neighbor in empirical distribution for each x
      @ In, Xhigh, np.array, right-nearest neighbor in empirical distribution for each x
      @ In, Ylow, np.array, value at left-nearest neighbor in empirical distribution for each x
      @ In, Yhigh, np.array, value at right-nearest neighbor in empirical distribution for each x
      @ In, inMask, np.array, boolean mask in "y" where the distribution values apply
      @ Out, y, np.array, same "y" but with values inserted
    """
    # treat potential divide-by-zeroes specially
    ## mask
    divZero = Xlow == Xhigh
    ## careful when using double masks
    y[[a[divZero] for a in np.where(inMask)]] =  0.5*(Yhigh[divZero] + Ylow[divZero])
    # interpolate all other points as y = low + slope*frac
    ## mask
    okay = np.logical_not(divZero)
    ## empirical CDF change in y, x
    dy = Yhigh[okay] - Ylow[okay]
    dx = Xhigh[okay] - Xlow[okay]
    ## distance from x to low is fraction through dx
    frac = x[inMask][okay] - Xlow[okay]
    ## careful when using double masks
    y[[a[okay] for a in np.where(inMask)]] = Ylow[okay] + dy/dx * frac
    return y

  def _normalizeThroughCDF(self, data, params):
    """
      Normalizes "data" using a Gaussian normal plus CDF of data
      @ In, data, np.array, data to normalize with
      @ In, params, dict, CDF parameters (as obtained by "generateCDF")
      @ Out, normed, np.array, normalized data
    """
    normed = self._sampleCDF(data, params)
    normed = self.normEngine.ppf(normed)
    return normed

  def _sampleCDF(self,x,params):
    """
      Samples the CDF defined in 'params' to get values
      @ In, x, float, value at which to sample inverse CDF
      @ In, params, dict, CDF parameters (as constructed by "_trainCDF")
      @ Out, y, float, value of inverse CDF at x
    """
    # TODO could this be covered by an empirical distribution from Distributions?
    # set up I/O
    x = np.atleast_1d(x)
    y = np.zeros(x.shape)
    # create masks for data outside range (above, below), inside range of empirical CDF
    belowMask = x <= params['bins'][0]
    aboveMask = x >= params['bins'][-1]
    inMask = np.logical_and(np.logical_not(belowMask), np.logical_not(aboveMask))
    # outside CDF set to min, max CDF values
    y[belowMask] = params['cdf'][0]
    y[aboveMask] = params['cdf'][-1]
    # for points in the CDF linearly interpolate between empirical entries
    ## get indices where points should be inserted (gives higher value)
    indices = np.searchsorted(params['bins'],x[inMask])
    Xlow = params['bins'][indices-1]
    Ylow = params['cdf'][indices-1]
    Xhigh = params['bins'][indices]
    Yhigh = params['cdf'][indices]
    y = self._interpolateDist(x,y,Xlow,Xhigh,Ylow,Yhigh,inMask)
    # numerical errors can happen due to not-sharp 0 and 1 in empirical cdf
    ## also, when Crow dist is asked for ppf(1) it returns sys.max (similar for ppf(0))
    y[y >= 1.0] = 1.0 - np.finfo(float).eps
    y[y <= 0.0] = np.finfo(float).eps
    return y

  def _sampleICDF(self,x,params):
    """
      Samples the inverse CDF defined in 'params' to get values
      @ In, x, float, value at which to sample inverse CDF
      @ In, params, dict, CDF parameters (as constructed by "_trainCDF")
      @ Out, y, float, value of inverse CDF at x
   """
    # TODO could this be covered by an empirical distribution from Distributions?
    # set up I/O
    x = np.atleast_1d(x)
    y = np.zeros(x.shape)
    # create masks for data outside range (above, below), inside range of empirical CDF
    belowMask = x <= params['cdf'][0]
    aboveMask = x >= params['cdf'][-1]
    inMask = np.logical_and(np.logical_not(belowMask), np.logical_not(aboveMask))
    # outside CDF set to min, max CDF values
    y[belowMask] = params['bins'][0]
    y[aboveMask] = params['bins'][-1]
    # for points in the CDF linearly interpolate between empirical entries
    ## get indices where points should be inserted (gives higher value)
    indices = np.searchsorted(params['cdf'],x[inMask])
    Xlow = params['cdf'][indices-1]
    Ylow = params['bins'][indices-1]
    Xhigh = params['cdf'][indices]
    Yhigh = params['bins'][indices]
    y = self._interpolateDist(x,y,Xlow,Xhigh,Ylow,Yhigh,inMask)
    return y

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

  def _trainCDF(self,data):
    """
      Constructs a CDF from the given data
      @ In, data, np.array(float), values to fit to
      @ Out, params, dict, essential parameters for CDF
    """
    # caluclate number of bins
    nBins = self._computeNumberOfBins(data)
    # construct histogram
    counts, edges = np.histogram(data, bins = nBins, normed = True)
    # bin widths
    widths = edges[1:] - edges[:-1]
    # numerical CDF
    integrated = np.cumsum(counts)*np.average(widths)
    # set lowest value as first entry,
    ## from Jun implementation, min of CDF set to starting point for numerical issues
    cdf = np.insert(integrated, 0, integrated[0])
    # store parameters
    params = {'bins':edges,
              'cdf':cdf}
              #'binSearch':neighbors.NearestNeighbors(n_neighbors=2).fit([[b] for b in edges]),
              #'cdfSearch':neighbors.NearestNeighbors(n_neighbors=2).fit([[c] for c in cdf])}
    return params

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

  def _trainVARMA(self,data):
    """
      Train correlated ARMA model on white noise ARMA, with Fourier already removed
      @ In, data, np.array(np.array(float)), data on which to train with shape (# pivot values, # targets)
      @ Out, results, statsmodels.tsa.arima_model.ARMAResults, fitted VARMA
    """
    Pmax = self.Pmax
    Qmax = self.Qmax
    model = sm.tsa.VARMAX(endog=data, order=(Pmax,Qmax))
    results = model.fit(disp=False)
    return model

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

