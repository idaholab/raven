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
  Specific ROM implementation for ARMA (Autoregressive Moving Average) ROM
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import collections
import numpy as np
import statsmodels.api as sm # VARMAX is in sm.tsa
import functools
from statsmodels.tsa.arima_model import ARMA as smARMA
from scipy.linalg import solve_discrete_lyapunov
from sklearn import linear_model
from scipy.signal import find_peaks
from scipy.stats import rv_histogram
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import randomUtils, xmlUtils, mathUtils,utils
import Distributions
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------


class ARMA(supervisedLearning):
  r"""
    Autoregressive Moving Average model for time series analysis. First train then evaluate.
    Specify a Fourier node in input file if detrending by Fourier series is needed.

    Time series Y: Y = X + \sum_{i}\sum_k [\delta_ki1*sin(2pi*k/basePeriod_i)+\delta_ki2*cos(2pi*k/basePeriod_i)]
    ARMA series X: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
  """
  ### INHERITED METHODS ###
  def __init__(self, messageHandler, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler: a MessageHandler object in charge of raising errors,
                           and printing messages
      @ In, kwargs: an arbitrary dictionary of keywords and values
    """
    # general infrastructure
    supervisedLearning.__init__(self, messageHandler, **kwargs)
    self.printTag = 'ARMA'
    self._dynamicHandling  = True # This ROM is able to manage the time-series on its own.
    # training storage
    self.trainingData      = {} # holds normalized ('norm') and original ('raw') training data, by target
    self.cdfParams         = {} # dictionary of fitted CDF parameters, by target
    self.armaResult        = {} # dictionary of assorted useful arma information, by target
    self.correlations      = [] # list of correlated variables
    self.fourierResults    = {} # dictionary of Fourier results, by target
    # training parameters
    self.fourierParams     = {} # dict of Fourier training params, by target (if requested, otherwise not present)
    self.P                 = kwargs.get('P', 3) # autoregressive lag
    self.Q                 = kwargs.get('Q', 3) # moving average lag
    self.segments          = kwargs.get('segments', 1)
    # data manipulation
    reseed=kwargs.get('reseedCopies',str(True)).lower()
    self.reseedCopies      = reseed not in utils.stringsThatMeanFalse()
    self.outTruncation = {'positive':set(),'negative':set()} # store truncation requests
    self.pivotParameterID  = kwargs['pivotParameter']
    self.pivotParameterValues = None  # In here we store the values of the pivot parameter (e.g. Time)
    self.seed              = kwargs.get('seed',None)
    self.preserveInputCDF  = kwargs.get('preserveInputCDF', False) # if True, then CDF of the training data will be imposed on the final sampled signal
    self._trainingCDF      = {} # if preserveInputCDF, these CDFs are scipy.stats.rv_histogram objects for the training data
    self.zeroFilterTarget  = None # target for whom zeros should be filtered out
    self.zeroFilterTol     = None # tolerance for zerofiltering to be considered zero, set below
    self.zeroFilterMask    = None # mask of places where zftarget is zero, or None if unused
    self.notZeroFilterMask = None # mask of places where zftarget is NOT zero, or None if unused
    self._minBins          = 20   # min number of bins to use in determining distributions, eventually can be user option, for now developer's pick
    #peaks
    self.peaks             = {} # dictionary of peaks information, by target
    # signal storage
    self._signalStorage    = collections.defaultdict(dict) # various signals obtained in the training process

    # check zeroFilterTarget is one of the targets given
    if self.zeroFilterTarget is not None and self.zeroFilterTarget not in self.target:
      self.raiseAnError('Requested ZeroFilter on "{}" but this target was not found among the ROM targets!'.format(self.zeroFilterTarget))

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

    self.setEngine(randomUtils.newRNG(),seed=self.seed,count=0)

    # FIXME set the numpy seed
      ## we have to do this because VARMA.simulate does not accept a random number generator,
      ## but instead uses numpy directly.  As a result, for now, we have to seed numpy.
      ## Because we use our RNG to set the seed, though, it should follow the global seed still.
    self.raiseADebug('Setting ARMA seed to',self.seed)
    randomUtils.randomSeed(self.seed,engine=self.randomEng)

    # check for correlation
    correlated = kwargs.get('correlate',None)
    if correlated is not None:
      np.random.seed(self.seed)
      # store correlated targets
      for var in correlated:
        if var not in self.target:
          self.raiseAnError(IOError,'Variable "{}" requested in "correlate" but not found among the targets!'.format(var))
      # NOTE: someday, this could be expanded to include multiple sets of correlated variables.
      self.correlations = correlated

    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")

    # can only handle one scaling input currently
    if len(self.features) != 1:
      self.raiseAnError(IOError,"The ARMA can only currently handle a single feature, which scales the outputs!")

    # read off of paramInput for more detailed inputs # TODO someday everything should read off this!
    paramInput = kwargs['paramInput']
    for child in paramInput.subparts:
      # read truncation requests (really value limits, not truncation)
      if child.getName() == 'outTruncation':
        # did the user request positive or negative?
        domain = child.parameterValues['domain']
        # if a recognized request, store it for later
        if domain in self.outTruncation:
          self.outTruncation[domain] = self.outTruncation[domain] | set(child.value)
        # if unrecognized, error out
        else:
          self.raiseAnError(IOError,'Unrecognized "domain" for "outTruncation"! Was expecting "positive" '+\
                                    'or "negative" but got "{}"'.format(domain))
      # additional info for zerofilter
      elif child.getName() == 'ZeroFilter':
        self.zeroFilterTarget = child.value
        if self.zeroFilterTarget not in self.target:
          self.raiseAnError(IOError,'Requested zero filtering for "{}" but not found among targets!'.format(self.zeroFilterTarget))
        self.zeroFilterTol = child.parameterValues.get('tol',1e-16)
      # read SPECIFIC parameters for Fourier detrending
      elif child.getName() == 'SpecificFourier':
        # clear old information
        periods = None
        # what variables share this Fourier?
        variables = child.parameterValues['variables']
        # check for variables that aren't targets
        missing = set(variables) - set(self.target)
        if len(missing):
          self.raiseAnError(IOError,
                            'Requested SpecificFourier for variables {} but not found among targets!'.format(missing))
        # record requested Fourier periods
        for cchild in child.subparts:
          if cchild.getName() == 'periods':
            periods = cchild.value
        # set these params for each variable
        for v in variables:
          self.raiseADebug('recording specific Fourier settings for "{}"'.format(v))
          if v in self.fourierParams:
            self.raiseAWarning('Fourier params for "{}" were specified multiple times! Using first values ...'
                               .format(v))
            continue
          self.fourierParams[v] = periods
      elif child.getName() == 'Peaks':
        # read peaks information for each target
        peak={}
        # creat an empty list for each target
        threshold = child.parameterValues['threshold']
        peak['threshold']=threshold
        # read the threshold for the peaks and store it in the dict
        period = child.parameterValues['period']
        peak['period']=period
        # read the period for the peaks and store it in the dict
        windows=[]
        # creat an empty list to store the windows' information
        for cchild in child.subparts:
          if cchild.getName() == 'window':
            tempDict={}
            window = cchild.value
            width = cchild.parameterValues['width']
            tempDict['window']=window
            tempDict['width']=width
            # for each window in the windows, we create a dictionary. Then store the
            # peak's width, the index of stating point and ending point in time unit
            windows.append(tempDict)
        peak['windows']=windows
        target = child.parameterValues['target']
        # target is the key to reach each peak information
        self.peaks[target]=peak

    # read GENERAL parameters for Fourier detrending
    ## these apply to everyone without SpecificFourier nodes
    ## use basePeriods to check if Fourier node present
    basePeriods = paramInput.findFirst('Fourier')
    if basePeriods is not None:
      # read periods
      basePeriods = basePeriods.value
      if len(set(basePeriods)) != len(basePeriods):
        self.raiseAnError(IOError,'Some <Fourier> periods have been listed multiple times!')
      # set to any variable that doesn't already have a specific one
      for v in set(self.target) - set(self.fourierParams.keys()):
        self.raiseADebug('setting general Fourier settings for "{}"'.format(v))
        self.fourierParams[v] = basePeriods

  def __getstate__(self):
    """
      Obtains state of object for pickling.
      @ In, None
      @ Out, d, dict, stateful dictionary
    """
    d = supervisedLearning.__getstate__(self)
    eng=d.pop("randomEng")
    randCounts = eng.get_rng_state()
    d['crow_rng_counts'] = randCounts
    return d

  def __setstate__(self,d):
    """
      Sets state of object from pickling.
      @ In, d, dict, stateful dictionary
      @ Out, None
    """
    rngCounts = d.pop('crow_rng_counts')
    self.__dict__.update(d)
    self.setEngine(randomUtils.newRNG(),seed=None,count=rngCounts)
    if self.reseedCopies:
      randd = np.random.randint(1,2e9)
      self.reseed(randd)

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
      self._signalStorage[target]['original'] = copy.deepcopy(timeSeriesData)
      # if we're enforcing the training CDF, we should store it now
      if self.preserveInputCDF:
        self._trainingCDF[target] = mathUtils.trainEmpiricalFunction(timeSeriesData, minBins=self._minBins)
      # if this target governs the zero filter, extract it now
      if target == self.zeroFilterTarget:
        self.notZeroFilterMask = self._trainZeroRemoval(timeSeriesData,tol=self.zeroFilterTol) # where zeros or less than zeros are
        self.zeroFilterMask = np.logical_not(self.notZeroFilterMask) # where data are
      # if we're removing Fourier signal, do that now.

      maskPeakRes = np.ones(len(timeSeriesData), dtype=bool)
      # Make a full mask
      if target in self.peaks:
        deltaT=self.pivotParameterValues[-1]-self.pivotParameterValues[0]
        deltaT=deltaT/(len(self.pivotParameterValues)-1)
        # change the peak information in self.peak from time unit into index by divided the timestep
        # deltaT is the time step calculated by (ending point - stating point in time)/(len(time)-1)
        self.peaks[target]['period']=int(self.peaks[target]['period']/deltaT)
        for i in range(len(self.peaks[target]['windows'])):
          self.peaks[target]['windows'][i]['window'][0]=int(self.peaks[target]['windows'][i]['window'][0]/deltaT)
          self.peaks[target]['windows'][i]['window'][1]=int(self.peaks[target]['windows'][i]['window'][1]/deltaT)
          self.peaks[target]['windows'][i]['width']=int(self.peaks[target]['windows'][i]['width']/deltaT)
        groupWin , maskPeakRes=self._peakGroupWindow(timeSeriesData, windowDict=self.peaks[target] )
        self.peaks[target]['groupWin']=groupWin
        self.peaks[target]['mask']=maskPeakRes

      if target in self.fourierParams:
        self.raiseADebug('... analyzing Fourier signal  for target "{}" ...'.format(target))
        self.fourierResults[target] = self._trainFourier(self.pivotParameterValues,
                                                         self.fourierParams[target],
                                                         timeSeriesData,
                                                         masks=[maskPeakRes],  # In future, a consolidated masking system for multiple signal processors can be implemented.
                                                         zeroFilter = target == self.zeroFilterTarget)
        self._signalStorage[target]['fourier'] = copy.deepcopy(self.fourierResults[target]['predict'])
        timeSeriesData -= self.fourierResults[target]['predict']
        self._signalStorage[target]['nofourier'] = copy.deepcopy(timeSeriesData)
      # zero filter application
      ## find the mask for the requested target where values are nonzero
      if target == self.zeroFilterTarget:
        # artifically force signal to 0 post-fourier subtraction where it should be zero
        targetVals[:,t][self.notZeroFilterMask] = 0.0
        self._signalStorage[target]['zerofilter'] = copy.deepcopy(timeSeriesData)

    # Transform data to obatain normal distrbuted series. See
    # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
    # Applied Energy, 87(2010) 843-855
    for t,target in enumerate(self.target):
      # if target correlated with the zero-filter target, truncate the training material now?
      timeSeriesData = targetVals[:,t]
      self.raiseADebug('... analyzing ARMA properties for target "{}" ...'.format(target))
      self.cdfParams[target] = self._trainCDF(timeSeriesData)
      # normalize data
      normed = self._normalizeThroughCDF(timeSeriesData, self.cdfParams[target])
      self._signalStorage[target]['gaussianed'] = copy.deepcopy(normed[:])
      # check if this target is part of a correlation set, or standing alone
      if target in self.correlations:
        # store the data and train it separately in a moment
        ## keep data in order of self.correlations
        correlationData[:,self.correlations.index(target)] = normed
      else:
        # go ahead and train it now
        ## if using zero filtering and target is the zero-filtered, only train on the masked part
        if target == self.zeroFilterTarget:
          # don't bother training the part that's all zeros; it'll still be all zeros
          # just train the data portions
          normed = normed[self.zeroFilterMask]
        self.raiseADebug('... ... training "{}"...'.format(target))
        self.armaResult[target] = self._trainARMA(normed,masks=[maskPeakRes])
        self.raiseADebug('... ... finished training target "{}"'.format(target))

    # now handle the training of the correlated armas
    if len(self.correlations):
      self.raiseADebug('... ... training correlated: {} ...'.format(self.correlations))
      # if zero filtering, then all the correlation data gets split
      if self.zeroFilterTarget in self.correlations:
        # split data into the zero-filtered and non-zero filtered
        unzeroed = correlationData[self.zeroFilterMask]
        zeroed = correlationData[self.notZeroFilterMask]
        ## throw out the part that's all zeros (axis 1, row corresponding to filter target)
        zeroed = np.delete(zeroed, self.correlations.index(self.zeroFilterTarget), 1)
        self.raiseADebug('... ... ... training unzeroed ...')
        unzVarma, unzNoise, unzInit = self._trainVARMA(unzeroed)
        self.raiseADebug('... ... ... training zeroed ...')
        ## the VAR fails if only 1 variable is non-constant, so we need to decide whether "zeroed" is actually an ARMA
        ## -> instead of a VARMA
        if zeroed.shape[1] == 1:
          # then actually train an ARMA instead
          zVarma = self._trainARMA(zeroed,masks=None)
          zNoise = None # NOTE this is used to check whether an ARMA was trained later!
          zInit = None
        else:
          zVarma, zNoise, zInit = self._trainVARMA(zeroed)
        self.varmaResult = (unzVarma, zVarma) # NOTE how for zero-filtering we split the results up
        self.varmaNoise = (unzNoise, zNoise)
        self.varmaInit = (unzInit, zInit)
      else:
        varma, noiseDist, initDist = self._trainVARMA(correlationData)
        # FUTURE if extending to multiple VARMA per training, these will need to be dictionaries
        self.varmaResult = (varma,)
        self.varmaNoise = (noiseDist,)
        self.varmaInit = (initDist,)

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
      # start with the random gaussian signal
      if target in self.correlations:
        # where is target in correlated data
        corrIndex = self.correlations.index(target)
        # check if we have zero-filtering in play here
        if len(self.varmaResult) > 1:
          # where would the filter be in the index lineup had we included it in the zeroed varma?
          filterTargetIndex = self.correlations.index(self.zeroFilterTarget)
          # if so, we need to sample both VARMAs
          # have we already taken the correlated sample yet?
          if correlatedSample is None:
            # if not, take the samples now
            unzeroedSample = self._generateVARMASignal(self.varmaResult[0],
                                                 numSamples = self.zeroFilterMask.sum(),
                                                 randEngine = self.normEngine.rvs,
                                                 rvsIndex = 0)
            ## zero sampling is dependent on whether the trained model is a VARMA or ARMA
            if self.varmaNoise[1] is not None:
              zeroedSample = self._generateVARMASignal(self.varmaResult[1],
                                                   numSamples = self.notZeroFilterMask.sum(),
                                                   randEngine = self.normEngine.rvs,
                                                   rvsIndex = 1)
            else:
              result = self.varmaResult[1]
              sample = self._generateARMASignal(result,
                                                numSamples = self.notZeroFilterMask.sum(),
                                                randEngine = self.randomEng)
              zeroedSample = np.zeros((self.notZeroFilterMask.sum(),1))
              zeroedSample[:,0] = sample
            correlatedSample = True # placeholder, signifies we've sampled the correlated distribution
          # reconstruct base signal from samples
          ## initialize
          signal = np.zeros(len(self.pivotParameterValues))
          ## first the data from the non-zero portions of the original signal
          signal[self.zeroFilterMask] = unzeroedSample[:,corrIndex]
          ## then the data from the zero portions (if the filter target, don't bother because they're zero anyway)
          if target != self.zeroFilterTarget:
            # fix offset since we didn't include zero-filter target in zeroed correlated arma
            indexOffset = 0 if corrIndex < filterTargetIndex else -1
            signal[self.notZeroFilterMask] = zeroedSample[:,corrIndex+indexOffset]
        # if no zero-filtering (but still correlated):
        else:
          ## check if sample taken yet
          if correlatedSample is None:
            ## if not, do so now
            correlatedSample = self._generateVARMASignal(self.varmaResult[0],
                                                         numSamples = len(self.pivotParameterValues),
                                                         randEngine = self.normEngine.rvs,
                                                         rvsIndex = 0)
          # take base signal from sample
          signal = correlatedSample[:,self.correlations.index(target)]
      # if NOT correlated
      else:
        result = self.armaResult[target] # ARMAResults object
        # generate baseline ARMA + noise
        # are we zero-filtering?
        if target == self.zeroFilterTarget:
          sample = self._generateARMASignal(result,
                                            numSamples = self.zeroFilterMask.sum(),
                                            randEngine = self.randomEng)

          ## if so, then expand result into signal space (functionally, put back in all the zeros)
          signal = np.zeros(len(self.pivotParameterValues))
          signal[self.zeroFilterMask] = sample
        else:
          ## if not, no extra work to be done here!
          sample = self._generateARMASignal(result,
                                            numSamples = len(self.pivotParameterValues),
                                            randEngine = self.randomEng)
          signal = sample
      # END creating base signal
      # DEBUG adding arbitrary variables for debugging, TODO find a more elegant way, leaving these here as markers
      #returnEvaluation[target+'_0base'] = copy.copy(signal)
      # denoise
      signal = self._denormalizeThroughCDF(signal,self.cdfParams[target])
      # DEBUG adding arbitrary variables
      #returnEvaluation[target+'_1denorm'] = copy.copy(signal)
      #debuggFile.writelines('signal_arma,'+','.join(str(x) for x in signal)+'\n')

      # Add fourier trends
      if target in self.fourierParams:
        signal += self.fourierResults[target]['predict']
        # DEBUG adding arbitrary variables
        #returnEvaluation[target+'_2fourier'] = copy.copy(signal)
        #debuggFile.writelines('signal_fourier,'+','.join(str(x) for x in self.fourierResults[target]['predict'])+'\n')
      if target in self.peaks:
        signal = self._transformBackPeaks(signal,windowDict=self.peaks[target])
      # if enforcing the training data CDF, apply that transform now
      if self.preserveInputCDF:
        signal = self._transformThroughInputCDF(signal, self._trainingCDF[target])

      # Re-zero out zero filter target's zero regions
      if target == self.zeroFilterTarget:
        # DEBUG adding arbitrary variables
        #returnEvaluation[target+'_3zerofilter'] = copy.copy(signal)
        signal[self.notZeroFilterMask] = 0.0

      # Domain limitations
      for domain,requests in self.outTruncation.items():
        if target in requests:
          if domain == 'positive':
            signal = np.absolute(signal)
          elif domain == 'negative':
            signal = -np.absolute(signal)
        # DEBUG adding arbitrary variables
        #returnEvaluation[target+'_4truncated'] = copy.copy(signal)

      # store results
      ## FIXME this is ASSUMING the input to ARMA is only ever a single scaling factor.
      signal *= featureVals[0]
      # DEBUG adding arbitrary variables
      #returnEvaluation[target+'_5scaled'] = copy.copy(signal)

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
    randomUtils.randomSeed(seed,engine=self.randomEng)
    self.seed=seed

  ### UTILITY METHODS ###
  def _computeNumberOfBins(self,data):
    """
      Uses the Freedman-Diaconis rule for histogram binning
      -> For relatively few samples, this can cause unnatural flat-lining on low, top end of CDF
      @ In, data, np.array, data to bin
      @ Out, n, integer, number of bins
    """
    # leverage the math utils implementation
    n, _ = mathUtils.numBinsDraconis(data, low=self._minBins, alternateOkay=True)
    return n

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

  def _generateARMASignal(self, model, numSamples=None,randEngine=None):
    """
      Generates a synthetic history from fitted parameters.
      @ In, model, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA such as otained from _trainARMA
      @ In, numSamples, int, optional, number of samples to take (default to pivotParameters length)
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ Out, hist, np.array(float), synthetic ARMA signal
    """
    if numSamples is None:
      numSamples =  len(self.pivotParameterValues)
    if randEngine is None:
      randEngine=self.randomEng
    hist = sm.tsa.arma_generate_sample(ar = np.append(1., -model.arparams),
                                       ma = np.append(1., model.maparams),
                                       nsample = numSamples,
                                       distrvs = functools.partial(randomUtils.randomNormal,engine=randEngine),
                                       # functool.partial provide the random number generator as a function
                                       # with normal distribution and take engine as the positional arguments keywords.
                                       sigma = np.sqrt(model.sigma2),
                                       burnin = 2*max(self.P,self.Q)) # @epinas, 2018
    return hist

  def _generateFourierSignal(self, pivots, periods):
    """
      Generate fourier signal as specified by the input file
      @ In, pivots, np.array, pivot values (e.g. time)
      @ In, periods, list, list of Fourier periods (1/frequency)
      @ Out, fourier, array, shape = [n_timeStep, n_basePeriod]
    """
    fourier = np.zeros((pivots.size, 2*len(periods))) # sin, cos for each period
    for p, period in enumerate(periods):
      hist = 2. * np.pi / period * pivots
      fourier[:, 2 * p] = np.sin(hist)
      fourier[:, 2 * p + 1] = np.cos(hist)
    return fourier

  def _generateVARMASignal(self, model, numSamples=None, randEngine=None, rvsIndex=None):
    """
      Generates a set of correlated synthetic histories from fitted parameters.
      @ In, model, statsmodels.tsa.statespace.VARMAX, fitted VARMA such as otained from _trainVARMA
      @ In, numSamples, int, optional, number of samples to take (default to pivotParameters length)
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ In, rvsIndex, int, optional, if provided then will take from list of varmaNoise and varmaInit distributions
      @ Out, hist, np.array(float), synthetic ARMA signal
    """
    if numSamples is None:
      numSamples =  len(self.pivotParameterValues)
    # sample measure, state shocks
    ## TODO it appears that measure shock always has a 0 variance multivariate normal, so just create it
    measureShocks = np.zeros([numSamples,len(self.correlations)])
    ## state shocks come from sampling multivariate
    noiseDist = self.varmaNoise
    initDist = self.varmaInit
    if rvsIndex is not None:
      noiseDist = noiseDist[rvsIndex]
      initDist = initDist[rvsIndex]
    stateShocks = np.array([noiseDist.rvs() for _ in range(numSamples)])
    # pick an intial by sampling multinormal distribution
    init = np.array(initDist.rvs())
    obs, states = model.ssm.simulate(numSamples,
                                     initial_state = init,
                                     measurement_shocks = measureShocks,
                                     state_shocks = stateShocks)
    return obs

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

  def _sampleCDF(self, x, params):
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

  def _trainARMA(self,data,masks=None):
    r"""
      Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
      @ In, data, np.array(float), data on which to train
      @ In, masks, np.array, optional, boolean mask where is the signal should be train by ARMA
      @ Out, results, statsmodels.tsa.arima_model.ARMAResults, fitted ARMA
    """
    if masks == None:
      masks = []
    if len(masks)>1:
      fullMask = np.logical_and.reduce(*masks)
      data=data[fullMask]
    elif len(masks)==1:
      fullMask =masks[0]
      data=data[fullMask]
    results = smARMA(data, order = (self.P, self.Q)).fit(disp = False)
    return results

  def _trainCDF(self,data):
    """
      Constructs a CDF from the given data
      @ In, data, np.array(float), values to fit to
      @ Out, params, dict, essential parameters for CDF
    """
    # caluclate number of bins
    nBins = self._computeNumberOfBins(data)
    # construct histogram
    counts, edges = np.histogram(data, bins = nBins, density = False)
    counts = np.array(counts) / float(len(data))
    # numerical CDF, normalizing to 0..1
    cdf = np.cumsum(counts)
    # set lowest value as first entry,
    ## from Jun implementation, min of CDF set to starting point for ?numerical issues?
    #cdf = np.insert(cdf, 0, cdf[0]) # Jun
    cdf = np.insert(cdf, 0, 0) # trying something else
    # store parameters
    params = {'bins': edges,
              'pdf' : counts * nBins,
              'cdf' : cdf}
              #'binSearch':neighbors.NearestNeighbors(n_neighbors=2).fit([[b] for b in edges]),
              #'cdfSearch':neighbors.NearestNeighbors(n_neighbors=2).fit([[c] for c in cdf])}
    return params

  def _trainFourier(self, pivotValues, periods, values, masks=None,zeroFilter=False):
    """
      Perform fitting of Fourier series on self.timeSeriesDatabase
      @ In, pivotValues, np.array, list of values for the independent variable (e.g. time)
      @ In, periods, list, list of the base periods
      @ In, values, np.array, list of values for the dependent variable (signal to take fourier from)
      @ In, masks, np.array, optional, boolean mask where is the signal should be train by Fourier
      @ In, zeroFilter, bool, optional, if True then apply zero-filtering for fourier fitting
      @ Out, fourierResult, dict, results of this training in keys 'residues', 'fourierSet', 'predict', 'regression'
    """
    # XXX fix for no order
    if masks is None:
      masks = []

    fourierSignalsFull = self._generateFourierSignal(pivotValues, periods)
    # fourierSignals dimensions, for each key (base):
    #   0: length of history
    #   1: evaluations, in order and flattened:
    #                 0:   sin(2pi*t/period[0]),
    #                 1:   cos(2pi*t/period[0]),
    #                 2:   sin(2pi*t/period[1]),
    #                 3:   cos(2pi*t/period[1]), ...
    fourierEngine = linear_model.LinearRegression(normalize=False)
    for mask in masks:
      fourierSignalsFull = fourierSignalsFull[mask, :]
      values = values[mask]


    # if using zero-filter, cut the parts of the Fourier and values that correspond to the zero-value portions
    if zeroFilter:
      values = values[self.zeroFilterMask]
      fourierSignals = fourierSignalsFull[self.zeroFilterMask, :]
    else:
      fourierSignals = fourierSignalsFull

    # fit the signal
    fourierEngine.fit(fourierSignals, values)

    # get signal intercept
    intercept = fourierEngine.intercept_
    # get coefficient map for A*sin(ft) + B*cos(ft)
    waveCoefMap = collections.defaultdict(dict) # {period: {sin:#, cos:#}}
    for c, coef in enumerate(fourierEngine.coef_):
      period = periods[c//2]
      waveform = 'sin' if c % 2 == 0 else 'cos'
      waveCoefMap[period][waveform] = coef
    # convert to C*sin(ft + s)
    ## since we use fitting to get A and B, the magnitudes can be deceiving.
    ## this conversion makes "C" a useful value to know the contribution from a period
    coefMap = {}
    signal=np.ones(len(pivotValues)) * intercept
    for period, coefs in waveCoefMap.items():
      A = coefs['sin']
      B = coefs['cos']
      C, s = mathUtils.convertSinCosToSinPhase(A, B)
      coefMap[period] = {'amplitude': C, 'phase': s}
      signal+=mathUtils.evalFourier(period,C,s,pivotValues)
    # re-add zero-filtered
    if zeroFilter:
      signal[self.notZeroFilterMask] = 0.0


    # store results
    fourierResult = {'regression': {'intercept':intercept,
                                    'coeffs'   :coefMap,
                                    'periods'  :periods},
                     'predict': signal}
    return fourierResult

  def _trainMultivariateNormal(self,dim,means,cov):
    """
      Trains multivariate normal distribution for future sampling
      @ In, dim, int, number of dimensions
      @ In, means, np.array, distribution mean
      @ In, cov, np.ndarray, dim x dim matrix of covariance terms
      @ Out, dist, Distributions.MultivariateNormal, distribution
    """
    dist = Distributions.MultivariateNormal()
    dist.method = 'pca'
    dist.dimension = dim
    dist.rank = dim
    dist.mu = means
    dist.covariance = np.ravel(cov)
    dist.messageHandler = self.messageHandler
    dist.initializeDistribution()
    return dist

  def _trainVARMA(self,data):
    """
      Train correlated ARMA model on white noise ARMA, with Fourier already removed
      @ In, data, np.array(np.array(float)), data on which to train with shape (# pivot values, # targets)
      @ Out, results, statsmodels.tsa.arima_model.ARMAResults, fitted VARMA
      @ Out, stateDist, Distributions.MultivariateNormal, MVN from which VARMA noise is taken
      @ Out, initDist, Distributions.MultivariateNormal, MVN from which VARMA initial state is taken
    """
    model = sm.tsa.VARMAX(endog=data, order=(self.P, self.Q))
    self.raiseADebug('... ... ... fitting VARMA ...')
    results = model.fit(disp=False,maxiter=1000)
    lenHist,numVars = data.shape
    # train multivariate normal distributions using covariances, keep it around so we can control the RNG
    ## it appears "measurement" always has 0 covariance, and so is all zeros (see _generateVARMASignal)
    ## all the noise comes from the stateful properties
    stateDist = self._trainMultivariateNormal(numVars,np.zeros(numVars),model.ssm['state_cov'])
    # train initial state sampler
    ## Used to pick an initial state for the VARMA by sampling from the multivariate normal noise
    #    and using the AR and MA initial conditions.  Implemented so we can control the RNG internally.
    #    Implementation taken directly from statsmodels.tsa.statespace.kalman_filter.KalmanFilter.simulate
    ## get mean
    smoother = model.ssm
    mean = np.linalg.solve(np.eye(smoother.k_states) - smoother['transition',:,:,0],
                           smoother['state_intercept',:,0])
    ## get covariance
    r = smoother['selection',:,:,0]
    q = smoother['state_cov',:,:,0]
    selCov = r.dot(q).dot(r.T)
    cov = solve_discrete_lyapunov(smoother['transition',:,:,0], selCov)
    # FIXME it appears this is always resulting in a lowest-value initial state.  Why?
    initDist = self._trainMultivariateNormal(len(mean),mean,cov)
    # NOTE: uncomment this line to get a printed summary of a lot of information about the fitting.
    #self.raiseADebug('VARMA model training summary:\n',results.summary())
    return model, stateDist, initDist

  def _trainZeroRemoval(self, data, tol=1e-10):
    """
      A test for SOLAR GHI data.
      @ In, data, np.array, original signal
      @ In, tol, float, optional, tolerance below which to consider 0
      @ Out, mask, np.ndarray(bool), mask where zeros occur
    """
    # where should the data be truncated?
    mask = data < tol
    return mask

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained! Cannot write to DataObject.')
    rlz = {}
    # set up pivot parameter index
    pivotID = self.pivotParameterID
    pivotVals = self.pivotParameterValues
    rlz[self.pivotParameterID] = self.pivotParameterValues
    # set up sample counter ID
    ## ASSUMPTION: data object is EMPTY!
    if writeTo.size > 0:
      self.raiseAnError(ValueError,'Target data object has "{}" entries, but require an empty object to write ROM to!'.format(writeTo.size))
    counterID = writeTo.sampleTag
    counterVals = np.array([0])
    # Training signals
    for target, signals in self._signalStorage.items():
      for name, signal in signals.items():
        varName = '{}_{}'.format(target,name)
        writeTo.addVariable(varName, np.array([]), classify='meta', indices=[pivotID])
        rlz[varName] = signal
    # add realization
    writeTo.addRealization(rlz)

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused (kept for compatability)
      @ In, skip, list, optional, unused (kept for compatability)
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError, 'ROM is not yet trained! Cannot write to DataObject.')
    root = writeTo.getRoot()
    # - Fourier coefficients (by period, waveform)
    for target, fourier in self.fourierResults.items():
      targetNode = root.find(target)
      if targetNode is None:
        targetNode = xmlUtils.newNode(target)
        root.append(targetNode)
      fourierNode = xmlUtils.newNode('Fourier')
      targetNode.append(fourierNode)
      fourierNode.append(xmlUtils.newNode('SignalIntercept', text='{:1.9e}'.format(fourier['regression']['intercept'])))
      for period in fourier['regression']['periods']:
        periodNode = xmlUtils.newNode('period', text='{:1.9e}'.format(period))
        fourierNode.append(periodNode)
        periodNode.append(xmlUtils.newNode('frequency', text='{:1.9e}'.format(1.0/period)))
        for stat, value in sorted(list(fourier['regression']['coeffs'][period].items()), key=lambda x:x[0]):
          periodNode.append(xmlUtils.newNode(stat, text='{:1.9e}'.format(value)))
    # - ARMA std
    for target, arma in self.armaResult.items():
      targetNode = root.find(target)
      if targetNode is None:
        targetNode = xmlUtils.newNode(target)
        root.append(targetNode)
      armaNode = xmlUtils.newNode('ARMA_params')
      targetNode.append(armaNode)
      armaNode.append(xmlUtils.newNode('std', text=np.sqrt(arma.sigma2)))
      # TODO covariances, P and Q, etc
    for target,peakInfo in self.peaks.items():
      targetNode = root.find(target)
      if targetNode is None:
        targetNode = xmlUtils.newNode(target)
        root.append(targetNode)
      peakNode = xmlUtils.newNode('Peak_params')
      targetNode.append(peakNode)
      if 'groupWin' in peakInfo.keys():
        for group in peakInfo['groupWin']:
          groupnode=xmlUtils.newNode('peak')
          groupnode.append(xmlUtils.newNode('Amplitude', text='{}'.format(np.array(group['Amp']).mean())))
          groupnode.append(xmlUtils.newNode('Index', text='{}'.format(np.array(group['Ind']).mean())))
          peakNode.append(groupnode)




  def _transformThroughInputCDF(self, signal, originalDist, weights=None):
    """
      Transforms a signal through the original distribution
      @ In, signal, np.array(float), signal to transform
      @ In, originalDist, scipy.stats.rv_histogram, distribution to transform through
      @ In, weights, np.array(float), weighting for samples (assumed uniform if not given)
      @ Out, new, np.array, new signal after transformation
    """
    # first build a histogram object of the sampled data
    dist = mathUtils.trainEmpiricalFunction(signal, minBins=self._minBins, weights=weights)
    # transform data through CDFs
    new = originalDist.ppf(dist.cdf(signal))
    return new

  ### Segmenting and Clustering ###
  def  isClusterable(self):
    """
      Allows ROM to declare whether it has methods for clustring. Default is no.
      @ In, None
      @ Out, isClusterable, bool, if True then has clustering mechanics.
    """
    # clustering methods have been added
    return True

  def _getMeanFromGlobal(self, settings, pickers, targets=None):
    """
      Derives segment means from global trends
      @ In, settings, dict, as per getGlobalRomSegmentSettings
      @ In, pickers, list(slice), picks portion of signal of interest
      @ In, targets, list, optional, targets to include (default is all)
      @ Out, results, list(dict), mean for each target per picker
    """
    if 'long Fourier signal' not in settings:
      return []
    if isinstance(pickers, slice):
      pickers = [pickers]
    if targets == None:
      targets = settings['long Fourier signal'].keys()
    results = [] # one per "pickers"
    for picker in pickers:
      res = dict((target, signal['predict'][picker].mean()) for target, signal in settings['long Fourier signal'].items())
      results.append(res)
    return results

  def getLocalRomClusterFeatures(self, featureTemplate, settings, picker=None, **kwargs):
    """
      Provides metrics aka features on which clustering compatibility can be measured.
      This is called on LOCAL subsegment ROMs, not on the GLOBAL template ROM
      @ In, featureTemplate, str, format for feature inclusion
      @ In, settings, dict, as per getGlobalRomSegmentSettings
      @ In, picker, slice, indexer for segmenting data
      @ In, kwargs, dict, arbitrary keyword arguments
      @ Out, features, dict, {target_metric: np.array(floats)} features to cluster on
    """
    # algorithm for providing Fourier series and ARMA white noise variance and #TODO covariance
    features = {}
    # include Fourier if available
    for target, fourier in self.fourierResults.items():
      for period in fourier['regression']['periods']:
        # amp
        amp = fourier['regression']['coeffs'][period]['amplitude']
        ID = '{}_{}'.format(period, 'amp')
        feature = featureTemplate.format(target=target, metric='Fourier', id=ID)
        features[feature] = amp
        # phase
        ## not great for clustering, but sure, why not
        phase = fourier['regression']['coeffs'][period]['phase']
        ID = '{}_{}'.format(period, 'phase')
        feature = featureTemplate.format(target=target, metric='Fourier', id=ID)
        features[feature] = phase

    # signal variance, ARMA (not varma)
    for target, arma in self.armaResult.items():
      feature = featureTemplate.format(target=target, metric='arma', id='std')
      features[feature] = np.sqrt(arma.sigma2)
    # segment means
    # since we've already detrended globally, get the means from that (if present)
    if 'long Fourier signal' in settings:
      assert picker is not None
      results = self._getMeanFromGlobal(settings, picker)
      for target, mean in results[0].items():
        feature = featureTemplate.format(target=target, metric="global", id="mean")
        features[feature] = mean
    return features

  def getGlobalRomSegmentSettings(self, trainingDict, divisions):
    """
      Allows the ROM to perform some analysis before segmenting.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL subsegment ROMs!
      @ In, trainingDict, dict, data for training, full and unsegmented
      @ In, divisions, tuple, (division slice indices, unclustered spaces)
      @ Out, settings, object, arbitrary information about ROM clustering settings
      @ Out, trainingDict, dict, adjusted training data (possibly unchanged)
    """
    trainingDict = copy.deepcopy(trainingDict) # otherwise we destructively tamper with the input data object
    settings = {}
    targets = list(self.fourierParams.keys())
    # set up for input CDF preservation on a global scale
    if self.preserveInputCDF:
      inputDists = {}
      for target in targets:
        if target == self.pivotParameterID:
          continue
        targetVals = trainingDict[target][0]
        inputDists[target] = mathUtils.trainEmpiricalFunction(targetVals, minBins=self._minBins)
      settings['input CDFs'] = inputDists
    # do global Fourier analysis on combined signal for all periods longer than the segment
    if self.fourierParams:
      # determine the Nyquist length for the clustered params
      slicers = divisions[0]
      pivotValues = trainingDict[self.pivotParameterID][0]
      # use the first segment as typical of all of them, NOTE might be bad assumption
      delta = pivotValues[slicers[0][-1]] - pivotValues[slicers[0][0]]
      # any Fourier longer than the delta should be trained a priori, leaving the reaminder
      #    to be specific to individual ROMs
      full = {}      # train these periods on the full series
      segment = {}   # train these periods on the segments individually
      for target in targets:
        if target == self.pivotParameterID:
          continue
        # only do separation for targets for whom there's a Fourier request
        if target in self.fourierParams:
          # NOTE: assuming training on only one history!
          targetVals = trainingDict[target][0]
          # if zero filtering in play, set the masks now
          ## TODO I'm not particularly happy with having to remember to do this; can we automate it more?
          zeroFiltering = target == self.zeroFilterTarget
          if zeroFiltering:
            self.notZeroFilterMask = self._trainZeroRemoval(targetVals, tol=self.zeroFilterTol) # where zeros are not
            self.zeroFilterMask = np.logical_not(self.notZeroFilterMask) # where zeroes are
          periods = np.asarray(self.fourierParams[target])
          full = periods[periods > delta]
          segment[target] = periods[np.logical_not(periods > delta)]
          if len(full):
            # train Fourier on longer periods
            self.fourierResults[target] = self._trainFourier(pivotValues,
                                                             full,
                                                             targetVals,
                                                             zeroFilter=zeroFiltering)
            # remove longer signal from training data
            signal = self.fourierResults[target]['predict']
            targetVals -= signal
            trainingDict[target][0] = targetVals
      # store the segment-based periods in the settings to return
      settings['segment Fourier periods'] = segment
      settings['long Fourier signal'] = self.fourierResults
    return settings, trainingDict

  def adjustLocalRomSegment(self, settings):
    """
      Adjusts this ROM to account for it being a segment as a part of a larger ROM collection.
      Call this before training the subspace segment ROMs
      Note this is called on the LOCAL subsegment ROMs, NOT on the GLOBAL templateROM from the ROMcollection!
      @ In, settings, object, arbitrary information about ROM clustering settings from getGlobalRomSegmentSettings
      @ Out, None
    """
    # some Fourier periods have already been handled, so reset the ones that actually are needed
    newFourier = settings.get('segment Fourier periods', None)
    if newFourier is not None:
      for target in list(self.fourierParams.keys()):
        periods = newFourier.get(target, [])
        # if any sub-segment Fourier remaining, send it through
        if len(periods):
          self.fourierParams[target] = periods
        else:
          # otherwise, remove target from fourierParams so no Fourier is applied
          self.fourierParams.pop(target,None)
    # disable CDF preservation on subclusters
    ## Note that this might be a good candidate for a user option someday,
    ## but right now we can't imagine a use case that would turn it on
    self.preserveInputCDF = False

  def finalizeLocalRomSegmentEvaluation(self, settings, evaluation, picker):
    """
      Allows global settings in "settings" to affect a LOCAL evaluation of a LOCAL ROM
      Note this is called on the LOCAL subsegment ROM and not the GLOBAL templateROM.
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, preliminary evaluation from the local segment ROM as {target: [values]}
      @ In, picker, slice, indexer for data range of this segment
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    # add back in Fourier
    if 'long Fourier signal' in settings:
      for target, signal in settings['long Fourier signal'].items():
        sig = signal['predict'][picker]
        evaluation[target][picker] += sig
    return evaluation

  def finalizeGlobalRomSegmentEvaluation(self, settings, evaluation, weights=None):
    """
      Allows any global settings to be applied to the signal collected by the ROMCollection instance.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL supspace segment ROMs!
      @ In, settings, dict, as from getGlobalRomSegmentSettings
      @ In, evaluation, dict, {target: np.ndarray} evaluated full (global) signal from ROMCollection
      @ In, weights, np.array(float), optional, if included then gives weight to histories for CDF preservation
      @ Out, evaluation, dict, {target: np.ndarray} adjusted global evaluation
    """
    # backtransform signal
    ## how nicely does this play with zerofiltering?
    if self.preserveInputCDF:
      for target, dist in settings['input CDFs'].items():
        evaluation[target] = self._transformThroughInputCDF(evaluation[target], dist, weights)
    return evaluation

  ### Peak Picker ###
  def _peakPicker(self,signal,low):
    """
      Peak picker, this method find the local maxima index inside the signal by comparing the
      neighboring values. Threshold of peaks is required to output the height of each peak.
      @ In, signal, np.array(float), signal to transform
      @ In, low, float, required height of peaks.
      @ Out, peaks, np.array, indices of peaks in x that satisfy all given conditions
      @ Out, heights, np.array, boolean mask where is the residual signal
    """
    peaks, properties = find_peaks(signal, height=low)
    heights = properties['peak_heights']
    return peaks,heights

  def rangeWindow(self,windowDict):
    """
      Collect the window index in to groups and store the information in dictionariy for each target
      @ In, windowDict, dict, dictionary for specefic target peaks
      @ Out, rangeWindow, list, list of dictionaries which store the window index for each target
    """
    rangeWindow = []
    windowType = len(windowDict['windows'])
    windows = windowDict['windows']
    period = windowDict['period']
    for i in range(windowType):
      windowRange={}
      bgP=(windows[i]['window'][0]-1)%period
      endP=(windows[i]['window'][1]+2)%period
      timeInd=np.arange(len(self.pivotParameterValues))
      bgPInd  = np.where(timeInd%period==bgP )[0].tolist()
      endPInd = np.where(timeInd%period==endP)[0].tolist()
      if bgPInd[0]>endPInd[0]:
        tail=endPInd[0]
        endPInd.pop(0)
        endPInd.append(tail)
      windowRange['bg']=bgPInd
      windowRange['end']=endPInd
      rangeWindow.append(windowRange)
    return rangeWindow

  def _peakGroupWindow(self,signal,windowDict):
    """
      Collect the peak information in to groups, define the residual signal.
      Including the index and amplitude of each peak found in the window.
      @ In, signal, np.array(float), signal to transform
      @ In, windowDict, dict, dictionary for specefic target peaks
      @ Out, groupWin, list, list of dictionaries which store the peak information
      @ Out, maskPeakRes, np.array, boolean mask where is the residual signal
    """
    groupWin = []
    maskPeakRes = np.ones(len(signal), dtype=bool)
    rangeWindow = self.rangeWindow(windowDict)
    low = windowDict['threshold']
    windows = windowDict['windows']
    for i in range(len(windowDict['windows'])):
      bg  = rangeWindow[i]['bg']
      end = rangeWindow[i]['end']
      peakInfo   = {}
      indLocal   = []
      ampLocal   = []
      for j in range(min(len(bg), len(end))):
        ##FIXME this might ignore one window, because the amount of the
        # staring points and the ending points might be different here,
        # we choose the shorter one to make sure each window is complete.
        # Future developer can extend the head and tail of the signal to
        # include all the posible windows
        bgLocal = bg[j]
        endLocal = end[j]
        peak, height = self._peakPicker(signal[bgLocal:endLocal], low=low)
        if len(peak) ==1:
          indLocal.append(int(peak))
          ampLocal.append(float(height))
          maskBg=int(peak)+bgLocal-int(np.floor(windows[i]['width']/2))
          maskEnd=int(peak)+bgLocal+int(np.ceil(windows[i]['width']/2))
          maskPeakRes[maskBg:maskEnd]=False
        elif len(peak) >1:
          indLocal.append(int(peak[np.argmax(height)]))
          ampLocal.append(float(height[np.argmax(height)]))
          maskBg=int(peak[np.argmax(height)])+bgLocal-int(np.floor(windows[i]['width']/2))
          maskEnd=int(peak[np.argmax(height)])+bgLocal+int(np.ceil(windows[i]['width']/2))
          maskPeakRes[maskBg:maskEnd]=False
      peakInfo['Ind'] = indLocal
      peakInfo['Amp'] = ampLocal
      groupWin.append(peakInfo)
    return groupWin , maskPeakRes

  def _transformBackPeaks(self,signal,windowDict):
    """
      Transforms a signal by regenerate the peaks signal
      @ In, signal, np.array(float), signal to transform
      @ In, windowDict, dict, dictionary for specefic target peaks
      @ Out, signal, np.array(float), new signal after transformation
    """
    groupWin = windowDict['groupWin']
    windows  = windowDict['windows']
    rangeWindow=self.rangeWindow(windowDict)
    for i in range(len(windowDict['windows'])):
      prbExist = len(groupWin[i]['Ind'])/len(rangeWindow[i]['bg'])
      # (amount of peaks that collected in the windows)/(the amount of windows)
      # this is the probability to check if we should add a peak in each type of window
      histAmp = np.histogram(groupWin[i]['Amp'])
      # generate the distribution of the amplitude for this type of peak
      histInd = np.histogram(groupWin[i]['Ind'])
      # generate the distribution of the position( relative index) in the window
      for j in range(min(len(rangeWindow[i]['bg']),len(rangeWindow[i]['end']))):
        # the length of the starting points and ending points might be different
        bgLocal = rangeWindow[i]['bg'][j]
        # choose the starting index for specific window
        exist = np.random.choice(2, 1, p=[1-prbExist,prbExist])
        # generate 1 or 0 base on the prbExist
        if exist == 1:
          Amp = rv_histogram(histAmp).rvs()
          Ind = int(rv_histogram(histInd).rvs())
          # generate the amplitude and the relative position base on the distribution
          SigInd = bgLocal+Ind
          SigInd = int(SigInd%len(self.pivotParameterValues))
          signal[SigInd] = Amp
          # replace the signal with peak in this window
          maskBg = SigInd-int(np.floor(windows[i]['width']/2))
          maskEnd = SigInd+int(np.ceil(windows[i]['width']/2))
          # replace the signal inside the width of this peak by interpolation
          if maskBg > 0 and maskEnd < len(self.pivotParameterValues)-1:
            # make sure the window is inside the range of the signal
            bgValue = signal[maskBg-1]
            endVaue = signal[maskEnd+1]
            valueBg=np.interp(range(maskBg,SigInd), [maskBg-1,SigInd], [bgValue,  Amp])
            signal[maskBg:SigInd]=valueBg
            valueEnd=np.interp(range(SigInd+1,maskEnd+1), [SigInd,maskEnd+1],   [Amp,endVaue])
            signal[SigInd+1:maskEnd+1]=valueEnd
      return signal

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

  def setEngine(self,eng,seed=None,count=None):
    """
     Set up the random engine for arma
     @ In, eng, instance, random number generator
     @ In, seed, int, optional, the seed, if None then use the global seed from ARMA
     @ In, count, int, optional, advances the state of the generator, if None then use the current ARMA.randomEng count
     @ Out, None
    """
    if seed is None:
      seed=self.seed
    seed=abs(seed)
    eng.seed(seed)
    if count is None:
      count=self.randomEng.get_rng_state()
    eng.forward_seed(count)
    self.randomEng=eng
