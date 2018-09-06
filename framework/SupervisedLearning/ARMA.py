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
from scipy.linalg import solve_discrete_lyapunov
from sklearn import linear_model, neighbors
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import randomUtils
import Distributions
from .SupervisedLearning import supervisedLearning
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
    # general infrastructure
    supervisedLearning.__init__(self,messageHandler,**kwargs)
    self.printTag          = 'ARMA'
    self._dynamicHandling  = True # This ROM is able to manage the time-series on its own.
    # training storage
    self.trainingData      = {} # holds normalized ('norm') and original ('raw') training data, by target
    self.cdfParams         = {} # dictionary of fitted CDF parameters, by target
    self.armaResult        = {} # dictionary of assorted useful arma information, by target
    self.correlations      = [] # list of correlated variables
    self.fourierResults    = {} # dictionary of Fourier results, by target
    # training parameters
    self.fourierParams     = {} # dict of Fourier training params, by target (if requested, otherwise not present)
    self.Pmax              = kwargs.get('Pmax', 3) # bounds for autoregressive lag
    self.Pmin              = kwargs.get('Pmin', 0)
    self.Qmax              = kwargs.get('Qmax', 3) # bounds for moving average lag
    self.Qmin              = kwargs.get('Qmin', 0)
    self.segments          = kwargs.get('segments', 1)
    # data manipulation
    self.reseedCopies      = kwargs.get('reseedCopies',True)
    self.outTruncation = {'positive':set(),'negative':set()} # store truncation requests
    self.pivotParameterID  = kwargs['pivotParameter']
    self.pivotParameterValues = None  # In here we store the values of the pivot parameter (e.g. Time)
    self.seed              = kwargs.get('seed',None)
    self.zeroFilterTarget  = None # target for whom zeros should be filtered out
    self.zeroFilterTol     = None # tolerance for zerofiltering to be considered zero, set below
    self.zeroFilterMask    = None # mask of places where zftarget is zero, or None if unused

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
        orders = None
        # what variables share this Fourier?
        variables = child.parameterValues['variables']
        # check for variables that aren't targets
        missing = set(variables) - set(self.target)
        if len(missing):
          self.raiseAnError(IOError,
                            'Requested SpecificFourier for variables {} but not found among targets!'.format(missing))
        # record requested Fourier periods, orders
        for cchild in child.subparts:
          if cchild.getName() == 'periods':
            periods = cchild.value
          elif cchild.getName() == 'orders':
            orders = cchild.value
        # sanity check
        if len(periods) != len(orders):
          self.raiseADebug(IOError,'"periods" and "orders" need to have the same number of entries' +\
                                   'for variable group "{}"!'.format(variables))
        # set these params for each variable
        for v in variables:
          self.raiseADebug('recording specific Fourier settings for "{}"'.format(v))
          if v in self.fourierParams:
            self.raiseAWarning('Fourier params for "{}" were specified multiple times! Using first values ...'
                               .format(v))
            continue
          self.fourierParams[v] = {'periods': periods,
                                   'orders': dict(zip(periods,orders))}

    # read GENERAL parameters for Fourier detrending
    ## these apply to everyone without SpecificFourier nodes
    ## use basePeriods to check if Fourier node present
    basePeriods = paramInput.findFirst('Fourier')
    if basePeriods is not None:
      # read periods
      basePeriods = basePeriods.value
      if len(set(basePeriods)) != len(basePeriods):
        self.raiseAnError(IOError,'Some <Fourier> periods have been listed multiple times!')
      # read orders
      baseOrders = self.initOptionDict.get('FourierOrder', [1]*len(basePeriods))
      if len(basePeriods) != len(baseOrders):
        self.raiseAnError(IOError,'{} Fourier periods were requested, but {} Fourier order expansions were given!'
                                   .format(len(basePeriods),len(baseOrders)))
      # set to any variable that doesn't already have a specific one
      for v in set(self.target) - set(self.fourierParams.keys()):
        self.raiseADebug('setting general Fourier settings for "{}"'.format(v))
        self.fourierParams[v] = {'periods': basePeriods,
                                 'orders': dict(zip(basePeriods,baseOrders))}

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
    # DEBUG FILE -> uncomment lines with this file in it to get series information.  This should be made available
    #    through a RomTrainer SolutionExport or something similar, or perhaps just an Output DataObject, in the future.
    writeTrainDebug = False
    if writeTrainDebug:
      debugfile = open('debugg_varma.csv','w')
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
      if writeTrainDebug:
        debugfile.writelines('{}_original,'.format(target)+','.join(str(d) for d in timeSeriesData)+'\n')
      # if this target governs the zero filter, extract it now
      if target == self.zeroFilterTarget:
        self.notZeroFilterMask = self._trainZeroRemoval(timeSeriesData,tol=self.zeroFilterTol) # where zeros are not
        self.zeroFilterMask = np.logical_not(self.notZeroFilterMask) # where zeroes are
      # if we're removing Fourier signal, do that now.
      if target in self.fourierParams:
        self.raiseADebug('... analyzing Fourier signal  for target "{}" ...'.format(target))
        self.fourierResults[target] = self._trainFourier(self.pivotParameterValues,
                                                         self.fourierParams[target]['periods'],
                                                         self.fourierParams[target]['orders'],
                                                         timeSeriesData,
                                                         zeroFilter = target == self.zeroFilterTarget)
        if writeTrainDebug:
          debugfile.writelines('{}_fourier,'.format(target)+','.join(str(d) for d in self.fourierResults[target]['predict'])+'\n')
        timeSeriesData -= self.fourierResults[target]['predict']
        if writeTrainDebug:
          debugfile.writelines('{}_nofourier,'.format(target)+','.join(str(d) for d in timeSeriesData)+'\n')
      # zero filter application
      ## find the mask for the requested target where values are nonzero
      if target == self.zeroFilterTarget:
        # artifically force signal to 0 post-fourier subtraction where it should be zero
        targetVals[:,t][self.notZeroFilterMask] = 0.0
        if writeTrainDebug:
          debugfile.writelines('{}_zerofilter,'.format(target)+','.join(str(d) for d in timeSeriesData)+'\n')


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
      if writeTrainDebug:
        debugfile.writelines('{}_normed,'.format(target)+','.join(str(d) for d in normed)+'\n')
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
        self.raiseADebug('... ... training ...')
        self.armaResult[target] = self._trainARMA(normed)
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
          zVarma = self._trainARMA(zeroed)
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

    if writeTrainDebug:
      debugfile.close()

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
                                                randEngine = self.normEngine.rvs)
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
                                            randEngine = self.normEngine.rvs)
          ## if so, then expand result into signal space (functionally, put back in all the zeros)
          signal = np.zeros(len(self.pivotParameterValues))
          signal[self.zeroFilterMask] = sample
        else:
          ## if not, no extra work to be done here!
          sample = self._generateARMASignal(result,
                                            numSamples = len(self.pivotParameterValues),
                                            randEngine = self.normEngine.rvs)
          signal = sample
      # END creating base signal
      # DEBUGG adding arbitrary variables for debugging, TODO find a more elegant way, leaving these here as markers
      #returnEvaluation[target+'_0base'] = copy.copy(signal)
      # denoise
      signal = self._denormalizeThroughCDF(signal,self.cdfParams[target])
      # DEBUGG adding arbitrary variables
      #returnEvaluation[target+'_1denorm'] = copy.copy(signal)
      #debuggFile.writelines('signal_arma,'+','.join(str(x) for x in signal)+'\n')

      # Add fourier trends
      if target in self.fourierParams:
        signal += self.fourierResults[target]['predict']
        # DEBUGG adding arbitrary variables
        #returnEvaluation[target+'_2fourier'] = copy.copy(signal)
        #debuggFile.writelines('signal_fourier,'+','.join(str(x) for x in self.fourierResults[target]['predict'])+'\n')

      # Re-zero out zero filter target's zero regions
      if target == self.zeroFilterTarget:
        # DEBUGG adding arbitrary variables
        #returnEvaluation[target+'_3zerofilter'] = copy.copy(signal)
        signal[self.notZeroFilterMask] = 0.0

      # Domain limitations
      for domain,requests in self.outTruncation.items():
        if target in requests:
          if domain == 'positive':
            signal = np.absolute(signal)
          elif domain == 'negative':
            signal = -np.absolute(signal)
        # DEBUGG adding arbitrary variables
        #returnEvaluation[target+'_4truncated'] = copy.copy(signal)

      # store results
      ## FIXME this is ASSUMING the input to ARMA is only ever a single scaling factor.
      signal *= featureVals[0]
      # DEBUGG adding arbitrary variables
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
    randomUtils.randomSeed(seed)

  ### UTILITY METHODS ###
  def _computeNumberOfBins(self,data):
    """
      Uses the Freedman-Diaconis rule for histogram binning
      -> For relatively few samples, this can cause unnatural flat-lining on low, top end of CDF
      @ In, data, np.array, data to bin
      @ Out, n, integer, number of bins
    """
    # Freedman-Diaconis
    iqr = np.percentile(data,75) - np.percentile(data,25)
    # see if we can use Freedman-Diaconis
    if iqr > 0.0:
      size = 2.0 * iqr / np.cbrt(data.size)
      # tend towards too many bins, not too few
      # also don't use less than 20 bins, it makes some pretty sketchy CDFs otherwise
      n = max(int(np.ceil((max(data) - min(data))/size)),20)
    else:
      self.raiseAWarning('While computing CDF, 25 and 75 percentile are the same number; using Root instead of Freedman-Diaconis.')
      n = max(int(np.ceil(np.sqrt(data.size))),20)
    n *= 100
    self.raiseADebug('... ... bins for ARMA empirical CDF:',n)
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

  def _trainFourier(self, pivotValues, basePeriod, order, values, zeroFilter=False):
    """
      Perform fitting of Fourier series on self.timeSeriesDatabase
      @ In, pivotValues, np.array, list of values for the independent variable (e.g. time)
      @ In, basePeriod, list, list of the base periods
      @ In, order, dict, Fourier orders to extract for each base period
      @ In, values, np.array, list of values for the dependent variable (signal to take fourier from)
      @ In, zeroFilter, bool, optional, if True then apply zero-filtering for fourier fitting
      @ Out, fourierResult, dict, results of this training in keys 'residues', 'fOrder', 'predict'
    """
    fourierSeriesOriginal = self._generateFourierSignal(pivotValues,
                                                   basePeriod,
                                                   order)
    fourierEngine = linear_model.LinearRegression()

    # if using zero-filter, cut the parts of the Fourier and values that correspond to the zero-value portions
    if zeroFilter:
      values = values[self.zeroFilterMask]
      fourierSeriesAll = dict((period,vals[self.zeroFilterMask]) for period,vals in fourierSeriesOriginal.items())
    else:
      fourierSeriesAll = fourierSeriesOriginal

    # get the combinations of fourier signal orders to consider
    temp = [range(1,order[bp]+1) for bp in order]
    fourOrders = list(itertools.product(*temp)) # generate the set of combinations of the Fourier order

    criterionBest = np.inf
    fSeriesBest = []
    fourierResult={'residues': 0,
                   'fOrder': []}

    # for all combinations of Fourier periods and orders ...
    for fOrder in fourOrders:
      # generate container for Fourier series evaluation
      fSeries = np.zeros(shape=(values.size,2*sum(fOrder)))
      # running indices for orders and sine/cosine coefficients
      indexTemp = 0
      # for each base period requested ...
      for index,bp in enumerate(order):
        # store the series values for the given periods
        fSeries[:,indexTemp:indexTemp+fOrder[index]*2] = fourierSeriesAll[bp][:,0:fOrder[index]*2]
        # update the running index
        indexTemp += fOrder[index]*2
      # find the correct magnitudes to best fit the data
      ## note in the zero-filter case, this is fitting the truncated data
      fourierEngine.fit(fSeries,values)
      # determine the (normalized) error associated with this best fit
      r = (fourierEngine.predict(fSeries)-values)**2
      if r.size > 1:
        r = sum(r)
      # TODO any reason to scale this error? values.size should be the same for every order, so all scales same
      r = r/values.size
      # TODO is anything gained by the copy and deepcopy for r?
      criterionCurrent = copy.copy(r)
      if  criterionCurrent< criterionBest:
        fourierResult['fOrder'] = copy.deepcopy(fOrder)
        fSeriesBest = copy.deepcopy(fSeries)
        fourierResult['residues'] = copy.deepcopy(r)
        criterionBest = copy.deepcopy(criterionCurrent)

    # retrain the best-fitting set of orders
    fourierEngine.fit(fSeriesBest,values)
    # produce the best-fitting signal
    fourierSignal = np.asarray(fourierEngine.predict(fSeriesBest))
    # if zero-filtered, put zeroes back into the Fourier series
    if zeroFilter:
      signal = np.zeros(pivotValues.size)
      signal[self.zeroFilterMask] = fourierSignal
    else:
      signal = fourierSignal
    fourierResult['predict'] = signal
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
    Pmax = self.Pmax
    Qmax = self.Qmax
    model = sm.tsa.VARMAX(endog=data, order=(Pmax,Qmax))
    self.raiseADebug('... ... ... fitting VARMA ...')
    results = model.fit(disp=False,maxiter=1000)
    lenHist,numVars = data.shape
    # train multivariate normal distributions using covariances, keep it around so we can control the RNG
    ## it appears "measurement" always has 0 covariance, and so is all zeros (see _generateVARMASignal)
    ## all the noise comes from the stateful properties
    stateDist = self._trainMultivariateNormal(numVars,np.zeros(numVars),model.ssm['state_cov'.encode('ascii')])
    # train initial state sampler
    ## Used to pick an initial state for the VARMA by sampling from the multivariate normal noise
    #    and using the AR and MA initial conditions.  Implemented so we can control the RNG internally.
    #    Implementation taken directly from statsmodels.tsa.statespace.kalman_filter.KalmanFilter.simulate
    ## get mean
    smoother = model.ssm
    mean = np.linalg.solve(np.eye(smoother.k_states) - smoother['transition'.encode('ascii'),:,:,0],
                           smoother['state_intercept'.encode('ascii'),:,0])
    ## get covariance
    r = smoother['selection'.encode('ascii'),:,:,0]
    q = smoother['state_cov'.encode('ascii'),:,:,0]
    selCov = r.dot(q).dot(r.T)
    cov = solve_discrete_lyapunov(smoother['transition'.encode('ascii'),:,:,0], selCov)
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

