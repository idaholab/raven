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
  AutoRegressive Moving Average time series analysis
"""
import copy
import collections
import numpy as np

from utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, utils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

import Distributions
from .TimeSeriesAnalyzer import TimeSeriesAnalyzer


# utility methods
class ARMA(TimeSeriesAnalyzer):
  r"""
    AutoRegressive Moving Average time series analyzer algorithm
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = [] # TODO

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(ARMA, cls).getInputSpecification()
    specs.name = 'arma' # NOTE lowercase because ARMA already has Fourier and no way to resolve right now
    specs.description = r"""TimeSeriesAnalysis algorithm for determining the stochastic
                            characteristics of signal with time-invariant variance"""
    specs.addSub(InputData.parameterInputFactory('SignalLag', contentType=InputTypes.FloatListType,
                 descr=r"""the number of terms in the AutoRegressive term to retain in the
                           regression; "P" in literature."""))
    specs.addSub(InputData.parameterInputFactory('NoiseLag', contentType=InputTypes.FloatListType,
                 descr=r"""the number of terms in the Moving Average term to retain in the
                           regression; "Q" in literature."""))
    return specs

  #
  # API Methods
  #
  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    # general infrastructure
    TimeSeriesAnalyzer.__init__(self, *args, **kwargs)
    self._P = None # number of AR terms
    self._Q = None # number of MA terms
    self._d = 0    # TODO differences option
    self._minBins = 20
    self._gaussianize = True # TODO user option?
    # normalization engine
    self.normEngine = Distributions.returnInstance('Normal', self)
    self.normEngine.mean = 0.0
    self.normEngine.sigma = 1.0
    self.normEngine.upperBoundUsed = False
    self.normEngine.lowerBoundUsed = False
    self.normEngine.initializeDistribution()

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, None
    """
    TimeSeriesAnalyzer.handleInput(self, spec)
    self._P = spec.findFirst('SignalLag').value
    self._Q = spec.findFirst('NoiseLag').value

  def characterize(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict, characteristic parameters
    """
    # settings:
    # P: number of AR terms to use (signal lag)
    # Q: number of MA terms to use (noise lag)
    # gaussianize: whether to "whiten" noise before training
    params = {}
    # Transform data to obatain normal distrbuted series. See
    # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
    # Applied Energy, 87(2010) 843-855
    # -> then train independent ARMAs
    for tg, target in enumerate(targets):
      params[target] = {}
      history = signal[:, tg]
      if settings['gaussianize']:
        params[target]['cdf'] = mathUtils.characterizeCDF(history, binOps=2, minBins=self._minBins)
        normed = mathUtils.gaussianize(history, params[target]['cdf'])
      else:
        normed = history
      # TODO correlation (VARMA) as well as singular
      P = settings['P']
      Q = settings['Q']
      d = settings.get('d', 0)
      # lazy import statsmodels
      import statsmodels.api
      model = statsmodels.tsa.arima.model.ARIMA(
          normed,
          order=(P, d, Q))
      # TODO
      # model = statsmodels.tsa.statespace.sarimax.SARIMAX(
      #     normed,
      #     order=(P, d, Q),
      #     seasonal_order = (0, 0, 0, 0))
      res = model.fit(low_memory=True)
      # NOTE additional interesting arguments to model.fit:
      # -> method_kwargs passes arguments to scipy.optimize.fmin_l_bfgs_b() as kwargs
      #   -> disp: int, 0 or 50 or 100, in order of increasing verbosity for fit solve
      #   -> pgtol: gradient norm tolerance before quitting solve (usually not the limiter)
      #   -> factr: "factor" for exiting solve, roughly as f_new - f_old / scaling <= factr * eps
      #              default is 1e10 (loose solve), medium is 1e7, extremely tight is 1e1
      #   e.g. method_kwargs={'disp': 1, 'pgtol': 1e-9, 'factr': 10.0})
      params[target]['arma'] = {'const': res.params[0], # exog/intercept/constant
                                'ar': res.arparams,
                                'ma': res.maparams,
                                'var': res.params[-1], # variance
                                'model': model}
    return params

  def generate(self, params, pivot, randEngine):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    synthetic = np.zeros((len(pivot), len(params)))
    for t, (target, data) in enumerate(params.items()):
      armaData = data['arma']
      modelParams = np.hstack([[armaData.get('const', 0)],
                               armaData['ar'],
                               armaData['ma'],
                               [armaData.get('var', 1)]])
      # TODO back-transform through CDF!
      synthetic[:, t] = armaData['model'].simulate(modelParams, synthetic.shape[0])
    return synthetic

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.characterize
      @ Out, None
    """
    raise NotImplementedError
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)
      base.append(xmlUtils.newNode('fit_intercept', text=f'{float(info["intercept"]):1.9e}'))
      for period in info['coeffs']:
        periodNode = xmlUtils.newNode('waveform', attrib={'period': f'{period:1.9e}',
                                                          'frequency': f'{(1.0/period):1.9e}'})
        base.append(periodNode)
        for stat, value in sorted(list(info['coeffs'][period].items()), key=lambda x:x[0]):
          periodNode.append(xmlUtils.newNode(stat, text=f'{value:1.9e}'))

  #
  # Utility Methods
  #
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

  def _sampleICDF(self, x, params):
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

  def _trainCDF(self, data, binOps=2):
    """
      Constructs a CDF from the given data
      @ In, data, np.array(float), values to fit to
      @ In, binOps, int, optional, determines algorithm to use for binning (2 is sqrt)
      @ Out, params, dict, essential parameters for CDF
    """
    bins, _ = mathUtils.numBinsDraconis(data, low=self._minBins, alternateOkay=True, binOps=binOps)
    counts, edges = np.histogram(data, bins=bins, density=False)
    counts = np.array(counts) / float(len(data))
    cdf = np.cumsum(counts)
    ## from Jun implementation: add min of CDF as 0 for ... numerical issues?
    cdf = np.insert(cdf, 0, 0)
    params = {'bins': edges,
              'counts': counts,
              'pdf': counts * bins,
              'cdf': cdf,
              'lens': len(data)}
    return params
