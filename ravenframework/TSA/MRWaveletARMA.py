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
import scipy as sp
import pandas as pd

from .. import Decorators

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .. import Distributions
from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer
from .ARMA import ARMA
from .Wavelet import Wavelet


# utility methods
class MRWaveletARMA(TimeSeriesGenerator, TimeSeriesCharacterizer):
  r"""
    AutoRegressive Moving Average time series analyzer algorithm
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = []

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(MRWaveletARMA, cls).getInputSpecification()
    specs.name = 'MRWaveletARMA' # NOTE lowercase because ARMA already has Fourier and no way to resolve right now
    specs.addSub(InputData.parameterInputFactory('Levels', contentType=InputTypes.IntegerType,
              descr=r"""the number of wavelet decomposition levels for our signal. If level is 0,
                    it doesn't """))
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
    super().__init__(*args, **kwargs)
    self._nlevels = 0

    self.SFparams = {
        "seasonal": False, # set to True if you want a SARIMA model
        "start_p": 3,
        "start_q": 3,
        "start_P": 1,
        "start_Q": 1,
        "max_p": 4,
        "max_q": 4,
        "max_P": 4,
        "max_Q": 4,
        "max_d": 2,
        "max_D": 2,
        "max_order": 5,
        "num_cores": 3,
        "ic": 'bic',
        "test": 'kpss',
      }

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    self._nlevels = spec.findFirst('Levels').value

    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    return settings

  def characterize(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict, characteristic parameters
    """
    # TODO extend to continuous wavelet transform
    try:
      from darts import models as dmodels
      from darts import TimeSeries
    except ModuleNotFoundError as exc:
      print("This RAVEN TSA Module requires the DARTS library to be installed in the current python environment")
      raise ModuleNotFoundError from exc

    SFAA = dmodels.StatsForecastAutoARIMA(**self.SFparams)

    params = {}
    for i, target in enumerate(targets):
      print(f"Performing Wavelet Decomposition on Signal {target}")
      params[target] = {}

      wavelet_settings = {}
      wavelet_settings['family'] = 'db8'
      wavelet_settings['levels'] = self._nlevels

      Wavelet_obj = Wavelet()
      tmp_signal = np.expand_dims(signal[:,i], axis=1)
      wavelet_res = Wavelet_obj.characterize(tmp_signal, pivot, [target], wavelet_settings)

      # TODO: all of this could be within the Wavelet characterize?
      coeff_a  = wavelet_res[target]['results']['coeff_a']
      coeffs_d = wavelet_res[target]['results']['coeff_d']
      params[target]['wavelets'] = wavelet_res[target]['results']
      params[target]['full'] = signal[:,i]
      params[target]['trend'] = coeff_a
      params[target]['residual'] = params[target]['full'] - params[target]['trend']

      # FIXME: temporary
      settings['gaussianize'] = True
      params[target]['decomposition_levels'] = {}

      for lvl, decomp in enumerate(coeffs_d):
        print(f"... Creating ARMA for Lvl {lvl} of {len(coeffs_d)}")
        wavelet = f'Wavelet_{lvl}'

        params[target][wavelet] = {}

        if settings.get('gaussianize', True):
          # Transform data to obatain normal distrbuted series. See
          # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
          # Applied Energy, 87(2010) 843-855
          # -> then train independent ARMAs
          params[target][wavelet]['cdf'] = mathUtils.characterizeCDF(decomp, binOps=2, minBins=20) # FIXME: minBins
          tmp_decomp = mathUtils.gaussianize(decomp, params[target][wavelet]['cdf'])
        else:
          tmp_decomp = decomp


        dataframe = pd.DataFrame()
        dataframe['Target'] = tmp_decomp
        dataframe['Pivot'] = np.arange(len(pivot))
        timeSeries = TimeSeries.from_dataframe(dataframe, time_col='Pivot')

        fittedARIMA = SFAA.fit(timeSeries,)
        ARIMA = fittedARIMA.model.model_['arma']
        ARIMA_order = tuple( ARIMA[i] for i in [0, 5, 1, 2, 6, 3, 4] )

        arma_settings = {}
        arma_settings['P'] = ARIMA_order[0]
        arma_settings['d'] = ARIMA_order[1]
        arma_settings['Q'] = ARIMA_order[2]
        arma_settings['reduce_memory'] = False
        arma_settings['seed'] = None
        arma_settings['gaussianize'] = settings['gaussianize']

        ARMA_obj = ARMA()
        tmp_decomp = np.expand_dims(tmp_decomp, axis=1)
        arma_params = ARMA_obj.characterize(tmp_decomp, pivot, [wavelet], arma_settings)

        #TODO: getting some errors here...
        # "ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals"
        # "UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters"
        #   they are just warnings, so gonne trek further
        params[target]['decomposition_levels'][wavelet] = arma_params
    return params

  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    names = []
    return names

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    return rlz

  def getResidual(self, initial, params, pivot, settings):
    """
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
    """
    raise NotImplementedError('ARMA cannot provide a residual yet; it must be the last TSA used!')

  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, settings for this ROM
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    # TODO if there's not a model, generate one?
    # -> I think we need the training signal though ...
    synthetic = np.zeros((len(pivot), len(params)))
    for t, (target, data) in enumerate(params.items()):

      # first add back the zero-th order wavelet used as the trend
      synthetic[:,t] += data['trend']

      # next generate a new synthetic time series for each decomp level
      for w, (lvl, subdata) in enumerate(data['decomposition_levels'].items()):
        ARMA_obj = ARMA()
        decomp_synthetic = ARMA_obj.generate(subdata, pivot, settings)

        synthetic[:,t] += decomp_synthetic[:,0]

    return synthetic

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.characterize
      @ Out, None
    """
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)

  # clustering
  def getClusteringValues(self, nameTemplate: str, requests: list, params: dict) -> dict:
    """
      Provide the characteristic parameters of this ROM for clustering with other ROMs
      @ In, nameTemplate, str, formatting string template for clusterable params (target, metric id)
      @ In, requests, list, list of requested attributes from this ROM
      @ In, params, dict, parameters from training this ROM
      @ Out, features, dict, params as {paramName: value}
    """
    # nameTemplate convention:
    # -> target is the trained variable (e.g. Signal, Temperature)
    # -> metric is the algorithm used (e.g. Fourier, ARMA)
    # -> id is the subspecific characteristic ID (e.g. sin, AR_0)
    features = {}
    return features

  def setClusteringValues(self, fromCluster, params):
    """
      Interpret returned clustering settings as settings for this algorithm.
      Acts somewhat as the inverse of getClusteringValues.
      @ In, fromCluster, list(tuple), (target, identifier, values) to interpret as settings
      @ In, params, dict, trained parameter settings
      @ Out, params, dict, updated parameter settings
    """
    return params

  # utils
  def _generateNoise(self, model, initDict, size):
    """
      Generates purturbations for ARMA sampling.
      @ In, model, statsmodels.tsa.arima.model.ARIMA, trained ARIMA model
      @ In, initDict, dict, mean and covariance of initial sampling distribution
      @ In, size, int, length of time-like variable
      @ Out, msrShocks, np.array, measurement shocks
      @ Out, stateShocks, np.array, state shocks
      @ Out, initialState, np.array, initial random state
    """
    # measurement shocks -> these are usually near 0 but not exactly
    # note in statsmodels.tsa.statespace.kalman_filter, mean of measure shocks is 0s
    msrCov = model['obs_cov']
    msrShocks = randomUtils.randomMultivariateNormal(msrCov, size=size)
    # state shocks -> these are the significant noise terms
    # note in statsmodels.tsa.statespace.kalman_filter, mean of state shocks is 0s
    stateCov = model['state_cov']
    stateShocks = randomUtils.randomMultivariateNormal(stateCov, size=size)
    # initial state
    initMean = initDict['mean']
    initCov = initDict['cov']
    initialState = randomUtils.randomMultivariateNormal(initCov, size=1, mean=initMean)
    return msrShocks, stateShocks, initialState
