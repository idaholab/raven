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

from .. import Decorators

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .. import Distributions
from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer


# utility methods
class ARMA(TimeSeriesGenerator, TimeSeriesCharacterizer):
  r"""
    AutoRegressive Moving Average time series analyzer algorithm
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = ['ar',
               'ma',
               'sigma2',
               'const']

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
    specs.description = r"""characterizes the signal using Auto-Regressive and Moving Average
        coefficients to stochastically fit the training signal.
        The ARMA representation has the following form:
        \begin{equation*}
          A_t = \sum_{i=1}^P \phi_i A_{t-i} + \epsilon_t + \sum_{j=1}^Q \theta_j \epsilon_{t-j},
        \end{equation*}
        where $t$ indicates a discrete time step, $\phi$ are the signal lag (or auto-regressive)
        coefficients, $P$ is the number of signal lag terms to consider, $\epsilon$ is a random noise
        term, $\theta$ are the noise lag (or moving average) coefficients, and $Q$ is the number of
        noise lag terms to consider. The ARMA algorithms are developed in RAVEN using the
        \texttt{statsmodels} Python library."""
    specs.addParam('reduce_memory', param_type=InputTypes.BoolType, required=False,
                   descr=r"""activates a lower memory usage ARMA training. This does tend to result
                         in a slightly slower training time, at the benefit of lower memory usage. For
                         example, in one 1000-length history test, low memory reduced memory usage by 2.3
                         MiB, but increased training time by 0.4 seconds. No change in results has been
                         observed switching between modes. Note that the ARMA must be
                         retrained to change this property; it cannot be applied to serialized ARMAs.
                         """, default=False)
    specs.addSub(InputData.parameterInputFactory('SignalLag', contentType=InputTypes.IntegerType,
                 descr=r"""the number of terms in the AutoRegressive term to retain in the
                       regression; typically represented as $P$ in literature."""))
    specs.addSub(InputData.parameterInputFactory('NoiseLag', contentType=InputTypes.IntegerType,
                 descr=r"""the number of terms in the Moving Average term to retain in the
                       regression; typically represented as $Q$ in literature."""))
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
    self._minBins = 20 # this feels arbitrary; used for empirical distr. of data

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['P'] = spec.findFirst('SignalLag').value
    settings['Q'] = spec.findFirst('NoiseLag').value
    settings['reduce_memory'] = spec.parameterValues.get('reduce_memory', settings['reduce_memory'])

    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'gaussianize' not in settings:
      settings['gaussianize'] = True
    if 'engine' not in settings:
      settings['engine'] = randomUtils.newRNG()
    if 'reduce_memory' not in settings:
      settings['reduce_memory'] = False
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
    # lazy import statsmodels
    import statsmodels.api
    # settings:
    #   P: number of AR terms to use (signal lag)
    #   Q: number of MA terms to use (noise lag)
    #   gaussianize: whether to "whiten" noise before training
    # set seed for training
    seed = settings['seed']
    if seed is not None:
      randomUtils.randomSeed(seed, engine=settings['engine'], seedBoth=True)

    params = {}
    for tg, target in enumerate(targets):
      params[target] = {}
      history = signal[:, tg]
      if settings.get('gaussianize', True):
        # Transform data to obatain normal distrbuted series. See
        # J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
        # Applied Energy, 87(2010) 843-855
        # -> then train independent ARMAs
        params[target]['cdf'] = mathUtils.characterizeCDF(history, binOps=2, minBins=self._minBins)
        normed = mathUtils.gaussianize(history, params[target]['cdf'])
      else:
        normed = history
      # TODO correlation (VARMA) as well as singular -> maybe should be independent TSA algo?
      P = settings['P']
      Q = settings['Q']
      d = settings.get('d', 0)
      # TODO just use SARIMAX?
      model = statsmodels.tsa.arima.model.ARIMA(normed, order=(P, d, Q), trend='c')
      res = model.fit(low_memory=settings['reduce_memory'])
      # NOTE on low_memory use, test using SyntheticHistory.ARMA test:
      #   case    | time used (s) | memory used (MiB)
      #   low mem | 2.570851      | 0.5
      #   no arg  | 2.153929      | 2.8
      #   using low_memory, fit() takes an extra 0.4 seconds and uses 2 MB less
      # NOTE additional interesting arguments to model.fit:
      # -> method_kwargs passes arguments to scipy.optimize.fmin_l_bfgs_b() as kwargs
      #   -> disp: int, 0 or 50 or 100, in order of increasing verbosity for fit solve
      #   -> pgtol: gradient norm tolerance before quitting solve (usually not the limiter)
      #   -> factr: "factor" for exiting solve, roughly as f_new - f_old / scaling <= factr * eps
      #              default is 1e10 (loose solve), medium is 1e7, extremely tight is 1e1
      #   e.g. method_kwargs={'disp': 1, 'pgtol': 1e-9, 'factr': 10.0})
      ## get initial state distribution stats
      # taken from old statsmodels.tsa.statespace.kalman_filter.KalmanFilter.simulate
      smoother = model.ssm
      initMean = np.linalg.solve(np.eye(smoother.k_states) - smoother['transition',:,:,0], smoother['state_intercept',:,0])
      r = smoother['selection',:,:,0]
      q = smoother['state_cov',:,:,0]
      selCov = r.dot(q).dot(r.T)
      initCov = sp.linalg.solve_discrete_lyapunov(smoother['transition',:,:,0], selCov)
      initDist = {'mean': initMean, 'cov': initCov}
      params[target]['arma'] = {'const': res.params[res.param_names.index('const')], # exog/intercept/constant
                                'ar': -res.polynomial_ar[1:],     # AR
                                'ma': res.polynomial_ma[1:],     # MA
                                'var': res.params[res.param_names.index('sigma2')],  # variance
                                'initials': initDist,   # characteristics for sampling initial states
                                'model': model}
      if not settings['reduce_memory']:
        params[target]['arma']['results'] = res
    return params

  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    names = []
    for target in settings['target']:
      base = f'{self.name}__{target}'
      names.append(f'{base}__constant')
      names.append(f'{base}__variance')
      for p in range(settings['P']):
        names.append(f'{base}__AR__{p}')
      for q in range(settings['Q']):
        names.append(f'{base}__MA__{q}')
    return names

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    for target, info in params.items():
      base = f'{self.name}__{target}'
      rlz[f'{base}__constant'] = info['arma']['const']
      rlz[f'{base}__variance'] = info['arma']['var']
      for p, ar in enumerate(info['arma']['ar']):
        rlz[f'{base}__AR__{p}'] = ar
      for q, ma in enumerate(info['arma']['ma']):
        rlz[f'{base}__MA__{q}'] = ma
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
    # FIXME how to get a useful residual?
    # -> the "residual" of the ARMA is ideally white noise, not a 0 vector, even if perfectly fit
    #    so what does it mean to provide the residual from the ARMA training?
    # in order to use "predict" (in-sample forecasting) can't be in low-memory mode
    # if settings['reduce_memory']:
    #   raise RuntimeError('Cannot get residual of ARMA if in reduced memory mode!')
    # for tg, (target, data) in enumerate(params.items()):
    #   armaData = data['arma']
    #   modelParams = np.hstack([[armaData.get('const', 0)],
    #                            armaData['ar'],
    #                            armaData['ma'],
    #                            [armaData.get('var', 1)]])
    #   new = armaData['model'].predict(modelParams)

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
      armaData = data['arma']
      modelParams = np.hstack([[armaData.get('const', 0)],
                               armaData['ar'],
                               armaData['ma'],
                               [armaData.get('var', 1)]])
      msrShocks, stateShocks, initialState = self._generateNoise(armaData['model'], armaData['initials'], synthetic.shape[0])
      # measurement shocks
      # statsmodels if we don't provide them.
      # produce sample
      new = armaData['model'].simulate(modelParams,
                                       synthetic.shape[0],
                                       measurement_shocks=msrShocks,
                                       state_shocks=stateShocks,
                                       initial_state=initialState)
      if settings.get('gaussianize', True):
        # back-transform through CDF
        new = mathUtils.degaussianize(new, params[target]['cdf'])
      synthetic[:, t] = new
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
      base.append(xmlUtils.newNode('constant', text=f'{float(info["arma"]["const"]):1.9e}'))
      for p, ar in enumerate(info['arma']['ar']):
        base.append(xmlUtils.newNode(f'AR_{p}', text=f'{float(ar):1.9e}'))
      for q, ma in enumerate(info['arma']['ma']):
        base.append(xmlUtils.newNode(f'MA_{q}', text=f'{float(ma):1.9e}'))
      base.append(xmlUtils.newNode('variance', text=f'{float(info["arma"]["var"]):1.9e}'))

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
    for target, info in params.items():
      data = info['arma']
      if 'ar' in requests:
        for p, phi in enumerate(data['ar']):
          key = nameTemplate.format(target=target, metric=self.name, id=f'ar_{p}')
          features[key] = phi
      if 'ma' in requests:
        for q, theta in enumerate(data['ma']):
          key = nameTemplate.format(target=target, metric=self.name, id=f'ma_{q}')
          features[key] = theta
      if 'const' in requests:
        key = nameTemplate.format(target=target, metric=self.name, id='const')
        features[key] = data['const']
      if 'var' in requests:
        key = nameTemplate.format(target=target, metric=self.name, id='var')
        features[key] = data['var']
      # TODO mean? should always be 0
      # TODO CDF properties? might change noise a lot ...
    return features

  def setClusteringValues(self, fromCluster, params):
    """
      Interpret returned clustering settings as settings for this algorithm.
      Acts somewhat as the inverse of getClusteringValues.
      @ In, fromCluster, list(tuple), (target, identifier, values) to interpret as settings
      @ In, params, dict, trained parameter settings
      @ Out, params, dict, updated parameter settings
    """
    # TODO this needs to be fast, as it's done a lot.
    for target, identifier, value in fromCluster:
      value = float(value)
      if identifier in ['const', 'var']:
        params[target]['arma'][identifier] = value
      elif identifier.startswith('ar_'):
        index = int(identifier.split('_')[1])
        params[target]['arma']['ar'][index] = value
      elif identifier.startswith('ma_'):
        index = int(identifier.split('_')[1])
        params[target]['arma']['ma'][index] = value
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
