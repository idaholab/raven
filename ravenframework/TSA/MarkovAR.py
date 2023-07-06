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
  Markov-switching autoregressive time series analyzer algorithm
"""
import re
import numpy as np

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer


# utility methods
class MarkovAR(TimeSeriesGenerator, TimeSeriesCharacterizer):
  r"""
    AutoRegressive Moving Average time series analyzer algorithm
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = ['ar',
               'sigma2',
               'const']
  _acceptsMissingValues = True

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(MarkovAR, cls).getInputSpecification()
    specs.name = 'markovar'
    specs.description = r"""characterizes the signal using autoregressive coefficients conditioned
        on the state of a hidden Markov model (HMM) to stochastically fit the training signal.
        The AR representation has the following form:
        \begin{equation*}
          Y_t = \mu^{(S_t)} \sum_{i=1}^P \phi_i^{(S_t)} \left(Y_{t-i} - \mu^{(S_{t-i})}\right) + \epsilon_t^{(S_t)},
        \end{equation*}
        where $t$ indicates a discrete time step, $\phi$ are the signal lag (or auto-regressive)
        coefficients, $P$ is the number of signal lag terms to consider, $\epsilon$ is a random noise
        term, and $S_t$ is the HMM state at time $t$. The HMM state is determined by the transition
        probabilities between states, which are conditioned on the previous state. The transition
        probabilities are stored in a transition matrix $T$, where $T_{ij}$ is the probability of
        transitioning from state $i$ to state $j$."""
    specs.addSub(InputData.parameterInputFactory('SignalLag', contentType=InputTypes.IntegerType,
                 descr=r"""the number of terms in the AutoRegressive term to retain in the
                       regression; typically represented as $P$ in literature."""))
    specs.addSub(InputData.parameterInputFactory('MarkovStates', contentType=InputTypes.IntegerType,
                 descr=r"""the number of states in the hidden Markov model."""))
    return specs

  #
  # API Methods
  #
  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['order'] = spec.findFirst('SignalLag').value
    settings['regimes'] = spec.findFirst('MarkovStates').value
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'engine' not in settings:
      settings['engine'] = randomUtils.newRNG()
    return settings

  def fit(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict, characteristic parameters
    """
    # lazy import statsmodels
    import statsmodels.api as sm
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
      order = settings['order']
      regimes = settings['regimes']
      history = signal[:, tg]
      history = history[~np.isnan(history)]  # manually drop masked values

      # FIXME For now, we assume that all of the AR, variance, and trend terms are switching.
      # It could be good to let the user decide which terms are switching and which are not, but
      # maybe default to all switching. Setting some terms to non-switching would reduce the number
      # of parameters to estimate, which should help with fitting convergence.
      model = sm.tsa.MarkovAutoregression(endog=history,
                                          k_regimes=regimes,
                                          order=order,
                                          switching_ar=True,
                                          switching_variance=True,
                                          switching_trend=True)
      res = model.fit()

      rawParams = {k: v for k, v in zip(res.model.param_names, res.params)}
      parsedParams = self._parseParams(rawParams, order, regimes)
      # The model stores the transition probabilities in a flattened array, so we need to reshape
      # it into a matrix.
      parsedParams['transitionMatrix'] = res.regime_transition.reshape(regimes, regimes).T

      # The model fit can fail silently, resulting in a model with some NaN values. Raise an error
      # if this happens.
      for name, value in parsedParams.items():
        if np.any(np.isnan(value)):
          raise RuntimeError(f'Failed to fit MarkovAR model for target "{target}". The parameters '
                             f'"{name}" contain NaN values: {value}.')

      params[target]['MarkovAR'] = parsedParams
    return params

  def _parseParams(self, params, order, kRegimes):
    """
      Parses the raw parameter names from the statsmodels model into a more accessible format.
      @ In, params, dict, raw parameters from statsmodels model
      @ In, order, int, AR order
      @ In, kRegimes, int, number of regimes in HMM
      @ Out, parsedParams, dict, parsed parameters
    """
    # We'll use regex to parse the parameter names.
    # This pattern will extract the parameter name, and optionally the state number.
    pattern = re.compile('^([\w\.]+)(?:\[(\d+)\])?$')
    parsedParams = {'const': np.zeros(kRegimes),
                    'sigma2': np.zeros(kRegimes),
                    'ar': np.zeros((kRegimes, order))}
    for name, value in params.items():
      matches = pattern.findall(name)
      if len(matches) != 1:
        # This should happen for the transition probability parameters, which are of the form
        # p[0->0], p[0->1], p[1->0], p[1->1], etc.
        # We're saving the transition probability matrix separately, so we can ignore these.
        continue
      # All other parameters should match the pattern exactly once, yielding a list of length 1
      # containing a tuple of 2 values, corresponding to the parameter name and the state number.
      # If the matched state number is the empty string, then it's a non-switching parameter.
      paramName, state = matches[0]
      if state != '':
        state = int(state)
        isSwitching = True
      else:
        isSwitching = False
      # The parameter name may be 'sigma2', 'const', or an autoregressive term of the form 'ar.L#'.
      # If the parameter is 'sigma2' or 'const', we're done.
      # If the parameter is an autoregressive term, we need to extract the lag number and create a
      # list of autoregressive terms in the correct order.
      if '.L' in paramName:  # it's an autoregressive parameter
        lag = int(paramName.split('.L')[1])
        if isSwitching:
          parsedParams['ar'][state, lag - 1] = value
        else:
          parsedParams['ar'][:, lag - 1] = value
      else:  # either 'const' or 'sigma2'
        if isSwitching:
          parsedParams[paramName][state] = value
        else:
          parsedParams[paramName][:] = value
    return parsedParams

  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    names = []
    for target in settings['target']:
      for regime in range(settings['regimes']):
        base = f'{self.name}__{target}'
        names.append(f'{base}__constant__state{regime}')
        names.append(f'{base}__variance__state{regime}')
        for p in range(settings['order']):
          names.append(f'{base}__AR__{p}__state{regime}')
      # TODO add transition matrix info?
    return names

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    paramNames = ['const', 'sigma2', 'ar']
    for target, info in params.items():
      base = f'{self.name}__{target}'
      for name in paramNames:
        for regime, value in enumerate(info['MarkovAR'][name]):
          if name == 'ar':  # value will be an array of lagged AR terms
            for p, val in enumerate(value):
              rlz[f'{base}__{name}__{p}__state{regime}'] = val
          else:  # sigma2 or const
            rlz[f'{base}__{name}__state{regime}'] = value
      # TODO add transition matrix info?
    return rlz

  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, settings for this ROM
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    synthetic = np.zeros((len(pivot), len(params)))
    for t, (target, data) in enumerate(params.items()):
      modelData = data['MarkovAR']
      # TODO produce our own shocks?
      # msrShocks, stateShocks, initialState = self._generateNoise(armaData['model'], armaData['initials'], synthetic.shape[0])
      new, _ = self._generateSignal(synthetic.shape[0], modelData, settings)
      synthetic[:, t] = new

    return synthetic

  def _generateSignal(self, size, params, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, size, int, length of series to generate
      @ In, params, dict, parsed model parameters
      @ In, settings, dict, settings for this ROM
      @ Out, synth, np.array(float), synthetic signal
      @ Out, states, np.array(int), Markov states of the synthetic signal
    """
    burn = min(100, size)  # FIXME burn-in should probably be a function of the AR order
                           # and expected state duration. Using 100 for now, or size if size < 100.
    regimes = settings['regimes']
    order = settings['order']
    transition = params['transitionMatrix']
    consts = params['const']
    ar = params['ar']
    noiseScale = np.sqrt(params['sigma2'])

    state = 0  # initial state
    possible_states = np.arange(regimes, dtype=int)
    synth = np.zeros(size + burn)
    states = np.zeros(size + burn, dtype=int)
    for t in range(order):
      states[t] = np.random.choice(possible_states, p=transition[state])
      # Generating just a few of noise values to get things going
      synth[t] = consts[state] + np.random.normal(scale=noiseScale[state])
    for t in range(order, size + burn):
      state = np.random.choice(possible_states, p=transition[state])
      states[t] = state
      synth[t] = consts[state] + np.random.normal(scale=noiseScale[state])
      for i in range(min(t, order)):
        synth[t] += ar[state, i] * (synth[t-i-1] - consts[states[t-i-1]])

    # remove burn-in period
    synth = synth[burn:]
    states = states[burn:]

    return synth, states

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
      regimes = info['MarkovAR']['transitionMatrix'].shape[0]
      for r in range(regimes):
        regime = xmlUtils.newNode(f'State_{r}')
        base.append(regime)
        regime.append(xmlUtils.newNode('constant', text=f'{float(info["MarkovAR"]["const"][r]):1.9e}'))
        regime.append(xmlUtils.newNode('variance', text=f'{float(info["MarkovAR"]["sigma2"][r]):1.9e}'))
        for p, ar in enumerate(info['MarkovAR']['ar'][r]):
          regime.append(xmlUtils.newNode(f'AR_{p+1}', text=f'{float(ar):1.9e}'))
      for (i, j), p in np.ndenumerate(info['MarkovAR']['transitionMatrix']):
        base.append(xmlUtils.newNode(f'Transition_{i}to{j}', text=f'{float(p):1.9e}'))

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
    raise NotImplementedError('MarkovAR should not be used with clustering!')

  def setClusteringValues(self, fromCluster, params):
    """
      Interpret returned clustering settings as settings for this algorithm.
      Acts somewhat as the inverse of getClusteringValues.
      @ In, fromCluster, list(tuple), (target, identifier, values) to interpret as settings
      @ In, params, dict, trained parameter settings
      @ Out, params, dict, updated parameter settings
    """
    raise NotImplementedError('MarkovAR should not be used with clustering!')

  # # utils
  # def _generateNoise(self, model, initDict, size):
  #   """
  #     Generates purturbations for ARMA sampling.
  #     @ In, model, statsmodels.tsa.arima.model.ARIMA, trained ARIMA model
  #     @ In, initDict, dict, mean and covariance of initial sampling distribution
  #     @ In, size, int, length of time-like variable
  #     @ Out, msrShocks, np.array, measurement shocks
  #     @ Out, stateShocks, np.array, state shocks
  #     @ Out, initialState, np.array, initial random state
  #   """
  #   # measurement shocks -> these are usually near 0 but not exactly
  #   # note in statsmodels.tsa.statespace.kalman_filter, mean of measure shocks is 0s
  #   msrCov = model['obs_cov']
  #   msrShocks = randomUtils.randomMultivariateNormal(msrCov, size=size)
  #   # state shocks -> these are the significant noise terms
  #   # note in statsmodels.tsa.statespace.kalman_filter, mean of state shocks is 0s
  #   stateCov = model['state_cov']
  #   stateShocks = randomUtils.randomMultivariateNormal(stateCov, size=size)
  #   # initial state
  #   initMean = initDict['mean']
  #   initCov = initDict['cov']
  #   initialState = randomUtils.randomMultivariateNormal(initCov, size=1, mean=initMean)
  #   return msrShocks, stateShocks, initialState
