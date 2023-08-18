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

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesTransformer


# utility methods
class MarkovAR(TimeSeriesGenerator, TimeSeriesTransformer):
  r"""
    AutoRegressive Moving Average time series analyzer algorithm
  """
  # class attribute
  _acceptsMissingValues = False

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(MarkovAR, cls).getInputSpecification()
    specs.name = 'MarkovAR'
    specs.description = r"""characterizes the signal using autoregressive (AR) coefficients conditioned
        on the state of a hidden Markov model (HMM) to stochastically fit the training signal.
        The Markov-switching autoregressive model (MSAR) has the following form:
        \begin{equation*}
          Y_t = \mu_{S_t} \sum_{i=1}^p \phi_{i,{S_t}} \left(Y_{t-i} - \mu_{S_{t-i}}\right) + \varepsilon_{t,{S_t}},
        \end{equation*}
        where $t$ indicates a discrete time step, $\phi$ are the signal lag (or auto-regressive)
        coefficients, $p$ is the number of signal lag terms to consider, $\varepsilon$ is a random noise
        term with mean 0 and variance $\sigma^2_{S_t}$, and $S_t$ is the HMM state at time $t$.
        The HMM state is determined by the transition probabilities between states, which are conditioned
        on the previous state. The transition probabilities are stored in a transition matrix $P$,
        where entry $p_{ij}$ is the probability of transitioning from state $i$ to state $j$ conditional
        on being in state $i$. For a MSAR model with HMM state dimensionality $r$, the transition matrix
        $P$ is of size $r \times r$. Each of the mean, autoregressive, and noise variance terms may be
        switching or non-switching parameters."""
    specs.addParam('switching_ar', param_type=InputTypes.BoolType, required=False, default=True,
                  descr=r"""indicates whether the autoregressive coefficients are switching parameters.""")
    specs.addParam('switching_variance', param_type=InputTypes.BoolType, required=False, default=True,
                  descr=r"""indicates whether the noise variance is a switching parameter.""")
    specs.addParam('switching_trend', param_type=InputTypes.BoolType, required=False, default=True,
                  descr=r"""indicates whether the mean is a switching parameter.""")
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
    settings['switching_ar'] = spec.parameterValues.get('switching_ar', True)
    settings['switching_variance'] = spec.parameterValues.get('switching_variance', True)
    settings['switching_trend'] = spec.parameterValues.get('switching_trend', True)
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
    # set seed for training
    seed = settings.get('seed')
    if seed is not None:
      randomUtils.randomSeed(seed, engine=settings['engine'], seedBoth=True)

    params = {}
    for tg, target in enumerate(targets):
      params[target] = {}
      order = settings['order']
      regimes = settings['regimes']
      history = signal[:, tg]

      # We default to having the AR, variance, and trend terms as switching parameters. Users can set
      # which terms are switching parameters using the input settings. If there is reason to believe
      # that some terms should not be switching parameters, then the user should set them as such
      # as this simplifies the model that needs to be fit and should improve convergence and estimation.
      # However, using all switching parameters is the most general case, so we default to that.
      model = sm.tsa.MarkovAutoregression(endog=history,
                                          k_regimes=regimes,
                                          order=order,
                                          switching_ar=settings['switching_ar'],
                                          switching_variance=settings['switching_variance'],
                                          switching_trend=settings['switching_trend'])
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
      # The model fit object is stored only to get the residuals. Is there a better way to hold on
      # to those values?
      params[target]['MarkovAR']['model'] = res
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
      # There's no immediate use for the state information, so we'll just discard it for now.
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
    regimes = settings['regimes']
    order = settings['order']
    transition = params['transitionMatrix']
    consts = params['const']
    ar = params['ar']
    noiseScale = np.sqrt(params['sigma2'])

    engine = settings['engine']
    rngSeed = settings.get('seed', None)
    if rngSeed is not None:
      engine.seed(rngSeed)

    # Burn-in when generating with a MSAR model accomplished two things. (1) It gives the Markov model
    # time to reach a high-probability state, and (2) it helps separate the AR signal from its initial
    # conditions, which do not follow the AR model. We can handle (1) by starting in the state with
    # the greatest expected duration, which is also the state with the greatest probability of
    # remaining in the state, which is the state with the greatest diagonal element in the transition
    # matrix.
    initState = np.diag(transition).argmax()
    # We'll use a burn-in period to address (2). In ravenframework/SupervisedLearning/ARMA.py,
    # the burn-in period is set to twice the maximum AR or MA order. We'll do the same here.
    burn = 2 * order
    states = np.zeros(size + burn, dtype=int)
    states[0] = initState
    # To transition from one state to another, we need to perform a weighted random choice according
    # to the transition probabilities. We can do this by creating a cumulative distribution function
    # for the transition probabilities of each state (cumulative sum of transition probabilities) and
    # using numpy's searchsorted function to map a uniform random number from U(0, 1) to a state number.
    transCDFs = np.cumsum(transition, axis=1)
    rands = engine.uniform(size=size+burn-1)
    for t in range(1, size + burn):
      states[t] = np.searchsorted(transCDFs[states[t - 1]], rands[t - 1])

    # Start building up the synthetic signal, starting with the noise
    synth = engine.normal(loc=0., scale=noiseScale[states], size=size+burn)

    # Build up the AR terms one time step at a time
    for t in range(order, size + burn):
      synth[t] += np.sum(ar[states[t]][::-1] * synth[t-order:t])

    # Finally add in the state constants
    synth += consts[states]

    # There's a chance that the model that has been fit is not stationary, leading to potential divergence
    # of the synthetic signal. We can check for this by looking for non-finite values in the synthetic
    # signal.
    if not np.all(np.isfinite(synth)):
      raise RuntimeError('Synthetic signal contains non-finite values!')

    # Remove burn-in period
    synth = synth[burn:]
    states = states[burn:]

    return synth, states

  def getResidual(self, initial, params, pivot, settings):
    """
      Removes trained signal from data and find residual
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
    """
    residual = initial.copy()
    for tg, (target, data) in enumerate(params.items()):
      # The model fit results contain the fit residuals, so we can just use those.
      # The length of the residuals array is shorter than the length of the original signal by the
      # AR order (not really sure why), so we need to pad the end of the residual array with zeros.
      modelResid = data['MarkovAR']['model'].resid
      residual[:, tg] = np.r_[modelResid, np.zeros(len(residual) - len(modelResid))]
    return residual

  def getComposite(self, initial, params, pivot, settings):
    """
      Combines two component signals to form a composite signal. This is essentially the inverse
      operation of the getResidual method.
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, composite, np.array, resulting composite signal
    """
    synthetic = initial + self.generate(params, pivot, settings)
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
