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
  Exponential (step) response: fit of a sum of exponential modes relaxing towards a steady value
"""
import itertools
import numpy as np

from ..utils import InputData, InputTypes, xmlUtils
from .TimeSeriesAnalyzer import TimeSeriesCharacterizer, TimeSeriesTransformer, TimeSeriesGenerator


class ExponentialResponse(TimeSeriesTransformer, TimeSeriesCharacterizer, TimeSeriesGenerator):
  """
    Fits the response of a (linear, multi time-constant) system to a step-like excitation:
    y(t) = y_inf + sum_i b_i exp(-(t-t0)/tau_i) for t >= t0, y(t) = y(t0) for t < t0
    with free amplitudes b_i ('exponential' model) or with the amplitudes of N first-order lags in
    series ('lagged' model)
  """
  _acceptsMissingValues = True
  _gridPoints = {1: 60, 2: 30, 3: 18} # number of log-spaced tau grid points used to initialize the search
  _nStarts = 5                        # number of best grid points used as starting points of the optimization

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(ExponentialResponse, cls).getInputSpecification()
    specs.name = 'ExponentialResponse'
    specs.description = r"""fits the response of a system to a step-like excitation as a sum of
      exponential modes relaxing towards a steady value:
      \begin{equation*}
        y(t) = y_{\infty} + \sum_{i=1}^{N} b_i \exp\left(-\frac{t-t_0}{\tau_i}\right), \quad t \geq t_0
      \end{equation*}
      with $y(t) = y(t_0) = y_{\infty} + \sum_i b_i$ for $t < t_0$. The time constants $\tau_i$
      characterize the inertia of the system (e.g. thermal inertia), $y_{\infty}$ is the extrapolated
      steady value and $b_i$ are the amplitudes of the modes. For given time constants the model is linear
      in $y_{\infty}$ and $b_i$, which are computed by least squares; the time constants are found by a
      deterministic search (log-spaced grid followed by a bounded optimization), so no seed is needed.
      The time constants are sorted in ascending order (fastest mode first) and two consecutive time
      constants must differ at least by the factor \xmlNode{minTauRatio}, which prevents degenerate
      solutions (nearly equal time constants with large amplitudes of opposite sign).
      With \xmlNode{model} equal to \xmlString{lagged}, the response is the one of $N$ first-order lags in series
      (e.g. thermal masses heated one through the other), whose step response has a zero initial slope
      (S-shaped for $N>1$):
      \begin{equation*}
        y(t) = y_{\infty} + (y_0 - y_{\infty}) \sum_{i=1}^{N} c_i \exp\left(-\frac{t-t_0}{\tau_i}\right), \quad
        c_i = \prod_{j \neq i} \frac{\tau_i}{\tau_i - \tau_j}
      \end{equation*}
      so that only $y_{\infty}$ and the initial value $y_0$ are fitted by least squares
      (the amplitudes $b_i = (y_0 - y_{\infty}) c_i$ are reported).
      The following characterization variables are produced for each target:
      \texttt{steady}, \texttt{initial}, \texttt{tau<i>}, \texttt{amplitude<i>}, \texttt{tau<i>\_atBound}
      and \texttt{rmse}. \texttt{tau<i>\_atBound} is 1 if the time constant is at (within 1\% of)
      \xmlNode{tauMin} or \xmlNode{tauMax}, 0 otherwise: a time constant at the bounds indicates that the
      corresponding mode cannot be identified from the provided signal (e.g. the signal is too short with
      respect to it)."""
    modelType = InputTypes.makeEnumType('ExpResponseModel', 'ExpResponseModelType', ['exponential', 'lagged'])
    specs.addSub(InputData.parameterInputFactory('model', contentType=modelType,
                 descr=r"""type of response: \xmlString{exponential} (independent modes with free amplitudes)
                 or \xmlString{lagged} (first-order lags in series, zero initial slope).""",
                 default='exponential'))
    specs.addSub(InputData.parameterInputFactory('nTerms', contentType=InputTypes.IntegerType,
                 descr=r"""number of exponential modes (time constants) to fit.""", default=1))
    specs.addSub(InputData.parameterInputFactory('startTime', contentType=InputTypes.FloatType,
                 descr=r"""value of the pivot parameter at which the response starts ($t_0$). Signal values
                 before it are not used for the fit. If not provided, the first pivot value is used."""))
    specs.addSub(InputData.parameterInputFactory('tauMin', contentType=InputTypes.FloatType,
                 descr=r"""lower bound for the time constants. If not provided, one tenth of the smallest
                 pivot step is used."""))
    specs.addSub(InputData.parameterInputFactory('tauMax', contentType=InputTypes.FloatType,
                 descr=r"""upper bound for the time constants. If not provided, 100 times the duration of
                 the fitted signal is used."""))
    specs.addSub(InputData.parameterInputFactory('minTauRatio', contentType=InputTypes.FloatType,
                 descr=r"""minimum ratio between two consecutive time constants (only used if
                 \xmlNode{nTerms} > 1).""", default=2.0))
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

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    for name in ['model', 'nTerms', 'startTime', 'tauMin', 'tauMax', 'minTauRatio']:
      node = spec.findFirst(name)
      if node is not None:
        settings[name] = node.value
    settings = self.setDefaults(settings)
    if settings['nTerms'] < 1:
      raise IOError(f'{self.name}: <nTerms> must be greater than 0. Got {settings["nTerms"]}!')
    if settings['minTauRatio'] <= 1.0:
      raise IOError(f'{self.name}: <minTauRatio> must be greater than 1. Got {settings["minTauRatio"]}!')
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    settings.setdefault('model', 'exponential')
    settings.setdefault('nTerms', 1)
    settings.setdefault('startTime', None)
    settings.setdefault('tauMin', None)
    settings.setdefault('tauMax', None)
    settings.setdefault('minTauRatio', 2.0)
    return settings

  def fit(self, signal, pivot, targets, settings, trainedParams=None):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, additional settings specific to this algorithm
      @ In, trainedParams, dict, running dict of trained algorithm params
      @ Out, params, dict, characteristic parameters
    """
    pivot = np.asarray(pivot, dtype=float)
    t0 = pivot[0] if settings['startTime'] is None else settings['startTime']
    params = {}
    for tg, target in enumerate(targets):
      mask = np.isfinite(signal[:, tg]) & (pivot >= t0)
      if mask.sum() < 2 * settings['nTerms'] + 2:
        raise ValueError(f'{self.name}: not enough valid points after startTime to fit {settings["nTerms"]} '
                         f'exponential term(s) for target "{target}"!')
      t = pivot[mask] - t0
      y = signal[mask, tg]
      tauMin, tauMax = self._tauBounds(t, settings)
      lagged = settings['model'] == 'lagged'
      taus, amplitudes, steady, sse = self._fitExponentials(t, y, settings['nTerms'], tauMin, tauMax,
                                                            settings['minTauRatio'], lagged)
      atBound = (taus <= tauMin * 1.01) | (taus >= tauMax / 1.01)
      params[target] = {'model': {'startTime': t0,
                                  'steady': steady,
                                  'initial': steady + np.sum(amplitudes),
                                  'taus': taus,
                                  'amplitudes': amplitudes,
                                  'atBound': atBound.astype(float),
                                  'rmse': np.sqrt(sse / len(y))}}
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
      names.extend([f'{base}__steady', f'{base}__initial'])
      for i in range(settings['nTerms']):
        names.extend([f'{base}__tau{i+1}', f'{base}__amplitude{i+1}', f'{base}__tau{i+1}_atBound'])
      names.append(f'{base}__rmse')
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
      model = info['model']
      rlz[f'{base}__steady'] = model['steady']
      rlz[f'{base}__initial'] = model['initial']
      for i, (tau, amp, bound) in enumerate(zip(model['taus'], model['amplitudes'], model['atBound'])):
        rlz[f'{base}__tau{i+1}'] = tau
        rlz[f'{base}__amplitude{i+1}'] = amp
        rlz[f'{base}__tau{i+1}_atBound'] = bound
      rlz[f'{base}__rmse'] = model['rmse']
    return rlz

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
    return initial - self.generate(params, pivot, settings)

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
    return initial + self.generate(params, pivot, settings)

  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, synthetic, np.array(float), synthetic estimated model signal
    """
    pivot = np.asarray(pivot, dtype=float)
    synthetic = np.zeros((len(pivot), len(params)))
    for tg, (_, info) in enumerate(params.items()):
      model = info['model']
      t = np.maximum(pivot - model['startTime'], 0.0)
      synthetic[:, tg] = model['steady'] + self._modes(t, model['taus']) @ model['amplitudes']
    return synthetic

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.characterize
      @ Out, None
    """
    for target, info in params.items():
      model = info['model']
      base = xmlUtils.newNode(target)
      writeTo.append(base)
      for name in ['startTime', 'steady', 'initial']:
        base.append(xmlUtils.newNode(name, text=f'{float(model[name]):1.9e}'))
      for i, (tau, amp, bound) in enumerate(zip(model['taus'], model['amplitudes'], model['atBound'])):
        base.append(xmlUtils.newNode(f'tau{i+1}', text=f'{float(tau):1.9e}'))
        base.append(xmlUtils.newNode(f'amplitude{i+1}', text=f'{float(amp):1.9e}'))
        base.append(xmlUtils.newNode(f'tau{i+1}_atBound', text=f'{int(bound)}'))
      base.append(xmlUtils.newNode('rmse', text=f'{float(model["rmse"]):1.9e}'))

  #
  # Utility Methods
  #
  @staticmethod
  def _modes(t, taus):
    """
      Evaluates the exponential modes
      @ In, t, np.array, time since the start of the response (>= 0)
      @ In, taus, np.array, time constants
      @ Out, modes, np.array, shaped [len(t), len(taus)], exp(-t/tau_i)
    """
    return np.exp(-np.outer(t, 1.0 / np.asarray(taus, dtype=float)))

  @staticmethod
  def _tauBounds(t, settings):
    """
      Computes the bounds of the time constants
      @ In, t, np.array, time since the start of the response
      @ In, settings, dict, algorithm settings
      @ Out, (tauMin, tauMax), tuple(float, float), bounds
    """
    steps = np.diff(np.unique(t))
    tauMin = settings['tauMin'] if settings['tauMin'] is not None else 0.1 * steps.min()
    tauMax = settings['tauMax'] if settings['tauMax'] is not None else 100.0 * (t.max() - t.min())
    if not 0.0 < tauMin < tauMax:
      raise IOError(f'ExponentialResponse: invalid time constant bounds tauMin={tauMin}, tauMax={tauMax}!')
    return tauMin, tauMax

  @staticmethod
  def _lagWeights(taus):
    """
      Weights of the exponential modes in the step response of first-order lags in series
      @ In, taus, np.array, distinct time constants
      @ Out, weights, np.array, c_i = prod_{j != i} tau_i / (tau_i - tau_j) (they sum to 1)
    """
    taus = np.asarray(taus, dtype=float)
    return np.array([np.prod([ti / (ti - tj) for j, tj in enumerate(taus) if j != i]) for i, ti in enumerate(taus)])

  def _linearSolve(self, t, y, taus, lagged=False):
    """
      Least-squares solution of steady value and amplitudes for given time constants
      @ In, t, np.array, time since the start of the response
      @ In, y, np.array, signal values
      @ In, taus, np.array, time constants
      @ In, lagged, bool, optional, if True the amplitudes are the ones of first-order lags in series
      @ Out, (steady, amplitudes, sse), tuple(float, np.array, float), steady value, amplitudes and sum of
             squared errors
    """
    modes = self._modes(t, taus)
    if lagged:
      weights = self._lagWeights(taus)
      basis = np.column_stack([np.ones_like(t), modes @ weights])
    else:
      basis = np.column_stack([np.ones_like(t), modes])
    coefs = np.linalg.lstsq(basis, y, rcond=None)[0]
    sse = float(np.sum((basis @ coefs - y)**2))
    amplitudes = coefs[1] * weights if lagged else coefs[1:]
    return coefs[0], amplitudes, sse

  def _fitExponentials(self, t, y, nTerms, tauMin, tauMax, minRatio, lagged=False):
    """
      Finds the time constants minimizing the squared error (variable projection)
      @ In, t, np.array, time since the start of the response
      @ In, y, np.array, signal values
      @ In, nTerms, int, number of exponential modes
      @ In, tauMin, float, lower bound for the time constants
      @ In, tauMax, float, upper bound for the time constants
      @ In, minRatio, float, minimum ratio between two consecutive time constants
      @ In, lagged, bool, optional, if True the response of first-order lags in series is fitted
      @ Out, (taus, amplitudes, steady, sse), tuple(np.array, np.array, float, float), sorted time constants,
             amplitudes, steady value and sum of squared errors
    """
    from scipy.optimize import minimize
    logMin, logMax, logRatio = np.log(tauMin), np.log(tauMax), np.log(minRatio)
    if nTerms > 1 and logMin + (nTerms - 1) * logRatio > logMax:
      raise IOError(f'{self.name}: the range [tauMin, tauMax] is too narrow for {nTerms} time constants '
                    f'separated by a factor {minRatio}!')
    # parametrization: log(tau_1) and the log-increments between consecutive time constants (>= log(minRatio))
    # (time constants beyond tauMax are clipped, keeping the minimum ratio, so that the objective stays finite)
    caps = logMax - logRatio * np.arange(nTerms - 1, -1, -1)
    toLogTaus = lambda x: np.minimum(np.cumsum(x), caps)
    objective = lambda x: self._linearSolve(t, y, np.exp(toLogTaus(x)), lagged)[2]
    # deterministic initialization on a log-spaced grid of time constants respecting the minimum ratio
    grid = np.linspace(logMin, logMax, self._gridPoints.get(nTerms, 10))
    starts = []
    for combo in itertools.combinations(grid, nTerms):
      x = np.concatenate([[combo[0]], np.diff(combo)])
      if np.all(x[1:] >= logRatio - 1e-12):
        starts.append((objective(x), x))
    starts.sort(key=lambda start: start[0])
    bounds = [(logMin, caps[0])] + [(logRatio, logMax - logMin)] * (nTerms - 1)
    bestSse, bestX = starts[0]
    for _, x0 in starts[:self._nStarts]:
      result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
      if np.isfinite(result.fun) and result.fun < bestSse:
        bestSse, bestX = result.fun, result.x
    taus = np.exp(toLogTaus(bestX))
    steady, amplitudes, sse = self._linearSolve(t, y, taus, lagged)
    return taus, amplitudes, steady, sse
