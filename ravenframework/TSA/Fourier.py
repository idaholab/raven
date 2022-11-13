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
  Fourier time series analysis
  Note this determines the fit of desired bases, not a fast fourier transform
"""
import copy
import collections
import numpy as np
import sklearn.linear_model

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, utils
from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer


# utility methods
class Fourier(TimeSeriesGenerator, TimeSeriesCharacterizer):
  """
    Perform Fourier analysis; note this is not Fast Fourier, where all Fourier modes are used to fit a
    signal. Instead, detect the presence of specifically-requested Fourier bases.
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = ['sin',       # amplitude of sine coefficients
               'cos',       # amplitude of cosine coefficients
               'intercept'] # mean of signal

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(Fourier, cls).getInputSpecification()
    specs.name = 'fourier' # NOTE lowercase because ARMA already has Fourier and no way to resolve right now
    specs.description = r"""TimeSeriesAnalysis algorithm for determining the strength and phase of
                        specified Fourier periods within training signals. The Fourier signals take
                        the form $C\sin(\frac{2\pi}{k}+\phi)$, where $C$ is the calculated strength
                        or amplitude, $k$ is the user-specified period(s) to search for, and $\phi$
                        is the calculated phase shift. The resulting characterization and synthetic
                        history generation is deterministic given a single training signal."""
    specs.addSub(InputData.parameterInputFactory('periods', contentType=InputTypes.FloatListType,
                 descr=r"""Specifies the periods (inverse of frequencies) that should be searched
                 for within the training signal."""))
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
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['periods'] = spec.findFirst('periods').value
    return settings

  def characterize(self, signal, pivot, targets, settings, simultFit=True):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, additional settings specific to this algorithm
      @ In, simultFit, bool, optional, if False then fit Fourier individually
      @ Out, params, dict, characteristic parameters
    """
    periods = settings['periods']
    fourierSignals = self._generateBaseFourier(pivot, periods)
    # fourierSignals dimensions, for each key (base):
    #   0: length of history (aka time)
    #   1: evaluations, in order and flattened:
    #                 0:   sin(2pi*t/period[0]),
    #                 1:   cos(2pi*t/period[0]),
    #                 2:   sin(2pi*t/period[1]),
    #                 3:   cos(2pi*t/period[1]), ...
    # check collinearity
    cond = np.linalg.cond(fourierSignals) if simultFit else 30
    # fit
    params = {}
    for tg, target in enumerate(targets):
      history = signal[:, tg] # TODO need to keep in sync with SyntheticSignal ROM!
      if simultFit and cond < 30:
        print(f'Fourier fitting condition number is {cond:1.1e} for "{target}". ',
                        ' Calculating all Fourier coefficients at once.')
        fourierEngine = sklearn.linear_model.LinearRegression(normalize=False)
        fourierEngine.fit(fourierSignals, history)
        intercept = fourierEngine.intercept_
        coeffs = fourierEngine.coef_
      else:
        print(f'Fourier fitting condition number is {cond:1.1e} for "{target}"! ',
                        'Calculating iteratively instead of all at once.')
        # fourierSignals has shape (H, 2F) where H is history len and F is number of Fourier periods
        ## Fourier periods are in order from largest period to smallest, with sin then cos for each:
        ## [S0, C0, S1, C1, ..., SN, CN]
        H, F2 = fourierSignals.shape
        signalToFit = copy.deepcopy(history) # will be modified during analysis
        intercept = 0
        coeffs = np.zeros(F2) # amplitude coeffs for sine, cosine
        for fn in range(F2):
          fSignal = fourierSignals[:,fn] # Fourier base signal for this waveform
          eng = sklearn.linear_model.LinearRegression(normalize=False) # regressor
          eng.fit(fSignal.reshape(H,1), signalToFit)
          thisIntercept = eng.intercept_
          thisCoeff = eng.coef_[0]
          coeffs[fn] = thisCoeff
          intercept += thisIntercept
          # remove this signal from the signal to fit
          thisSignal = thisIntercept + thisCoeff * fSignal
          signalToFit -= thisSignal

      # get coefficient map for A*sin(ft) + B*cos(ft)
      waveCoefMap = collections.defaultdict(dict) # {period: {sin:#, cos:#}}
      for c, coef in enumerate(coeffs):
        period = periods[c//2]
        waveform = 'sin' if c % 2 == 0 else 'cos'
        waveCoefMap[period][waveform] = coef
      # convert to C*sin(ft + s)
      ## since we use fitting to get A and B, the magnitudes can be deceiving.
      ## this conversion makes "C" a useful value to know the contribution from a period
      coefMap = {}
      # TODO can we put off making and storing the signal until called on?
      # signal = np.ones(len(pivot)) * intercept
      for period, coefs in waveCoefMap.items():
        A = coefs['sin']
        B = coefs['cos']
        C, s = mathUtils.convertSinCosToSinPhase(A, B)
        coefMap[period] = {'amplitude': C, 'phase': s}
      #   signal += mathUtils.evalFourier(period, C, s, pivot)

      # store results
      params[target] = {'intercept': intercept,
                        'coeffs'   : coefMap} #,
                        # 'periods'  : self._periods}
      # END for target in targets
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
      names.append(f'{base}__fit_intercept')
      for period in settings['periods']:
        baseWave = f'{base}__period{period}'
        names.append(f'{baseWave}__amplitude')
        names.append(f'{baseWave}__phase')
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
      rlz[f'{base}__fit_intercept'] = info['intercept']
      for period in info['coeffs']:
        baseWave = f'{base}__period{period}'
        for stat, value in info['coeffs'][period].items():
          rlz[f'{baseWave}__{stat}'] = value
    return rlz


  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    synthetic = np.zeros((len(pivot), len(params)))
    for t, (target, data) in enumerate(params.items()):
      synthetic[:, t] += data['intercept']
      for period, coeffs in data['coeffs'].items():
        C = coeffs['amplitude']
        s = coeffs['phase']
        synthetic[:, t] += mathUtils.evalFourier(period, C, s, pivot)
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
      base.append(xmlUtils.newNode('fit_intercept', text=f'{float(info["intercept"]):1.9e}'))
      for period in info['coeffs']:
        periodNode = xmlUtils.newNode('waveform', attrib={'period': f'{period:1.9e}',
                                                          'frequency': f'{(1.0/period):1.9e}'})
        base.append(periodNode)
        for stat, value in sorted(list(info['coeffs'][period].items()), key=lambda x:x[0]):
          periodNode.append(xmlUtils.newNode(stat, text=f'{value:1.9e}'))

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
      if 'intercept' in requests:
        key = nameTemplate.format(target=target, metric=self.name, id='intercept')
        features[key] = info['intercept']
      if 'sin' in requests or 'cos' in requests:
        for period, data in info['coeffs'].items():
          C = data['amplitude']
          p = data['phase']
          if 'sin' in requests:
            sin = C * np.cos(p)
            key = nameTemplate.format(target=target, metric=self.name, id=f'{period}_sin')
            features[key] = sin
          if 'cos' in requests:
            cos = C * np.sin(p)
            key = nameTemplate.format(target=target, metric=self.name, id=f'{period}_cos')
            features[key] = cos
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
    # Consider restructering the "period", "sin", "cos" for faster manipulation.
    # need both sin and cos before we can set the params for amp, phase, so store here
    sincos = collections.defaultdict(dict)
    for target, identifier, value in fromCluster:
      value = float(value)
      if identifier == 'intercept':
        params[target]['intercept'] = value
      elif identifier.endswith('_sin'):
        period = float(identifier.split('_')[0])
        if period not in sincos[target]:
          sincos[target][period] = {}
        sincos[target][period]['sin'] = value
      elif identifier.endswith('_cos'):
        period = float(identifier.split('_')[0])
        if period not in sincos[target]:
          sincos[target][period] = {}
        sincos[target][period]['cos'] = value
    for target, tdata in sincos.items():
      for period, pdata in tdata.items():
        A = pdata['sin']
        B = pdata['cos']
        C, p = mathUtils.convertSinCosToSinPhase(A, B)
        if period not in params[target]['coeffs']:
          raise RuntimeError()
        params[target]['coeffs'][period] = {'amplitude': C, 'phase': p}
    return params

  #
  # Utility Methods
  #
  def _generateBaseFourier(self, pivots, periods):
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
