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
from .TimeSeriesAnalyzer import TimeSeriesCharacterizer


# utility methods
class AutoARMA(TimeSeriesCharacterizer):
  r"""
    AutoARMA time series characterizer algorithm
  """
  # class attribute
  _features = []
  ## Currently, AutoARMA is only a global algorithm. Applying it per segment is fine, but clusters
  ##   cause issues. Nominally (w/o auto-selection, ARMA models are trained per segment which means that
  ##   each segment model has the same number of coefficients but with different values. *With* auto
  ##   selection, each segment can have different number of coefficients if the algorithm chooses
  ##   different values of p and q for the segments. This can be mitigated by padding the coefficient
  ##   arrays according to largets p or q value; this works for Segmented ROMs. However, if we cluster
  ##   based on the coefficients then this causes issues if the coefficient arrays are padded to be
  ##   the same size. The zero or NaN values affect the resultant clusters and lead to bad synthetic
  ##   histories. Until a general solution is found, AutoARMA will remain a global algorithm
  ##   (i.e., not allowed if clustering).
  _canBeAppliedPerCluster = False

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(AutoARMA, cls).getInputSpecification()
    specs.name = 'autoarma'
    specs.description = r"""characterizes the signal \textit{before} using Auto-Regressive and
        Moving Average coefficients to stochastically fit the training signal. AutoARMA
        automatically determines the number of coefficients to use in the regression.
        The ARMA representation has the following form:
        \begin{equation*}
          A_t = \sum_{i=1}^P \phi_i A_{t-i} + \epsilon_t + \sum_{j=1}^Q \theta_j \epsilon_{t-j},
        \end{equation*}
        where $t$ indicates a discrete time step, $\phi$ are the signal lag (or auto-regressive)
        coefficients, $P$ is the number of signal lag terms to consider, $\epsilon$ is a random noise
        term, $\theta$ are the noise lag (or moving average) coefficients, and $Q$ is the number of
        noise lag terms to consider. The AutoARMA algorithm determines the optimal value of $P$
        and $Q$ for the each signal. The AutoARMA algorithms are developed in RAVEN using the
        \texttt{statsforecast} Python library."""
    specs.addSub(InputData.parameterInputFactory('P_upper', contentType=InputTypes.IntegerType,
                 descr=r"""upper bound for the number of terms in the AutoRegressive term to retain
                       in the regression; typically represented as $P$ or Signal Lag in
                       literature."""))
    specs.addSub(InputData.parameterInputFactory('Q_upper', contentType=InputTypes.IntegerType,
                 descr=r"""upper bound for the number of terms in the Moving Average term to retain
                       in the regression; typically represented as $Q$ in Noise Lag
                       literature."""))
    opt_criterion = InputTypes.makeEnumType("criterion", "criterionType", ['aic', 'aicc','bic'])
    specs.addSub(InputData.parameterInputFactory('criterion', contentType=opt_criterion,
                 descr=r"""information criterion used to determine optimal ARMA parameters. The
                 options are `aic` for Akaike Information Criterion, `aicc` for corrected AIC which
                 is used when number of observations is small, and `bic` for Bayesian Information
                 Criterion. Default is `aicc`."""))
    specs.addSub(InputData.parameterInputFactory('use_approximation', contentType=InputTypes.BoolType,
                 descr=r"""if True, this uses the default version of the AutoARIMA algorithm
                 within `statsforecast` which uses heuristics to find an approximate solution
                 in much faster time. This previously led to different answers between Linux and
                 Windows, but may be a good option if the alternative is taking too long.
                 Default is False."""))
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
    # maximum value that P+Q can have as an upper bound
    self._maxCombinedPQ = 5
    self._use_approximation = False
    self._ic = 'aicc'

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)

    for sub in spec.subparts:
      settings[sub.name] = sub.value
    # AutoARMA is currently only a global TSA algorithm; this is enforced in TSAUser

    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'global' not in settings:
      settings['global'] = True
    if 'engine' not in settings:
      settings['engine'] = randomUtils.newRNG()
    if 'P_upper' not in settings:
      settings['P_upper'] = self._maxCombinedPQ
    if 'Q_upper' not in settings:
      settings['Q_upper'] = self._maxCombinedPQ
    if 'criterion' not in settings:
      settings['criterion'] = self._ic
    if 'use_approximation' not in settings:
      settings['use_approximation'] = self._use_approximation
    return settings

  def fit(self, signal, pivot, targets, settings, trainedParams=None):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ In, trainedParams, dict, running dict of trained algorithm params
      @ Out, params, dict, characteristic parameters
    """
    # set seed for training
    seed = settings['seed']
    if seed is not None:
      randomUtils.randomSeed(seed, engine=settings['engine'])

    try:
      from statsforecast.models import AutoARIMA
      from statsforecast.arima import arima_string
      import re
    except ModuleNotFoundError as exc:
      print("This RAVEN TSA Module requires the statsforecast library to be installed in the current python environment")
      raise ModuleNotFoundError from exc

    maxOrder = np.max(np.r_[settings['P_upper'], settings['Q_upper'], self._maxCombinedPQ])
    statsforecastParams = {
        "seasonal": False, # set to True if you want a SARIMA model
        "stationary": True, # NOTE: basically ignored 'd' because it should be applied as a TSA Transformer
        "start_p": 0,
        "start_q": 0,
        "max_p": settings['P_upper'],
        "max_q": settings['Q_upper'],
        "max_order": maxOrder,
        "ic": settings['criterion'],
        "stepwise": settings['use_approximation'],
        "approximation": settings['use_approximation'],
      }

    params = {}
    for tg, target in enumerate(targets):
      params[target] = {}
      history = signal[:, tg]
      mask = ~np.isnan(history)

      SFAA = AutoARIMA(**statsforecastParams)
      fittedARIMA = SFAA.fit(y=history[mask])

      arma_str = re.findall(r'\(([^\\)]+)\)', arima_string(fittedARIMA.model_))[0]
      p_opt,d_opt,q_opt = [int(a) for a in arma_str.split(',')]

      params[target]['P_opt'] = p_opt
      params[target]['D_opt'] = d_opt
      params[target]['Q_opt'] = q_opt
      del SFAA, fittedARIMA

    return params

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    for target, info in params.items():
      base = f'{self.name}__{target}'
      rlz[f'{base}__P_opt'] = info['P_opt']
      rlz[f'{base}__D_opt'] = info['D_opt']
      rlz[f'{base}__Q_opt'] = info['Q_opt']
    return rlz

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
      base.append(xmlUtils.newNode('P_opt', text=f'{int(info["P_opt"]):d}'))
      base.append(xmlUtils.newNode('D_opt', text=f'{int(info["D_opt"]):d}'))
      base.append(xmlUtils.newNode('Q_opt', text=f'{int(info["Q_opt"]):d}'))

