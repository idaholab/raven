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
Filters which mask values based on some criterion.
"""

import abc
import numpy as np

from ..TimeSeriesAnalyzer import TimeSeriesTransformer
from ...utils import xmlUtils


class FilterBase(TimeSeriesTransformer):
  """ Base class for transformers which filter or mask data """
  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    return settings

  @abc.abstractmethod
  def criterion(self, signal):
    """
      Criterion for being masked. Evaluates to True if the value should be masked and evaluates to
      False otherwise.
      @ In, signal, numpy.ndarray, data array
      @ Out, mask, numpy.ndarray, numpy array of boolean values that masks values of X
    """

  def fit(self, signal, pivot, targets, settings):
    """
      Fits the algorithm/model using the provided time series ("signal") using methods specific to
      the algorithm.
      @ In, signal, np.array, time-dependent series
      @ In, pivot, np.array, time-like parameter
      @ In, targets, list(str), names of targets
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, params, dict, characterization of signal; structure as:
                           params[target variable][characteristic] = value
    """
    params = {}
    for tg, target in enumerate(targets):
      history = signal[:, tg]
      mask = self.criterion(history)
      # save the masked (hidden) values
      hiddenValues = history[mask]
      params[target] = {'mask': mask, 'hiddenValues': hiddenValues}
    return params

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
    for t, (target, data) in enumerate(params.items()):
      mask = data['mask']
      residual[:, t] = np.ma.MaskedArray(residual[:, t], mask=mask, fill_value=np.nan).filled()
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
    composite = initial.copy()
    for t, (target, data) in enumerate(params.items()):
      # Put the hidden values back into the composite signal
      composite[data['mask'], t] = data['hiddenValues']
    return composite

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, parameters from training as from self.fit
      @ Out, None
    """
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)


class ZeroFilter(FilterBase):
  """ Masks any values that are near zero """

  @classmethod
  def getInputSpecification(cls):
    """
      Define input spec for this class.
      @ In, None
      @ Out, specs, InputData.ParameterInput, input specification
    """
    specs = super().getInputSpecification()
    specs.name = 'zerofilter'
    specs.description = r"""masks values that are near zero. The masked values are replaced with NaN
    values. Caution should be used when using this algorithm because not all algorithms can handle
    NaN values! A warning will be issued if NaN values are detected in the input of an algorithm that
    does not support them."""
    return specs

  def criterion(self, signal):
    """
      Criterion for being masked. Evaluates to True if the value should be masked and evaluates to
      False otherwise.
      @ In, signal, numpy.ndarray, data array
      @ Out, mask, numpy.ndarray, numpy array of boolean values that masks values of X
    """
    return np.isclose(signal, 0)
