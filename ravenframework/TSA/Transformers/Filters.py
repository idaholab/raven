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
from ...utils import xmlUtils, InputData, InputTypes


class FilterBase(TransformerMixin):
  """ Base class for transformers which filter or mask data """

  @classmethod
  def getInputSpecification(cls):
    """
      Define input spec for this class.
      @ In, None
      @ Out, specs, InputData.ParameterInput, input specification
    """
    specs = super().getInputSpecification()
    specs.addParam('fill', param_type=InputTypes.FloatOrStringType, required=False, default='drop',
                   descr=r"""fill strategy for masked values. May be one of 'drop' or a float value.
                             If 'drop', the masked values are dropped and replaced with NaN. If a float value,
                             that value is used to fill the masked values.""")
    return specs

  def handleInput(self, spec):
    """
      @ In, maskFill, float or None, value used to replace masked values; if maskFill=None,
                                      the masked values will be dropped
    """
    settings = super().handleInput(spec)
    fill = spec.parameterValues.get('fill', 'drop')
    if isinstance(fill, str):
      if fill.lower() != 'drop':
        raise ValueError(f"An unsupported fill value for {spec.name} was provided. Must be one of 'drop' "
                          "or a numeric value.")
      settings['fillValue'] = np.nan
    else:
      settings['fillValue'] = fill
    return settings

  @abc.abstractmethod
  def criterion(self, signal, settings):
    """
      Criterion for being masked. Evaluates to True if the value should be masked and evaluates to
      False otherwise.
      @ In, signal, numpy.ndarray, data array
      @ In, settings, dict, initialization settings for this algorithm
      @ Out, mask, numpy.ndarray, numpy array of boolean values that masks values of X
    """
    pass

  def fit(self, X):
    """
      Fits the mask to the array using the defined criterion
      @ In, X, np.ndarray, array of data
      @ Out, self, FilterBase, class instance
    """
    params = {}
    for tg, target in enumerate(targets):
      history = signal[:, tg]
      mask = self.criterion(history, settings)
      # save the masked (hidden) values
      hiddenValues = history[mask]
      params[target] = {'mask': mask, 'hiddenValues': hiddenValues}
    return params

  def transform(self, X):
    """
      Applies mask to data
      @ In, X, np.ndarray, array of data
      @ Out, xMasked, np.ndarray, array of masked data
    """
    residual = initial.copy()
    for t, (target, data) in enumerate(params.items()):
      mask = data['mask']
      residual[:, t] = np.ma.MaskedArray(residual[:, t], mask=mask, fill_value=settings['fillValue']).filled()
    return residual

  def inverse_transform(self, X):
    """
      Restores the masked values to the data array X
      @ In, X, np.ndarray, array of data
      @ Out, xUnmasked, np.ndarray, array of data with the masked values restored
    """
    xUnmasked = np.ma.MaskedArray(X, mask=self._mask).filled(0) + self._hiddenValues.filled(0)
    return xUnmasked


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
    specs.addParam('tol', param_type=InputTypes.FloatType, required=False, default=1e-8,
                   descr=r"""absolute tolerance about zero for which to apply the filter""")
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['tol'] = spec.parameterValues.get('tol', 1e-8)
    return settings

  def criterion(self, signal, settings):
    """
      Criterion for being masked. Evaluates to True if the value should be masked and evaluates to
      False otherwise.
      @ In, signal, numpy.ndarray, data array
      @ In, settings, dict, initialization settings for this algorithm
      @ Out, mask, numpy.ndarray, numpy array of boolean values that masks values of X
    """
    return np.abs(signal) < settings['tol']
