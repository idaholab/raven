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
from ...utils import xmlUtils, InputTypes


class FilterBase(TimeSeriesTransformer):
  """ Base class for transformers which filter or mask data """

  @classmethod
  def getInputSpecification(cls):
    """
      Define input spec for this class.
      @ In, None
      @ Out, specs, InputData.ParameterInput, input specification
    """
    specs = super().getInputSpecification()
    maskFillEnumType = InputTypes.makeEnumType('maskFill', 'maskFillTypeType', ['NaN', 'None',])
    specs.addParam('maskFill',
                   param_type=maskFillEnumType,
                   required=False,
                   default='NaN',
                   descr=r"""defines the value used to replace masked values. If \xmlString{NaN},
                    then the masked values will be replaced with NaN values. If \xmlString{None},
                    then the masked values will be dropped entirely. Note that dropping the data
                    will result in collapsing the data into a 1D array, which may cause issues with
                    later TSA algorithms.""")
    return specs

  def __init__(self):
    """
      @ In, maskFill, float or None, value used to replace masked values; if maskFill=None,
                                      the masked values will be dropped
    """
    super().__init__()
    # NOTE using np.nan to fill the masked values is advantageous because it preserves
    ## the full shape and spacing of the masked array. However, this requires any
    ## subsequent models to be able to correctly handle NaN values! Using maskFill=None
    ## will drop the masked values instead of filling them with a different value.
    self._maskFill = np.nan
    self._mask = {}
    self._hiddenValues = {}

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)

    maskFill = spec.parameterValues.get('maskFill', np.nan)
    if maskFill == 'NaN':
      settings['maskFill'] = np.nan
    elif maskFill == 'None':
      settings['maskFill'] = None
    else:
      settings['maskFill'] = maskFill

    return settings

  # TODO remove this before commit
  # def setDefaults(self, settings):
  #   """
  #     Fills default values for settings with default values.
  #     @ In, settings, dict, existing settings
  #     @ Out, settings, dict, modified settings
  #   """
  #   settings = super().setDefaults(settings)
  #   if 'maskFill' not in settings:
  #     settings['maskFill'] = np.nan
  #   return settings

  @abc.abstractmethod
  def criterion(self, signal):
    """
      Criterion for being masked. Evaluates to False if the value should be
      masked and evaluates to True otherwise.
      @ In, signal, numpy.ndarray, data array
      @ In, tol, float, tolerance for the criterion
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
    self._maskFill = settings['maskFill']
    for tg, target in enumerate(targets):
      params[target] = {}  # There are no parameters to save and return
      history = signal[:, tg]
      self._mask[target] = self.criterion(history)
      # save the masked (hidden) values
      self._hiddenValues[target] = np.ma.MaskedArray(history, mask=~self._mask[target])
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
      signal = residual[:, t]
      masked = np.ma.MaskedArray(signal, mask=self._mask[target], fill_value=self._maskFill)
      if self._maskFill is None:
        masked = masked.compressed()
      else:
        masked = masked.filled()
      residual[:, t] = masked
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
      unmasked = np.ma.MaskedArray(composite[:, t], mask=self._mask[target]).filled(0) + self._hiddenValues[target].filled(0)
      composite[:, t] = unmasked
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
    specs.description = r"""masks values that are near zero. The masked values are either replaced
    with NaN values or dropped entirely. Caution should be used when using this algorithm because
    (a) not all algorithms can handle NaN values and (b) dropping masked values can change the
    auto-correlative structure of the data."""
    # maskFillEnumType = InputTypes.makeEnumType('maskFill', 'maskFillTypeType', ['NaN', 'None',])
    # specs.addParam('maskFill',
    #                param_type=maskFillEnumType,
    #                required=False,
    #                default='NaN',
    #                descr=r"""defines the value used to replace masked values. If \xmlString{NaN},
    #                 then the masked values will be replaced with NaN values. If \xmlString{None},
    #                 then the masked values will be dropped entirely. Note that dropping the data
    #                 will result in collapsing the data into a 1D array, which may cause issues with
    #                 later TSA algorithms.""")
    return specs

  def criterion(self, signal):
    """
      Criterion for being masked. Evaluates to False if the value should be
      masked and evaluates to True otherwise.
      @ In, signal, numpy.ndarray, data array
      @ In, tol, float, tolerance for the criterion
      @ Out, mask, numpy.ndarray, numpy array of boolean values that masks values of X
    """
    return np.isclose(signal, 0)
