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
Created on July 3, 2023
@author: j-bryan

Nth order differencing
"""

import numpy as np

from ..TimeSeriesAnalyzer import TimeSeriesTransformer
from ...utils import xmlUtils, InputTypes


class Differencing(TimeSeriesTransformer):
  """ TODO """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'differencing'
    specs.description = r"""applies Nth order differencing to the data."""
    specs.addParam('order', param_type=InputTypes.IntegerType, required=True,
                   descr="""differencing order.""", default=1)
    # TODO add initial value option
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['order'] = spec.parameterValues.get('order', 1)
    return settings

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
      # For differencing order N, we need to have the first N values of the signal
      targetSignal = signal[:settings['order']+1, tg]
      initValues = np.zeros(settings['order'])
      for i in range(settings['order']):
        initValues[i] = targetSignal[0]
        targetSignal = np.diff(targetSignal)
      params[target] = {'initValues': initValues, 'order': settings['order']}
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
    for tg, (target, data) in enumerate(params.items()):
      diffOrder = data['order']
      signal = initial[:, tg]
      for n in range(diffOrder):  # difference the signal N times
        signal = np.diff(signal)
      residual[:, tg] = np.concatenate((signal, [np.nan]*diffOrder))  # pad array with NaNs
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
    for tg, (target, data) in enumerate(params.items()):
      diffOrder = data['order']
      signal = initial[:, tg]
      signal = signal[~np.isnan(signal)]  # drop any masking values
      for n in range(diffOrder):  # integrate the signal N times
        signal = np.concatenate(([data['initValues'][-n-1]], signal)).cumsum()
      composite[:, tg] = signal[:len(composite)]  # truncate to original length
    return composite

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.fit
      @ Out, None
    """
    # Add model settings as subnodes to writeTO node
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)
      base.append(xmlUtils.newNode('order', text=info['order']))
