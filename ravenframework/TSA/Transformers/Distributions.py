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
  Created on August 3, 2023
  @author: j-bryan

  Data transformations that change one distribution to another.
"""

import numpy as np
import sklearn.preprocessing as skl

from ..TimeSeriesAnalyzer import TimeSeriesTransformer
from .ScikitLearnBase import SKLTransformer
from ...utils import InputTypes, xmlUtils, mathUtils


class Gaussianize(SKLTransformer):
  """ Uses scikit-learn's preprocessing.QuantileTransformer to transform data to a normal distribution """
  templateTransformer = skl.QuantileTransformer(output_distribution='normal')

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'gaussianize'
    specs.description = r"""transforms the data into a normal distribution using quantile mapping."""
    specs.addParam('nQuantiles', param_type=InputTypes.IntegerType,
                   descr=r"""number of quantiles to use in the transformation. If \xmlAttr{nQuantiles}
                   is greater than the number of data, then the number of data is used instead.""",
                   required=False, default=1000)
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['nQuantiles'] = spec.parameterValues.get('nQuantiles', settings['nQuantiles'])
    self.templateTransformer.set_params(n_quantiles=settings['nQuantiles'])
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'nQuantiles' not in settings:
      settings['nQuantiles'] = 1000
    return settings


class QuantileTransformer(Gaussianize):
  """ Wrapper of scikit-learn's QuantileTransformer """
  templateTransformer = skl.QuantileTransformer()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'quantiletransformer'
    specs.description = r"""transforms the data to fit a given distribution by mapping the data to
    a uniform distribution and then to the desired distribution."""
    distType = InputTypes.makeEnumType('outputDist', 'outputDistType', ['normal', 'uniform'])
    specs.addParam('outputDistribution', param_type=distType,
                   descr=r"""distribution to transform to.""",
                   required=False, default='normal')
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['outputDistribution'] = spec.parameterValues.get('outputDistribution', settings['outputDistribution'])
    self.templateTransformer.set_params(output_distribution=settings['outputDistribution'])
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'outputDistribution' not in settings:
      settings['outputDistribution'] = 'normal'
    return settings


class PreserveCDF(TimeSeriesTransformer):
  """
    Transformer that preserves the CDF of the input data and forces data given to the inverse
    transformation function to have the same CDF as the original data through quantile mapping.
  """
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

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'preserveCDF'
    specs.description = r"""forces generated data provided to the inverse transformation function to
                            have the same CDF as the original data through quantile mapping. If this
                            transformer is used as part of a SyntheticHistory ROM, it should likely
                            be used as the first transformer in the chain."""
    return specs

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
      params[target] = {}
      history = signal[:, tg]
      mask = ~np.isnan(history)
      params[target]['cdf'] = mathUtils.trainEmpiricalFunction(history[mask], minBins=self._minBins)
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
    # Nothing to do on the forward transformation
    return initial

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
    composite = np.zeros_like(initial)
    for tg, (target, data) in enumerate(params.items()):
      signal = initial[:, tg]
      originalDist = data['cdf']
      # Convert to distribution fit to training data
      dist, hist = mathUtils.trainEmpiricalFunction(signal, minBins=self._minBins)
      # transform data through CDFs
      transformed = originalDist[0].ppf(dist.cdf(signal))
      composite[:, tg] = transformed
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
