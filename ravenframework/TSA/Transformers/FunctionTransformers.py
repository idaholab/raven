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
Created on June 20, 2023
@author: j-bryan
Commonly used stateless transformation functions
"""

import numpy as np
import scipy.special as sps
import sklearn.preprocessing as skl

from .ScikitLearnBase import SKLTransformer
from ...utils import InputTypes


class LogTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for np.log/np.exp """
  templateTransformer = skl.FunctionTransformer(np.log, np.exp)

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'logtransformer'
    specs.description = r"""applies the natural logarithm to the data and inverts by applying the
                        exponential function."""
    return specs

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
    # Check for non-positive values in targets before handing off to super
    for t, (target, data) in enumerate(params.items()):
      if np.any(initial[:, t] <= 0):
        raise ValueError('Log transformation requires strictly positive values, and negative values '
                         f'were found in target "{target}"! If negative values were expected, perhaps '
                         'an ArcsinhTransformer would be more appropriate?')
    return super().getResidual(initial, params, pivot, settings)


class ArcsinhTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for np.arcsinh/np.sinh """
  templateTransformer = skl.FunctionTransformer(np.arcsinh, np.sinh)

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'arcsinhtransformer'
    specs.description = r"""applies the inverse hyperbolic sine to the data and inverts by applying
                        the hyperbolic sine."""
    return specs


class TanhTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for np.tanh/np.arctanh """
  templateTransformer = skl.FunctionTransformer(np.tanh, np.arctanh)

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'tanhtransformer'
    specs.description = r"""applies the hyperbolic tangent to the data and inverts by applying the
                        inverse hyperbolic tangent."""
    return specs


class SigmoidTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for scipy.special.expit/scipy.special.logit """
  templateTransformer = skl.FunctionTransformer(sps.expit, sps.logit)

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'sigmoidtransformer'
    specs.description = r"""applies the sigmoid (expit) function to the data and inverts by applying
                        the logit function."""
    return specs


class OutTruncation(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for limiting generated data to a specific range """
  templateTransformer = skl.FunctionTransformer()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'outtruncation'
    specs.description = r"""limits the data to either positive or negative values by "reflecting" the
                        out-of-range values back into the desired range."""
    domainType = InputTypes.makeEnumType('outDomain', 'outDomainType', ['positive', 'negative'])
    specs.addParam('domain', param_type=domainType, required=True)
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['domain'] = spec.parameterValues['domain']
    # Set the templateTransformer's inverse_func based on the specified domain
    if settings['domain'] == 'positive':
      self.templateTransformer.set_params(inverse_func=np.abs)
    else:  # negative
      self.templateTransformer.set_params(inverse_func=lambda x: -np.abs(x))
    return settings
