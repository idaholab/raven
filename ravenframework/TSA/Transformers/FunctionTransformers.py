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
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'logtransformer'
    specs.description = r"""applies the natural logarithm to the data and inverts by applying the
                        exponential function."""
    return specs

  def __init__(self):
    """ Constructor """
    super().__init__(skl.FunctionTransformer, func=np.log, inverse_func=np.exp)


class ArcsinhTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for np.arcsinh/np.sinh """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'arcsinhtransformer'
    specs.description = r"""applies the inverse hyperbolic sine to the data and inverts by applying
                        the hyperbolic sine."""
    return specs

  def __init__(self):
    """ Constructor """
    super().__init__(skl.FunctionTransformer, func=np.arcsinh, inverse_func=np.sinh)


class TanhTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for np.tanh/np.arctanh """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'tanhtransformer'
    specs.description = r"""applies the hyperbolic tangent to the data and inverts by applying the
                        inverse hyperbolic tangent."""
    return specs

  def __init__(self):
    """ Constructor """
    super().__init__(skl.FunctionTransformer, func=np.tanh, inverse_func=np.arctanh)


class SigmoidTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for scipy.special.expit/scipy.special.logit """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'sigmoidtransformer'
    specs.description = r"""applies the sigmoid (expit) function to the data and inverts by applying
                        the logit function."""
    return specs

  def __init__(self):
    """ Constructor """
    super().__init__(skl.FunctionTransformer, func=sps.expit, inverse_func=sps.logit)


class OutTruncation(SKLTransformer):
  """ Wrapper of scikit-learn's FunctionTransformer for limiting generated data to a specific range """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
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

  def __init__(self):
    """ Constructor """
    super().__init__(skl.FunctionTransformer)

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
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
