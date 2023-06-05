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
Commonly used stateless transformation functions
"""

import numpy as np
import scipy.special as sps
from sklearn.preprocessing import FunctionTransformer


def functionTransformerFactory(func, inverse_func):
  """
    Utility function for creating FunctionTransformer classes with common
    transforming functions

    @ In, func, callable, forward transformation function
    @ In, inverse_fun, callable, inverse transformation function
    @ Out, UserFunctionTransformer, class, custom transformer class
  """
  class UserFunctionTransformer(FunctionTransformer):
    """ Custom FunctionTransformer class """
    def __init__(self):
      super().__init__(func=func, inverse_func=inverse_func)
  return UserFunctionTransformer


LogTransformer = functionTransformerFactory(np.log, np.exp)
ArcsinhTransformer = functionTransformerFactory(np.arcsinh, np.sinh)
TanhTransformer = functionTransformerFactory(np.tanh, np.arctanh)
SigmoidTransformer = functionTransformerFactory(sps.expit, sps.logit)
# TODO what are other commonly used FunctionTransformers?

# These replicate the <outTruncation> option in the ARMA ROM
OutTruncationPositive = functionTransformerFactory(None, np.abs)
OutTruncationNegative = functionTransformerFactory(None, lambda x: -1 * np.abs(x))

__all__ = [
  "LogTransformer",
  "ArcsinhTransformer",
  "TanhTransformer",
  "SigmoidTransformer",
  "OutTruncationPositive",
  "OutTruncationNegative"
]
