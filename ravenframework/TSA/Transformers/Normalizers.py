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

Wrappers for scikit-learn preprocessing scalers.
"""

import sklearn.preprocessing as skl
import numpy as np
from scipy.stats import iqr
from copy import deepcopy

from ..TimeSeriesAnalyzer import TimeSeriesTransformer
from .ScikitLearnBase import SKLTransformer, SKLCharacterizer
from ...utils import InputTypes, xmlUtils


class MaxAbsScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.MaxAbsScaler """
  _features = ['scale']
  templateTransformer = skl.MaxAbsScaler()

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
    specs.name = 'maxabsscaler'
    specs.description = r"""scales the data to the interval $[-1, 1]$. This is done by dividing by
    the largest absolute value of the data."""
    return specs


class MinMaxScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.MinMaxScaler """
  _features = ['dataMin', 'dataMax']
  templateTransformer = skl.MinMaxScaler()

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
    specs.name = 'minmaxscaler'
    specs.description = r"""scales the data to the interval $[0, 1]$. This is done by subtracting the
                        minimum value from each point and dividing by the range."""
    return specs


class RobustScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.RobustScaler """
  _features = ['center', 'scale']
  templateTransformer = skl.RobustScaler()

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
    specs.name = 'robustscaler'
    specs.description = r"""centers and scales the data by subtracting the median and dividing by
    the interquartile range."""
    return specs


class StandardScaler(SKLCharacterizer):
  """ Wrapper of sklearn.preprocessing.StandardScaler """
  _features = ['mean', 'scale']
  templateTransformer = skl.StandardScaler()

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
    specs.name = 'standardscaler'
    specs.description = r"""centers and scales the data by subtracting the mean and dividing by
    the standard deviation."""
    return specs
