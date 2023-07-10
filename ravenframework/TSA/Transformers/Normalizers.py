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

from .ScikitLearnBase import SKLTransformer, SKLCharacterizer
from ...utils import InputTypes


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
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
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
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
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
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
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
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.name = 'standardscaler'
    specs.description = r"""centers and scales the data by subtracting the mean and dividing by
    the standard deviation."""
    return specs


class QuantileTransformer(SKLTransformer):
  """ Wrapper of scikit-learn's QuantileTransformer """
  templateTransformer = skl.QuantileTransformer()

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
    specs.name = 'quantiletransformer'
    specs.description = r"""transforms the data to fit a given distribution by mapping the data to
    a uniform distribution and then to the desired distribution."""
    specs.addParam('nQuantiles', param_type=InputTypes.IntegerType,
                   descr=r"""number of quantiles to use in the transformation. If \xmlAttr{nQuantiles}
                   is greater than the number of data, then the number of data is used instead.""",
                   required=False, default=1000)
    distType = InputTypes.makeEnumType('outputDist', 'outputDistType', ['normal', 'uniform'])
    specs.addParam('outputDistribution', param_type=distType,
                   descr=r"""distribution to transform to. Must be either 'normal' or 'uniform'.""",
                   required=False, default='normal')
    return specs

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['nQuantiles'] = spec.parameterValues.get('nQuantiles', settings['nQuantiles'])
    settings['outputDistribution'] = spec.parameterValues.get('outputDistribution', settings['outputDistribution'])
    self.templateTransformer.set_params(n_quantiles=settings['nQuantiles'],
                                        output_distribution=settings['outputDistribution'])
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
    if 'outputDistribution' not in settings:
      settings['outputDistribution'] = 'normal'
    return settings
