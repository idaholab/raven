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
Provides API and utilities for extending the Metric with custom options.

Created on May 25, 2021
@author: talbpaul
"""

from utils import InputData, InputTypes # this lets inheritors access these directly
from Metrics import Metric, factory
from .PluginBase import PluginBase

class PlotPlugin(PluginBase, Metric):
  """
    Defines a specialized class from which plugin metrics may inherit.
  """
  _interfaceFactory = factory

  #####################
  # API
  #
  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable user inputs for this class.
      @ In, None
      @ Out, specs, InputData.ParameterInput,
    """
    specs = super().getInputSpecification()
    specs.description = r"""Base class for Metric plugins"""
    return specs
