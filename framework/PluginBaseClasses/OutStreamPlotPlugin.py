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
Provides API and utilities for extending the OutStream Plot with custom plotting options.

Created on April 2, 2021
@author: talbpaul
"""
from utils import InputData, InputTypes

from OutStreams.PlotInterfaces import PlotInterface, factory
from .PluginBase import PluginBase

class PlotPlugin(PluginBase, PlotInterface):
  """
    Defines a specialized class from which plugin plots may inherit.
  """
  _interfaceFactory = factory
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin. This list needs to be populated by the derived class
  _methodsToCheck = ['run']

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
    specs = super(PlotPlugin, cls).getInputSpecification()
    specs.description = r"""Base class for OutStream Plot plugins"""
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()

  def handleInput(self, spec):
    """
      Reads in data from the input file
      @ In, spec, InputData.ParameterInput, input information
      @ Out, None
    """
    super().handleInput(spec)

  def run(self):
    """
      Main run method.
      @ In, sources, list(DataObject.DataSet), DataSets containing source data
      @ In, figures, dict, matplotlib.pyplot.figure, figure on which to plot
      @ Out, None
    """
    pass

  #####################
  # RAVEN Utilities
  #
