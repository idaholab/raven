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
Created on March 11, 2021
@author: talbpaul
"""
from __future__ import division, print_function , unicode_literals, absolute_import
import abc

from utils import InputData, InputTypes

from OutStreams import OutStreamBase
from .PluginBase import PluginBase

class PlotPlugin(PluginBase, OutStreamBase):
  """
    Defines a specialized class from which plugin plots may inherit.
  """
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin. This list needs to be populated by the derived class
  _methodsToCheck = ['plot']
  entityType = 'OutStreams'

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
    specs.addParam('interactive', param_type=InputTypes.BoolType)
    formatsEnum = InputTypes.makeEnumType('PlotSaveFormats', 'PlotSaveFormats',
        ['screen', 'pdf', 'png']) # TODO others?
    specs.addSub(InputData.parameterInputFactory('format', contentType=formatsEnum,
        descr=r"""Determines the format that the plot should be saved to disk, and whether it should be shown on screen."""))
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super(PlotPlugin, self).__init__()
    self._saveFormat = None   # list(str), how to save file to disk
    self._interactive = None  # bool, whether to be interactive after plotting

  def _handleInput(self, spec):
    """
      Reads in data from the input file
      @ In, spec, InputData.ParameterInput, input information
      @ Out, None
    """
    super(PlotPlugin, self)._handleInput()
    self._interactive = spec.parameterValues.get('interactive', False)
    for sub in spec:
      if sub.getName() == 'format':
        self._format = sub.value

  def addOutput(self):
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