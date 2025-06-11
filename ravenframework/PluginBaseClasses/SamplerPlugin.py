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

Created on June 10, 2025
@author: wangc
"""
from abc import abstractmethod

from ..utils import InputData, InputTypes # this lets inheritors access these directly
from ..Samplers import Sampler, factory
from .PluginBase import PluginBase

class SamplerPlugin(PluginBase, Sampler):
  """
    Defines a specialized class from which plugin Sampler may inherit.
  """
  _methodsToCheck = ['getInputSpecification']
  entityType = 'Sampler'
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
    specs = super(SamplerPlugin, cls).getInputSpecification()
    specs.description = r"""Base class for Sampler plugins"""
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Plugin SAMPLER'
    self.samplingType = None
    self.limit = None

  @abstractmethod
  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """

  @abstractmethod
  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """

  def _localHandleFailedRuns(self, failedRuns):
    """
      Specialized method for samplers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns) > 0:
      self.raiseADebug('  Continuing with reduced-size Monte-Carlo sampling.')

  def flush(self):
    """
      Reset Stratified attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()

  # TODO: Identify Common Functions that will be used by Sampler Plugin
