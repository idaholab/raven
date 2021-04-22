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
Created on Nov 14, 2013

@author: alfoa, talbpaul
"""
from abc import abstractmethod, ABCMeta

from utils.utils import metaclass_insert
from BaseClasses import BaseEntity

class OutStreamEntity(metaclass_insert(ABCMeta, BaseEntity)):
  """
    OUTSTREAM CLASS
    This class is a general base class for outstream action classes
    For example, a matplotlib interface class or Print class, etc.
  """
  ###################
  # API
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OutStreamManager'
    self.type = 'Base'        # identifying type

  def _handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super()._handleInput(spec)

  def initialize(self, stepEntities):
    """
      Initialize the OutStream for a new Step
      @ In, stepEntities, dict, the Entities used in the current Step. Sources are taken from this.
      @ Out, None
    """
    pass

  @abstractmethod
  def addOutput(self):
    """
      Function to craft a new output (for example a CSV file or a plot)
      @ In, None
      @ Out, None
    """
