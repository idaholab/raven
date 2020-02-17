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
  Base class for step sizing strategies in optimization paths

  Created 2020-01
  @author: talbpaul
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils, InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------


class StepManipulator(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Base class for handling step sizing in optimization paths
  """
  requiredInformation = [] # data required to make step size/direction determination
  optionalInformation = [] # optional data to make step size/direction determination

  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    # TODO
    ## Instance Variable Initialization
    # public
    # _protected
    # __private
    # additional methods

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    pass

  def initialize(self, optVars, **kwargs):
    """ TODO """

  ###############
  # Run Methods #
  ###############
  @abc.abstractmethod
  def initialStepSize(self, **kwargs):
    """
      Calculates the first step size to use in the optimization path.
      @ In, kwargs, dict, keyword-based specifics as required by individual step sizers
      @ Out, stepSize, float, new step size
    """

  @abc.abstractmethod
  def step(self, prevOpt, **kwargs):
    """
      Calculates a new step size to use in the optimization path.
      @ In, prevOpt, dict, previous optimal point
      @ In, kwargs, dict, keyword-based specifics as required by individual step sizers
      @ Out, newOpt, dict, new optimal point
      @ Out, stepSize, float, new step size
    """

  @abc.abstractmethod
  def fixConstraintViolations(self, proposed, previous, fixInfo):
    """
      Given constraint violations, update the desired optimal point to consider.
      @ In, proposed, dict, proposed new optimal point
      @ In, previous, dict, previous optimal point
      @ In, fixInfo, dict, contains record of progress in fixing search
      @ Out, proposed, new proposed point
      @ Out, stepSize, new step size taken
      @ Out, fixInfo, updated fixing info
    """



  ###################
  # Utility Methods #
  ###################