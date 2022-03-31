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
  Defines acceptance conditions when comparing successive optimal points.

  Reworked 2020-01
  @author: talbpaul
"""
import abc

from ...utils import utils, InputData, InputTypes

class AcceptanceCondition(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    GradientApproximators use provided information to both select points
    required to estimate gradients as well as calculate the estimates.
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    specs.description = 'Base class for acceptance conditions in the GradientDescent Optimizer.'
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
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

  def initialize(self):
    """
      After construction, finishes initialization of this approximator.
      @ In, None
      @ Out, None
    """
    pass

  ###############
  # Run Methods #
  ###############
  @abc.abstractmethod
  def checkImprovement(self, new, old):
    """
      Determines if a new value is sufficiently improved over the old
      @ In, new, float, new value
      @ In, old, float, old value
      @ Out, acceptable, bool, True if acceptable value
    """

  ###################
  # Utility Methods #
  ###################


