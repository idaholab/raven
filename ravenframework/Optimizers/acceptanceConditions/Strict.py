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
  Defines STRICT acceptance conditions when comparing successive optimal points.

  Reworked 2020-01
  @author: talbpaul
"""
from ...utils import utils, InputData, InputTypes
from .AcceptanceCondition import AcceptanceCondition

class Strict(AcceptanceCondition):
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
    specs = super(Strict, cls).getInputSpecification()
    specs.description = r"""if this node is present, indicates that a Strict acceptance policy for
        potential new optimal points should be enforced; that is, for a potential optimal point to
        become the new point from which to take another iterative optimizer step, the new response value
        must be improved over the old response value. Otherwise, the potential opt point is rejected
        and the search continues with the previously-discovered optimal point."""
    return specs
  ###############
  # Run Methods #
  ###############
  def checkImprovement(self, new, old):
    """
      Determines if a new value is sufficiently improved over the old
      @ In, new, float, new value
      @ In, old, float, old value
      @ Out, acceptable, bool, True if acceptable value
    """
    return new < old

  ###################
  # Utility Methods #
  ###################


