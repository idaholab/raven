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
from utils import utils, InputData, InputTypes
from .AcceptanceCondition import AcceptanceCondition

class Strict(AcceptanceCondition):
  """
    GradientApproximators use provided information to both select points
    required to estimate gradients as well as calculate the estimates.
  """
  ##########################
  # Initialization Methods #
  ##########################
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


