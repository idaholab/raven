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
  Implementation of FiniteDifference gradient approximation
"""
import copy
import numpy as np
from utils import InputData, InputTypes, randomUtils, mathUtils
from .GradientApproximater import GradientApproximater

class FiniteDifference(GradientApproximater):
  """
    Uses FiniteDifference approach to approximating gradients
  """
  ##########################
  # Initialization Methods #
  ##########################
  ###############
  # Run Methods #
  ###############
  def chooseEvaluationPoints(self, opt, stepSize):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """
    dh = self._proximity * stepSize
    evalPoints = []
    evalInfo = []

    directions = np.asarray(randomUtils.random(self.N) < 0.5) * 2 - 1
    for o, optVar in enumerate(self._optVars):
      optValue = opt[optVar]
      new = copy.deepcopy(opt)
      delta = dh * directions[o]
      new[optVar] = optValue + delta
      evalPoints.append(new)
      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
                       'delta': delta})


    return evalPoints, evalInfo

  def evaluate(self, opt, grads, infos, objVar):
    """
      Approximates gradient based on evaluated points.
      @ In, opt, dict, current opt point (normalized)
      @ In, grads, list(dict), evaluated neighbor points
      @ In, infos, list(dict), info about evaluated neighbor points
      @ In, objVar, string, objective variable
      @ Out, magnitude, float, magnitude of gradient
      @ Out, direction, dict, versor (unit vector) for gradient direction
    """
    gradient = {}
    for g, pt in enumerate(grads):
      info = infos[g]
      delta = info['delta']
      activeVar = info['optVar']
      lossDiff = np.atleast_1d(mathUtils.diffWithInfinites(pt[objVar], opt[objVar]))
      grad = lossDiff/delta
      gradient[activeVar] = grad
    # obtain the magnitude and versor of the gradient to return
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))
    return magnitude, direction, foundInf


  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """
    return self.N


  ###################
  # Utility Methods #
  ###################
