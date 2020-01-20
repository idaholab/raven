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
import abc
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
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(FiniteDifference, cls).getInputSpecification()
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    GradientApproximater.__init__(self)
    ## Instance Variable Initialization
    # public
    # _protected
    # __private
    # additional methods

  def initialize(self, optVars, proximity):
    """
      After construction, finishes initialization of this approximator.
      @ In, optVars, list(str), list of optimization variable names
      @ In, proximity, float, percentage of step size away that neighbor samples should be taken
      @ Out, None
    """
    GradientApproximater.initialize(self, optVars, proximity)

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
    for o, (optVar, optValue) in enumerate(opt.items()):
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
      # TODO FIXME flip sign for maximization?? Should be in the optimizer methinks.
      grad = (lossDiff) / delta
      gradient[activeVar] = grad
    # get the magnitude, versor of the gradient
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    return magnitude, direction, foundInf


  ###################
  # Utility Methods #
  ###################


