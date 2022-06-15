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
from ...utils import randomUtils, mathUtils
from .GradientApproximator import GradientApproximator

class FiniteDifference(GradientApproximator):
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
    specs.description = r"""if node is present, indicates that gradient approximation should be performed
        using Finite Difference approximation. Finite difference makes use of orthogonal perturbations
        in each dimension of the input space to estimate the local gradient, requiring a total of $N$
        perturbations, where $N$ is dimensionality of the input space. For example, if the input space
        $\mathbf{i} = (x, y, z)$ for objective function $f(\mathbf{i})$, then FiniteDifference chooses
        three perturbations $(\alpha, \beta, \gamma)$ and evaluates the following perturbation points:
        \begin{itemize}
          \item $f(x+\alpha, y, z)$,
          \item $f(x, y+\beta, z)$,
          \item $f(x, y, z+\gamma)$
        \end{itemize}
        and evaluates the gradient $\nabla f = (\nabla^{(x)} f, \nabla^{(y)} f, \nabla^{(z)} f)$ as
        \begin{equation*}
          \nabla^{(x)}f \approx \frac{f(x+\alpha, y, z) - f(x, y, z)}{\alpha},
        \end{equation*}
        and so on for $ \nabla^{(y)}f$ and $\nabla^{(z)}f$.
          """

    return specs

  ###############
  # Run Methods #
  ###############
  def chooseEvaluationPoints(self, opt, stepSize, constraints=None):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ In, constraints, dict, optional, boundary and functional constraints to respect when
                                         choosing new sampling points
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """
    dh = self._proximity * stepSize
    evalPoints = []
    evalInfo = []

    directions = np.atleast_1d(randomUtils.random(self.N) < 0.5) * 2 - 1
    for o, optVar in enumerate(self._optVars):
      # pick a new grad eval point
      optValue = opt[optVar]
      new = copy.deepcopy(opt)
      delta = dh * directions[o] # note this is NORMALIZED space delta
      new[optVar] = optValue + delta
      # constraint handling
      if constraints is not None:
        # all constraints speak DENORM space, not NORM space
        denormed = constraints['denormalize'](new)
        altPoint = self._handleConstraints(denormed, constraints['denormalize'](opt), optVar, constraints)
        # need NORM point, delta
        new = constraints['normalize'](altPoint)
        delta = new[optVar] - opt[optVar]
      # store as samplable point
      evalPoints.append(new)
      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
                       'delta': delta})

    return evalPoints, evalInfo

  def _handleConstraints(self, newPoint, original, optVar, constraints):
    """
      Allows the FiniteDifference to handle grad points that might violate constraints.
      Note this should be generalized as much as possible to the base class, if the different
      gradient approximation algorithms can find common ground in this algorithm.
      @ In, newPoint, np.array, desired new sampling point
      @ In, original, np.array, current opt point from which the new point is derived
      @ In, optVar, string, name of optimization variable being perturbed
      @ In, constraints, dict, boundary and functional constraints passed through
      @ Out, newPoint, np.array, potentially-adjusted new gradient sampling point
    """
    new = newPoint[optVar]
    orgval = original[optVar]
    delta = new - orgval
    # TODO div 0 protection? Can it ever be 0?
    scale = abs(delta)
    dist = constraints['boundary'][optVar]
    lower = dist.lowerBound
    upper = dist.upperBound
    info = constraints['inputs'] # has constants and such
    origDelta = delta # save the starting delta so we can keep track of it
    # check the new point to see if we're good or need to do something
    okay = self._checkConstraints(newPoint, optVar, lower, upper, constraints['functional'], info)
    if okay:
      return newPoint
    # we're not okay, so let's check if we're ok by flipping the delta direction
    delta = - origDelta
    newPoint[optVar] = orgval + delta
    okay = self._checkConstraints(newPoint, optVar, lower, upper, constraints['functional'], info)
    if okay:
      return newPoint
    # well, that didn't work, so now we try cutting delta (in the original direction)
    # first get the workable distance (bounded by distance to boundary)
    if origDelta < 0:
      delta =  - min(abs(lower - orgval), abs(origDelta))
    else:
      delta =  min(upper - orgval, origDelta)
    flipped = False   # have we checked the other side of the opt point?
    while not okay:
      okay = self._checkConstraints(newPoint, optVar, lower, upper, constraints['functional'], info)
      if not okay:
        delta /= 2
        if abs(delta) / scale < 1e-2:
          if not flipped:
            delta = - origDelta
            flipped = True
          else:
            raise RuntimeError(f'Could not find acceptable value for {optVar}: start {orgval:1.8e}, wanted {new:1.8e}, rejected all options via constraints.')
        else:
          newPoint[optVar] = orgval + delta

    return newPoint


  def _checkConstraints(self, point, optVar, lower, upper, constraints, info):
    """
      Checks for constraint violations in point.
      @ In, point, np.array, proposed sampling point
      @ In, optVar, string, name of optimization variable being perturbed
      @ In, lower, float, lower limit
      @ In, upper, float, upper limit
      @ In, constraints, dict, functional constraints imposed by user or similar
      @ In, info, dict, other useful info such as constants, etc
      @ Out, allOkay, bool, True if no constraints violated
    """
    allOkay = lower < point[optVar] < upper
    if allOkay:
      for constraint in constraints:
        info.update(point)
        okay = constraint.evaluate('constrain', info)
        allOkay &= okay

    return allOkay


  def evaluate(self, opt, grads, infos, objVar):
    """
      Approximates gradient based on evaluated points.
      @ In, opt, dict, current opt point (normalized)
      @ In, grads, list(dict), evaluated neighbor points
      @ In, infos, list(dict), info about evaluated neighbor points
      @ In, objVar, string, objective variable
      @ Out, magnitude, float, magnitude of gradient
      @ Out, direction, dict, versor (unit vector) for gradient direction
      @ Out, foundInf, bool, if True then infinity calculations were used
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
