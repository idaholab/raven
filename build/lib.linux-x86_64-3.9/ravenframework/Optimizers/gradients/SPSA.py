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
  Simultaneous Perturbation Stochastic Approximation gradient estimation algorithms
  Author: gairabhi
"""
import numpy as np
from ...utils import randomUtils, mathUtils
from .GradientApproximator import GradientApproximator

class SPSA(GradientApproximator):
  """
    Single-point (zeroth-order) gradient approximation.
    Note that SPSA is a larger algorithm; this is simply the gradient approximation part of it.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(SPSA, cls).getInputSpecification()
    specs.description = r"""if node is present, indicates that gradient approximation should be performed
        using the Simultaneous Perturbation Stochastic Approximation (SPSA).
        SPSA makes use of a single perturbation as a zeroth-order gradient approximation,
        requiring exactly $1$
        perturbation regardless of the dimensionality of the input space. For example, if the input space
        $\mathbf{i} = (x, y, z)$ for objective function $f(\mathbf{i})$, then SPSA chooses
        a single perturbation point $(\epsilon^{(x)}, \epsilon^{(y)}, \epsilon^{(z)})$ and evaluates
        the following perturbation point:
        \begin{itemize}
          \item $f(x+\epsilon^{(x)}, y+\epsilon^{(y)}, z+\epsilon^{(z)})$
        \end{itemize}
        and evaluates the gradient $\nabla f = (\nabla^{(x)} f, \nabla^{(y)} f, \nabla^{(z)} f)$ as
        \begin{equation*}
          \nabla^{(x)}f \approx \frac{f(x+\epsilon^{(x)}, y+\epsilon^{(y)}, z+\epsilon^{(z)})) -
              f(x, y, z)}{\epsilon^{(x)}},
        \end{equation*}
        and so on for $ \nabla^{(y)}f$ and $\nabla^{(z)}f$. This approximation is much less robust
        than FiniteDifference or CentralDifference, but has the benefit of being dimension agnostic.
          """

    return specs

  def chooseEvaluationPoints(self, opt, stepSize, constraints=None):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ In, constraints, dict, optional, constraints to check against when choosing new sample points
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """
    dh = self._proximity * stepSize
    perturb = np.atleast_1d(randomUtils.randPointsOnHypersphere(self.N))
    delta = {}
    new = {}
    for i, var in enumerate(self._optVars):
      delta[var] = perturb[i] * dh
      new[var] = opt[var] + delta[var]
    # only one point needed for SPSA, but still needs to store as a list
    evalPoints = [new]
    evalInfo = [{'type': 'grad',
                 'delta': delta}]

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
    lossDiff = np.atleast_1d(mathUtils.diffWithInfinites(grads[0][objVar], opt[objVar]))
    for var in self._optVars:
      # don't assume delta is unchanged; calculate it here
      delta = grads[0][var] - opt[var]
      gradient[var] = lossDiff / delta
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))

    return magnitude, direction, foundInf

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
      @ In, None
      @ Out, None
    """
    # SPSA always uses 1 point, regardless
    return 1
