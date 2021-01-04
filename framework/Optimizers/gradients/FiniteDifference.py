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

    directions = np.atleast_1d(randomUtils.random(self.N) < 0.5) * 2 - 1
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
