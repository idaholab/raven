"""
  Central difference approximation algorithms
  Author:--gairabhi
"""
import copy
from utils import mathUtils

from .GradientApproximater import GradientApproximater


class CentralDifference(GradientApproximater):
  """
    Enables gradient estimation via central differencing
  """
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
    # submit a positive and negative side of the opt point for each dimension
    for _, optVar in enumerate(self._optVars):
      optValue = opt[optVar]
      neg = copy.deepcopy(opt)
      pos = copy.deepcopy(opt)
      delta = dh
      neg[optVar] = optValue - delta
      pos[optVar] = optValue + delta

      evalPoints.append(neg)
      evalInfo.append({'type': 'grad',
                      'optVar': optVar,
                      'delta': delta,
                      'side': 'negative'})

      evalPoints.append(pos)
      evalInfo.append({'type': 'grad',
                      'optVar': optVar,
                      'delta': delta,
                      'side': 'positive'})
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
    for v, var in enumerate(self._optVars):
      # get the positive and negative sides for this var
      neg = None
      pos = None
      # TODO this search could get expensive in high dimensions!
      for g, grad in enumerate(grads):
        info = infos[g]
        if info['optVar'] == var:
          if info['side'] == 'negative':
            neg = grad
          else:
            pos = grad
          if neg and pos:
            break
      # dh for pos and neg (note we don't assume delta was unchanged, we recalculate it)
      dhNeg = opt[var] - neg[var]
      dhPos = pos[var] - opt[var]
      # 3-point central difference doesn't use opt point, since it cancels out
      # also the terms are weighted by the dh on each side
      gradient[var] = 1/(2*dhNeg) * pos[objVar] - 1/(2*dhPos) * neg[objVar]

    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))
    return magnitude, direction, foundInf

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """
    return self.N * 2
