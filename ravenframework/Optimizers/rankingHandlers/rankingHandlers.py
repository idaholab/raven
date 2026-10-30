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
  Ranking handlers for multi-objective Genetic Algorithms (NSGA-II family).

  A ranking handler is the mix-and-match operator that turns a population's objective values
  (and, depending on the policy, its constraint values or a scalar fitness) into non-dominated
  front ranks. It is the policy layer sitting on top of the numeric core in
  ravenframework.utils.frontUtils; it does NOT compute constraint values g(x) (that is the job of
  Optimizers/constraintHandling), it decides HOW objectives and violations combine into ranks.

  Selecting one of these operators via <rankingAlgorithm type="..."> under <GAparams> lets the user
  swap the survival criterion without touching the sampler. New ranking policies are added by
  defining a function with the signature below and registering it in the __rankingHandlers dict.

  Implemented handlers (with their literature source):
    1. constrainedDomination  -- Deb et al. 2002 constrained non-dominated sorting (default).
    2. feasibleFirstPenalty   -- Deb 2000 parameter-less penalty, ranked as a scalar fitness.
    3. epsilonConstrained     -- Takahama & Sato epsilon-constrained relaxation of (1).
    4. stochasticRanking      -- Runarsson & Yao; constraints ignored per-generation with prob pf.

  Handler contract:
    def handler(caller, objVals, constraintVals, minMask, epsilon, fitVals=None, **kwargs) -> list[int]
      @ In, caller, object, the optimizer instance (for logging / attribute access)
      @ In, objVals, np.array, (nPoints, nObjectives) external-space objective values
      @ In, constraintVals, np.array or None, (nPoints, nConstraints) g(x); >= 0 feasible; None ->
                            rank by objectives only (used by the stochastic-ranking gate)
      @ In, minMask, np.array, per-objective bool, True where the objective is minimized
      @ In, epsilon, float, epsilon-constrained relaxation tolerance (0.0 = strict Deb dominance)
      @ In, fitVals, np.array or None, (nPoints,) scalar fitness, required by penalty ranking only
      @ Out, ranks, list, non-dominated front rank per point (1 = best front)

  @authors: Mohammad Abdo (@Jimmy-INL)
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils import frontUtils
# Internal Modules End------------------------------------------------------------------------------

def constrainedDomination(caller, objVals, constraintVals, minMask, epsilon=0.0, fitVals=None, **kwargs):
  """
    Deb et al. (2002) constrained non-dominated sorting: feasible solutions dominate infeasible
    ones, two infeasible solutions are ordered by total constraint violation, and two feasible
    solutions are ordered by ordinary Pareto dominance. This is faithful NSGA-II and uses NO scalar
    fitness. When constraintVals is None (stochastic-ranking gate) it ranks by objectives only.
    @ In, caller, object, the optimizer instance (unused; kept for a uniform handler signature)
    @ In, objVals, np.array, (nPoints, nObjectives) external-space objective values
    @ In, constraintVals, np.array or None, (nPoints, nConstraints) g(x); >= 0 feasible; None -> objectives only
    @ In, minMask, np.array, per-objective bool, True where minimized
    @ In, epsilon, float, optional, epsilon-constrained tolerance (0.0 = strict)
    @ In, fitVals, np.array or None, optional, ignored (domination does not use scalar fitness)
    @ Out, ranks, list, non-dominated front rank per point
  """
  return frontUtils.rankNonDominatedFrontiers(objVals,
                                              constraintVals=constraintVals,
                                              minMask=minMask,
                                              epsilon=epsilon)

def epsilonConstrained(caller, objVals, constraintVals, minMask, epsilon=0.0, fitVals=None, **kwargs):
  """
    Takahama & Sato epsilon-constrained ranking: identical to constrainedDomination but treats
    solutions whose total constraint violation does not exceed epsilon as feasible, relaxing the
    strict feasible-first boundary. The epsilon value comes from the <constraintEpsilon> node
    (caller._constraintEpsilon), passed through here; selecting this operator is a discoverable,
    self-documenting way to request that relaxation.
    @ In, caller, object, the optimizer instance (unused; uniform signature)
    @ In, objVals, np.array, (nPoints, nObjectives) external-space objective values
    @ In, constraintVals, np.array or None, (nPoints, nConstraints) g(x); >= 0 feasible
    @ In, minMask, np.array, per-objective bool, True where minimized
    @ In, epsilon, float, epsilon-constrained tolerance
    @ In, fitVals, np.array or None, optional, ignored
    @ Out, ranks, list, non-dominated front rank per point
  """
  return frontUtils.rankNonDominatedFrontiers(objVals,
                                              constraintVals=constraintVals,
                                              minMask=minMask,
                                              epsilon=epsilon)

def stochasticRanking(caller, objVals, constraintVals, minMask, epsilon=0.0, fitVals=None, **kwargs):
  """
    Runarsson & Yao stochastic ranking. The per-generation coin that decides whether constraints
    are ignored is drawn upstream in NSGAII._rankingConstraintVals (which passes constraintVals=None
    on the "ignore constraints" draw); this handler therefore only needs to honor whatever
    constraintVals it is handed, ranking by objectives alone when it is None and by constrained
    domination otherwise. Kept as a named operator so the user can request the policy explicitly.
    @ In, caller, object, the optimizer instance (unused; uniform signature)
    @ In, objVals, np.array, (nPoints, nObjectives) external-space objective values
    @ In, constraintVals, np.array or None, g(x) for this generation, or None to ignore constraints
    @ In, minMask, np.array, per-objective bool, True where minimized
    @ In, epsilon, float, optional, epsilon-constrained tolerance
    @ In, fitVals, np.array or None, optional, ignored
    @ Out, ranks, list, non-dominated front rank per point
  """
  return frontUtils.rankNonDominatedFrontiers(objVals,
                                              constraintVals=constraintVals,
                                              minMask=minMask,
                                              epsilon=epsilon)

def feasibleFirstPenalty(caller, objVals, constraintVals, minMask, epsilon=0.0, fitVals=None, **kwargs):
  """
    Deb (2000) parameter-less penalty ranking, provided for comparison against constrained
    domination. The scalar penalized fitness (feasible: -a*obj; infeasible: -a*obj_worst - b*violation)
    is computed by the <fitness> operator BEFORE this handler runs; here it is ranked larger-is-better
    via the isFitness path of frontUtils. Because the penalty already folds the constraints into the
    scalar, constraint values are NOT passed separately (frontUtils rejects constraintVals together
    with isFitness). This is a genuinely different survival criterion than Deb-2002 domination.
    @ In, caller, object, the optimizer instance (for the missing-fitness error message)
    @ In, objVals, np.array, (nPoints, nObjectives) external-space objective values (unused for ranking here)
    @ In, constraintVals, np.array or None, g(x); unused (already folded into fitVals)
    @ In, minMask, np.array, per-objective bool (unused; penalty ranks the scalar)
    @ In, epsilon, float, optional, unused
    @ In, fitVals, np.array, (nPoints, nObjectives) scalar fitness per objective; required
    @ Out, ranks, list, non-dominated front rank per point (larger fitness = better front)
  """
  if fitVals is None:
    caller.raiseAnError(IOError, 'rankingAlgorithm "feasibleFirstPenalty" requires a scalar fitness '
                        'to rank, but none was provided. Add a <fitness> node under <GAparams> '
                        '(e.g. <fitness type="feasibleFirst"/>), or switch <rankingAlgorithm> to '
                        '"constrainedDomination".')
  fit = np.asarray(fitVals, dtype=float)
  return frontUtils.rankNonDominatedFrontiers(fit, isFitness=True)

__rankingHandlers = {}
__rankingHandlers['constrainedDomination'] = constrainedDomination
__rankingHandlers['feasibleFirstPenalty'] = feasibleFirstPenalty
__rankingHandlers['epsilonConstrained'] = epsilonConstrained
__rankingHandlers['stochasticRanking'] = stochasticRanking

def returnInstance(cls, name):
  """
    Method designed to return the ranking-handler function registered under name.
    @ In, cls, class type, the caller (for raiseAnError on an unknown name)
    @ In, name, string, name of the ranking handler
    @ Out, __rankingHandlers[name], function, the ranking handler
  """
  if name not in __rankingHandlers:
    cls.raiseAnError(IOError, "{} is not a valid option for <rankingAlgorithm>. Valid options are: {}. "
                     "Please review the spelling of the ranking algorithm.".format(name, list(__rankingHandlers.keys())))
  return __rankingHandlers[name]
