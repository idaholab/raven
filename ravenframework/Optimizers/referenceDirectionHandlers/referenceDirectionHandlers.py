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
  Reference-direction handlers for the NSGA-III many-objective genetic algorithm.

  A reference-direction handler builds the set of structured reference points on the unit
  simplex that NSGA-III uses to preserve diversity in objective space (in place of NSGA-II's
  crowding distance). It is the mix-and-match operator selected via
  <referenceDirections type="..."> under <MultiObjectiveGeneticAlgorithm type="NSGA-III">.
  New schemes (e.g. Riesz s-energy) are added by defining a function with the signature below
  and registering it in the __referenceDirectionHandlers dict.

  Implemented handlers (with their literature source):
    1. dasDennis -- Das & Dennis (1998) simplex-lattice design, with the two-layer (boundary +
       inside) construction of Deb & Jain (2014) for many objectives (default).

  Handler contract:
    def handler(caller, numObjectives, populationSize, **kwargs) -> np.ndarray
      @ In, caller, object, the optimizer instance (for logging / attribute access)
      @ In, numObjectives, int, number of objectives (simplex dimension)
      @ In, populationSize, int, target population size (used to pick the lattice divisions)
      @ Out, directions, np.ndarray, (nRef, numObjectives) reference points on the unit simplex
                         (each row sums to 1); the number of rows is >= 1 and is set by the design,
                         not forced to equal populationSize.

  @authors: Mohammad Abdo (@Jimmy-INL)
"""
# External Modules----------------------------------------------------------------------------------
from math import comb
import numpy as np
# External Modules End------------------------------------------------------------------------------


def _simplexLattice(numObjectives, divisions):
  """
    Das & Dennis (1998) simplex-lattice (a.k.a. boundary) design: all points with nonnegative
    integer coordinates summing to `divisions`, scaled by 1/divisions so each point lies on the
    unit simplex (coordinates sum to 1). Produces comb(divisions + numObjectives - 1, numObjectives - 1)
    points.
    @ In, numObjectives, int, simplex dimension (number of objectives)
    @ In, divisions, int, number of divisions p along each objective
    @ Out, points, np.ndarray, (nPoints, numObjectives) lattice points on the unit simplex
  """
  points = []
  def recurse(remaining, depth, acc):
    """
      Recursively enumerate compositions of `divisions` into `numObjectives` nonnegative parts.
      @ In, remaining, int, budget left to distribute over the remaining coordinates
      @ In, depth, int, current coordinate index being filled
      @ In, acc, list, coordinates fixed so far
      @ Out, None
    """
    if depth == numObjectives - 1:
      acc.append(remaining)
      points.append(np.array(acc, dtype=float) / divisions)
      acc.pop()
      return
    for i in range(remaining + 1):
      acc.append(i)
      recurse(remaining - i, depth + 1, acc)
      acc.pop()
  recurse(divisions, 0, [])
  return np.asarray(points, dtype=float)


def dasDennis(caller, numObjectives, populationSize, **kwargs):
  """
    Structured reference directions following Das & Dennis (1998) and the two-layer construction
    of Deb & Jain (2014).

    A single boundary layer is generated with the largest number of divisions H whose point count
    comb(H + M - 1, M - 1) does not exceed the population size. For many objectives (M > 5) the
    single-layer boundary design places almost all points on the simplex edges, so Deb & Jain add a
    second, inside layer: a coarser boundary design shrunk toward the centroid of the simplex by a
    factor tau = 0.5 via x_i <- (1 - tau)/M + tau * x_i (so the inside points still sum to 1). The
    two layers are concatenated.

    @ In, caller, object, the optimizer instance (for logging)
    @ In, numObjectives, int, number of objectives
    @ In, populationSize, int, target population size (sets the boundary-layer divisions)
    @ Out, directions, np.ndarray, (nRef, numObjectives) reference points on the unit simplex
  """
  tau = 0.5  # Deb & Jain (2014) inside-layer shrink factor toward the simplex centroid

  # Boundary layer: largest divisions H1 whose lattice size fits within the population.
  divisionsBoundary = 1
  while comb(divisionsBoundary + numObjectives, numObjectives - 1) <= populationSize:
    divisionsBoundary += 1
  directions = _simplexLattice(numObjectives, divisionsBoundary)

  # Inside layer (Deb & Jain 2014): only meaningful for M > 2, and only when it buys new points.
  if numObjectives > 2:
    divisionsInside = 1
    while (len(directions) + comb(divisionsInside + numObjectives, numObjectives - 1)
           <= populationSize):
      divisionsInside += 1
    divisionsInside -= 1
    if divisionsInside >= 1:
      inside = _simplexLattice(numObjectives, divisionsInside)
      inside = (1.0 - tau) / numObjectives + tau * inside
      directions = np.vstack([directions, inside])

  if directions.size == 0:
    directions = np.eye(numObjectives)
  return directions


__referenceDirectionHandlers = {}
__referenceDirectionHandlers['dasDennis'] = dasDennis


def returnInstance(cls, name):
  """
    Return the reference-direction handler registered under `name`.
    @ In, cls, class type, the caller (for raiseAnError on an unknown name)
    @ In, name, str, the <referenceDirections type="..."> value
    @ Out, __referenceDirectionHandlers[name], function, the reference-direction handler
  """
  if name not in __referenceDirectionHandlers:
    cls.raiseAnError(IOError, "{} is not a valid option for <referenceDirections>. Valid options are: {}. "
                     "Please review the spelling of the reference-direction handler.".format(
                       name, list(__referenceDirectionHandlers.keys())))
  return __referenceDirectionHandlers[name]
