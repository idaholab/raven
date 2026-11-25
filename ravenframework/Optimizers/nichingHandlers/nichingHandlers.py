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
  Niching handlers for the NSGA-III many-objective genetic algorithm.

  A niching handler fills the final (splitting) front's survivor slots by balancing the reference
  directions: it repeatedly picks the least-represented reference direction and takes one of its
  associated candidates. It is the mix-and-match operator selected via <niching type="..."> under
  <MultiObjectiveGeneticAlgorithm type="NSGA-III">.

  Implemented handlers (with their literature source):
    1. referencePoint              -- Deb & Jain (2014) Algorithm 4 niche-preservation: for the chosen
       reference direction, if its niche count rho == 0 take the associated candidate of minimum
       perpendicular distance, otherwise take a RANDOM associated candidate (default).
    2. referencePointDeterministic -- same selection order, but always take the minimum-distance
       candidate (no randomness), so a run is reproducible without seeding the niche tie-break.

  Handler contract:
    def handler(caller, front, slots, normalized, selectedIndices, nicheCounts,
                referenceDirections, associate, **kwargs) -> (list, dict, np.ndarray)
      @ In, caller, object, the optimizer instance (for logging and the RNG, caller.raiseAWarning etc.)
      @ In, front, list, global indices of the splitting front's members
      @ In, slots, int, number of survivors to select from this front
      @ In, normalized, np.ndarray, (nTotal, nObjectives) normalized objectives for the whole combined pop
      @ In, selectedIndices, list, global indices already selected from earlier (fully fitting) fronts
      @ In, nicheCounts, np.ndarray, (nRef,) niche counts contributed by earlier fronts (not mutated)
      @ In, referenceDirections, np.ndarray, (nRef, nObjectives) reference points
      @ In, associate, callable, association handler (caller, points, referenceDirections) -> (idx, dist)
      @ Out, selected, list, global indices chosen from this front (len == min(slots, len(front)))
      @ Out, distanceCache, dict, {globalIndex: perpendicular distance} for the chosen points
      @ Out, countsUpdate, np.ndarray, (nRef,) added niche counts from this front's selections

  @authors: Mohammad Abdo (@Jimmy-INL)
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils import randomUtils
# Internal Modules End------------------------------------------------------------------------------


def _niche(caller, front, slots, normalized, selectedIndices, nicheCounts,
           referenceDirections, associate, randomizeWhenOccupied):
  """
    Shared NSGA-III niche-preservation loop. `randomizeWhenOccupied` selects the literature behavior
    (random candidate when the niche is already occupied) versus the deterministic variant.
    @ In, caller, object, optimizer instance (for the RNG via caller.getRandomState / randomUtils)
    @ In, front, list, global indices of the splitting front
    @ In, slots, int, survivors to pick from this front
    @ In, normalized, np.ndarray, normalized objectives for the whole combined population
    @ In, selectedIndices, list, indices already selected from earlier fronts
    @ In, nicheCounts, np.ndarray, niche counts from earlier fronts
    @ In, referenceDirections, np.ndarray, reference points
    @ In, associate, callable, association handler
    @ In, randomizeWhenOccupied, bool, True for Deb-Jain random pick when rho>0, False for min-distance
    @ Out, selected, list, chosen global indices
    @ Out, distanceCache, dict, {globalIndex: distance}
    @ Out, countsUpdate, np.ndarray, added niche counts
  """
  numRef = len(referenceDirections)
  rho = nicheCounts.astype(float, copy=True)
  countsUpdate = np.zeros(numRef, dtype=int)
  selected = []
  distanceCache = {}

  # Reference-direction niche counts seeded by the already-selected (earlier-front) members.
  if selectedIndices:
    assocSelected, _ = associate(caller, normalized[selectedIndices], referenceDirections)
    for idx in assocSelected:
      rho[idx] += 1

  # Associate every member of the splitting front; group candidates by reference direction.
  assocFront, distFront = associate(caller, normalized[front], referenceDirections)
  candidates = {i: [] for i in range(numRef)}
  for globalIdx, directionIdx, distance in zip(front, assocFront, distFront):
    candidates[directionIdx].append((globalIdx, distance))

  while len(selected) < slots:
    finiteRho = rho[np.isfinite(rho)]
    if finiteRho.size == 0:
      break
    minCount = np.min(finiteRho)
    candidateDirs = [idx for idx in range(numRef) if np.isfinite(rho[idx]) and rho[idx] == minCount]
    if not candidateDirs:
      break
    # Among the least-represented directions, deterministically take the lowest-index one that still
    # has an unselected candidate (ties on rho are broken by direction index for reproducibility).
    chosenDir = None
    pool = None
    for direction in candidateDirs:
      available = [item for item in candidates[direction] if item[0] not in selected]
      if available:
        chosenDir = direction
        pool = available
        break
      rho[direction] = np.inf  # exhausted: never pick this direction again
    if chosenDir is None:
      continue

    if rho[chosenDir] == 0 or not randomizeWhenOccupied:
      # Empty niche (Deb-Jain) or deterministic variant: take the minimum-distance candidate.
      chosen = min(pool, key=lambda item: (item[1], item[0]))
    else:
      # Occupied niche (Deb-Jain): take a random associated candidate. Draw through randomUtils so
      # the choice obeys RAVEN's seeded RNG, consistent with the parent-selection/crossover operators.
      choice = int(randomUtils.randomIntegers(0, len(pool) - 1, caller))
      chosen = pool[choice]

    globalIdx, distance = chosen
    selected.append(globalIdx)
    distanceCache[globalIdx] = distance
    rho[chosenDir] += 1
    countsUpdate[chosenDir] += 1

  return selected, distanceCache, countsUpdate


def referencePoint(caller, front, slots, normalized, selectedIndices, nicheCounts,
                   referenceDirections, associate, **kwargs):
  """
    Deb & Jain (2014) niche preservation: min-distance when the niche is empty, random when occupied.
    @ In, caller, object, optimizer instance
    @ In, front, list, splitting-front global indices
    @ In, slots, int, survivors to select
    @ In, normalized, np.ndarray, normalized objectives
    @ In, selectedIndices, list, earlier-front selections
    @ In, nicheCounts, np.ndarray, earlier-front niche counts
    @ In, referenceDirections, np.ndarray, reference points
    @ In, associate, callable, association handler
    @ Out, (selected, distanceCache, countsUpdate), tuple, see module contract
  """
  return _niche(caller, front, slots, normalized, selectedIndices, nicheCounts,
                referenceDirections, associate, randomizeWhenOccupied=True)


def referencePointDeterministic(caller, front, slots, normalized, selectedIndices, nicheCounts,
                                referenceDirections, associate, **kwargs):
  """
    Deterministic variant of referencePoint: always take the minimum-distance candidate. Produces
    reproducible survivor sets without seeding the niche tie-break (useful for gold-based tests).
    @ In, caller, object, optimizer instance
    @ In, front, list, splitting-front global indices
    @ In, slots, int, survivors to select
    @ In, normalized, np.ndarray, normalized objectives
    @ In, selectedIndices, list, earlier-front selections
    @ In, nicheCounts, np.ndarray, earlier-front niche counts
    @ In, referenceDirections, np.ndarray, reference points
    @ In, associate, callable, association handler
    @ Out, (selected, distanceCache, countsUpdate), tuple, see module contract
  """
  return _niche(caller, front, slots, normalized, selectedIndices, nicheCounts,
                referenceDirections, associate, randomizeWhenOccupied=False)


__nichingHandlers = {}
__nichingHandlers['referencePoint'] = referencePoint
__nichingHandlers['referencePointDeterministic'] = referencePointDeterministic


def returnInstance(cls, name):
  """
    Return the niching handler registered under `name`.
    @ In, cls, class type, the caller (for raiseAnError on an unknown name)
    @ In, name, str, the <niching type="..."> value
    @ Out, __nichingHandlers[name], function, the niching handler
  """
  if name not in __nichingHandlers:
    cls.raiseAnError(IOError, "{} is not a valid option for <niching>. Valid options are: {}. "
                     "Please review the spelling of the niching handler.".format(
                       name, list(__nichingHandlers.keys())))
  return __nichingHandlers[name]
