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
  Repository of utils for non-dominated and Pareto frontier methods
  Created  Feb 18, 2020
  @authors: Diego Mandelli and Mohammad Abdo
"""
# External Imports
import numpy as np
import xarray as xr
# Internal Imports

def _constraintViolation(constraintVals):
  """Return total positive constraint violation for each point. RAVEN constraints are feasible when g(x) >= 0."""
  if constraintVals is None:
    return None
  values = np.asarray(constraintVals)
  if values.ndim == 1:
    values = values.reshape((-1, 1))
  if values.size == 0:
    return np.zeros(values.shape[0], dtype=float)
  return np.sum(np.maximum(0.0, -values), axis=1)


def _applyObjectiveDirections(data, minMask=None):
  """Convert objective values to minimization-space for dominance checks using an explicit direction mask."""
  data = np.array(data, dtype=float, copy=True)
  if minMask is None:
    return data
  minMask = np.asarray(minMask, dtype=bool)
  if len(minMask) != data.shape[1]:
    raise IOError("rankNonDominatedFrontiers method: Data features do not match minMask dimensions")
  data[:, ~minMask] *= -1.0
  return data


def _dominatesForMinimization(candidate, other, candidateViolation=0.0, otherViolation=0.0):
  """Evaluate Deb constrained dominance for minimization-space objective values."""
  candidateFeasible = candidateViolation <= 0.0
  otherFeasible = otherViolation <= 0.0
  if candidateFeasible and not otherFeasible:
    return True
  if not candidateFeasible and otherFeasible:
    return False
  if not candidateFeasible and not otherFeasible:
    return candidateViolation < otherViolation
  return np.all(candidate <= other) and np.any(candidate < other)


def _rankNonDominatedFrontiersConstrained(data, constraintVals, minMask=None):
  """Rank fronts using objective dominance plus Deb constrained-dominance rules."""
  data = np.asarray(data, dtype=float)
  if data.ndim != 2:
    raise IOError("rankNonDominatedFrontiers method: data must be a 2-D array")
  violation = _constraintViolation(constraintVals)
  if violation is None:
    violation = np.zeros(data.shape[0], dtype=float)
  if len(violation) != data.shape[0]:
    raise IOError("rankNonDominatedFrontiers method: constraint rows do not match data rows")
  directedData = _applyObjectiveDirections(data, minMask)

  ranks = np.zeros(data.shape[0], dtype=int)
  remaining = set(range(data.shape[0]))
  rank = 0
  while remaining:
    rank += 1
    front = []
    for candidate in sorted(remaining):
      dominated = False
      for other in remaining:
        if other == candidate:
          continue
        if _dominatesForMinimization(directedData[other], directedData[candidate], violation[other], violation[candidate]):
          dominated = True
          break
      if not dominated:
        front.append(candidate)
    if not front:
      raise RuntimeError("No non-dominated front could be identified.")
    for index in front:
      ranks[index] = rank
      remaining.remove(index)
  return ranks.tolist()


def nonDominatedFrontier(data, returnMask, minMask=None, isFitness=False):
  """
    This method identifies the set of non-dominated points (nEfficientPoints).

    If returnMask=True, a True/False mask (nonDominatedFrontierMask) is returned.
    Non-dominated points pFront can be obtained as follows:
      mask = nonDominatedFrontier(data, True)
      pFront = data[np.array(mask)]

    If returnMask=False, an array of integer values containing the indexes of the non-dominated points is returned.
    Non-dominated points pFront can be obtained as follows:
      mask = nonDominatedFrontier(data, False)
      pFront = data[np.array(mask)]

    @ In, data, np.array, data matrix (nPoints, nCosts) containing the data points
    @ In, returnMask, bool, type of data to be returned: indices (False) or True/False mask (True)
    @ In, minMask, np.array, array (nCosts,) of boolean values: True (dimension needs to be minimized), False (dimension needs to be maximized)
    @ In, isFitness, boolean, if True, thus means the data is fitness values, otherwise, objective values (i.e., do not include penalties from constraint violation))
    @ Out, nonDominatedFrontierMask, np.array, data matrix (nPoints,), array of boolean values if returnMask=True
    @ Out, nonDominatedFrontier, np.array, data matrix (nEfficientPoints,), integer array of indexes if returnMask=False

    Reference: Adapted from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
  """

  if minMask is None:
    pass
  elif minMask is not None and len(minMask) != data.shape[1]:
    raise IOError("nonDominatedFrontier method: Data features do not match minMask dimensions: data has shape " + str(data.shape) + " while minMask has shape " + str(minMask.shape))
  elif not isFitness:
    for index, elem in enumerate(minMask):
      if not elem:
        data[:, index] = -1. * data[:, index]

  nPoints = data.shape[0]
  nonDominatedFrontier = np.arange(nPoints)
  nextPointIndex = 0

  while nextPointIndex < np.shape(data)[0]:
    if not isFitness:
      nondominatedPointMask = np.any(data < data[nextPointIndex], axis=1) | np.all(data == data[nextPointIndex], axis=1)

    else:
      nondominatedPointMask = np.any(data > data[nextPointIndex], axis=1) | np.all(data == data[nextPointIndex], axis=1)
    nonDominatedFrontier = nonDominatedFrontier[nondominatedPointMask]
    data = data[nondominatedPointMask]
    nextPointIndex = np.sum(nondominatedPointMask[:nextPointIndex]) + 1

  if returnMask:
    nonDominatedFrontierMask = np.zeros(nPoints, dtype=bool)
    nonDominatedFrontierMask[nonDominatedFrontier] = True
    return nonDominatedFrontierMask
  else:
    return nonDominatedFrontier


def rankNonDominatedFrontiers(data, isFitness=False, constraintVals=None, minMask=None):
  """
    This method ranks the non-dominated fronts by omitting the first front from the data
    and searching the remaining data for a new one recursively.
    @ In, data, np.array, data matrix (nPoints, nObjectives) containing the multi-objective
                          evaluations of each point/individual, element (i,j)
                          means jth objective/fitness function at the ith point/individual
    @ In, isFitness, bool, optional, if True rank larger values as better fitness values.
    @ In, constraintVals, np.array, optional, constraint evaluations g(x); rows with all g >= 0 are feasible.
    @ In, minMask, np.array, optional, True for minimized objectives and False for maximized objectives.
    @ Out, nonDominatedRank, list, a list of length nPoints that has the ranking
                                  of the front passing through each point
  """
  if constraintVals is not None:
    if isFitness:
      raise IOError("rankNonDominatedFrontiers method: constrained ranking expects objective values with minMask, not fitness values")
    return _rankNonDominatedFrontiersConstrained(data, constraintVals, minMask=minMask)

  nonDominatedRank = np.zeros(data.shape[0], dtype=int)
  mask = np.ones(data.shape[0], dtype=bool)
  rank = 0

  while np.any(mask):
    rank += 1
    # Get non-dominated points from remaining data
    if not isFitness:
      currentFront = nonDominatedFrontier(data[mask].copy(), False, minMask=minMask)
    else:
      currentFront = nonDominatedFrontier(data[mask], False, [False] * data.shape[1], isFitness=isFitness)
    # Convert indices back to original data space
    originalIndices = np.where(mask)[0][currentFront]
    # Assign rank
    nonDominatedRank[originalIndices] = rank
    # Update mask to remove current front
    mask[originalIndices] = False

  return nonDominatedRank.tolist()


def crowdingDistance(rank, popSize, objectiveValues):
  """
    Calculate the NSGA-II crowding distance for each front.

    Crowding distance is an objective-space diversity estimate (Deb et al., 2002):
    within each non-dominated front the points are sorted along every objective and
    each interior point accrues the normalized gap between its neighbours. The two
    extreme points of each front (per objective) are assigned an infinite distance so
    they are always preserved. Note this is a measure of objective-space spread, so it
    must be fed objective values, not transformed/penalized fitness values.

    @ In, rank, np.array or xr.DataArray, array which contains the front ID for each element of the population
    @ In, popSize, int, size of population
    @ In, objectiveValues, np.array, matrix (nPoints, nObjectives) of objective values for each individual
    @ Out, crowdDist, np.array, array of crowding distances
  """
  if isinstance(rank, xr.DataArray):
    rank = rank.data

  crowdDist = np.zeros(popSize)
  fronts = np.unique(rank)
  fronts = fronts[fronts != np.inf]

  # Keep track of which points are on each front
  frontIndices = {f: [] for f in fronts}
  for i, r in enumerate(rank):
    frontIndices[r].append(i)

  for f in fronts:
    front = frontIndices[f]  # Get indices of current front
    numObjectives = objectiveValues.shape[1]
    numPoints = len(front)

    # Special case: fronts with <= 2 points; every member is a boundary point
    if numPoints <= 2:
      crowdDist[front] = np.inf
      continue

    # For each objective, calculate crowding distance contribution
    for obj in range(numObjectives):
      # Sort points in current front by current objective
      sortedFront = [i for i in front]
      sortedIndices = np.argsort(objectiveValues[sortedFront, obj], kind='stable')
      sortedFront = [sortedFront[i] for i in sortedIndices]

      # Only the actual boundary points (first and last after sorting) get infinity
      crowdDist[sortedFront[0]] = np.inf   # Minimum boundary
      crowdDist[sortedFront[-1]] = np.inf  # Maximum boundary

      # Skip normalization if all values are identical
      objMax = objectiveValues[sortedFront, obj].max()
      objMin = objectiveValues[sortedFront, obj].min()
      if objMax == objMin:
        continue

      # Calculate normalized distances for interior points
      for i in range(1, numPoints - 1):
        # Skip if already set to infinity (can happen if point is boundary in another objective)
        if crowdDist[sortedFront[i]] != np.inf:
          nextObjValue = objectiveValues[sortedFront[i + 1], obj]
          prevObjValue = objectiveValues[sortedFront[i - 1], obj]
          # Add normalized distance for this objective
          crowdDist[sortedFront[i]] += (nextObjValue - prevObjValue) / (objMax - objMin)

  return crowdDist
