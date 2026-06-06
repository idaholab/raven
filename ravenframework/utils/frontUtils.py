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


def _dominatesForMinimization(candidate, other, candidateViolation=0.0, otherViolation=0.0, epsilon=0.0):
  """Evaluate Deb constrained dominance for minimization-space objective values.
     With epsilon > 0 this becomes epsilon-constrained dominance (Takahama & Sato):
     total constraint violations up to epsilon are treated as feasible, so near-boundary
     solutions compete on objectives instead of being strictly dominated by feasible ones."""
  candidateFeasible = candidateViolation <= epsilon
  otherFeasible = otherViolation <= epsilon
  if candidateFeasible and not otherFeasible:
    return True
  if not candidateFeasible and otherFeasible:
    return False
  if not candidateFeasible and not otherFeasible:
    return candidateViolation < otherViolation
  return np.all(candidate <= other) and np.any(candidate < other)


def _fastNonDominatedSortConstrained(directedData, violation, epsilon=0.0):
  """
    Fast non-dominated sort (Deb et al., 2002) with Deb constrained dominance.

    Computes, for every individual p, the set of individuals it dominates and the
    count of individuals that dominate it, in a single O(M*N^2) pairwise pass
    (M objectives, N individuals); fronts are then peeled by decrementing the
    domination counters. This yields ranks identical to the recursive peeling
    ranker but avoids its O(N^3) cost on large populations.

    @ In, directedData, np.array, (nPoints, nObjectives) objective values already
                                  converted to minimization space.
    @ In, violation, np.array, (nPoints,) total positive constraint violation per point.
    @ Out, ranks, list, 1-based front index for each point.
  """
  nPoints = directedData.shape[0]
  dominated = [[] for _ in range(nPoints)]   # individuals dominated by p
  dominationCount = np.zeros(nPoints, dtype=int)  # individuals that dominate p
  ranks = np.zeros(nPoints, dtype=int)
  currentFront = []
  for p in range(nPoints):
    for q in range(p + 1, nPoints):
      if _dominatesForMinimization(directedData[p], directedData[q], violation[p], violation[q], epsilon):
        dominated[p].append(q)
        dominationCount[q] += 1
      elif _dominatesForMinimization(directedData[q], directedData[p], violation[q], violation[p], epsilon):
        dominated[q].append(p)
        dominationCount[p] += 1
    if dominationCount[p] == 0:
      ranks[p] = 1
      currentFront.append(p)
  rank = 1
  while currentFront:
    nextFront = []
    for p in currentFront:
      for q in dominated[p]:
        dominationCount[q] -= 1
        if dominationCount[q] == 0:
          ranks[q] = rank + 1
          nextFront.append(q)
    rank += 1
    currentFront = nextFront
  return ranks.tolist()


def _rankNonDominatedFrontiersConstrained(data, constraintVals, minMask=None, epsilon=0.0):
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
  return _fastNonDominatedSortConstrained(directedData, violation, epsilon)


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


def rankNonDominatedFrontiers(data, isFitness=False, constraintVals=None, minMask=None, epsilon=0.0):
  """
    This method ranks the non-dominated fronts by omitting the first front from the data
    and searching the remaining data for a new one recursively.
    @ In, data, np.array, data matrix (nPoints, nObjectives) containing the multi-objective
                          evaluations of each point/individual, element (i,j)
                          means jth objective/fitness function at the ith point/individual
    @ In, isFitness, bool, optional, if True rank larger values as better fitness values.
    @ In, constraintVals, np.array, optional, constraint evaluations g(x); rows with all g >= 0 are feasible.
    @ In, minMask, np.array, optional, True for minimized objectives and False for maximized objectives.
    @ In, epsilon, float, optional, epsilon-constrained relaxation: total constraint violations up to
                          epsilon are treated as feasible (0.0 = strict Deb constrained dominance).
    @ Out, nonDominatedRank, list, a list of length nPoints that has the ranking
                                  of the front passing through each point
  """
  if constraintVals is not None:
    if isFitness:
      raise IOError("rankNonDominatedFrontiers method: constrained ranking expects objective values with minMask, not fitness values")
    return _rankNonDominatedFrontiersConstrained(data, constraintVals, minMask=minMask, epsilon=epsilon)

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


def crowdingDistance(rank, popSize, objectiveValues, normalizationBounds=None):
  """
    Calculate the NSGA-II crowding distance for each front.

    Crowding distance is an objective-space diversity estimate (Deb et al., 2002):
    within each non-dominated front the points are sorted along every objective and
    each interior point accrues the normalized gap between its neighbours. The two
    extreme points of each front (per objective) are assigned an infinite distance so
    they are always preserved. Note this is a measure of objective-space spread, so it
    must be fed objective values, not transformed/penalized fitness values.

    By default each objective is normalized by the range observed within the front. A
    population-level normalization (the min/max of each objective over the whole
    population) can be supplied via normalizationBounds so crowding distances are
    comparable across fronts and generations rather than rescaled per front.

    @ In, rank, np.array or xr.DataArray, array which contains the front ID for each element of the population
    @ In, popSize, int, size of population
    @ In, objectiveValues, np.array, matrix (nPoints, nObjectives) of objective values for each individual
    @ In, normalizationBounds, np.array, optional, (2, nObjectives) array whose first row is the
                               per-objective minimum and second row the per-objective maximum used to
                               normalize gaps; if None each front is normalized by its own range.
    @ Out, crowdDist, np.array, array of crowding distances
  """
  if isinstance(rank, xr.DataArray):
    rank = rank.data

  if normalizationBounds is not None:
    normalizationBounds = np.asarray(normalizationBounds, dtype=float)

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

      # Normalize by the population range when provided, else by this front's range.
      if normalizationBounds is not None:
        objMin = normalizationBounds[0, obj]
        objMax = normalizationBounds[1, obj]
      else:
        objMax = objectiveValues[sortedFront, obj].max()
        objMin = objectiveValues[sortedFront, obj].min()
      # Skip normalization if the range is degenerate
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


def updateParetoArchive(archiveObjectives, newObjectives, minMask=None, maxArchiveSize=None):
  """
    Update a Pareto archive with new candidate objective vectors.

    Stacks the current archive and the new candidates, retains only the mutually
    non-dominated set (in minimization space after applying minMask), and optionally
    truncates to maxArchiveSize by removing the most crowded points first; boundary
    points (infinite crowding distance) are always retained. An external archive lets
    NSGA-II report the best Pareto front found over the whole run, even if a later
    generation's (mu+lambda) elitism happens to drop a previously found point.

    @ In, archiveObjectives, np.array, (nArchive, nObjectives) current archive objective vectors
    @ In, newObjectives, np.array, (nNew, nObjectives) candidate objective vectors
    @ In, minMask, np.array, optional, True per minimized objective and False per maximized objective
    @ In, maxArchiveSize, int, optional, maximum number of points to retain (None = unbounded)
    @ Out, combined, np.array, (nArchive+nNew, nObjectives) stacked objective vectors
    @ Out, keptIndices, list, sorted indices into combined that form the updated archive
  """
  archiveObjectives = np.asarray(archiveObjectives, dtype=float)
  newObjectives = np.asarray(newObjectives, dtype=float)
  pieces = [arr for arr in (archiveObjectives, newObjectives) if arr.size]
  if not pieces:
    return np.empty((0, 0)), []
  combined = np.vstack(pieces)
  nonDominatedMask = nonDominatedFrontier(combined.copy(), returnMask=True, minMask=minMask)
  keptIndices = list(np.where(nonDominatedMask)[0])
  if maxArchiveSize is not None and len(keptIndices) > maxArchiveSize:
    keptObjectives = combined[keptIndices]
    ranks = np.ones(len(keptIndices), dtype=int)
    crowding = crowdingDistance(ranks, len(keptIndices), keptObjectives)
    # Keep the least crowded points (largest crowding distance); boundaries are +inf.
    order = sorted(range(len(keptIndices)), key=lambda i: crowding[i], reverse=True)
    keptIndices = [keptIndices[i] for i in order[:maxArchiveSize]]
  return combined, sorted(keptIndices)


def _meanNearestDistance(fromPoints, toPoints):
  """
    Mean over fromPoints of the Euclidean distance to the nearest point in toPoints.
    @ In, fromPoints, np.array, (nFrom, nObjectives) objective vectors to measure from
    @ In, toPoints, np.array, (nTo, nObjectives) objective vectors to measure to
    @ Out, value, float, average nearest-neighbour distance (np.inf if toPoints is empty)
  """
  fromPoints = np.atleast_2d(np.asarray(fromPoints, dtype=float))
  toPoints = np.atleast_2d(np.asarray(toPoints, dtype=float))
  if fromPoints.size == 0:
    return 0.0
  if toPoints.size == 0:
    return np.inf
  # distance from each fromPoint to every toPoint, then take the minimum per fromPoint
  diffs = fromPoints[:, None, :] - toPoints[None, :, :]
  dists = np.sqrt(np.sum(diffs ** 2, axis=2))
  return float(np.mean(np.min(dists, axis=1)))


def generationalDistance(obtainedFront, referenceFront):
  """
    Generational distance (GD): the average Euclidean distance from each obtained
    point to the nearest point of the reference (true) Pareto front. It measures how
    close the obtained solutions are to the reference front; smaller is better.
    @ In, obtainedFront, list or np.array, (nObtained, nObjectives) objective vectors found by the optimizer
    @ In, referenceFront, list or np.array, (nReference, nObjectives) reference/true Pareto front
    @ Out, gd, float, generational distance
  """
  return _meanNearestDistance(obtainedFront, referenceFront)


def invertedGenerationalDistance(obtainedFront, referenceFront):
  """
    Inverted generational distance (IGD): the average Euclidean distance from each
    reference (true) Pareto-front point to the nearest obtained point. Unlike GD it
    rewards both convergence and coverage of the whole reference front; smaller is
    better and it is the standard quality indicator when the true front is known.
    @ In, obtainedFront, list or np.array, (nObtained, nObjectives) objective vectors found by the optimizer
    @ In, referenceFront, list or np.array, (nReference, nObjectives) reference/true Pareto front
    @ Out, igd, float, inverted generational distance
  """
  return _meanNearestDistance(referenceFront, obtainedFront)


def hypervolume(points, reference):
  """
    Exact hypervolume indicator of a point set in MINIMIZATION objective space.

    The hypervolume is the measure (area in 2-D, volume in 3-D, ...) of the region
    that is dominated by ``points`` and bounded above by ``reference``. It is the
    standard unary quality indicator for multi-objective optimization: it is the
    only widely used indicator that is strictly Pareto-compliant (a set that
    dominates another never has a smaller hypervolume), so a monotonically
    increasing hypervolume is direct evidence of convergence + spread improvement.

    This routine works entirely in minimization space, so smaller objective values
    are better and the reference point must be weakly dominated by every point
    (i.e. ``reference[k] >= point[k]`` for every objective ``k``; points that do
    not strictly dominate the reference contribute nothing). It is computed with
    the Hypervolume-by-Slicing-Objectives (HSO) recursion of While et al. (2006),
    which is exact for any number of objectives.

    @ In, points, list or np.array, (nPoints, nObjectives) objective vectors in minimization space
    @ In, reference, list or np.array, (nObjectives,) reference point, no better than any point in every objective
    @ Out, hv, float, hypervolume dominated by the point set relative to the reference
  """
  reference = np.asarray(reference, dtype=float).reshape(-1)
  pts = np.asarray(points, dtype=float)
  if pts.ndim == 1:
    pts = pts.reshape((1, -1))
  if pts.size == 0:
    return 0.0
  if pts.shape[1] != reference.shape[0]:
    raise IOError("hypervolume method: points and reference have mismatched dimensions")
  # Only points that strictly dominate the reference in every objective contribute volume.
  contributing = pts[np.all(pts < reference, axis=1)]
  if contributing.shape[0] == 0:
    return 0.0
  return _hypervolumeHSO(contributing, reference)


def _hypervolumeHSO(points, reference):
  """
    Recursive Hypervolume by Slicing Objectives (HSO) in minimization space.
    Slices the dominated region along the first objective; within each slice the
    cross-section is the hypervolume of the projected points in the remaining
    objectives, recursing down to the 1-D base case. Assumes every point strictly
    dominates the reference in every (remaining) objective.
    @ In, points, np.array, (nPoints, nRemainingObjectives) objective vectors (minimization space)
    @ In, reference, np.array, (nRemainingObjectives,) reference point for the remaining objectives
    @ Out, hv, float, hypervolume of the slice
  """
  nObjectives = reference.shape[0]
  if nObjectives == 1:
    return float(reference[0] - np.min(points[:, 0]))
  ordered = points[np.argsort(points[:, 0], kind='stable')]
  nPoints = ordered.shape[0]
  hv = 0.0
  for i in range(nPoints):
    lowerEdge = ordered[i, 0]
    upperEdge = ordered[i + 1, 0] if i + 1 < nPoints else reference[0]
    width = upperEdge - lowerEdge
    if width <= 0.0:
      continue
    # Cross-section of the slice = hypervolume of points 0..i projected onto the remaining objectives.
    hv += width * _hypervolumeHSO(ordered[:i + 1, 1:], reference[1:])
  return hv
