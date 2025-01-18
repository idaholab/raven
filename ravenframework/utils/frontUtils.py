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



def nonDominatedFrontier(data, returnMask,minMask=None, isFitness=False):
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
    for index,elem in enumerate(minMask):
      if not elem:
        data[:,index] = -1. * data[:,index]

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
    nextPointIndex = np.sum(nondominatedPointMask[:nextPointIndex])+1
  if returnMask:
    nonDominatedFrontierMask = np.zeros(nPoints, dtype = bool)
    nonDominatedFrontierMask[nonDominatedFrontier] = True
    return nonDominatedFrontierMask
  else:
    return nonDominatedFrontier

def rankNonDominatedFrontiers(data,isFitness=False):
  """
    This method ranks the non-dominated fronts by omitting the first front from the data
    and searching the remaining data for a new one recursively.
    @ In, data, np.array, data matrix (nPoints, nObjectives) containing the multi-objective
                          evaluations of each point/individual, element (i,j)
                          means jth objective/fitness function at the ith point/individual
    @ out, nonDominatedRank, list, a list of length nPoints that has the ranking
                                  of the front passing through each point
  """
  nonDominatedRank = np.zeros(data.shape[0], dtype=int)
  mask = np.ones(data.shape[0], dtype=bool)
  rank = 0

  while np.any(mask):
    rank += 1
    # Get non-dominated points from remaining data
    if not isFitness:
      currentFront = nonDominatedFrontier(data[mask], False)
    else:
      currentFront = nonDominatedFrontier(data[mask], False, [False] * data.shape[1], isFitness=isFitness)
    # Convert indices back to original data space
    originalIndices = np.where(mask)[0][currentFront]
    # Assign rank
    nonDominatedRank[originalIndices] = rank
    # Update mask to remove current front
    mask[originalIndices] = False

  return nonDominatedRank.tolist()

def crowdingDistance(rank, popSize, fitness):
  """
    Method designed to calculate the crowding distance for each front.
    @ In, rank, np.array or xr.DataArray, array which contains the front ID for each element of the population
    @ In, popSize, int, size of population
    @ In, fitness, np.array, matrix contains fitness values for each element of the population
    @ Out, crowdDist, np.array, array of crowding distances
  """
  if isinstance(rank,xr.DataArray):
    rank = rank.data
  crowdDist = np.zeros(popSize)
  fronts = np.unique(rank)
  fronts = fronts[fronts != np.inf]

  # Keep track of which points are on each front
  frontIndices = {f: [] for f in fronts}
  for i, r in enumerate(rank):
    frontIndices[r].append(i)

  for f in fronts:
    # front = np.where(rank == f)[0]  # Get indices of current front
    front = frontIndices[f]  # Get indices of current front
    numObjectives = fitness.shape[1]
    numPoints = len(front)
    if numPoints <= 2:  # If front has 2 or fewer points, set to infinity
      crowdDist[frontIndices[f]] = np.inf
      continue
    for obj in range(numObjectives):
      # Sort points in current front by current objective
      sortedFront = [i for i in front]
      sortedIndices = np.argsort(fitness[sortedFront, obj])
      sortedFront = [sortedFront[i] for i in sortedIndices]

      # Set boundary points to infinity
      crowdDist[sortedFront[0]] = np.inf
      crowdDist[sortedFront[-1]] = np.inf

      # Skip normalization if all values are identical
      fMax = fitness[sortedFront, obj].max()
      fMin = fitness[sortedFront, obj].min()
      if fMax == fMin:
        continue

      # Calculate normalized distances with epsilon protection
      for i in range(1, numPoints - 1):
        nextObjValue = fitness[sortedFront[i + 1], obj]
        prevObjValue = fitness[sortedFront[i - 1], obj]
        crowdDist[sortedFront[i]] += (nextObjValue - prevObjValue) / (fMax - fMin)
  return crowdDist