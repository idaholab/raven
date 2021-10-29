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
# Internal Imports



def nonDominatedFrontier(data, returnMask, minMask=None):
  """
    This method is designed to identify the set of non-dominated points (nEfficientPoints)

    If returnMask=True, then a True/False mask (isEfficientMask) is returned
    Non-dominated points pFront can be obtained as follows:
      mask = nonDominatedFrontier(data,True)
      pFront = data[np.array(mask)]

    If returnMask=False, then an array of integer values containing the indexes of the non-dominated points is returned
    Non-dominated points pFront can be obtained as follows:
      mask = nonDominatedFrontier(data,False)
      pFront = data[np.array(mask)]

    @ In, data, np.array, data matrix (nPoints, nCosts) containing the data points
    @ In, returnMask, bool, type of data to be returned: indices (False) or True/False mask (True)
    @ Out, minMask, np.array, array (nCosts,1) of boolean values: True (dimension need to be minimized), False (dimension need to be maximized)
    @ Out, isEfficientMask , np.array, data matrix (nPoints,1), array  of boolean values if returnMask=True
    @ Out, isEfficient, np.array, data matrix (nEfficientPoints,1), integer array of indexes if returnMask=False

    Reference: the following code has been adapted from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
  """
  if minMask is None:
    pass
  elif minMask is not None and minMask.shape[0] != data.shape[1]:
    raise IOError("nonDominatedFrontier method: Data features do not match minMask dimensions: data has shape " + str(data.shape) + " while minMask has shape " + str(minMask.shape))
  else:
    for index,elem in np.ndenumerate(minMask):
      if not elem:
        data[:,index] = -1. * data[:,index]

  isEfficient = np.arange(data.shape[0])
  nPoints = data.shape[0]
  nextPointIndex = 0
  while nextPointIndex<len(data):
    nondominatedPointMask = np.any(data<data[nextPointIndex], axis=1)
    nondominatedPointMask[nextPointIndex] = True
    isEfficient = isEfficient[nondominatedPointMask]
    data = data[nondominatedPointMask]
    nextPointIndex = np.sum(nondominatedPointMask[:nextPointIndex])+1
  if returnMask:
    isEfficientMask = np.zeros(nPoints, dtype = bool)
    isEfficientMask[isEfficient] = True
    return isEfficientMask
  else:
    return isEfficient

def rankNonDominatedFrontiers(data):
  """
    This method ranks the non dominated fronts by omitting thr first front from the data
    and searching the remaining data for a new one recursively.
    @ In, data, np.array, data matrix (nPoints, nObjectives) containing the multi-objective
                          evaluations of each point/individual, element (i,j)
                          means jth objective function at the ith point/individual
    @ out, nonDominatedRank, list, a list of length nPoints that has the ranking
                                  of the front passing through each point
  """
  nonDominatedRank = np.zeros(data.shape[0],dtype=int)
  rank = 0
  indicesDominated = list(np.arange(data.shape[0]))
  indicesNonDominated = []
  rawData = data
  while np.shape(data)[0] > 0:
    rank += 1
    indicesNonDominated = list(nonDominatedFrontier(data, False))
    if rank > 1:
      for i in range(len(indicesNonDominated)):
        indicesNonDominated[i] = indicesDominated[indicesNonDominated[i]]
    indicesDominated = list(set(indicesDominated)-set(indicesNonDominated))
    data = rawData[indicesDominated]
    nonDominatedRank[indicesNonDominated] = rank
  nonDominatedRank = list(nonDominatedRank)
  return nonDominatedRank

def crowdingDistance(rank, popSize, objectives):
  """
    Method designed to calculate the crowding distance for each front
    @ In, rank, np.array, array which contains the front ID for each element of the population
    @ In, popSize, int, size of population
    @ In, objectives, np.array, matrix contains objective values for each element of the population
    @ Out, crowdDist, np.array, array of crowding distances
  """
  crowdDist = np.zeros(popSize)
  fronts = np.unique(rank)
  fronts = fronts[fronts!=np.inf]

  for f in range(len(fronts)):
    front = np.where(np.asarray(rank)==f+1)[0]
    fMax = np.max(objectives[front, :], axis=0)
    fMin = np.min(objectives[front, :], axis=0)
    for obj in range(np.shape(objectives)[1]):
      sortedRank = np.argsort(objectives[front, obj])
      crowdDist[front[sortedRank[0]]] = crowdDist[front[sortedRank[-1]]] = np.inf
      for i in range(1, len(front)-1):
        crowdDist[front[sortedRank[i]]] = crowdDist[front[sortedRank[i]]] + (objectives[front[sortedRank[i+1]], obj] - objectives[front[sortedRank[i-1]], obj]) / (fMax[obj]-fMin[obj])
  return crowdDist
