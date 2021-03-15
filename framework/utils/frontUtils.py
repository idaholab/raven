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
  @authors: Diego Mandelli
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
    raise IOError("nonDominatedFrontier method: minMask has shape " + str(data.shape) + " while minMask has shape " + str(minMask.shape))
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
