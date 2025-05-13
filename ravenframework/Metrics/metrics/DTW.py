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
Created on August 20 2016

@author: mandd
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import scipy.spatial.distance as spatialDistance
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .MetricInterface import MetricInterface
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class DTW(MetricInterface):
  """
    Dynamic Time Warping Metric
    Class for measuring similarity between two histories X and Y. Note that it is not required for these histories
    to have the same length
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    orderInputType = InputTypes.makeEnumType("order","orderType",["0","1"])
    inputSpecification.addSub(InputData.parameterInputFactory("order",contentType=orderInputType),quantity=InputData.Quantity.one)
    inputSpecification.addSub(InputData.parameterInputFactory("localDistance",contentType=InputTypes.StringType),quantity=InputData.Quantity.one)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # order of DTW calculation, 0 specifices a classical DTW, and 1 specifies derivative DTW
    # In this case the DTW distance is calculated on the gradient of the two time series
    self.order = None
    # the ID of distance function to be employed to determine the local distance evaluation of two time series
    # Available options are provided by scipy pairwise distances, i.e. cityblock, cosine, euclidean, manhattan.
    self.localDistance = None
    # True indicates the metric needs to be able to handle dynamic data
    self._dynamicHandling = True
    # True indicates the metric needs to be able to handle pairwise data
    self._pairwiseHandling = True

  def handleInput(self, paramInput):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, paramInput, InputData.parameterInput, input spec
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == "order":
        self.order = int(child.value)
      elif child.getName() == "localDistance":
        self.localDistance = child.value

  def run(self, x, y, axis=0, returnPath=False, **kwargs):
    """
      This method computes DTW distance between two inputs x and y based on given metric
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided,
        the array will be reshaped via x.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_time_steps)
      @ In, y, numpy.ndarray, array containing data of y, if 1D array is provided,
        the array will be reshaped via y.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_time_steps)
      @ In, returnPath, bool, optional, return the DTW path
      @ In, axis, integer, optional, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows").
        If metric postprocessor is used, the first dimension is the RAVEN_sample_ID,
        and the second dimension is the pivotParameter if HistorySet is provided.
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, float (if returnPath=False), DTW distance value
      @ Out, value, tuple=(distance,path) (if returnPath=True), tuple containing: 1) DTW distance value, 2) DTW path along the distance matrix 
    """
    assert (isinstance(x, np.ndarray))
    assert (isinstance(x, np.ndarray))
    if axis == 1:
      assert(x.shape[1] == y.shape[1]), self.raiseAnError(IOError, "The second dimension of first input is not \
              the same as the second dimension of second input!")
    tempX = x.T
    tempY = y.T

    if len(tempX.shape) == 1:
      tempX = tempX.reshape(-1,1)
    if len(tempY.shape) == 1:
      tempY = tempY.reshape(-1,1)

    X = np.empty(tempX.shape)
    Y = np.empty(tempY.shape)

    if self.order == 1:
      for index in range(tempX.shape[1]):
        X[:,index] = np.gradient(tempX[:,index])
        Y[:,index] = np.gradient(tempY[:,index])
    else:
      X = tempX
      Y = tempY

    if returnPath:
      value = self.dtwDistance(X, Y, returnPath=True)
    else:
      value = self.dtwDistance(X, Y)
    return value

  def dtwDistance(self, x, y, returnPath=False):
    """
      This method actually calculates the distance between two histories x and y
      @ In, x, numpy.ndarray, data matrix for x
      @ In, y, numpy.ndarray, data matrix for y
      @ In, returnPath, bool, optional, return the DTW path
      @ Out, value, float, distance between x and y calculates as the sum of the local distance
                           associated with each element of the DTW path
      @ Out, path, numpy.ndarray (P,2), DTW path along the D matrix where P is the length of the DTW path (if returnPath=True)
    """
    # Initialize the distance matrix
    r, c = len(x[:,0]), len(y[:,0])
    D0 = np.zeros((r+1, c+1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    # Compute the local distance between each element of the two histories x and y.
    # This matrix has shape (n,m) where n and m are the length of the two time series
    D1 = spatialDistance.cdist(x,y, metric=self.localDistance)
    D0[1:, 1:] = D1

    # Populate the cost matrix by iteratively creating the path from (0,0) to (n,m) that minimizes
    # the sum of the local distances of the elements that are part of the path
    for i in range(1, r+1):
      for j in range(1, c+1):
        # Given a path element at coordinate (i,j), the next element is the
        # minimum of the elements of coordinates (i-1,j), (i,j-1), (i-1,j-1)
        D0[i, j] += min(D0[i-1, j], D0[i, j-1], D0[i-1, j-1])

    value = D0[r, c]

    if returnPath:
      path = self.tracePath(D0)
      return value, path
    else:
      return value

  def tracePath(self, D):
    """
      This method calculates the indexes of the time warping path given a local distance matrix D.
      The DTW path is a path in the distance matrix that starts at (0,0) and ends at (n,m).
      The length P of the path is case dipendent.
      @ In, D,  numpy.ndarray (2D), local distance matrix D
      @ Out, warpingPath, numpy.ndarray (P,2), DTW path along the D matrix where P is the length of the DTW path
    """
    i, j = D.shape
    i = i-1
    j = j-1
    warpingPath = []
    while i > 0 or j > 0:
      warpingPath.append((i-1, j-1))
      if i > 0 and j > 0:
        min_cost = min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        if min_cost == D[i-1, j-1]:
          i, j = i-1, j-1
        elif min_cost == D[i-1, j]:
          i = i-1
        else:
          j = j-1
      elif i > 0:
        i = i-1
      else:
        j = j-1
    warpingPath.reverse()

    return np.array(warpingPath)
