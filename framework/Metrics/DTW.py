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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import scipy.spatial.distance as spatialDistance
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric
#Internal Modules End--------------------------------------------------------------------------------

class DTW(Metric):
  """
    Dynamic Time Warping Metric
    Class for measuring similarity between two variables X and Y, i.e. two temporal sequences
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Metric.__init__(self)
    # order of DTW calculation, 0 specifices a classical DTW, and 1 specifies derivative DTW
    self.order            = None
    # the ID of distance function to be employed to determine the local distance evaluation of two time series
    # Available options are provided by scipy pairwise distances, i.e. cityblock, cosine, euclidean, manhattan.
    self.localDistance    = None
    # True indicates the metric needs to be able to handle dynamic data
    self._dynamicHandling = True
    # True indicates the metric needs to be able to handle pairwise data
    self._pairwiseHandling = True

  def _localReadMoreXML(self, xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.requiredKeywords = set(['order','localDistance'])
    self.wrongKeywords = set()
    for child in xmlNode:
      if child.tag == 'order':
        if child.text in ['0','1']:
          self.order = float(child.text)
        else:
          self.raiseAnError(IOError,'DTW metrics - specified order ' + str(child.text) + ' is not recognized (allowed values are 0 or 1)')
        self.requiredKeywords.remove('order')
      if child.tag == 'localDistance':
        self.localDistance = child.text
        self.requiredKeywords.remove('localDistance')
      if child.tag not in self.requiredKeywords:
        self.wrongKeywords.add(child.tag)

    if self.requiredKeywords:
      self.raiseAnError(IOError,'The DTW metrics is missing the following parameters: ' + str(self.requiredKeywords))
    if not self.wrongKeywords:
      self.raiseAnError(IOError,'The DTW metrics block contains parameters that are not recognized: ' + str(self.wrongKeywords))

  def __evaluateLocal__(self, x, y, weights = None, axis = 0, **kwargs):
    """
      This method computes DTW distance between two inputs x and y based on given metric
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided,
        the array will be reshaped via x.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_time_steps)
      @ In, y, numpy.ndarray, array containing data of y, if 1D array is provided,
        the array will be reshaped via y.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_time_steps)
      @ In, weights, array_like (numpy.array or list), optional weights associated
        with input, shape (n_samples) if axis = 0, otherwise shape (n_time_steps)
      @ In, axis, integer, optional, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows").
        If metric postprocessor is used, the first dimension is the RAVEN_sample_ID,
        and the second dimension is the pivotParameter if HistorySet is provided.
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, float, metric result
    """
    assert (isinstance(x, np.ndarray))
    assert (isinstance(x, np.ndarray))
    tempX = copy.copy(x)
    tempY = copy.copy(y)
    if axis == 0:
      assert (len(x) == len(y))
    elif axis == 1:
      assert(x.shape[1] == y.shape[1]), self.raiseAnError(IOError, "The second dimension of first input is not \
              the same as the second dimension of second input!")
      tempX = tempX.T
      tempY = tempY.T
    else:
      self.raiseAnError(IOError, "Valid axis value should be '0' or '1' for the evaluate method of metric", self.name)

    if len(tempX.shape) == 1:
      tempX = tempX.reshape(1,-1)
    if len(tempY.shape) == 1:
      tempY = tempY.reshape(1,-1)
    X = np.empty(tempX.shape)
    Y = np.empty(tempY.shape)
    for index in range(len(tempX)):
      if self.order == 1:
        X[index] = np.gradient(tempX[index])
        Y[index] = np.gradient(tempY[index])
      else:
        X[index] = tempX[index]
        Y[index] = tempY[index]
    value = self.dtwDistance(X, Y)
    return value

  def dtwDistance(self, x, y):
    """
      This method actually calculates the distance between two histories x and y
      @ In, x, numpy.ndarray, data matrix for x
      @ In, y, numpy.ndarray, data matrix for y
      @ Out, value, float, distance between x and y
    """
    r, c = len(x[0,:]), len(y[0,:])
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    D1 = spatialDistance.cdist(x.T,y.T, metric=self.localDistance)
    C = D1.copy()
    for i in range(r):
      for j in range(c):
        D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
      path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
      path = range(len(x)), np.zeros(len(x))
    else:
      path = self.tracePath(D0)
    return D1[-1, -1]

  def tracePath(self, D):
    """
      This method calculate the time warping path given a local distance matrix D
      @ In, D,  numpy.ndarray (2D), local distance matrix D
      @ Out, p, numpy.ndarray (1D), path along horizontal direction
      @ Out, q, numpy.ndarray (1D), path along vertical direction
    """
    i,j = np.array(D.shape) - 2
    p,q = [i], [j]
    while ((i > 0) or (j > 0)):
      tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
      if (tb == 0):
        i -= 1
        j -= 1
      elif (tb == 1):
        i -= 1
      else:
        j -= 1
      p.insert(0, i)
      q.insert(0, j)
    return np.array(p), np.array(q)
