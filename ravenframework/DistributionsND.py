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
Created on Nov 22, 2023
@author: j-bryan
"""

import sys
import numpy as np
import scipy.stats
import scipy.linalg


class NDimensionalDistribution:
  def updateRNGparameter(self, tol, initDisc):
    pass

  def returnDimensionality(self):
    pass

  def returnLowerBound(self, dim):
    return sys.float_info.min

  def returnUpperBound(self, dim):
    return sys.float_info.max

  def marginal(self, x, variable):
    pass

  def cdf(self, x):
    pass

  def inverseCdf(self, x, rand):
    pass

  def pdf(self, x):
    pass

  def cellIntegral(self, coordinate, dxs):
    pass

  def inverseMarginal(self, x, variable):
    pass



class BasicMultiDimensionalInverseWeight(NDimensionalDistribution):
  pass


class BasicMultiDimensionalCartesianSpline(NDimensionalDistribution):
  pass


class BasicMultivariateNormal(NDimensionalDistribution):
  def __init__(self, cov, mean, covType='rel', rank=None):
    """
      Constructor

      @ In, cov, np.ndarray, the covariance matrix
      @ In, mean, np.ndarray, the mean vector
      @ In, covType, string, optional, the type of covariance matrix provided. May be 'abs' or 'rel'.
      @ In, rank, int, optional, the reduced dimension
      @ Out, None
    """
    covSymmetric = 0.5 * (cov + cov.T)  # make sure it is symmetric
    self._covarianceType = covType
    self._rank = len(mean) if rank is None else rank
    self._mu = mean
    if self._rank > len(mean):
      raise ValueError("The provided rank is larger than the given problem's dimension, it should be less or equal!")

    U, S, V = scipy.linalg.svd(covSymmetric)
    self._inverseTransformationMatrix = np.diag(S ** -0.5) @ V.T
    self._transformationMatrix = np.linalg.inv(self._inverseTransformationMatrix)

  def pdf(self, x):
    """
      Probability density function

      @ In, x, np.ndarray, the coordinates where the pdf needs to be evaluated
      @ Out, pdf, float, the pdf value
    """
    xTrans = self._transformationMatrix @ x
    return self._distribution.pdf(xTrans)

  def pdfInTransformedSpace(self, x):
    return self._distribution.pdf(x)

  def cdf(self, x):
    xTrans = self._transformationMatrix @ x
    return self.cdf(xTrans)

  def inverseCdf(self, x, rand):
    raise NotImplementedError

  def getTransformationMatrixDimensions(self, coordinateIndex):
    raise NotImplementedError

  def getTransformationMatrix(self, coordinateIndex=None):
    """
      Get the transformation matrix for the given coordinate index, if provided

      @ In, coordinateIndex, list, optional, the coordinate index
      @ Out, transformationMatrix, np.ndarray, the transformation matrix
    """
    raise NotImplementedError

  def getInverseTransformationMatrixDimensions(coordinateIndex):
    raise NotImplementedError

  def getInverseTransformationMatrix(self, coordinateIndex):
    pass

  def getSingularValues(self, coordinateIndex):
    return self._pca.singular_values_

  def coordinateInverseTransformed(self, coordinate, coordinateIndex=None):
    return self._inverseTransformationMatrix @ coordinate

  def cellProbabilityWeight(self, coordinate, dxs):
    raise NotImplementedError

  def cellIntegral(coordinate, dxs):
    raise NotImplementedError

  def marginalCdfForPCA(self, x):
    raise NotImplementedError

  def inverseMarginalForPCA(self, x):
    # bound between sys.float_info.epsilon and 1-sys.float_info.epsilon
    raise NotImplementedError

  def inverseMarginal(self, x, variable):
    raise NotImplementedError
