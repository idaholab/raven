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
  This file contains the mathematical methods used in the framework.
  Specifically for metric algorithms (distances, etc)
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import copy
import numpy as np
from scipy import stats

def calculateStats(data):
  """
    Calculate statistics on a numeric array data
    and return them in a dictionary
    @ In, data, list or numpy.array, the data
    @ Out, ret, dict, the dictionary containing the stats
  """
  ret = {}
  ret["mean"] = np.mean(data)
  ret["variance"] = np.var(data)
  ret["sampleVariance"] = stats.tvar(data)
  ret["stdev"] = stats.tstd(data)
  ret["skewness"] = stats.skew(data)
  ret["kurtosis"] = stats.kurtosis(data)
  return ret

def diffWithInfinites(a, b):
  """
    Calculates the difference a-b and treats infinites.  We consider infinites to have equal values, but
    inf - (- inf) = inf.
    @ In, a, float, first value (could be infinite)
    @ In, b, float, second value (could be infinite)
    @ Out, res, float, b-a (could be infinite)
  """
  if abs(a) == np.inf or abs(b) == np.inf:
    if a == b:
      res = 0 #not mathematically rigorous, but useful algorithmically
    elif a > b:
      res = np.inf
    else: # b > a
      res = -np.inf
  else:
    res = a - b
  return res

def relativeDiff(f1, f2):
  """
    Given two floats, safely compares them to determine relative difference.
    @ In, f1, float, first value (the value to compare to f2, "measured")
    @ In, f2, float, second value (the value being compared to, "actual")
    @ Out, relativeDiff, float, (safe) relative difference
  """
  if not isinstance(f1, float):
    try:
      f1 = float(f1)
    except ValueError:
      raise RuntimeError('Provided argument to compareFloats could not be cast as a float!  First argument is %s type %s' %(str(f1),type(f1)))
  if not isinstance(f2, float):
    try:
      f2 = float(f2)
    except ValueError:
      raise RuntimeError('Provided argument to compareFloats could not be cast as a float!  Second argument is %s type %s' %(str(f2),type(f2)))
  diff = abs(diffWithInfinites(f1, f2))
  #"scale" is the relative scaling factor
  scale = f2
  #protect against div 0
  if f2 == 0.0:
    #try using the "measured" for scale
    if f1 != 0.0:
      scale = f1
    #at this point, they're both equal to zero, so just divide by 1.0
    else:
      scale = 1.0
  if abs(scale) == np.inf:
    #no mathematical rigor here, but typical algorithmic use cases
    if diff == np.inf:
      return np.inf # assumption: inf/inf = 1
    else:
      return 0.0 # assumption: x/inf = 0 for all finite x
  return diff/abs(scale)

def compareFloats(f1, f2, tol=1e-6):
  """
    Given two floats, safely compares them to determine equality to provided relative tolerance.
    @ In, f1, float, first value (the value to compare to f2, "measured")
    @ In, f2, float, second value (the value being compared to, "actual")
    @ In, tol, float, optional, relative tolerance to determine match
    @ Out, compareFloats, bool, True if floats close enough else False
  """
  diff = relativeDiff(f1,f2)
  return diff < tol

def computeTruncatedTotalLeastSquare(X, Y, truncationRank):
  """
    Compute Total Least Square and truncate it till a rank = truncationRank
    @ In, X, numpy.ndarray, the first 2D matrix
    @ In, Y, numpy.ndarray, the second 2D matrix
    @ In, truncationRank, int, optional, the truncation rank
    @ Out, (dX,dy), tuple, the Leasted squared matrices X and Y
  """
  V = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)[-1]
  rank = min(int(truncationRank), V.shape[0])
  VV = V[:rank, :].conj().T.dot(V[:rank, :])
  dX = X.dot(VV)
  dY = Y.dot(VV)
  return dX, dY

def hyperdiagonal(lengths):
  """
    Obtains the length of a diagonal of a hyperrectangle given the lengths of the sides.  Useful for high-dimensional distance scaling.
    @ In, lengths, list(float), lengths of the sides of the ND rectangle
    @ Out, diag, float, the length of the diagonal between furthest-separated corners of the hypercube
  """
  try:
    return np.sqrt(np.sum(lengths*lengths))
  except TypeError:
    lengths = np.asarray(lengths)
    return np.sqrt(np.sum(lengths*lengths))

def calculateMagnitudeAndVersor(vector, normalizeInfinity=True):
  """
    Generates a magnitude and versor for provided vector.
    @ In, vector, list or np.array, potentially mixed float/np.array elements
    @ In, normalizeInfinity, bool, optional, if True then normalize vector if infinites present
    @ Out, mag, float, magnitude of vector
    @ Out, versor, np.array, vector divided by magnitude
    @ Out, foundInf, bool, if True than infinity calcs were used
  """
  # protect original data
  vector = copy.deepcopy(vector)
  # check if infinites were detected
  foundInf = False
  mag = calculateMultivectorMagnitude(vector)
  if normalizeInfinity and mag == np.inf:
    foundInf = True
    for e, entry in enumerate(vector):
      # if we're working with infinites, then recalculate by "dividing by infinity"
      vector[e][-np.inf < entry < np.inf] = 0.0
      # since np.inf / np.inf is nan, manually define quotient as 1
      vector[e][entry == np.inf] = 1.0
      vector[e][entry == -np.inf] = -1.0
    mag = calculateMultivectorMagnitude(vector)
  # create versor (if divisor is not zero)
  if mag != 0.0:
    for e, entry in enumerate(vector):
      vector[e] = entry / mag
      # fix up vector/scalars: len 1 vectors are scalars
      #if len(entry) == 1:
      #  vector[e] = float(vector[e])
  return mag, vector, foundInf

def calculateMultivectorMagnitude(vector):
  """
    Given a list of potentially mixed scalars and numpy arrays, obtains the magnitude as if every
    entry were part of one larger array
    @ In, vector, list, mixed float/np.array elements where all scalars should be treated with same weighting
    @ Out, mag, float, magnitude of combined elements
  """
  # reshape so numpy can perform Frobenius norm
  # -> do this by calculating the norm of vector entries first
  # --> note that np norm fails on length-1 arrays so we protect against that
  new = [np.linalg.norm(x) if len(np.atleast_1d(x)) > 1 else np.atleast_1d(x)[0] for x in vector]
  mag = np.linalg.norm(new)
  return mag

def angleBetweenVectors(a, b):
  """
    Calculates the angle between two N-dimensional vectors in DEGREES
    @ In, a, np.array, vector of floats
    @ In, b, np.array, vector of floats
    @ Out, ang, float, angle in degrees
  """
  nVar = len(a)
  # if either vector is all zeros, then angle between is also
  normA = np.linalg.norm(a)
  normB = np.linalg.norm(b)
  if normA == 0:
    ang = 0
  elif normB == 0:
    ang = 0
  else:
    dot = np.dot(a, b) / normA / normB
    ang = np.arccos(np.clip(dot, -1, 1))
  ang = np.rad2deg(ang)
  return ang

def computeTruncatedSingularValueDecomposition(X, truncationRank):
  """
    Compute Singular Value Decomposition and truncate it till a rank = truncationRank
    @ In, X, numpy.ndarray, the 2D matrix on which the SVD needs to be performed
    @ In, truncationRank, int or float, optional, the truncation rank:
                                                  * -1 = no truncation
                                                  *  0 = optimal rank is computed
                                                  *  >1  user-defined truncation rank
                                                  *  >0. and < 1. computed rank is the number of the biggest sv needed to reach
                                                                  the energy identified by truncationRank
    @ Out, (U, s, V), tuple of numpy.ndarray, (left-singular vectors matrix, singular values, right-singular vectors matrix)
  """
  U, s, V = np.linalg.svd(X, full_matrices=False)
  V = V.conj().T

  if truncationRank is 0:
    omeg = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
    rank = np.sum(s > np.median(s) * omeg(np.divide(*sorted(X.shape))))
  elif truncationRank > 0 and truncationRank < 1:
    rank = np.searchsorted(np.cumsum(s / s.sum()), truncationRank) + 1
  elif truncationRank >= 1 and isinstance(truncationRank, int):
    rank = min(truncationRank, U.shape[1])
  else:
    rank = X.shape[1]
  U = U[:, :rank]
  V = V[:, :rank]
  s = s[:rank]
  return U, s, V

def computeEigenvaluesAndVectorsFromLowRankOperator(lowOperator, Y, U, s, V, exactModes=True):
  """
    Compute the eigenvalues and eigenvectors of the high-dim operator
    from the low-dim operator and the matrix Y.
    The lowe-dim operator can be computed with the following numpy-based
    expression: U.T.conj().dot(Y).dot(V) * np.reciprocal(s)
    @ In, lowOperator, numpy.ndarray, the lower rank operator (a tilde)
    @ In, Y, numpy.ndarray, the input matrix Y
    @ In, U, numpy.ndarray,  2D matrix that contains the left-singular vectors of X, stored by column
    @ In, s, numpy.ndarray,  1D array  that contains the singular values of X
    @ In, V, numpy.ndarray,  2D matrix that contains the right-singular vectors of X, stored by column
    @ In, exactModes, bool, optional, if True the exact modes get computed otherwise the projected ones are (Default = True)
    @ Out, (eigvals,eigvects), tuple (numpy.ndarray,numpy.ndarray), eigenvalues and eigenvectors
  """
  lowrankEigenvals, lowrankEigenvects = np.linalg.eig(lowOperator)
  # Compute the eigvects and eigvals of the high-dimensional operator
  eigvects = ((Y.dot(V) * np.reciprocal(s)).dot(lowrankEigenvects)) if exactModes else U.dot(lowrankEigenvects)
  eigvals  = lowrankEigenvals.astype(complex)
  return eigvals, eigvects

def computeAmplitudeCoefficients(mods, Y, eigs, optmized):
  """
    @ In, mods, numpy.ndarray, 2D matrix that contains the modes (by column)
    @ In, Y, numpy.ndarray, 2D matrix that contains the input matrix (by column)
    @ In, eigs, numpy.ndarray, 1D array that contains the eigenvalues
    @ In, optmized, bool, if True  the amplitudes are computed minimizing the error between the mods and all entries (columns) in Y
                          if False the amplitudes are computed minimizing the error between the mods and the 1st entry (columns) in Y (faster)
    @ Out, amplitudes, numpy.ndarray, 1D array containing the amplitude coefficients
  """
  if optmized:
    L = np.concatenate([mods.dot(np.diag(eigs**i)) for i in range(Y.shape[1])], axis=0)
    amplitudes = np.linalg.lstsq(L, np.reshape(Y, (-1, ), order='F'))[0]
  else:
    amplitudes = np.linalg.lstsq(mods, Y.T[0])[0]
  return amplitudes
