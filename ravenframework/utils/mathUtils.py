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
 Some of the methods were in the PostProcessor.py
 created on 03/26/2015
 @author: senrs
"""

import math
import functools
import copy
import scipy
from scipy import interpolate, stats
import numpy as np
import six
from numpy import linalg

from .utils import UreturnPrintTag, UreturnPrintPostTag
from .graphStructure import graphObject

from .. import MessageHandler # makes sure getMessageHandler is defined
mh = getMessageHandler()

# dict of builtin types, filled by getBuiltinTypes
_bTypes = None

def normal(x, mu=0.0, sigma=1.0):
  """
    Computation of normal pdf
    @ In, x, list or np.array, x values
    @ In, mu, float, optional, mean
    @ In, sigma, float, optional, sigma
    @ Out, returnNormal, list or np.array, pdf
  """
  return stats.norm.pdf(x, mu, sigma)

def normalCdf(x, mu=0.0, sigma=1.0):
  """
    Computation of normal cdf
    @ In, x, list or np.array, x values
    @ In, mu, float, optional, mean
    @ In, sigma, float, optional, sigma
    @ Out, cdfReturn, list or np.array, cdf
  """
  return stats.norm.cdf(x, mu, sigma)

def skewNormal(x, alphafactor, xi, omega):
  """
    Computation of skewness normal
    @ In, x, list or np.array, x values
    @ In, alphafactor, float, the alpha factor (shape)
    @ In, xi, float, xi (location)
    @ In, omega, float, omega factor (scale)
    @ Out, returnSkew, float, skew
  """
  returnSkew = (2.0/omega)*normal((x-xi)/omega)*normalCdf(alphafactor*(x-xi)/omega)

  return returnSkew

def createInterp(x, y, lowFill, highFill, kind='linear'):
  """
    Creates an interpolation function that uses lowFill and highFill whenever a value is requested that lies outside of the range specified by x.
     @ In, x, list or np.array, x values
     @ In, y, list or np.array, y values
     @ In, lowFill, float, minimum interpolated value
     @ In, highFill, float, maximum interpolated value
     @ In, kind, string, optional, interpolation type (default=linear)
     @ Out, interp, function(float) returns float, an interpolation function that takes a single float value and return its interpolated value using lowFill or highFill when the input value is outside of the interpolation range.
  """
  sv = str(scipy.__version__).split('.')
  if int(sv[0])> 0 or int(sv[1]) >= 17:
    interp = interpolate.interp1d(x, y, kind, bounds_error=False, fill_value=([lowFill], [highFill]))
    return interp
  interp = interpolate.interp1d(x, y, kind)
  low = x[0]
  def myInterp(value):
    """
      @ In, value, float, value to interpolate
      @ Out, interpolatedValue, float, interpolated value corresponding to value
    """
    try:
      return interp(value)+0.0 ## why plus 0.0? Could this be done by casting as a float?
                               ## maljdp: I believe this is catching edge cases
                               ## in order to throw them into the except clause
                               ## below, but I am not sure it is the best
                               ## solution here.
    except ValueError:
      if value <= low:
        return lowFill
      else:
        return highFill

  return myInterp

def countBins(sortedData, binBoundaries):
  """
    This method counts the number of data items in the sorted_data
    Returns an array with the number.  ret[0] is the number of data
    points <= binBoundaries[0], ret[len(binBoundaries)] is the number
    of points > binBoundaries[len(binBoundaries)-1]
    @ In, sortedData, list or np.array,the data to be analyzed
    @ In, binBoundaries, list or np.array, the bin boundaries
    @ Out, ret, list, the list containing the number of bins
  """
  binIndex = 0
  sortedIndex = 0
  ret = [0]*(len(binBoundaries)+1)
  while sortedIndex < len(sortedData):
    while not binIndex >= len(binBoundaries) and \
          sortedData[sortedIndex] > binBoundaries[binIndex]:
      binIndex += 1
    ret[binIndex] += 1
    sortedIndex += 1

  return ret

def log2(x):
  """
   Compute log2
   @ In, x, float, the coordinate x
   @ Out, logTwo, float, log2
  """
  logTwo = math.log(x, 2)

  return logTwo

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

def hyperdiagonal(lengths):
  """
    Obtains the length of a diagonal of a hyperrectangle given the lengths of the sides.  Useful for high-dimensional distance scaling.
    @ In, lengths, list(float), lengths of the sides of the ND rectangle
    @ Out, diag, float, the length of the diagonal between farthest-separated corners of the hypercube
  """
  try:
    return np.sqrt(np.sum(lengths*lengths))
  except TypeError:
    lengths = np.asarray(lengths)
    return np.sqrt(np.sum(lengths*lengths))

def historySnapShoots(valueDict, numberOfTimeStep):
  """
    Method do to compute the temporal slices of each history
    @ In, valueDict, dict, the dictionary containing the data to be sliced {'varName':[history1Values,history2Values, etc.]}
    @ In, numberOfTimeStep, int, number of time samples of each history
    @ Out, outDic, list, it contains the temporal slice of all histories
  """
  outDict = []
  numberOfRealizations = len(list(valueDict.values())[-1])
  outPortion, inPortion = {}, {}
  numberSteps = - 1
  # check consistency of the dictionary
  for variable, value in valueDict.items():
    if len(value) != numberOfRealizations:
      return "historySnapShoots method: number of realizations are not consistent among the different parameters!!!!"
    if type(value).__name__ in 'list':
      # check the time-step size
      outPortion[variable] = np.asarray(value)
      if numberSteps == -1:
        numberSteps = functools.reduce(lambda x, y: x*y, list(list(outPortion.values())[-1].shape))/numberOfRealizations
      if len(list(outPortion[variable].shape)) != 2:
        return "historySnapShoots method: number of time steps are not consistent among the different histories for variable "+variable
      if functools.reduce(lambda x, y:
        x*y, list(list(outPortion.values())[-1].shape))/numberOfRealizations != numberSteps :
        return "historySnapShoots method: number of time steps are not consistent among the different histories for variable "+variable+". Expected "+str(numberSteps)+" /= "+ sum(list(outPortion[variable].shape))/numberOfRealizations
    else:
      inPortion [variable] = np.asarray(value)
  for ts in range(numberOfTimeStep):
    realizationSnap = {}
    realizationSnap.update(inPortion)
    for variable in outPortion:
      realizationSnap[variable] = outPortion[variable][:,ts]
    outDict.append(realizationSnap)

  return outDict

# def historySetWindow(vars,numberOfTimeStep):
#   """
#     Method do to compute the temporal slices of each history
#     @ In, vars, HistorySet, is an historySet
#     @ In, numberOfTimeStep, int, number of time samples of each history
#     @ Out, outDic, list, it contains the temporal slice of all histories
#   """
#
#   outKeys = vars.getParaKeys('outputs')
#   inpKeys = vars.getParaKeys('inputs')
#
#   outDic = []
#
#   for t in range(numberOfTimeStep):
#     newVars={}
#     for key in inpKeys+outKeys:
#       newVars[key]=np.zeros(0)
#
#     hs = vars.getParametersValues('outputs')
#     for history in hs:
#       for key in inpKeys:
#         newVars[key] = np.append(newVars[key],vars.getParametersValues('inputs')[history][key])
#
#       for key in outKeys:
#         newVars[key] = np.append(newVars[key],vars.getParametersValues('outputs')[history][key][t])
#
#     outDic.append(newVars)
#
#   return outDic

def normalizationFactors(values, mode='z'):
  """
    Method to normalize data based on various criteria.
    @ In, values, list,  data for which to obtain normalization factors
    @ In, mode, str, the mode of normalization to perform, e.g.: z = z-score
      standardization, 'scale' = 0,1 scaling of the data, anything else will
      be ignored and the values returned will not alter the data, namely offset
      of zero and a scale of 1.
    @ Out, (offset,scale), 2-tuple of floats, the first represents the offset
      for the data, and the latter represents the scaling factor.
      i.e., (x - offset)/ scale
  """
  if mode is None:
    mode = 'none'

  if mode == 'z':
    offset = np.average(values)
    scale = np.std(values)
  elif mode == 'scale':
    offset = np.min(values)
    scale = np.max(values) - offset
  else:
    offset = 0.0
    scale = 1.0

  # All of the values must be the same, okay just take the scale of the data
  # to be the maximum value
  if scale == 0:
    scale = np.max(np.absolute(values))

  # All of the values must be zero, okay use 1 to prevent a 0/0 issue
  if scale == 0:
    scale = 1.0

  return (offset, scale)

#
# I need to convert it in multi-dimensional
# Not a priority yet. Andrea
#
# def computeConcaveHull(coordinates,alphafactor):
#   """
#    Method to compute the Concave Hull of a cloud of points
#    @ In, coordinates, matrix-like, (M,N) -> M = number of coordinates, N, number of dimensions
#    @ In, alphafactorfactor, float, shape factor tollerance to influence the gooeyness of the border.
#   """
#   def add_edge(edges, edge_points, coords, i, j):
#     """
#     Add a line between the i-th and j-th points,
#     if not in the list already
#     """
#     if (i, j) in edges or (j, i) in edges: return
#     edges.add( (i, j) )
#     edge_points.append(coords[ [i, j] ])
#
#   #coords = np.array([point.coords[0] for point in points])
#
#   tri = Delaunay(coordinates)
#   edges = set()
#   edge_points = []
#   # loop over triangles:
#   # ia, ib, ic = indices of corner points of the
#   # triangle
#   for ia, ib, ic in tri.simplices:
#     pa = coordinates[ia]
#     pb = coordinates[ib]
#     pc = coordinates[ic]
#
#     # Lengths of sides of triangle
#     a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
#     b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
#     c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
#
#     # Semiperimeter of triangle
#     s = (a + b + c)/2.0
#
#     # Area of triangle by Heron's formula
#     area = math.sqrt(s*(s-a)*(s-b)*(s-c))
#     circum_r = a*b*c/(4.0*area)
#
#     # Here's the radius filter.
#     #print circum_r
#     if circum_r < 1.0/alphafactor:
#       add_edge(edges, edge_points, coordinates, ia, ib)
#       add_edge(edges, edge_points, coordinates, ib, ic)
#       add_edge(edges, edge_points, coordinates, ic, ia)
#
#   m = geometry.MultiLineString(edge_points)
#   triangles = list(polygonize(m))
#   return cascaded_union(triangles), edge_points

def convertNumpyToLists(inputDict):
  """
    Method aimed to convert a dictionary containing numpy
    arrays or a single numpy array in list
    @ In, inputDict, dict or numpy array,  object whose content needs to be converted
    @ Out, response, dict or list, same object with its content converted
  """
  returnDict = inputDict
  if isinstance(inputDict, dict):
    for key, value in inputDict.items():
      if isinstance(value, np.ndarray):
        returnDict[key] = value.tolist()
      elif isinstance(value, dict):
        returnDict[key] = (convertNumpyToLists(value))
      else:
        returnDict[key] = value
  elif isinstance(inputDict, np.ndarray):
    returnDict = inputDict.tolist()

  return returnDict

def interpolateFunction(x, y, option, z=None, returnCoordinate=False):
  """
    Method to interpolate 2D/3D points
    @ In, x, ndarray or cached_ndarray, the array of x coordinates
    @ In, y, ndarray or cached_ndarray, the array of y coordinates
    #FIXME missing option
    @ In, z, ndarray or cached_ndarray, optional, the array of z coordinates
    @ In, returnCoordinate, bool, optional, true if the new coordinates need to be returned
    @ Out, i, ndarray or cached_ndarray or tuple, the interpolated values
  """
  options = copy.copy(option)
  if x.size <= 2:
    xi = x
  else:
    xi = np.linspace(x.min(), x.max(), int(options['interpPointsX']))
  if z is not None:
    if y.size <= 2:
      yi = y
    else:
      yi = np.linspace(y.min(), y.max(), int(options['interpPointsY']))
    xig, yig = np.meshgrid(xi, yi)
    try:
      if ['nearest', 'linear', 'cubic'].count(options['interpolationType']) > 0 or z.size <= 3:
        if options['interpolationType'] != 'nearest' and z.size > 3:
          zi = interpolate.griddata((x,y), z, (xi[None,:], yi[:,None]), method=options['interpolationType'])
        else:
          zi = interpolate.griddata((x,y), z, (xi[None,:], yi[:,None]), method='nearest')
      else:
        rbf = interpolate.Rbf(x, y, z, function=str(str(options['interpolationType']).replace('Rbf', '')), epsilon=int(options.pop('epsilon',2)), smooth=float(options.pop('smooth',0.0)))
        zi  = rbf(xig, yig)
    except Exception as ae:
      if 'interpolationTypeBackUp' in options.keys():
        print(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        zi = interpolateFunction(x, y, z, options)
      else:
        raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
    if returnCoordinate:
      return xig, yig, zi
    else:
      return zi
  else:
    try:
      if ['nearest', 'linear', 'cubic'].count(options['interpolationType']) > 0 or y.size <= 3:
        if options['interpolationType'] != 'nearest' and y.size > 3:
          yi = interpolate.griddata((x), y, (xi[:]), method=options['interpolationType'])
        else:
          yi = interpolate.griddata((x), y, (xi[:]), method='nearest')
      else:
        xig, yig = np.meshgrid(xi, yi)
        rbf = interpolate.Rbf(x, y, function=str(str(options['interpolationType']).replace('Rbf', '')), epsilon=int(options.pop('epsilon',2)), smooth=float(options.pop('smooth',0.0)))
        yi  = rbf(xi)
    except Exception as ae:
      if 'interpolationTypeBackUp' in options.keys():
        print(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        yi = interpolateFunction(x,y,options)
      else:
        raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
    if returnCoordinate:
      return xi, yi
    else:
      return yi

def distance(points, pt):
  """
    Calculates the Euclidean distances between the points in "points" and the point "pt".
    @ In, points, np.array(tuple/list/array), list of points
    @ In, pt, tuple/list/array(int/float), point of distance
    @ Out, distance, np.array(float), distances
  """
  return np.linalg.norm(points-pt, axis=1)

def numpyNearestMatch(findIn, val):
  """
    Given an array, find the entry that most nearly matches the given value.
    @ In, findIn, np.array, the array to look in
    @ In, val, float or other compatible type, the value for which to find a match
    @ Out, returnMatch, tuple, index where match is and the match itself
  """
  findIn = np.asarray(findIn)
  if len(findIn.shape) == 1:
    findIn = findIn.reshape(-1,1)
  dist = distance(findIn,val)
  idx = dist.argmin()
  returnMatch = idx,findIn[idx]

  return returnMatch

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
      raise RuntimeError(f'Provided argument to compareFloats could not be cast as a float!  First argument is {f1} type {type(f1)}')
  if not isinstance(f2, float):
    try:
      f2 = float(f2)
    except ValueError:
      raise RuntimeError(f'Provided argument to compareFloats could not be cast as a float!  Second argument is {f2} type {type(f2)}')
  diff = abs(diffWithInfinites(f1, f2))
  # "scale" is the relative scaling factor
  scale = f2
  # protect against div 0
  if f2 == 0.0:
    # try using the "measured" for scale
    if f1 != 0.0:
      scale = f1
    # at this point, they're both equal to zero, so just divide by 1.0
    else:
      scale = 1.0
  if abs(scale) == np.inf:
    # no mathematical rigor here, but typical algorithmic use cases
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
  diff = relativeDiff(f1, f2)

  return diff < tol

def NDInArray(findIn, val, tol=1e-12):
  """
    checks a numpy array of numpy arrays for a near match, then returns info.
    @ In, findIn, np.array, numpy array of numpy arrays (both arrays can be any length)
    @ In, val, tuple/list/numpy array, entry to look for in findIn
    @ In, tol, float, optional, tolerance to check match within
    @ Out, (bool,idx,looking) -> (found/not found, index where found or None, findIn entry or None)
  """
  if len(findIn)<1:
    return False, None, None
  found = False
  for idx,looking in enumerate(findIn):
    num = looking - val
    den = np.array(val)
    #div 0 error
    for i, v in enumerate(den):
      if v == 0.0:
        if looking[i] != 0:
          den[i] = looking[i]
        elif looking[i] + den[i] != 0.0:
          den[i] = 0.5*(looking[i] + den[i])
        else:
          den[i] = 1
    if np.all(abs(num / den)<tol):
      found = True
      break
  if not found:
    return False, None, None
  return found, idx, looking

def numBinsDraconis(data, low=None, alternateOkay=True, binOps=None):
  """
    Determine  Bin size and number of bins determined by Freedman Diaconis rule (https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule)
    @ In, data, np.array, data to be binned
    @ In, low, int, minimum number of bins
    @ In, alternateOkay, bool, if True then can use alternate method if Freeman Draconis won't work
    @ In, binOps, int, optional, optional method choice for computing optimal bins
    @ Out, numBins, int, optimal number of bins
    @ Out, binEdges, np.array, location of the bins
  """
  # binOps: default to draconis, but allow other options
  # TODO additional options could be easily added in the future.
  # option 2: square root rule
  if binOps == 2:
    numBins = int(np.ceil(np.sqrt(data.size)))
  # default option: try draconis, then fall back on square root rule
  else:
    try:
      iqr = np.percentile(data, 75) - np.percentile(data, 25)
    # Freedman Diaoconis assumes there's a difference between the 75th and 25th percentile (there usually is)
      if iqr > 0.0:
        size = 2.0 * iqr / np.cbrt(data.size)
        numBins = int(np.ceil((max(data) - min(data))/size))
      else:
        raise TypeError
    except:
    # if there's not, with approval we can use the sqrt of the number of entries instead
      if alternateOkay:
        numBins = int(np.ceil(np.sqrt(data.size)))
      else:
        raise ValueError('When computing bins using Freedman-Diaconis the 25th and 75th percentiles are the same, and "alternate" is not enabled!')
  # if a minimum number of bins have been suggested, check that we use enough
  if low is not None:
    numBins = max(numBins, low)
  # for convenience, find the edges of the bins as well
  binEdges = np.linspace(start=np.asarray(data).min(), stop=np.asarray(data).max(), num=numBins+1)

  return numBins, binEdges

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

def computeTruncatedSingularValueDecomposition(X, truncationRank, full=False, conj=True):
  """
    Compute Singular Value Decomposition and truncate it till a rank = truncationRank
    @ In, X, numpy.ndarray, the 2D matrix on which the SVD needs to be performed
    @ In, truncationRank, int or float, optional, the truncation rank:
                                                  * -1 = no truncation
                                                  *  0 = optimal rank is computed
                                                  *  >1  user-defined truncation rank
                                                  *  >0. and < 1. computed rank is the number of the biggest sv needed to reach the energy identified by truncationRank
    @ In, full, bool, optional, compute svd returning full matrices
    @ In, conj, bool, optional, compute conjugate of right-singular vectors matrix)
    @ Out, (U, s, V), tuple of numpy.ndarray, (left-singular vectors matrix, singular values, right-singular vectors matrix)
  """
  try:
    U, s, V = np.linalg.svd(X, full_matrices=full)
  except np.linalg.LinAlgError as ae:
    raise np.linalg.LinAlgError(str(ae))
  V = V.conj().T if conj else V.T

  if truncationRank == 0:
    omeg = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
    rank = np.sum(s > np.median(s) * omeg(np.divide(*sorted(X.shape))))
  elif truncationRank > 0 and truncationRank < 1:
    rank = np.searchsorted(np.cumsum(s / s.sum()), truncationRank) + 1
  elif truncationRank >= 1 and isinstance(truncationRank, int):
    rank = min(truncationRank, U.shape[1])
  else:
    rank = U.shape[1]
  U = U[:, :rank]
  V = V[:, :rank]
  s = np.diag(s)[:rank, :rank] if full else s[:rank]

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

def computeAmplitudeCoefficients(mods, Y, eigs, optimized):
  """
    @ In, mods, numpy.ndarray, 2D matrix that contains the modes (by column)
    @ In, Y, numpy.ndarray, 2D matrix that contains the input matrix (by column)
    @ In, eigs, numpy.ndarray, 1D array that contains the eigenvalues
    @ In, optimized, bool, if True  the amplitudes are computed minimizing the error between the mods and all entries (columns) in Y
                          if False the amplitudes are computed minimizing the error between the mods and the 1st entry (columns) in Y (faster)
    @ Out, amplitudes, numpy.ndarray, 1D array containing the amplitude coefficients
  """
  if optimized:
    L = np.concatenate([mods.dot(np.diag(eigs**i)) for i in range(Y.shape[1])], axis=0)
    amplitudes = np.linalg.lstsq(L, np.reshape(Y, (-1, ), order='F'))[0]
  else:
    amplitudes = np.linalg.lstsq(mods, Y.T[0])[0]

  return amplitudes

def trainEmpiricalFunction(signal, bins=None, minBins=None, weights=None):
  """
    Creates a scipy empirical distribution object with all the associated methods (pdf, cdf, ppf, etc).
    Note this is only partially covered (while extended to include weights) by methods in raven/framework/Metrics/MetricUtilities,
    and ideally those methods can be generalized and extended to be included here, or in Distributions.  See issue #908.
    @ In, signal, np.array(float), signal to create distribution for
    @ In, bins, int, optional, number of bins to use
    @ In, minBins, int, optional, minimum number of bins to use
    @ In, weights, np.array(float), optional, weights for each sample within the distribution
    @ Out, dist, scipy.stats.rv_histogram instance, distribution object instance based on input data
    @ Out, histogram, tuple, (counts, edges) the frequency and bins of the histogram
  """
  # determine the number of bins to use in the empirical distribution
  if bins is None:
    bins, _ = numBinsDraconis(signal, low=minBins)
  counts, edges = np.histogram(signal, bins=bins, density=False, weights=weights)
  counts = np.asarray(counts) / float(len(signal))
  dist = stats.rv_histogram((counts, edges))

  return dist, (counts, edges)

def convertSinCosToSinPhase(A, B):
  """
    Given coefficients A, B for the equation A*sin(kt) = B*cos(kt), returns
    the equivalent values C, p for the equation C*sin(kt + p)
    @ In, A, float, sine coefficient
    @ In, B, float, cosine coefficient
    @ Out, C, float, equivalent sine-only amplitude
    @ Out, p, float, phase shift of sine-only waveform
  """
  p = np.arctan2(B, A)
  C = A / np.cos(p)

  return C, p

def evalFourier(period, C, p, t):
  """
    Evaluate Fourier Singal by coefficients C, p, t for the equation C*sin(kt + p)
    @ In, C, float, equivalent sine-only amplitude
    @ In, p, float, phase shift of sine-only waveform
    @ In, t, np.array, list of values for the time
    @ Out fourier, np.array, results of the transfered signal
  """
  fourier = C * np.sin(2. * np.pi * t / period + p)

  return fourier

def orderClusterLabels(originalLabels):
  """
    Regulates labels such that the first unique one to appear is 0, second one is 1, and so on.
    e.g. [B, B, C, B, A, A, D] becomes [0, 0, 1, 0, 2, 2, 3]
    @ In, originalLabels, list, the original labeling system
    @ Out, labels, np.array(int), ordinal labels
  """
  labels = np.zeros(len(originalLabels), dtype=int)
  oldToNew = {}
  nextUsableLabel = 0
  for l, old in enumerate(originalLabels):
    new = oldToNew.get(old, None)
    if new is None:
      oldToNew[old] = nextUsableLabel
      new = nextUsableLabel
      nextUsableLabel += 1
    labels[l] = new

  return labels

# determining types
# NOTE: REQUIRES six and numpy, do not move to utils!
def isSingleValued(val, nanOk=True, zeroDOk=True):
  """
    Determine if a single-entry value (by traditional standards).
    Single entries include strings, numbers, NaN, inf, None
    NOTE that Python usually considers strings as arrays of characters.  Raven doesn't benefit from this definition.
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ In, zeroDOk, bool, optional, if True then a zero-d numpy array with a single-valued entry is A-OK
    @ Out, isSingleValued, bool, result
  """
  # TODO most efficient order for checking?
  if zeroDOk:
    # if a zero-d numpy array, then technically it's single-valued, but we need to get into the array
    val = npZeroDToEntry(val)

  return isAFloatOrInt(val,nanOk=nanOk) or isABoolean(val) or isAString(val) or (val is None)

def isAString(val):
  """
    Determine if a string value (by traditional standards).
    @ In, val, object, check
    @ Out, isAString, bool, result
  """
  return isinstance(val, six.string_types)

def isAFloatOrInt(val, nanOk=True):
  """
    Determine if a float or integer value
    Should be faster than checking (isAFloat || isAnInteger) due to checking against numpy.number
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ Out, isAFloatOrInt, bool, result
  """
  return isAnInteger(val, nanOk) or isAFloat(val, nanOk)

def isAFloat(val, nanOk=True):
  """
    Determine if a float value (by traditional standards).
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ Out, isAFloat, bool, result
  """
  if isinstance(val,(float,np.number)):
    # exclude ints, which are numpy.number
    if isAnInteger(val):
      return False
    # numpy.float32 (or 16) is niether a float nor a numpy.float (it is a numpy.number)
    if nanOk:
      return True
    elif val not in [np.nan, np.inf]:
      return True

  return False

def isAnInteger(val, nanOk=False):
  """
    Determine if an integer value (by traditional standards).
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ Out, isAnInteger, bool, result
  """
  if isinstance(val, six.integer_types) or isinstance(val, np.integer):
    # exclude booleans
    if isABoolean(val):
      return False
    return True
  # also include inf and nan, if requested
  if nanOk and isinstance(val,float) and val in [np.nan, np.inf]:
    return True

  return False

def isABoolean(val):
  """
    Determine if a boolean value (by traditional standards).
    @ In, val, object, check
    @ Out, isABoolean, bool, result
  """
  return isinstance(val, (bool, np.bool_))

def npZeroDToEntry(a):
  """
    Cracks the shell of the numpy array and gets the sweet sweet value inside
    @ In, a, object, thing to crack open (might be anything, hopefully a zero-d numpy array)
    @ Out, a, object, thing that was inside the thing in the first place
  """
  if isinstance(a, np.ndarray) and a.shape == ():
    # make the thing we're checking the thing inside to the numpy array
    a = a.item()

  return a

def toListFromNumpyOrC1array(array):
  """
    This method converts a numpy or c1darray into list
    @ In, array, numpy or c1array,  array to be converted
    @ Out, response, list, the casted value
  """
  response = array
  if type(array).__name__ == 'ndarray':
    response = array.tolist()
  elif type(array).__name__.split(".")[0] == 'c1darray':
    response = np.asarray(array).tolist()
  return response

def toListFromNumpyOrC1arrayIterative(array):
  """
    Method aimed to convert all the string-compatible content of
    an object (dict, list, or string) in type list from numpy and c1darray types (recursively call toBytes(s))
    @ In, array, object,  object whose content needs to be converted
    @ Out, response, object, a copy of the object in which the string-compatible has been converted
  """
  if isinstance(array, list):
    return [toListFromNumpyOrC1array(x) for x in array]
  elif isinstance(array, dict):
    if len(array.keys()) == 0:
      return None
    tempdict = {}
    for key, value in array.items():
      tempdict[np.toBytes(key)] = toListFromNumpyOrC1arrayIterative(value)
    return tempdict
  else:
    return np.toBytes(array)

def sizeMatch(var, sizeToCheck):
  """
    This method is aimed to check if a variable has an expected size
    @ In, var, python datatype, the first variable to compare
    @ In, sizeToCheck, int, the size this variable should have
    @ Out, sizeMatched, bool, is the size ok?
  """
  sizeMatched = True
  if len(np.atleast_1d(var)) != sizeToCheck:
    sizeMatched = False

  return sizeMatched

def readVariableGroups(xmlNode):
  """
    Reads the XML for the variable groups and initializes them
    Placed in mathUtils because it uses VariableGroups, which inherit from BaseClasses
    -> and hence all the rest of the required libraries.
    NOTE: maybe we should have a thirdPartyUtils that is different from utils and mathUtils?
    @ In, xmlNode, ElementTree.Element, xml node to read in
    @ Out, varGroups, dict, dictionary of variable groups (names to the variable lists to replace the names)
  """
  from .. import VariableGroups
  # first find all the names
  names = [node.attrib['name'] for node in xmlNode]

  # find dependencies
  deps = {}
  nodes = {}
  initials = []
  for child in xmlNode:
    name = child.attrib['name']
    nodes[name] = child
    if child.text is None:
      needs = [''] # needs to be an empty string, not simply []
    else:
      needs = [s.strip().strip('-+^%') for s in child.text.split(',')]
    for n in needs:
      if n not in deps and n not in names:
        deps[n] = []
    deps[name] = needs
    if len(deps[name]) == 0:
      initials.append(name)
  graph = graphObject(deps)
  # sanity checking
  if graph.isALoop():
    mh.error('mathUtils', IOError, 'VariableGroups have circular dependency!')
  # ordered list (least dependencies first)
  hierarchy = list(reversed(graph.createSingleListOfVertices(graph.findAllUniquePaths(initials))))

  # build entities
  varGroups = {}
  for name in hierarchy:
    if len(deps[name]):
      varGroup = VariableGroups.VariableGroup()
      varGroup.readXML(nodes[name], varGroups)
      varGroups[name] = varGroup

  return varGroups

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

def angleBetweenVectors(a, b):
  """
    Calculates the angle between two N-dimensional vectors in DEGREES
    @ In, a, np.array, vector of floats
    @ In, b, np.array, vector of floats
    @ Out, ang, float, angle in degrees
  """
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

def partialDerivative(f, x0, var, n = 1, h = None, target = None):
  """
    Compute the n-th partial derivative of function f
    with respect variable var (numerical differentation).
    The derivative is computed with a central difference
    approximation.
    @ In, f, instance, the function to differentiate (format f(d) where d is a dictionary)
    @ In, x0, dict, the dictionary containing the x0 coordinate
    @ In, var, str, the variable of the resulting partial derivative
    @ In, n, int, optional, the order of the derivative. (max 10)
    @ In, h, float, optional, the step size, default automatically computed
    @ In, target, str, optional, the target output in case the "f" returns a dict of outputs. Default (takes the first)
    @ Out, deriv, float, the partial derivative of function f
  """
  import numdifftools as nd
  assert(n <= 10)
  def func(x, var, target=None):
    """
      Simple function wrapper for using numdifftools
      @ In, x, float, the point at which the nth derivative is found
      @ In, var, str, the variable in the dictionary x0 corresponding
                      to the part derivative to compute
      @ In, target, str, optional, the target output in case the "f" returns a dict of outputs. Default (takes the first)
      @ Out, func, float, the evaluated function
    """
    d = copy.copy(x0)
    d[var] = x
    out = f(d)
    if isinstance(out, dict):
      return list(out.values())[0] if target is None else out[target]
    else:
      return out

  do = nd.Derivative(func, order=n)
  deriv = do(x0[var],var,target)
  return deriv

def derivatives(f, x0, var = None, n = 1, h = None, target=None):
  """
    Compute the n-th partial derivative of function f
    with respect variable var (numerical differentation).
    The derivative is computed with a central difference
    approximation.
    @ In, f, instance, the function to differentiate (format f(d) where d is a dictionary)
    @ In, x0, dict, the dictionary containing the x0 coordinate {key:scalar or array of len(1)}
    @ In, var, list, optional, the list of variables of the resulting partial derivative (if None, compute all)
    @ In, n, int, optional, the order of the derivative. (max 10)
    @ In, h, float, optional, the step size, default automatically computed
    @ In, target, str, optional, the target output in case the "f" returns a dict of outputs. Default (all targets)
    @ Out, deriv, dict, the partial derivative of function f
  """
  assert(n <= 10)
  assert(isinstance(var, list) or isinstance(var, type(None)))
  #XXX Possibly this should only be done sometimes
  for key in x0:
    x0[key] = np.atleast_1d(x0[key])
  assert(len(list(x0.values())[0]) == 1)
  #assert(first(x0.values()).size == 1)
  targets = [target]
  if target is None:
    checkWorking = f(x0)
    if isinstance(checkWorking, dict):
      targets = checkWorking.keys()
  deriv = {}
  for t in targets:
    for variable in (x0.keys() if var is None else var):
      name = variable if t is None else "d{}|d{}".format(t, variable)
      deriv[name] = partialDerivative(f, x0, variable, n = n, h = h, target=t)
  return deriv




# utility function for defaultdict
def giveZero():
  """
    Utility function for defaultdict to 0
    Needed only to avoid lambda pickling issues for defaultdicts
    @ In, None
    @ Out, giveZero, int, zero
  """
  return 0

##########################
# empirical distribution #
##########################
def characterizeCDF(data, binOps=None, minBins=1):
  """
    Constructs an empirical CDF from the given data
    @ In, data, np.array(float), values to fit CDF to
    @ In, binOps, int, setting for picking binning strategy
    @ In, minBins, int, minimum bins for empirical CDF
    @ Out, params, dict, essential parameters for CDF
  """
  # calculate number of bins
  # binOps=Length or value
  nBins, _ = numBinsDraconis(data, low=minBins, binOps=binOps)
  # construct histogram
  counts, edges = np.histogram(data, bins=nBins, density=False)
  counts = np.array(counts) / float(len(data))
  # numerical CDF, normalizing to 0..1
  cdf = np.cumsum(counts)
  # set lowest value as first entry,
  ## from Jun implementation, min of CDF set to 0 for ?numerical issues?
  cdf = np.insert(cdf, 0, 0)
  # store parameters
  params = {'bins': edges,
            'counts':counts,
            'pdf' : counts * nBins,
            'cdf' : cdf,
            'lens' : len(data)}

  return params

def gaussianize(data, cdf):
  """
    Transforms "data" via empirical CDF into Gaussian distribution
    @ In, data, np.array, values to "gaussianize"
    @ In, cdf, dict, CDF characteristics (as via "characterizeCDF")
    @ Out, normed, np.array, gaussian version of "data"
  """
  cdfVals = sampleCDF(data, cdf)
  normed = stats.norm.ppf(cdfVals) # TODO could use RAVEN dist, but this is more modular

  return normed

def degaussianize(data, cdf):
  """
    Transforms "data" via empirical CDF from Gaussian distribution
    Opposite of "gaussianize" above
    @ In, data, np.array, "normal" values to "degaussianize"
    @ In, cdf, dict, CDF characteristics (as via "characterizeCDF")
    @ Out, denormed, np.array, empirical version of "data"
  """
  cdfVals = stats.norm.cdf(data)
  denormed = sampleICDF(cdfVals, cdf)

  return denormed

def sampleCDF(x, cdfParams):
  """
    Samples the empirical distribution's CDF at requested value(s)
    @ In, x, float/np.array, value(s) at which to sample CDF
    @ In, cdf, dict, CDF parameters (as constructed by "characterizeCDF")
    @ Out, y, float/np.array, value of empirical CDF at x
  """
  # TODO could this be covered by an empirical distribution from Distributions?
  # set up I/O
  x = np.atleast_1d(x)
  y = np.zeros(x.shape)
  # create masks for data outside range (above, below), inside range of empirical CDF
  belowMask = x <= cdfParams['bins'][0]
  aboveMask = x >= cdfParams['bins'][-1]
  inMask = np.logical_and(np.logical_not(belowMask), np.logical_not(aboveMask))
  # outside CDF set to min, max CDF values
  y[belowMask] = cdfParams['cdf'][0]
  y[aboveMask] = cdfParams['cdf'][-1]
  # for points in the CDF linearly interpolate between empirical entries
  ## get indices where points should be inserted (gives higher value)
  indices = np.searchsorted(cdfParams['bins'], x[inMask])
  x0 = cdfParams['bins'][indices-1]
  y0 = cdfParams['cdf'][indices-1]
  xf = cdfParams['bins'][indices]
  yf = cdfParams['cdf'][indices]
  y = interpolateDist(x, y, x0, xf, y0, yf, inMask)
  # numerical errors can happen due to not-sharp 0 and 1 in empirical cdf
  ## also, when Crow dist is asked for ppf(1) it returns sys.max (similar for ppf(0))
  y[y >= 1.0] = 1.0 - np.finfo(float).eps
  y[y <= 0.0] = np.finfo(float).eps

  return y

def sampleICDF(x, cdfParams):
  """
    Samples the inverse CDF defined by "cdfParams" to get values
    @ In, x, float/np.array, value(s) at which to sample inverse CDF
    @ In, cdf, dict, CDF parameters (as constructed by "characterizeCDF")
    @ Out, y, float/np.array, value of empirical inverse CDF at x
  """
  x = np.atleast_1d(x)
  y = np.zeros(x.shape)
  # create masks for data outside range (above, below), inside range of empirical CDF
  belowMask = x <= cdfParams['cdf'][0]
  aboveMask = x >= cdfParams['cdf'][-1]
  inMask = np.logical_and(np.logical_not(belowMask), np.logical_not(aboveMask))
  # outside CDF set to min, max CDF values
  y[belowMask] = cdfParams['bins'][0]
  y[aboveMask] = cdfParams['bins'][-1]
  # for points in the CDF linearly interpolate between empirical entries
  ## get indices where points should be inserted (gives higher value)
  indices = np.searchsorted(cdfParams['cdf'], x[inMask])
  x0 = cdfParams['cdf'][indices - 1]
  y0 = cdfParams['bins'][indices - 1]
  xf = cdfParams['cdf'][indices]
  yf = cdfParams['bins'][indices]
  y = interpolateDist(x, y, x0, xf, y0, yf, inMask)

  return y

def interpolateDist(x, y, x0, xf, y0, yf, mask):
  """
    Interpolates values for samples "x" to get dependent values "y" given bins
    @ In, x, np.array, sampled points (independent var)
    @ In, y, np.array, sampled points (dependent var)
    @ In, x0, np.array, left-nearest neighbor in empirical distribution for each x
    @ In, xf, np.array, right-nearest neighbor in empirical distribution for each x
    @ In, y0, np.array, value at left-nearest neighbor in empirical distribution for each x
    @ In, yf, np.array, value at right-nearest neighbor in empirical distribution for each x
    @ In, mask, np.array, boolean mask in "y" where the distribution values apply
    @ Out, y, np.array, same "y" but with values inserted
  """
  # handle divide-by-zero problems first, specially
  # check for where div zero problems will occur
  divZeroMask = x0 == xf
  # careful with double masking -> doesn't always do what you think it does
  zMask = [a[divZeroMask] for a in np.where(mask)]
  y[tuple(zMask)] = 0.5 * (yf[divZeroMask] + y0[divZeroMask])
  ### interpolate all other points as y = low + slope * frac
  okayMask = np.logical_not(divZeroMask)
  dy = yf[okayMask] - y0[okayMask]
  dx = xf[okayMask] - x0[okayMask]
  frac = x[mask][okayMask] - x0[okayMask]
  okayWhere = [a[okayMask] for a in np.where(mask)]
  y[tuple(okayWhere)] = y0 + dy/dx * frac

  return y

def computeCrowdingDistance(trainSet):
  """
    This function will compute the Crowding distance coefficients among the input parameters
    @ In, trainSet, numpy.array, array contains values of input parameters
    @ Out, crowdingDist, numpy.array, crowding distances for given input parameters
  """
  dim = trainSet.shape[1]
  distMat = np.zeros((dim, dim))

  for i in range(dim):
    for j in range(i):
      distMat[i,j] = linalg.norm(trainSet[:,i] - trainSet[:,j])
      distMat[j,i] = distMat[i,j]

  crowdingDist = np.sum(distMat,axis=1)

  return crowdingDist

def rankData(x, w=None):
  """
    Method to rank the data (weighted and unweighted)
    @ In, x, numpy.array (or array-like), array containing values to rank
    @ In, w, numpy.array (or array-like), optional, array containing weights (if None, equally-weighted)
    @ Out, rank, numpy.array, the ranked features
  """
  weights = w if w is not None else np.ones(len(x))
  _, inverseArray, num = np.unique(stats.rankdata(x), return_counts = True, return_inverse = True )
  A = np.bincount(inverseArray, weights)
  rank = (np.cumsum(A) - A)[inverseArray]+((num + 1)/2 * (A/num))[inverseArray]

  return rank

def getBuiltinTypes(typ):
  """
    Method to get a dictionary of builtin types
    @ In, typ, str, the type to get
    @ Out, getBuiltinTypes, list, the list of type instances
  """
  import builtins as b
  global _bTypes
  if _bTypes is None:
    _bTypes = {k:[t] for k, t in b.__dict__.items() if isinstance(t, type)}
  return _bTypes.get(typ,[])

def getNumpyTypes(typ):
  """
    Method to get a dictionary of numpy types
    @ In, typ, str, the type to get
    @ Out, nTypes, list, the list of type instances
  """
  nTypes = []
  if typ in np.sctypes:
    nTypes = np.sctypes[typ]
  else:
    for t in np.sctypes['others']:
      if typ in t.__name__:
        nTypes.append(t)

  return nTypes
