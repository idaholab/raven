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
  Specifically for manipulating functions and data.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import functools
import copy
from scipy import interpolate, integrate
import numpy as np
from utils.utils import UreturnPrintTag, UreturnPrintPostTag


def simpson(f, a, b, n):
  """
    Simpson integration rule
    @ In, f, instance, the function to integrate
    @ In, a, float, lower bound
    @ In, b, float, upper bound
    @ In, n, int, number of integration steps
    @ Out, sumVar, float, integral
  """
  h = (b - a) / float(n)
  y = np.zeros(n+1)
  x = np.zeros(n+1)
  for i in range(0, n+1):
    x[i] = a + i*h
    y[i] = f(x[i])
  return integrate.simps(y, x)

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

  ## All of the values must be the same, okay just take the scale of the data
  ## to be the maximum value
  if scale == 0:
    scale = np.max(np.absolute(values))

  ## All of the values must be zero, okay use 1 to prevent a 0/0 issue
  if scale == 0:
    scale = 1.0

  return (offset, scale)

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
    xi = np.linspace(x.min(),x.max(),int(options['interpPointsX']))
  if z is not None:
    if y.size <= 2:
      yi = y
    else:
      yi = np.linspace(y.min(),y.max(),int(options['interpPointsY']))
    xig, yig = np.meshgrid(xi, yi)
    try:
      if ['nearest','linear','cubic'].count(options['interpolationType']) > 0 or z.size <= 3:
        if options['interpolationType'] != 'nearest' and z.size > 3:
          zi = interpolate.griddata((x,y), z, (xi[None,:], yi[:,None]), method=options['interpolationType'])
        else:
          zi = interpolate.griddata((x,y), z, (xi[None,:], yi[:,None]), method='nearest')
      else:
        rbf = interpolate.Rbf(x,y,z,function=str(str(options['interpolationType']).replace('Rbf', '')), epsilon=int(options.pop('epsilon',2)), smooth=float(options.pop('smooth',0.0)))
        zi  = rbf(xig, yig)
    except Exception as ae:
      if 'interpolationTypeBackUp' in options.keys():
        print(UreturnPrintTag('UTILITIES:')+': ' +UreturnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        zi = interpolateFunction(x,y,z,options)
      else:
        raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
    if returnCoordinate:
      return xig,yig,zi
    else:
      return zi
  else:
    try:
      if ['nearest','linear','cubic'].count(options['interpolationType']) > 0 or y.size <= 3:
        if options['interpolationType'] != 'nearest' and y.size > 3:
          yi = interpolate.griddata((x), y, (xi[:]), method=options['interpolationType'])
        else:
          yi = interpolate.griddata((x), y, (xi[:]), method='nearest')
      else:
        xig, yig = np.meshgrid(xi, yi)
        rbf = interpolate.Rbf(x, y,function=str(str(options['interpolationType']).replace('Rbf', '')),epsilon=int(options.pop('epsilon',2)), smooth=float(options.pop('smooth',0.0)))
        yi  = rbf(xi)
    except Exception as ae:
      if 'interpolationTypeBackUp' in options.keys():
        print(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('Warning') + '->   The interpolation process failed with error : ' + str(ae) + '.The STREAM MANAGER will try to use the BackUp interpolation type '+ options['interpolationTypeBackUp'])
        options['interpolationTypeBackUp'] = options.pop('interpolationTypeBackUp')
        yi = interpolateFunction(x,y,options)
      else:
        raise Exception(UreturnPrintTag('UTILITIES')+': ' +UreturnPrintPostTag('ERROR') + '-> Interpolation failed with error: ' +  str(ae))
    if returnCoordinate:
      return xi,yi
    else:
      return yi

def numpyNearestMatch(findIn, val):
  """
    Given an array, find the entry that most nearly matches the given value.
    @ In, findIn, np.array, the array to look in
    @ In, val, float or other compatible type, the value for which to find a match
    @ Out, returnMatch, tuple, index where match is and the match itself
  """
  dist = distance(findIn,val)
  idx = dist.argmin()
  #idx = np.sum(np.abs(findIn-val),axis=0).argmin()
  returnMatch = idx,findIn[idx]
  return returnMatch

def NDInArray(findIn, val, tol=1e-12):
  """
    checks a numpy array of numpy arrays for a near match, then returns info.
    @ In, findIn, np.array, numpy array of numpy arrays (both arrays can be any length)
    @ In, val, tuple/list/numpy array, entry to look for in findIn
    @ In, tol, float, optional, tolerance to check match within
    @ Out, (bool,idx,looking) -> (found/not found, index where found or None, findIn entry or None)
  """
  if len(findIn)<1:
    return False,None,None
  targ = []
  found = False
  for idx,looking in enumerate(findIn):
    num = looking - val
    den = np.array(val)
    #div 0 error
    for i,v in enumerate(den):
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
    return False,None,None
  return found,idx,looking

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
    for variable in outPortion.keys():
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

#   outKeys = vars.getParaKeys('outputs')
#   inpKeys = vars.getParaKeys('inputs')

#   outDic = []

#   for t in range(numberOfTimeStep):
#     newVars={}
#     for key in inpKeys+outKeys:
#       newVars[key]=np.zeros(0)

#     hs = vars.getParametersValues('outputs')
#     for history in hs:
#       for key in inpKeys:
#         newVars[key] = np.append(newVars[key],vars.getParametersValues('inputs')[history][key])

#       for key in outKeys:
#         newVars[key] = np.append(newVars[key],vars.getParametersValues('outputs')[history][key][t])

#     outDic.append(newVars)

#   return outDic

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