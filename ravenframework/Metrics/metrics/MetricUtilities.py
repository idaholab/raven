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
Created on September 18, 2017

@author: Joshua J. Cogliati
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import math
import scipy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import mathUtils
from ... import Files
from ... import Distributions
#Internal Modules End--------------------------------------------------------------------------------

def _countWeightInBins(sortedData, binBoundaries):
  """
    This method counts the number of data items in the sorted_data
    Returns an array with the number.  ret[0] is the number of data
    points <= binBoundaries[0], ret[len(binBoundaries)] is the number
    of points > binBoundaries[len(binBoundaries)-1]
    @ In, sortedData, list of (value,weight) of the data to be analyzed
    @ In, binBoundaries, list or np.array, the bin boundaries
    @ Out, ret, list, the list containing the number of bins
  """
  value = 0 #Read only
  weight = 1 #Read only
  binIndex = 0
  sortedIndex = 0
  ret = [0]*(len(binBoundaries)+1)
  while sortedIndex < len(sortedData):
    while not binIndex >= len(binBoundaries) and \
          sortedData[sortedIndex][value] > binBoundaries[binIndex]:
      binIndex += 1
    ret[binIndex] += sortedData[sortedIndex][weight]
    sortedIndex += 1
  return ret


def _getPDFandCDFfromWeightedData(data, weights, numBins, uniformBins, interpolation):
  """
    This method is used to convert weighted data into a PDF and CDF function.
    Basically, this does kernel density estimation of weighted data.
    @ In, data, np.array,  one dimentional array of the data to process
    @ In, weights, np.array, one dimentional array of the weights for the data
    @ In, numBins, int, the number of bins to use.
    @ In, uniformBins, bool, if True, use uniformly sized bins, otherwise use equal probability bins.
    @ In, interpolation, str, "linear" or "quadratic", depending on which interpolation is used
    @ Out, (dataStats, cdfFunc, pdfFunc), tuple, dataStats is dictionary with things like "mean" and "stdev", cdfFunction is a function that returns the CDF value and pdfFunc is a function that returns the PDF value.
  """
  # normalize weights
  weightSum = sum(weights)
  if not math.isclose(weightSum, 1.0):
    weights = weights / weightSum
  weightSum = sum(weights)
  # Sort the data
  sortedData = list(zip(data, weights))
  sortedData.sort() #Sort the data.
  value = 0 #Read only
  weight = 1 #Read only
  # Find data range
  low = sortedData[0][value]
  high = sortedData[-1][value]
  dataRange = high - low
  #Find the values to use between the histogram bins
  if uniformBins:
    minBinSize = dataRange/numBins
    bins = [low + x * minBinSize for x in range(1, numBins)]
  else:
    #Equal probability bins
    probPerBin = weightSum/numBins
    probSum = 0.0
    bins = [None]*(numBins-1)
    searchIndex = 0
    #Look thru the array, and find the next place where probSum > nextProb
    for i in range(numBins-1):
      nextProb = (i + 1) * probPerBin
      while probSum + sortedData[searchIndex][weight] < nextProb:
        probSum += sortedData[searchIndex][weight]
        searchIndex += 1
      bins[i] = sortedData[searchIndex][value]
    #Remove duplicates
    for i in reversed(range(len(bins))):
      if i > 1 and bins[i-1] == bins[i]:
        bins.pop(i)
    if len(bins) > 1:
      minBinSize = min(map(lambda x, y: x - y, bins[1:], bins[:-1]))
    else:
      minBinSize = dataRange
  #Count the amount of weight in each bin
  counts = _countWeightInBins(sortedData, bins)
  binBoundaries = [low] + bins + [high]
  countSum = sum(counts)
  assert -1e-4 < countSum - weightSum < 1e-4
  # Create CDF
  cdf = [0.0] * len(counts)
  midpoints = [0.0] * len(counts)
  cdfSum = 0.0
  for i in range(len(counts)):
    f0 = counts[i] / countSum
    cdfSum += f0
    cdf[i] = cdfSum
    midpoints[i] = (binBoundaries[i] + binBoundaries[i + 1]) / 2.0
  cdfFunc = mathUtils.createInterp(midpoints, cdf, 0.0, 1.0, interpolation)
  #Create PDF
  fPrimeData = [0.0] * len(counts)
  for i in range(len(counts)):
    h = binBoundaries[i + 1] - binBoundaries[i]
    nCount = counts[i] / countSum  # normalized count
    f0 = cdf[i]
    if i + 1 < len(counts):
      f1 = cdf[i + 1]
    else:
      f1 = 1.0
    if i + 2 < len(counts):
      f2 = cdf[i + 2]
    else:
      f2 = 1.0
    if interpolation == 'linear':
      fPrime = (f1 - f0) / h
    else:
      fPrime = (-1.5 * f0 + 2.0 * f1 + -0.5 * f2) / h
    fPrimeData[i] = fPrime
  pdfFunc = mathUtils.createInterp(midpoints, fPrimeData, 0.0, 0.0, interpolation)
  mean = np.average(data, weights = weights)
  dataStats = {"mean":mean,"minBinSize":minBinSize,"low":low,"high":high}
  return dataStats, cdfFunc, pdfFunc


def _convertToCommonFormat(data):
  """
    Convert either a distribution or a set of data to a (stats, cdf, pdf) pair
  """
  if isinstance(data, Distributions.Distribution):
    # data is a subclass of BoostDistribution, generate needed stats, and pass in cdf and pdf.
    stats = {"mean":data.untruncatedMean(),"stdev":data.untruncatedStdDev()}
    cdf = lambda x:data.cdf(x)
    pdf = lambda x:data.pdf(x)
    return stats, cdf, pdf
  if type(data).__name__ == "tuple":
    # data is (list,list), then it is a list of weights
    assert len(data) == 2
    points, weights = data
    assert len(points) == len(weights)
  elif '__len__' in dir(data):
    # data is list, then it is a list of data, generate uniform weights and begin
    points = data
    weights = [1.0/len(points)]*len(points)
  else:
    raise IOError("Unknown type in _convertToCommonFormat")
  #Sturges method for determining number of bins
  numBins = int(math.ceil(mathUtils.log2(len(points)) + 1))
  return _getPDFandCDFfromWeightedData(points, weights, numBins, False, 'linear')


def _getBounds(stats1, stats2):
  """
    Gets low and high bounds that captures the interesting bits of the two
    pieces of data.
    @ In, stats1, dict, dictionary with either "low" and "high" or "mean" and "stdev"
    @ In, stats2, dict, dictionary with either "low" and "high" or "mean" and "stdev"
    @ Out, (low, high), (float, float) Returns low and high bounds.
  """
  def getLowBound(stat):
    """
      Finds the lower bound from the statistics in stat.
      @ In, stat, dict, Dictionary with either "low" or "mean" and "stdev"
      @ Out, getLowBound, float, the lower bound to use.
    """
    if "low" in stat:
      return stat["low"]
    return stat["mean"] - 5*stat["stdev"]

  def getHighBound(stat):
    """
      Finds the higher bound from the statistics in stat.
      @ In, stat, dict, Dictionary with either "high" or "mean" and "stdev"
      @ Out, getLowBound, float, the lower bound to use.
    """
    if "high" in stat:
      return stat["high"]
    return stat["mean"] + 5*stat["stdev"]

  low = min(getLowBound(stats1), getLowBound(stats2))
  high = max(getHighBound(stats1), getHighBound(stats2))
  return (low,high)

def _getCDFAreaDifference(data1, data2):
  """
    Gets the area between the two CDFs in data1 and data2.
    The greater the area, the more different data1 and data2 are.
    @ In, data1, varies, The first data to use, see _convertToCommonFormat
    @ In, data2, varies, The second data to use, see _convertToCommonFormat
    @ Out, cdfAreaDifference, float, the area difference between the CDFs.
  """
  stats1, cdf1, pdf1 =_convertToCommonFormat(data1)
  stats2, cdf2, pdf2 =_convertToCommonFormat(data2)
  low, high = _getBounds(stats1, stats2)
  return scipy.integrate.quad(lambda x:abs(cdf1(x)-cdf2(x)),low,high,limit=1000)[0]

def _getPDFCommonArea(data1, data2):
  """
    Gets the area that the PDFs overlap in data1 and data2.
    The greater the area, the more similar data1 and data2 are.
    @ In, data1, varies, The first data to use, see _convertToCommonFormat
    @ In, data2, varies, The second data to use, see _convertToCommonFormat
    @ Out, pdfCommonArea, float, the common area between the PDFs.
  """
  stats1, cdf1, pdf1 =_convertToCommonFormat(data1)
  stats2, cdf2, pdf2 =_convertToCommonFormat(data2)
  low, high = _getBounds(stats1, stats2)
  return scipy.integrate.quad(lambda x:min(pdf1(x),pdf2(x)),low,high,limit=1000)[0]
