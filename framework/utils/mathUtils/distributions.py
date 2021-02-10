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
  Specifically for distribution actions.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np
from scipy import stats

def normal(x, mu=0.0, sigma=1.0):
  """
    Computation of normal pdf
    @ In, x, list or np.array, x values
    @ In, mu, float, optional, mean
    @ In, sigma, float, optional, sigma
    @ Out, returnNormal, list or np.array, pdf
  """
  return stats.norm.pdf(x,mu,sigma)

def normalCdf(x, mu=0.0, sigma=1.0):
  """
    Computation of normal cdf
    @ In, x, list or np.array, x values
    @ In, mu, float, optional, mean
    @ In, sigma, float, optional, sigma
    @ Out, cdfReturn, list or np.array, cdf
  """
  return stats.norm.cdf(x,mu,sigma)

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
  ## TODO additional options could be easily added in the future.
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
  # caluclate number of bins
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
    Interplotes values for samples "x" to get dependent values "y" given bins
    @ In, x, np.array, sampled points (independent var)
    @ In, y, np.array, sampled points (dependent var)
    @ In, x0, np.array, left-nearest neighbor in empirical distribution for each x
    @ In, xf, np.array, right-nearest neighbor in empirical distribution for each x
    @ In, y0, np.array, value at left-nearest neighbor in empirical distribution for each x
    @ In, yf, np.array, value at right-nearest neighbor in empirical distribution for each x
    @ In, mask, np.array, boolean mask in "y" where the distribution values apply
    @ Out, y, np.array, same "y" but with values inserted
  """
  ### handle divide-by-zero problems first, specially
  # check for where div zero prooblems will occur
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
