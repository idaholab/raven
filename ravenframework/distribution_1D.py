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
Created on Oct 10, 2023
@author: j-bryan
"""
import numpy as np
import scipy


class Distribution:
  def __init__(self, dist, xMin=None, xMax=None, **params):
    self.dist = dist
    self.params = params
    self._isTruncated = xMin is not None or xMax is not None
    # FIXME a and b aren't exactly the same as xMin and xMax
    self._xMin = xMin if xMin is not None else self.dist.a
    self._xMax = xMax if xMax is not None else self.dist.b

  def pdf(self, x):
    # If the distribution is not truncated, just return the pdf evaluation from dist
    if not self._isTruncated:
      return self.dist.pdf(x, **self.params)

    # Otherwise, return the truncated and rescaled pdf value
    if x < self._xMin or x > self._xMax:
      return 0

    scale = 1 / (self.dist.cdf(self._xMax, **self.params) - self.dist.cdf(self._xMin, **self.params))
    returnPdf = scale * self.dist.pdf(x, **self.params)
    return returnPdf

  def cdf(self, x):
    # If the distribution is not truncated, just return the pdf evaluation from dist
    if not self._isTruncated:
      return self.dist.cdf(x, **self.params)

    # Otherwise, return the truncated and rescaled cdf value
    if x < self._xMin:
      return 0
    elif x > self._xMax:
      return 1

    scale = 1 / (self.dist.cdf(self._xMax, **self.params) - self.dist.cdf(self._xMin, **self.params))
    returnCdf = (self.dist.cdf(x, **self.params) - self.dist.cdf(self._xMin, **self.params)) * scale
    return returnCdf

  def inverseCdf(self, x):
    if not self._isTruncated:
      return self.dist.ppf(x, **self.params)

    if x == 0:
      return self._xMin
    elif x == 1:
      return self._xMax

    cdfXMin = self.dist.cdf(self._xMin, **self.params)
    cdfXMax = self.dist.cdf(self._xMax, **self.params)
    returnInvCdf = self.dist.ppf(x * (cdfXMax - cdfXMin) + cdfXMin, **self.params)
    return returnInvCdf

  def untrCdfComplement(self, x):
    return self.dist.sf(x, **self.params)

  def untrQuantile(self, x):
    return self.dist.ppf(x, **self.params)

  def untrMean(self):
    return self.dist.mean(**self.params)

  def untrMedian(self):
    return self.dist.median(**self.params)

  def untrStdDev(self):
    return self.dist.std(**self.params)

  def untrMode(self):
    # scipy.stats distributions don't have a mode function. Implementing this is left to the subclasses.
    raise NotImplementedError("Mode is not implemented for scipy distributions.")

  def untrHazard(self, x):
    return self.dist.pdf(x, **self.params) / self.dist.sf(x, **self.params)


#**************************
# Continuous Distributions
#**************************


class BasicUniformDistribution(Distribution):
  def __init__(self, lowerBound, upperBound, xMin=None, xMax=None):
    """
    @ In, lowerBound, float, lower bound of the uniform distribution
    @ In, upperBound, float, upper bound of the uniform distribution
    @ Out, None
    """
    params = {'loc': lowerBound,
              'scale': upperBound - lowerBound}
    super().__init__(scipy.stats.uniform, xMin, xMax, **params)


class BasicNormalDistribution(Distribution):
  def __init__(self, mean, sd, xMin=None, xMax=None):
    params = {'loc': mean,
              'scale': sd}
    super().__init__(scipy.stats.norm, xMin, xMax, **params)

  def untrMode(self):
    return self.untrMean()


class BasicLogNormalDistribution(Distribution):
  def __init__(self, mu, sigma, low, xMin=None, xMax=None):
    params = {'s': sigma,
              'loc': low,
              'scale': mu}
    super().__init__(scipy.stats.lognorm, xMin, xMax, **params)

  def untrMode(self):
    return np.exp(self.params['scale'] - self.params['s'] ** 2)


class BasicLogisticDistribution(Distribution):
  def __init__(self, location, scale, xMin=None, xMax=None):
    params = {'loc': location,
              'scale': scale}
    super().__init__(scipy.stats.logistic, xMin, xMax, **params)

  def untrMode(self):
    return self.untrMean()


class BasicLaplaceDistribution(Distribution):
  def __init__(self, location, scale, xMin=None, xMax=None):
    params = {'loc': location,
              'scale': scale}
    super().__init__(scipy.stats.laplace, xMin, xMax, **params)

  def untrMode(self):
    return self.untrMean()


class BasicTriangularDistribution(Distribution):
  def __init__(self, x_peak, lower_bound, upper_bound, xMin=None, xMax=None):
    params = {'loc': lower_bound,
              'scale': upper_bound - lower_bound,
              'c': (x_peak - lower_bound) / (upper_bound - lower_bound)}
    super().__init__(scipy.stats.triang, xMin, xMax, **params)

  def untrMode(self):
    return self.params['loc'] + self.params['c'] * self.params['scale']


class BasicExponentialDistribution(Distribution):
  def __init__(self, lmbda, loc, xMin=None, xMax=None):
    params = {'scale': 1 / lmbda,  # scipy.stats.expon uses the scale parameter, which is 1 / lambda
              'loc': loc}
    super().__init__(scipy.stats.expon, xMin, xMax, **params)

  def untrMode(self):
    return self.params['loc']


class BasicWeibullDistribution(Distribution):
  def __init__(self, k, lmbda, low, xMin=None, xMax=None):
    params = {'c': k,
              'loc': low,
              'scale': lmbda}
    super().__init__(scipy.stats.weibull_min, xMin, xMax, **params)

  def untrMode(self):
    lmbda = self.params['scale']
    k = self.params['c']
    loc = self.params['loc']
    mode = lmbda * ((k - 1) / k) ** (1 / k) + loc if k > 1 else loc
    return mode


class BasicGammaDistribution(Distribution):
  def __init__(self, k, theta, low, xMin=None, xMax=None):
    params = {'a': k,
              'loc': low,
              'scale': theta}
    super().__init__(scipy.stats.gamma, xMin, xMax, **params)

  def untrMode(self):
    k = self.params['a']
    theta = self.params['scale']
    loc = self.params['loc']
    mode = (k - 1) * theta + loc if k >= 1 else loc
    return mode


class BasicBetaDistribution(Distribution):
  def __init__(self, alpha, beta, low, scale, xMin=None, xMax=None):
    params = {'a': alpha,
              'b': beta,
              'loc': low,
              'scale': scale}
    super().__init__(scipy.stats.beta, xMin, xMax, **params)

  def untrMode(self):
    a = self.params['a']
    b = self.params['b']
    scale = self.params['scale']
    loc = self.params['loc']

    if a > 1 and b > 1:
      mode = (a - 1) / (a + b - 2) * scale + loc
    elif a <= 1 and b > 1:
      mode = 0
    elif a > 1 and b <= 1:
      mode = scale + loc
    else:
      raise ValueError(f'The beta distribution is bimodal for a <= 1 and b <= 1. Given a = {a} and b = {b}.')

    return mode


#************************
# Discrete Distributions
#************************


class BasicPoissonDistribution(Distribution):
  def __init__(self, mu):
    params = {'mu': mu}
    dist = scipy.stats.poisson
    setattr(dist, 'pdf', dist.pmf)  # discrete distributions from scipy use pmf instead of pdf
    super().__init__(dist, **params)

  def untrMode(self):
    return np.floor(self.params['mu'])


class BasicBinomialDistribution(Distribution):
  def __init__(self, n, p):
    params = {'n': n,
              'p': p}
    dist = scipy.stats.binom
    setattr(dist, 'pdf', dist.pmf)  # discrete distributions from scipy use pmf instead of pdf
    super().__init__(dist, **params)

  def untrMode(self):
    return np.floor(self.params['n'] * self.params['p'])


class BasicBernoulliDistribution(Distribution):
  def __init__(self, p):
    params = {'p': p}
    dist = scipy.stats.bernoulli
    setattr(dist, 'pdf', dist.pmf)  # discrete distributions from scipy use pmf instead of pdf
    super().__init__(dist, **params)

  def untrMode(self):
    return 0 if self.params['p'] < 0.5 else 1


class BasicGeometricDistribution(Distribution):
  def __init__(self, p):
    params = {'p': p,
              'loc': -1}  # boost includes 0 in the support of the geometric distribution, scipy starts at 1
    dist = scipy.stats.geom
    setattr(dist, 'pdf', dist.pmf)  # discrete distributions from scipy use pmf instead of pdf
    super().__init__(dist, **params)

  def untrMode(self):
    return 1


class BasicConstantDistribution:
  """ The constant distribution is a degenerate distribution. There is no scipy class for it. """
  def __init__(self, value):
    self.value = value

  def pdf(self, x):
    return np.finfo[np.float64].max / 2

  def cdf(self, x):
    if x < self.value:
      return 0
    elif x == self.value:
      return 0.5
    else:
      return 1

  def ppf(self, x):
    return self.value

  # TODO need to finish implementing this distribution
