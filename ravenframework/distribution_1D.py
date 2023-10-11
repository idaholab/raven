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
import scipy


def truncate(func):
  return func


class Distribution:
  def __init__(self, dist, params):
    self.dist = dist
    self.params = params

    # scipy.stats distributions treat continuous and discrete distributions differently.
    # rv_continuous has a `pdf` method, while rv_discrete has a `pmf` method.
    if isinstance(dist, scipy.stats.rv_continuous):
      self._pdf = dist.pdf
    elif isinstance(dist, scipy.stats.rv_discrete):
      self._pdf = dist.pmf
    else:
      raise TypeError("DistributionBackend must be initialized with a scipy.stats distribution.")

  def pdf(self, x):
    return self._pdf(x, **self.params)

  def cdf(self, x):
    return self.dist.cdf(x, **self.params)

  def inverseCdf(self, x):
    return self.dist.ppf(x, **self.params)

  def cdfComplement(self, x):
    return self.dist.sf(x, **self.params)

  def quantile(self, x):
    return self.dist.ppf(x, **self.params)

  def mean(self):
    return self.dist.mean(**self.params)

  def median(self):
    return self.dist.median(**self.params)

  def mode(self):
    # scipy.stats distributions don't have a mode function.
    # The mode can be found by finding the maximum of the pdf.
    # However, this requires performing an optimization, which is not ideal.
    # For now, just raise an error.
    # raise NotImplementedError("Mode is not implemented for scipy distributions.")
    return self.dist.mean(**self.params)  # true for symmetric distributions

  def hazard(self, x):
    return self.dist.pdf(x, **self.params) / self.dist.sf(x, **self.params)

  def untrPdf(self, x):
    return self._pdf(x, **self.params)

  def untrCdf(self, x):
    return self.dist.cdf(x, **self.params)

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
    # scipy.stats distributions don't have a mode function.
    # The mode can be found by finding the maximum of the pdf.
    # However, this requires performing an optimization, which is not ideal.
    # For now, just raise an error.
    # raise NotImplementedError("Mode is not implemented for scipy distributions.")
    return self.dist.mean(**self.params)  # true for symmetric distributions

  def untrHazard(self, x):
    return self.dist.pdf(x, **self.params) / self.dist.sf(x, **self.params)


class TruncatedDistribution(Distribution):
  def __init__(self, dist, params):
    if 'xMin' not in params or 'xMax' not in params:
      raise ValueError("TruncatedDistribution must be initialized with values for both xMin and xMax.")
    super().__init__(dist, params)

  def pdf(self, x):
    scale = 1 / (self.cdf(self.params['xMax']) - self.cdf(self.params['xMin']))
    return scale * self._pdf(x, **self.params)

  def cdf(self, x):
    if x < self.params['xMin']:
      return 0
    elif x > self.params['xMax']:
      return 1
    else:
      scale = 1 / (self.cdf(self.params['xMax']) - self.cdf(self.params['xMin']))
      return scale * self.dist.cdf(x, **self.params)


class BasicUniformDistribution(Distribution):
  def __init__(self, xMin, xMax):
    params = {'loc': xMin,
              'scale': xMax - xMin}
    super().__init__(scipy.stats.uniform, params)


class BasicNormalDistribution(Distribution):
  def __init__(self, mean, sd, xMin=None, xMax=None):
    params = {'loc': mean,
              'scale': sd}
    if xMin is None and xMax is None:
      super().__init__(scipy.stats.norm, params)
    else:
      params['a'] = (xMin - mean) / sd
      params['b'] = (xMax - mean) / sd
      super().__init__(scipy.stats.truncnorm, params)


class BasicLogNormalDistribution(Distribution):
  def __init__(self, mu, sigma, low, xMin, xMax):
    params = {'s': sigma,
              'loc': low,
              'scale': mu}
    super().__init__(scipy.stats.lognorm, params)


class BasicLogisticDistribution(Distribution):
  def __init__(self, location, scale):
    params = {'loc': location,
              'scale': scale}
    super().__init__(scipy.stats.logistic, params)


class BasicLaplaceDistribution(Distribution):
  def __init__(self, location, scale):
    params = {'loc': location,
              'scale': scale}
    super().__init__(scipy.stats.laplace, params)


class BasicTriangularDistribution(Distribution):
  def __init__(self, x_peak, lower_bound, upper_bound):
    params = {'loc': lower_bound,
              'scale': upper_bound - lower_bound,
              'c': (x_peak - lower_bound) / (upper_bound - lower_bound)}
    super().__init__(scipy.stats.triang, params)


class BasicExponentialDistribution(Distribution):
  def __init__(self, lmbda, loc):
    params = {'scale': 1 / lmbda,  # scipy.stats.expon uses the scale parameter, which is 1 / lambda
              'loc': loc}
    super().__init__(scipy.stats.expon, params)


class BasicWeibullDistribution(Distribution):
  def __init__(self, k, lmbda, low):
    params = {'c': k,
              'loc': low,
              'scale': lmbda}
    super().__init__(scipy.stats.weibull_min, params)


class BasicGammaDistribution(Distribution):
  def __init__(self, k, theta, low):
    params = {'a': k,
              'loc': low,
              'scale': theta}
    super().__init__(scipy.stats.gamma, params)



# Truncated distributions in Crow
#   - normal
#   - lognormal
#   - laplace
#   - exponential
#   - Weibull
#   - gamma
#   - beta


# Truncated distributions in scipy
#   - normal
#   - exponential
#   - pareto
#   - weibull_min


# Need to implement:
#   - lognormal
#   - laplace
#   - gamma
#   - beta


"""

Required methods, taken from BoostDistribution (in Distributions.py) methods
  - cdf
  - inverseCdf
  -


Create new class ScipyDistribution in Distributions.py that implements the same interface
as BoostDistribution.

"""