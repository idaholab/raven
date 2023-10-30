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


#**************************
# Continuous Distributions
#**************************


class ContinuousDistribution:
  def __init__(self, dist, xMin=None, xMax=None):
    self.dist = dist
    self._xMin = xMin if xMin is not None else self.dist.support()[0]
    self._xMax = xMax if xMax is not None else self.dist.support()[1]

  def pdf(self, x):
    x = np.asarray(x)
    mask = np.logical_and(x >= self._xMin, x <= self._xMax)
    returnPdf = np.zeros_like(x)
    scale = 1 / (self.dist.cdf(self._xMax) - self.dist.cdf(self._xMin))
    returnPdf[mask] = scale * self.dist.pdf(x[mask])
    return returnPdf

  def cdf(self, x):
    x = np.asarray(x)
    mask = np.logical_and(x > self._xMin, x < self._xMax)
    returnCdf = np.zeros_like(x)
    cdfXmin = self.dist.cdf(self._xMin)
    scale = 1 / (self.dist.cdf(self._xMax) - cdfXmin)
    returnCdf[x <= self._xMin] = 0
    returnCdf[x >= self._xMax] = 1
    returnCdf[mask] = (self.dist.cdf(x[mask]) - cdfXmin) * scale
    return returnCdf

  def inverseCdf(self, x):
    cdfXMin = self.dist.cdf(self._xMin)
    cdfXMax = self.dist.cdf(self._xMax)
    returnInvCdf = self.dist.ppf(x * (cdfXMax - cdfXMin) + cdfXMin)
    return returnInvCdf

  def untrCdfComplement(self, x):
    return self.dist.sf(x)

  def untrQuantile(self, x):
    return self.dist.ppf(x)

  def untrMean(self):
    return self.dist.mean()

  def untrMedian(self):
    return self.dist.median()

  def untrStdDev(self):
    return self.dist.std()

  def untrMode(self):
    # scipy.stats distributions don't have a mode function. Implementing this is left to the subclasses.
    raise NotImplementedError("Mode is not implemented for scipy distributions.")

  def untrHazard(self, x):
    return self.dist.pdf(x) / self.dist.sf(x)


class BasicUniformDistribution(ContinuousDistribution):
  def __init__(self, lowerBound, upperBound, xMin=None, xMax=None):
    """
    @ In, lowerBound, float, lower bound of the uniform distribution
    @ In, upperBound, float, upper bound of the uniform distribution
    @ Out, None
    """
    super().__init__(scipy.stats.uniform(lowerBound, upperBound - lowerBound), xMin, xMax)


class BasicNormalDistribution(ContinuousDistribution):
  def __init__(self, mean, sd, xMin=None, xMax=None):
    super().__init__(scipy.stats.norm(mean, sd), xMin, xMax)

  def untrMode(self):
    # all values are equally probable, so the mode isn't really well defined here
    return self.untrMean()


class LogNormal:
  def __init__(self, mu, sigma, low):
    self.mu = mu
    self.sigma = sigma
    self.low = low
    self._norm = scipy.stats.norm(self.mu, self.sigma)

  def support(self):
    return self.low, np.inf

  def pdf(self, x):
    x = np.asarray(x)
    returnPdf = np.zeros_like(x)
    xMask = x > self.low
    returnPdf[xMask] = 1 / ((x[xMask] - self.low) * self.sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((np.log(x[xMask] - self.low) - self.mu) / self.sigma) ** 2)
    return returnPdf

  def cdf(self, x):
    x = np.asarray(x)
    returnCdf = np.zeros_like(x)
    xMask = x > self.low
    returnCdf[xMask] = self._norm.cdf(np.log(x[xMask] - self.low))
    return returnCdf

  def ppf(self, x):
    return np.exp(self._norm.ppf(x)) + self.low

  def sf(self, x):
    return self._norm.sf(np.log(x - self.low))

  def mean(self):
    return np.exp(self.mu + self.sigma ** 2 / 2) + self.low

  def median(self):
    return np.exp(self.mu) + self.low

  def std(self):
    return np.sqrt((np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2)) + self.low


class BasicLogNormalDistribution(ContinuousDistribution):
  def __init__(self, mu, sigma, low, xMin=None, xMax=None):
    super().__init__(LogNormal(mu, sigma, low), xMin, xMax)

  def untrMode(self):
    return np.exp(self.dist.mu - self.dist.sigma ** 2) + self.dist.low


class BasicLogisticDistribution(ContinuousDistribution):
  def __init__(self, location, scale, xMin=None, xMax=None):
    super().__init__(scipy.stats.logistic(location, scale), xMin, xMax)

  def untrMode(self):
    return self.untrMean()


class BasicLaplaceDistribution(ContinuousDistribution):
  def __init__(self, location, scale, xMin=None, xMax=None):
    super().__init__(scipy.stats.laplace(location, scale), xMin, xMax)

  def untrMode(self):
    return self.untrMean()


class BasicTriangularDistribution(ContinuousDistribution):
  def __init__(self, x_peak, lower_bound, upper_bound, xMin=None, xMax=None):
    c = (x_peak - lower_bound) / (upper_bound - lower_bound)
    loc = lower_bound
    scale = upper_bound - lower_bound
    super().__init__(scipy.stats.triang(c, loc, scale), xMin, xMax)

  def untrMode(self):
    c, loc, scale = self.dist.args
    return loc + c * scale


class BasicExponentialDistribution(ContinuousDistribution):
  def __init__(self, lmbda, loc, xMin=None, xMax=None):
    super().__init__(scipy.stats.expon(loc, 1 / lmbda), xMin, xMax)

  def untrMode(self):
    return self.dist.args[0]  # loc parameter


class BasicWeibullDistribution(ContinuousDistribution):
  def __init__(self, k, lmbda, low, xMin=None, xMax=None):
    super().__init__(scipy.stats.weibull_min(k, low, lmbda), xMin, xMax)

  def untrMode(self):
    k, loc, lmbda = self.dist.args
    mode = lmbda * ((k - 1) / k) ** (1 / k) + loc if k > 1 else loc
    return mode


class BasicGammaDistribution(ContinuousDistribution):
  def __init__(self, k, theta, low, xMin=None, xMax=None):
    super().__init__(scipy.stats.gamma(k, low, theta), xMin, xMax)

  def untrMode(self):
    a, loc, scale = self.dist.args
    mode = (a - 1) * scale + loc if a >= 1 else loc
    return mode


class BasicBetaDistribution(ContinuousDistribution):
  def __init__(self, alpha, beta, scale, low, xMin=None, xMax=None):
    super().__init__(scipy.stats.beta(alpha, beta, low, scale), xMin, xMax)

  def untrMode(self):
    a, b, loc, scale = self.dist.args

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
"""
Some of the discrete distributions must be implemented manually to match the behavior of the boost distributions used in crow.
The scipy classes for discrete distributions are true discrete distributions in that the probability at values not in
the support of the distribution is zero, while some of the discrete distributions in boost are really implemented as continuous
random variables. The distributions that I've seen that are effected are:
  - Geometric
  - Binomial
"""


class DiscreteDistribution:
  """
    The way discrete distributions are used throughout the rest of RAVEN is much simpler than for continuous
    distributions, so we reflect that simplicity here. For example, no truncation is used for the discrete
    distributions defined here. Also, the scipy interface for discrete distributions is slightly different than
    for continuous distributions, so having a separate base class is helpful.
  """
  def __init__(self, dist):
    self.dist = dist

  def pdf(self, x):
    return self.dist.pmf(x)

  def cdf(self, x):
    return self.dist.cdf(x)

  def inverseCdf(self, x):
    return self.dist.ppf(x)

  def untrCdfComplement(self, x):
    return self.dist.sf(x)

  def untrQuantile(self, x):
    return self.dist.ppf(x)

  def untrMean(self):
    return self.dist.mean()

  def untrMedian(self):
    return self.dist.median()

  def untrStdDev(self):
    return self.dist.std()

  def untrMode(self):
    # scipy.stats distributions don't have a mode function. Implementing this is left to the subclasses.
    raise NotImplementedError("Mode is not implemented for scipy distributions.")

  def untrHazard(self, x):
    return self.dist.pdf(x) / self.dist.sf(x)


class BasicPoissonDistribution(DiscreteDistribution):
  def __init__(self, mu):
    super().__init__(scipy.stats.poisson(mu))

  def untrMode(self):
    return self.dist.args[0]  # mu parameter


class BasicBinomialDistribution(DiscreteDistribution):  # TODO ppf function broken
  def __init__(self, n, p):
    super().__init__(scipy.stats.binom(n, p))

  def untrMode(self):
    n, p = self.dist.args
    return np.floor(n * p)

  def inverseCdf(self, x):
    # Solve numerically using the complement of the incomplete beta function
    def func(a, y):
      if a == self.dist.args[0]:
        q = 1
      else:
        q = 1 - scipy.special.betainc(a + 1, self.dist.args[0] - a, self.dist.args[1])
      return q - y

    if hasattr(x, '__len__'):
      roots = [scipy.optimize.root_scalar(func, bracket=[0, self.dist.args[0]], args=(xi), method='toms748').root for xi in x]
      iCdf = np.array([np.floor(root) if root < 0.5 else np.ceil(root) for root in roots])
    else:
      root = scipy.optimize.root_scalar(func, bracket=[0, self.dist.args[0]], args=(x), method='toms748').root
      iCdf = np.floor(root) if root < 0.5 else np.ceil(root)
    return iCdf


class BasicBernoulliDistribution(DiscreteDistribution):
  def __init__(self, p):
    super().__init__( scipy.stats.bernoulli(p))

  def untrMode(self):
    return 0 if self.dist.params[0] < 0.5 else 1


class Geometric:
  def __init__(self, p):
    self.p = p

  def support(self):
    return 0, np.inf

  def pmf(self, x):
    return np.power(1 - self.p, x - 1) * self.p

  def cdf(self, x):
    return -np.expm1(np.log1p(-self.p) * (x + 1))

  def sf(self, x):
    return np.exp(self.logsf(x))

  def logsf(self, x):
    return x * np.log1p(-self.p)

  def ppf(self, x):
    _ppf = (np.log1p(-x) / np.log1p(-self.p) - 1) * (x >= self.p)
    return _ppf

  def mean(self):
    return (1 - self.p ) / self.p

  def std(self):
    return np.sqrt(1 - self.p) / self.p

  def median(self):
    return self.ppf(0.5)

class BasicGeometricDistribution(DiscreteDistribution):
  """  """
  def __init__(self, p):
    super().__init__(Geometric(p))

  def untrMode(self):
    return 0


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


#***************************
# Multivariate Distributions
#***************************

class BasicMultivariateNormalDistribution(ContinuousDistribution):
  pass


def checkAnswer(msg, answer, expected, tol=1e-10):
  if abs(answer - expected) > tol:
    print('{}: {} != {}'.format(msg, answer, expected))


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from scipy.integrate import cumulative_trapezoid as cumtrapz
  from scipy.integrate import trapz, simpson

  bernoulli = BasicBernoulliDistribution(0.4)

  x = np.array([0, 1])
  y = bernoulli.pdf(x)
  print(np.sum(y))
  plt.plot(x, y, label='pdf')
  plt.show()
