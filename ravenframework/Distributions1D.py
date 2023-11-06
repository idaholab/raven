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
  """ Base class for distributions of continuous random variables """
  def __init__(self, dist, xMin=None, xMax=None):
    """
      Class constructor

      @ In, dist, scipy.stats.rv_continuous object, distribution to wrap
      @ In, xMin, float, optional, minimum value of the distribution's support
      @ In, xMax, float, optional, maximum value of the distribution's support
      @ Out, None
    """
    self.dist = dist
    self._xMin = xMin if xMin is not None else self.dist.support()[0]
    self._xMax = xMax if xMax is not None else self.dist.support()[1]
    self._cdfXMin = self.dist.cdf(self._xMin)
    self._cdfXMax = self.dist.cdf(self._xMax)

  def pdf(self, x):
    """
      Probability density function

      @ In, x, float or np.ndarray, value to evaluate at
      @ Out, returnPdf, float or np.ndarray, probability density function value
    """
    x = np.asarray(x)
    mask = np.logical_and(x >= self._xMin, x <= self._xMax)
    returnPdf = np.zeros_like(x)
    returnPdf[mask] = self.dist.pdf(x[mask]) / (self._cdfXMax - self._cdfXMin)
    # If the input x was a scalar, returnPdf will be 0-dim numpy array. If that's the case,
    # we want to return a scalar instead, so we extract the value from the array.
    if returnPdf.ndim == 0:
      returnPdf = returnPdf.item()
    return returnPdf

  def cdf(self, x):
    """
      Cumulative distribution function

      @ In, x, float, value to evaluate at
      @ Out, returnCdf, float, cumulative distribution function value
    """

    x = np.asarray(x)
    mask = np.logical_and(x > self._xMin, x < self._xMax)
    returnCdf = np.zeros_like(x)
    returnCdf[x <= self._xMin] = 0
    returnCdf[x >= self._xMax] = 1
    returnCdf[mask] = (self.dist.cdf(x[mask]) - self._cdfXMin) / (self._cdfXMax - self._cdfXMin)
    # If the input x was a scalar, returnCdf will be 0-dim numpy array. If that's the case,
    # we want to return a scalar instead, so we extract the value from the array.
    if returnCdf.ndim == 0:
      returnCdf = returnCdf.item()
    return returnCdf

  def inverseCdf(self, x):
    """
      Inverse cumulative distribution function

      @ In, x, float, quantile value to evaluate at
      @ Out, returnInvCdf, float, inverse cumulative distribution function value
    """
    x = np.asarray(x)
    mask = np.logical_and(x > 0, x < 1)
    returnInvCdf = np.zeros_like(x)
    returnInvCdf[mask] = self.dist.ppf(x[mask] * (self._cdfXMax - self._cdfXMin) + self._cdfXMin)
    returnInvCdf[x <= 0] = self._xMin
    returnInvCdf[x >= 1] = self._xMax
    # If the input x was a scalar, returnInvCdf will be 0-dim numpy array. If that's the case,
    # we want to return a scalar instead, so we extract the value from the array.
    if returnInvCdf.ndim == 0:
      returnInvCdf = returnInvCdf.item()
    return returnInvCdf

  def untrCdfComplement(self, x):
    """
      CDF complement (1 - CDF(x)). Also known as the survival function.

      @ In, x, float, value to evaluate at
      @ Out, sf, float, survival function value
    """
    return self.dist.sf(x)

  def untrQuantile(self, x):
    """
      Quantile function (inverse CDF) of the untruncated distribution.

      @ In, x, float, quantile value to evaluate at
      @ Out, ppf, float, quantile function value
    """
    return self.dist.ppf(x)

  def untrMean(self):
    """
      Mean of the untruncated distribution.

      @ In, None
      @ Out, mean, float, mean of the untruncated distribution
    """
    return self.dist.mean()

  def untrMedian(self):
    """
      Median of the untruncated distribution.

      @ In, None
      @ Out, median, float, median of the untruncated distribution
    """
    return self.dist.median()

  def untrStdDev(self):
    """
      Standard deviation of the untruncated distribution.

      @ In, None
      @ Out, std, float, standard deviation of the untruncated distribution
    """
    return self.dist.std()

  def untrMode(self):
    """
      Mode of the untruncated distribution.

      @ In, None
      @ Out, mode, float, mode of the untruncated distribution
    """
    # scipy.stats distributions don't have a mode function. Implementing this is left to the subclasses.
    raise NotImplementedError("Mode is not implemented for scipy distributions.")

  def untrHazard(self, x):
    """
      Hazard function of the untruncated distribution.
      The hazard function is the ratio of the PDF to the survival function.

      @ In, None
      @ Out, mean, float, mean of the untruncated distribution
    """
    return self.dist.pdf(x) / self.dist.sf(x)


class BasicUniformDistribution(ContinuousDistribution):
  """ Uniform distribution wrapper """
  def __init__(self, lowerBound, upperBound, xMin=None, xMax=None):
    """
      Class constructor

      @ In, lowerBound, float, lower bound of the uniform distribution
      @ In, upperBound, float, upper bound of the uniform distribution
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(scipy.stats.uniform(lowerBound, upperBound - lowerBound), xMin, xMax)


class BasicNormalDistribution(ContinuousDistribution):
  """ Normal distribution wrapper """
  def __init__(self, mean, sd, xMin=None, xMax=None):
    """
      Class constructor

      @ In, mean, float, mean
      @ In, sd, float, standard deviation
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(scipy.stats.norm(mean, sd), xMin, xMax)

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    # mode and mean are the same for the normal distribution
    return self.untrMean()


class LogNormal:
  """ Log-normal distribution"""
  def __init__(self, mu, sigma, low):
    """
      Class constructor

      @ In, mu, float, mean of normal distribution
      @ In, sigma, float, standard deviation of normal distribution
      @ In, low, float, lower bound of the distribution
      @ Out, None
    """
    self.mu = mu
    self.sigma = sigma
    self.low = low
    self._norm = scipy.stats.norm(self.mu, self.sigma)

  def support(self):
    """
      The support of the distribution

      @ In, None
      @ Out, (low, high), tuple, lower and upper bound of the distribution
    """
    return self.low, np.inf

  def pdf(self, x):
    """
      Probability density function (PDF)

      @ In, x, float, value to evaluate at
      @ Out, returnPdf, float, PDF value
    """
    x = np.asarray(x)
    returnPdf = np.zeros_like(x)
    # masking helps to not take logs of values less than or equal to zero
    xMask = x > self.low
    returnPdf[xMask] = 1 / ((x[xMask] - self.low) * self.sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((np.log(x[xMask] - self.low) - self.mu) / self.sigma) ** 2)
    return returnPdf

  def cdf(self, x):
    """
      Cumulative distribution function (CDF)

      @ In, x, float, value to evaluate at
      @ Out, returnCdf, float, CDF value
    """
    x = np.asarray(x)
    returnCdf = np.zeros_like(x)
    # masking helps to not take logs of values less than or equal to zero
    xMask = x > self.low
    returnCdf[xMask] = self._norm.cdf(np.log(x[xMask] - self.low))
    return returnCdf

  def ppf(self, x):
    """
      Percent point function, inverse CDF, or quantile function

      @ In, x, float, quantile value to evaluate at
      @ Out, ppf, float, PPF value
    """
    return np.exp(self._norm.ppf(x)) + self.low

  def sf(self, x):
    """
      Survival function (1-CDF)

      @ In, x, float, value to evaluate at
      @ Out, sf, float, survival function value
    """
    return self._norm.sf(np.log(x - self.low))

  def mean(self):
    """
      Mean of the distribution

      @ In, None
      @ Out, mean, float, mean of the distribution
    """
    return np.exp(self.mu + self.sigma ** 2 / 2) + self.low

  def median(self):
    """
      Median of the distribution

      @ In, None
      @ Out, median, float, median of the distribution
    """
    return np.exp(self.mu) + self.low

  def std(self):
    """
      Standard deviation of the distribution

      @ In, None
      @ Out, std, float, standard deviation of the distribution
    """
    return np.sqrt((np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2)) + self.low


class BasicLogNormalDistribution(ContinuousDistribution):
  """ Log-normal distribution wrapper """
  def __init__(self, mu, sigma, low, xMin=None, xMax=None):
    """
      Class constructor

      @ In, mu, float, mean of normal distribution
      @ In, sigma, float, standard deviation of normal distribution
      @ In, low, float, lower bound of the distribution (shift parameter)
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(LogNormal(mu, sigma, low), xMin, xMax)

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    return np.exp(self.dist.mu - self.dist.sigma ** 2) + self.dist.low


class BasicLogisticDistribution(ContinuousDistribution):
  """ Logistic distribution wrapper """
  def __init__(self, location, scale, xMin=None, xMax=None):
    """
      Class constructor

      @ In, location, float, location parameter
      @ In, scale, float, scale parameter
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(scipy.stats.logistic(location, scale), xMin, xMax)

  def untrMode(self):
    return self.untrMean()


class BasicLaplaceDistribution(ContinuousDistribution):
  """ Laplace distribution wrapper """
  def __init__(self, location, scale, xMin=None, xMax=None):
    """
      Class constructor

      @ In, location, float, location parameter
      @ In, scale, float, scale parameter
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(scipy.stats.laplace(location, scale), xMin, xMax)

  def untrMode(self):
    return self.untrMean()


class BasicTriangularDistribution(ContinuousDistribution):
  """ Triangular distribution wrapper """
  def __init__(self, xPeak, lowerBound, upperBound, xMin=None, xMax=None):
    """
      Class constructor

      @ In, xPeak, float, x value of the peak of the distribution
      @ In, lowerBound, float, x value of the lower bound of the distribution
      @ In, upperBound, float, x value of the upper bound of the distribution
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    # The scipy distribution uses a different parameterization
    #   - c is the location of the peak as a fraction of the distance between the lower and upper bounds
    #   - loc is the lower bound
    #   - scale is the distance between the lower and upper bounds
    c = (xPeak - lowerBound) / (upperBound - lowerBound)
    loc = lowerBound
    scale = upperBound - lowerBound
    super().__init__(scipy.stats.triang(c, loc, scale), xMin, xMax)

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    c, loc, scale = self.dist.args
    return loc + c * scale


class BasicExponentialDistribution(ContinuousDistribution):
  """ Exponential distribution wrapper """
  def __init__(self, lmbda, loc, xMin=None, xMax=None):
    """
      Class constructor

      @ In, lmbda, float, lambda scale parameter
      @ In, loc, float, location parameter
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    # The scipy distribution uses a parameterization of scale=1/lambda
    super().__init__(scipy.stats.expon(loc, 1 / lmbda), xMin, xMax)

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    return self.dist.args[0]  # loc parameter


class BasicWeibullDistribution(ContinuousDistribution):
  """ Weibull distribution wrapper """
  def __init__(self, k, lmbda, low, xMin=None, xMax=None):
    """
      Class constructor

      @ In, k, float, shape parameter
      @ In, lmbda, float, scale parameter
      @ In, low, float, lower bound of the distribution (shift parameter)
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(scipy.stats.weibull_min(k, low, lmbda), xMin, xMax)

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    k, loc, lmbda = self.dist.args
    mode = lmbda * ((k - 1) / k) ** (1 / k) + loc if k > 1 else loc
    return mode


class BasicGammaDistribution(ContinuousDistribution):
  """ Gamma distribution wrapper """
  def __init__(self, k, theta, low, xMin=None, xMax=None):
    """
      Class constructor

      @ In, k, float, shape parameter
      @ In, theta, float, scale parameter
      @ In, low, float, lower bound of the distribution (shift parameter)
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(scipy.stats.gamma(k, low, theta), xMin, xMax)

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    a, loc, scale = self.dist.args
    mode = (a - 1) * scale + loc if a >= 1 else loc
    return mode


class BasicBetaDistribution(ContinuousDistribution):
  """ Beta distribution wrapper """
  def __init__(self, alpha, beta, scale, low, xMin=None, xMax=None):
    """
      Class constructor

      @ In, alpha, float, shape parameter
      @ In, beta, float, shape parameter
      @ In, scale, float, scale parameter
      @ In, low, float, lower bound of the distribution (shift parameter)
      @ In, xMin, float, optional, lower bound of truncated distribution
      @ In, xMax, float, optional, upper bound of truncated distribution
      @ Out, None
    """
    super().__init__(scipy.stats.beta(alpha, beta, low, scale), xMin, xMax)

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
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
NOTE: Some of the discrete distributions must be implemented manually to match the behavior of the boost distributions used in crow.
The scipy classes for discrete distributions are true discrete distributions in that the probability at values not in
the support of the distribution is zero, while some of the discrete distributions in boost are really implemented as continuous
random variables. While this is true for the discrete distributions in general, only the way the Geometric distribution is used seems
to be effected.

Another behavior which does not match between scipy and boost distributions is the inverse CDF (ppf) function of the Binomial distribution.
The method used in boost to estimate this numerically has been implemented here as well so that the results match.
"""


class DiscreteDistribution:
  """
    The way discrete distributions are used throughout the rest of RAVEN is much simpler than for continuous
    distributions, so we reflect that simplicity here. For example, no truncation is used for the discrete
    distributions defined here. Also, the scipy interface for discrete distributions is slightly different than
    for continuous distributions (e.g. pmf vs pdf methods), so having a separate base class is helpful.
  """
  def __init__(self, dist):
    """
      Class constructor

      @ In, dist, scipy.stats.rv_discrete, discrete distribution
      @ Out, None
    """
    self.dist = dist

  def pdf(self, x):
    """
      Probability mass function. Mapped to pdf for consistency with ContinuousDistribution.

      @ In, x, float, point at which to evaluate the pmf
      @ Out, pmf, float, probability density function at x
    """
    return self.dist.pmf(x)

  def cdf(self, x):
    """
      Cumulative distribution function.

      @ In, x, float, point at which to evaluate the cdf
      @ Out, cdf, float, cumulative distribution function at x
    """
    return self.dist.cdf(x)

  def inverseCdf(self, x):
    """
      Inverse cumulative distribution function.

      @ In, x, float, point at which to evaluate the inverse cdf
      @ Out, inverseCdf, float, inverse cumulative distribution function at x
    """
    return self.dist.ppf(x)

  def untrCdfComplement(self, x):
    """
      Complementary cumulative distribution function.

      @ In, x, float, point at which to evaluate the complementary cdf
      @ Out, cdfComplement, float, complementary cumulative distribution function at x
    """
    return self.dist.sf(x)

  def untrQuantile(self, x):
    """
      Quantile function. Since truncation has not been implemented for discrete distributions,
      this is the same as inverseCdf.

      @ In, x, float, point at which to evaluate the quantile function
      @ Out, quantile, float, quantile function at x
    """
    return self.dist.ppf(x)

  def untrMean(self):
    """
      Mean of the distribution

      @ In, None
      @ Out, mean, float, mean of the distribution
    """
    return self.dist.mean()

  def untrMedian(self):
    """
      Median of the distribution

      @ In, None
      @ Out, median, float, median of the distribution
    """
    return self.dist.median()

  def untrStdDev(self):
    """
      Standard deviation of the distribution

      @ In, None
      @ Out, std, float, standard deviation of the distribution
    """
    return self.dist.std()

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    # scipy.stats distributions don't have a mode function. Implementing this is left to the subclasses.
    raise NotImplementedError("Mode is not implemented for scipy distributions.")

  def untrHazard(self, x):
    """
      Hazard function. Defined as pdf(x) / (1 - cdf(x)).

      @ In, x, float, point at which to evaluate the hazard function
      @ Out, hazard, float, hazard function at x
    """
    return self.dist.pdf(x) / self.dist.sf(x)


class BasicPoissonDistribution(DiscreteDistribution):
  """ Poisson distribution wrapper """
  def __init__(self, mu):
    """
      Class constructor

      @ In, mu, float, shape parameter equal to the mean and variance of the distribution
      @ Out, None
    """
    super().__init__(scipy.stats.poisson(mu))

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    return self.dist.args[0]  # mu parameter


class BasicBinomialDistribution(DiscreteDistribution):
  """ Binomial distribution wrapper """
  def __init__(self, n, p):
    """
      Class constructor

      @ In, n, int, number of trials
      @ In, p, float, probability of success
      @ Out, None
    """
    super().__init__(scipy.stats.binom(n, p))

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    n, p = self.dist.args
    return np.floor(n * p)

  def inverseCdf(self, x):
    """
      Inverse cumulative distribution function. The scipy implementation of this function does not match the behavior
      of the boost implementation, so we implement it manually here.

      @ In, x, float, point at which to evaluate the inverse cdf
      @ Out, inverseCdf, float, inverse cumulative distribution function at x
    """
    # Solve numerically using the complement of the incomplete beta function
    def func(a, y):
      """
        CDF defined with the complement of the incomplete beta function

        @ In, a, float, point at which to evaluate the CDF
        @ In, y, float, desired quantile value
        @ Out, diff, float, difference between the desired quantile value and the CDF evaluated at a
      """
      if a == self.dist.args[0]:
        q = 1
      else:
        q = 1 - scipy.special.betainc(a + 1, self.dist.args[0] - a, self.dist.args[1])
      diff = q - y
      return diff

    # The root finding problem being solved to get the inverse CDF is not vectorized, so we need to loop over the
    # elements of x if x is an array.
    if hasattr(x, '__len__'):
      # Uses the toms748 algorithm for root finding, as stated in the boost documentation at
      # https://www.boost.org/doc/libs/1_50_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/binomial_dist.html
      roots = [scipy.optimize.root_scalar(func, bracket=[0, self.dist.args[0]], args=(xi), method='toms748').root for xi in x]
      # Round "out": down if less than 0.5, up if greater than 0.5
      iCdf = np.array([np.floor(root) if root < 0.5 else np.ceil(root) for root in roots])
    else:
      root = scipy.optimize.root_scalar(func, bracket=[0, self.dist.args[0]], args=(x), method='toms748').root
      iCdf = np.floor(root) if root < 0.5 else np.ceil(root)
    return iCdf


class BasicBernoulliDistribution(DiscreteDistribution):
  """ Bernoulli distribution wrapper """
  def __init__(self, p):
    """
      Class constructor

      @ In, p, float, probability of occurrence
      @ Out, None
    """
    super().__init__( scipy.stats.bernoulli(p))

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    return 0 if self.dist.params[0] < 0.5 else 1


class Geometric:
  """
    Implementation of a geometric distribution with support over the positive real numbers (continuous).
  """
  def __init__(self, p):
    """
      Class constructor

      @ In, p, float, probability of success
      @ Out, None
    """
    self.p = p

  def support(self):
    """
      Gives the support of the distribution

      @ In, None
      @ Out, support, tuple, lower and upper bounds of the support
    """
    return 0, np.inf

  def pmf(self, x):
    """
      Probability mass function

      @ In, x, float, point at which to evaluate the pmf
      @ Out, pmf, float, probability mass function at x
    """
    return np.power(1 - self.p, x - 1) * self.p

  def cdf(self, x):
    """
      Cumulative distribution function

      @ In, x, float, point at which to evaluate the cdf
      @ Out, cdf, float, cumulative distribution function at x
    """
    return -np.expm1(np.log1p(-self.p) * (x + 1))

  def sf(self, x):
    """
      Survival function

      @ In, x, float, point at which to evaluate the survival function
      @ Out, sf, float, survival function at x
    """
    return np.exp(self.logsf(x))

  def logsf(self, x):
    """
      Log of the survival function

      @ In, x, float, point at which to evaluate the log of the survival function
      @ Out, logsf, float, log of the survival function at x
    """
    return x * np.log1p(-self.p)

  def ppf(self, x):
    """
      Percent point function (inverse of cdf)

      @ In, x, float, point at which to evaluate the ppf
      @ Out, ppf, float, percent point function at x
    """
    return (np.log1p(-x) / np.log1p(-self.p) - 1) * (x >= self.p)

  def mean(self):
    """
      Mean of the distribution

      @ In, None
      @ Out, mean, float, mean of the distribution
    """
    return (1 - self.p ) / self.p

  def std(self):
    """
      Standard deviation of the distribution

      @ In, None
      @ Out, std, float, standard deviation of the distribution
    """
    return np.sqrt(1 - self.p) / self.p

  def median(self):
    """
      Median of the distribution

      @ In, None
      @ Out, median, float, median of the distribution
    """
    return self.ppf(0.5)


class BasicGeometricDistribution(DiscreteDistribution):
  """ Geometric distribution wrapper """
  def __init__(self, p):
    """
      Class constructor

      @ In, p, float, probability of success
      @ Out, None
    """
    super().__init__(Geometric(p))

  def untrMode(self):
    """
      Mode of the untruncated distribution

      @ In, None
      @ Out, mode, float, mode of the distribution
    """
    return 0
