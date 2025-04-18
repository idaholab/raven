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
Created on Mar 7, 2013
@author: crisr
"""
import sys
import numpy as np
import scipy
from math import gamma
import os
import operator
import csv
from scipy.interpolate import UnivariateSpline
import scipy.stats
from numpy import linalg as LA
import copy
import math as math
import bisect
from collections import namedtuple

from .EntityFactoryBase import EntityFactory
from .BaseClasses import BaseEntity, InputDataUser
from .utils import utils
from .utils.randomUtils import random
from .utils import randomUtils
CrowDistribution1D = utils.findCrowModule('distribution1D')
from . import Distributions1D
from . import DistributionsND
from .utils import mathUtils, InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

def factorial(x):
  """
    Compute factorial
    @ In, x, float, the value
    @ Out, fact, float, the factorial
  """
  fact = gamma(x+1)
  return fact

"""
  Mapping between internal framework and Crow distribution name
"""
_FrameworkToCrowDistNames = { 'Uniform':'UniformDistribution',
                              'Normal':'NormalDistribution',
                              'Gamma':'GammaDistribution',
                              'Beta':'BetaDistribution',
                              'Triangular':'TriangularDistribution',
                              'Poisson':'PoissonDistribution',
                              'Binomial':'BinomialDistribution',
                              'Bernoulli':'BernoulliDistribution',
                              'Logistic':'LogisticDistribution',
                              'Custom1D':'Custom1DDistribution',
                              'Exponential':'ExponentialDistribution',
                              'Categorical':'Categorical',
                              'MarkovCategorical':'MarkovCategorical',
                              'LogNormal':'LogNormalDistribution',
                              'Weibull':'WeibullDistribution',
                              'NDInverseWeight': 'NDInverseWeightDistribution',
                              'NDCartesianSpline': 'NDCartesianSplineDistribution',
                              'MultivariateNormal' : 'MultivariateNormalDistribution',
                              'Laplace' : 'LaplaceDistribution',
                              'Geometric' : 'GeometricDistribution',
                              'LogUniform' : 'LogUniformDistribution',
                              'UniformDiscrete' : 'UniformDiscreteDistribution'
}


# Declaring namedtuple(DistributionTypes)
DistributionTypes = namedtuple('DistributionType', ['discrete', 'continuous'])
# Adding values
distType = DistributionTypes('Discrete', 'Continuous')

class DistributionsCollection(InputData.ParameterInput):
  """
    Class for reading in a collection of distributions
  """

DistributionsCollection.createClass("Distributions")


class Distribution(BaseEntity, InputDataUser):
  """
    A general class containing the distributions
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addSub(
      InputData.parameterInputFactory(
        'upperBound',
        descr=r"""the maximum value allowable by this distribution. For distributions that do not traditionally
          include an upper bound, including this node will truncate (and rebalance the probability) of the distribution,
          with this value as the maximum value.""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(
      InputData.parameterInputFactory(
        'lowerBound',
        descr=r"""the minimum value allowable by this distribution. For distributions that do not traditionally
          include a lower bound, including this node will truncate (and rebalance the probability) of the distribution,
          with this value as the minimum value.""",
        contentType=InputTypes.FloatType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.upperBoundUsed       = False  # True if the distribution is right truncated
    self.lowerBoundUsed       = False  # True if the distribution is left truncated
    self.hasInfiniteBound     = False  # True if the untruncated distribution has bounds of +- system max
    self.upperBound           = None   # Right bound
    self.lowerBound           = None   # Left bound
    self.__adjustmentType     = '' # this describe how the re-normalization to preserve the probability should be done for truncated distributions
    self.dimensionality       = None   # Dimensionality of the distribution (1D or ND)
    self.distType             = None   # Distribution type (continuous or discrete)
    self.memory               = False  # This variable flags if the distribution has history dependence in the sampling process (True) or not (False)
    self.printTag             = 'DISTRIBUTIONS'
    self.preferredPolynomials = None  # best polynomial for probability-weighted norm of error
    self.preferredQuadrature  = None  # best quadrature for probability-weighted norm of error
    self.compatibleQuadrature = [] #list of compatible quadratures
    self.convertToDistrDict   = {} #dict of methods keyed on quadrature types to convert points from quadrature measure and domain to distribution measure and domain
    self.convertToQuadDict    = {} #dict of methods keyed on quadrature types to convert points from distribution measure and domain to quadrature measure and domain
    self.measureNormDict      = {} #dict of methods keyed on quadrature types to provide scalar adjustment for measure transformation (from quad to distr)
    self.convertToDistrDict['CDFLegendre'] = self.CDFconvertToDistr
    self.convertToQuadDict ['CDFLegendre'] = self.CDFconvertToQuad
    self.measureNormDict   ['CDFLegendre'] = self.CDFMeasureNorm
    self.convertToDistrDict['CDFClenshawCurtis'] = self.CDFconvertToDistr
    self.convertToQuadDict ['CDFClenshawCurtis'] = self.CDFconvertToQuad
    self.measureNormDict   ['CDFClenshawCurtis'] = self.CDFMeasureNorm

  def __getstate__(self):
    """
      Get the pickling state
      @ In, None
      @ Out, pdict, dict, the namespace state
    """
    pdict = self.getInitParams()
    pdict['type'] = self.type
    return pdict

  def __setstate__(self,pdict):
    """
      Set the pickling state
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.__init__()
    self.upperBoundUsed   = pdict.pop('upperBoundUsed'  )
    self.lowerBoundUsed   = pdict.pop('lowerBoundUsed'  )
    self.hasInfiniteBound = pdict.pop('hasInfiniteBound')
    self.upperBound       = pdict.pop('upperBound'      )
    self.lowerBound       = pdict.pop('lowerBound'      )
    self.__adjustmentType = pdict.pop('adjustmentType'  )
    self.dimensionality   = pdict.pop('dimensionality'  )
    self.type             = pdict.pop('type'            )
    self._localSetState(pdict)
    self.initializeDistribution()

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      Default implementation, do nothing special
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    pass

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    upperBound = paramInput.findFirst('upperBound')
    if upperBound !=None:
      self.upperBound = upperBound.value
      self.upperBoundUsed = True
    lowerBound = paramInput.findFirst('lowerBound')
    if lowerBound !=None:
      self.lowerBound = lowerBound.value
      self.lowerBoundUsed = True
    if self.lowerBoundUsed and self.upperBoundUsed:
      if self.lowerBound == self.upperBound:
        self.raiseAnError(IOError, 'Lower bound for Distribution "'+self.name+'" is equal to the upper bound!')
      if self.lowerBound > self.upperBound:
        self.raiseAnError(IOError, 'Lower bound for Distribution "'+self.name+'" is greater than the upper bound!')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = {}
    retDict['type'] = _FrameworkToCrowDistNames[self.type]
    if self.lowerBoundUsed:
      retDict['xMin'] = self.lowerBound
    if self.upperBoundUsed:
      retDict['xMax'] = self.upperBound
    return retDict

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['upperBoundUsed'  ] = self.upperBoundUsed
    paramDict['lowerBoundUsed'  ] = self.lowerBoundUsed
    paramDict['hasInfiniteBound'] = self.hasInfiniteBound
    paramDict['upperBound'      ] = self.upperBound
    paramDict['lowerBound'      ] = self.lowerBound
    paramDict['adjustmentType'  ] = self.__adjustmentType
    paramDict['dimensionality'  ] = self.dimensionality
    return paramDict

  def rvsWithinCDFbounds(self,lowerBound,upperBound):
    """
      Function to get a random number from a truncated distribution
      @ In, lowerBound, float, lower bound
      @ In, upperBound, float, upper bound
      @ Out,randResult, float, random number
    """
    randResult = self._distribution.inverseCdf(float(random(1))*(upperBound-lowerBound)+lowerBound)
    return randResult

  def rvsWithinbounds(self,lowerBound,upperBound):
    """
      Function to get a random number from a truncated distribution
      @ In, lowerBound, float, lower bound
      @ In, upperBound, float, upper bound
      @ Out,randResult, float, random number
    """
    CDFupper = self._distribution.cdf(upperBound)
    CDFlower = self._distribution.cdf(lowerBound)
    randResult = self.rvsWithinCDFbounds(CDFlower,CDFupper)
    return randResult

  def convertToDistr(self,qtype,pts):
    """
      Converts points from the quadrature "qtype" standard domain to the distribution domain.
      @ In, qtype, string, type of quadrature to convert from
      @ In, pts, np.array, points to convert
      @ Out, convertToDistrDict, np.array, converted points
    """
    return self.convertToDistrDict[qtype](pts)

  def convertToQuad(self,qtype,pts):
    """
      Converts points from the distribution domain to the quadrature "qtype" standard domain.
      @ In, qtype, string, type of quadrature to convert to
      @ In, pts, np.array, points to convert
      @ Out, convertToQuadDict, np.array, converted points
    """
    return self.convertToQuadDict[qtype](pts)

  def measureNorm(self,qtype):
    """
      Provides the integral/jacobian conversion factor between the distribution domain and the quadrature domain.
      @ In,  qtype, string, type of quadrature to convert to
      @ Out, measureNormDict, float, conversion factor
    """
    return self.measureNormDict[qtype]()

  def _convertDistrPointsToCdf(self,pts):
    """
      Converts points in the distribution domain to [0,1].
      @ In, pts, array of floats, points to convert
      @ Out, cdfPoints, float/array of floats, converted points
    """
    try:
      return self.cdf(pts.real)
    except TypeError:
      return list(self.cdf(x) for x in pts)

  def _convertCdfPointsToDistr(self,pts):
    """
      Converts points in [0,1] to the distribution domain.
      @ In, pts, array of floats, points to convert
      @ Out, dist, float/array of floats, converted points
    """
    try:
      return self.ppf(pts.real)
    except TypeError:
      return list(self.ppf(x) for x in pts)

  def _convertCdfPointsToStd(self,pts):
    """
      Converts points in [0,1] to [-1,1], the uniform distribution's STANDARD domain.
      @ In, pts, array of floats, points to convert
      @ Out, stds, float/array of floats, converted points
    """
    try:
      return 2.0*pts.real-1.0
    except TypeError:
      return list(2.0*x-1.0 for x in pts)

  def _convertStdPointsToCdf(self,pts):
    """
      Converts points in [-1,1] to [0,1] (CDF domain).
      @ In, pts, array of floats, points to convert
      @ Out, cdfPoints, float/array of floats, converted points
    """
    try:
      return 0.5*(pts.real+1.0)
    except TypeError:
      return list(0.5*(x+1.0) for x in pts)

  def CDFconvertToQuad(self,pts):
    """
      Converts all the way from distribution domain to [-1,1] quadrature domain.
      @ In, pts, array of floats, points to convert
      @ Out, quads, float/array of floats, converted points
    """
    return self._convertCdfPointsToStd(self._convertDistrPointsToCdf(pts))

  def CDFconvertToDistr(self,pts):
    """
      Converts all the way from [-1,1] quadrature domain to distribution domain.
      @ In, pts, array of floats, points to convert
      @ Out, distr, float/array of floats, converted points
    """
    return self._convertCdfPointsToDistr(self._convertStdPointsToCdf(pts))

  def CDFMeasureNorm(self):
    """
      Integral norm/jacobian for [-1,1] Legendre quadrature.
      @ In, None
      @ Out, norm, float, normalization factor
    """
    norm = 1.0/2.0
    return norm

  def getDimensionality(self):
    """
      Function return the dimensionality of the distribution
      @ In, None
      @ Out, dimensionality, int, the dimensionality of the distribution
    """
    return self.dimensionality

  def getDistType(self):
    """
      Function return distribution type
      @ In, None
      @ Out, distType, string,  ('Continuous' or 'Discrete')
    """
    return self.distType

  def getMemory(self):
    """
      Function return the value of the memory variable
      @ In, None
      @ Out, memory, boolean, value which indicates if distribution has memory
    """
    return self.memory

  def reset(self):
    """
      Function that reset the distribution
      @ In, None
      @ Out, None
    """
    pass

  def initializeFromDict(self, inputDict):
    """
      Function which initializes the distribution given a the information contained in inputDict
      @ In, inputDict, dict, dictionary containing the values required to initialize the distribution
      @ Out, None
    """
    pass

class BoostDistribution(Distribution):
  """
    Base distribution class based on boost
  """

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dimensionality  = 1
    self.distType        = distType.continuous

  def cdf(self,x):
    """
      Function to get the cdf at a provided coordinate
      @ In, x, float, value to get the cdf at
      @ Out, retunrCdf, float, requested cdf
    """
    returnCdf = self._distribution.cdf(x)
    return returnCdf

  def ppf(self,x):
    """
      Function to get the inverse cdf at a provided coordinate
      @ In, x, float, value to get the inverse cdf at
      @ Out, retunrPpf, float, requested inverse cdf
    """
    returnPpf = self._distribution.inverseCdf(x)
    return returnPpf

  def pdf(self,x):
    """
      Function to get the pdf at a provided coordinate
      @ In, x, float, value to get the pdf at
      @ Out, returnPdf, float, requested pdf
    """
    returnPdf = self._distribution.pdf(x)
    return returnPdf

  def logPdf(self,x):
    """
      Function to get the log pdf at a provided coordinate
      @ In, x, float, value to get the pdf at
      @ Out, logPdf, float, requested log pdf
    """
    logPdf = np.log(self.pdf(x))
    return logPdf

  def untruncatedCdfComplement(self, x):
    """
      Function to get the untruncated  cdf complement at a provided coordinate
      @ In, x, float, value to get the untruncated  cdf complement  at
      @ Out, float, requested untruncated  cdf complement
    """
    return self._distribution.untrCdfComplement(x)

  def untruncatedHazard(self, x):
    """
      Function to get the untruncated  Hazard  at a provided coordinate
      @ In, x, float, value to get the untruncated  Hazard   at
      @ Out, float, requested untruncated  Hazard
    """
    return self._distribution.untrHazard(x)

  def untruncatedMean(self):
    """
      Function to get the untruncated  Mean
      @ In, None
      @ Out, float, requested Mean
    """
    return self._distribution.untrMean()

  def untruncatedStdDev(self):
    """
      Function to get the untruncated Standard Deviation
      @ In, None
      @ Out, float, requested Standard Deviation
    """
    return self._distribution.untrStdDev()

  def untruncatedMedian(self):
    """
      Function to get the untruncated  Median
      @ In, None
      @ Out, float, requested Median
    """
    return self._distribution.untrMedian()

  def untruncatedMode(self):
    """
      Function to get the untruncated  Mode
      @ In, None
      @ Out, untrMode, float, requested Mode
    """
    untrMode = self._distribution.untrMode()
    return untrMode

  def rvs(self, size=None):
    """
      Function to get random numbers
      @ In, size, int, optional, number of entries to return (one if None)
      @ Out, rvsValue, float or list, requested random number or numbers
    """
    size = size or 1
    rvsValue = self.ppf(random(size))
    return rvsValue

  def selectedRvs(self, discardedElems):
    """
      Function to get random numbers for discrete distribution which exclude discardedElems
      @ In, discardedElems, list, list of values to be discarded
      @ Out, rvsValue, float, requested random number
    """
    if not self.memory:
      self.raiseAnError(IOError,' The distribution '+ str(self.name) + ' does not support the method selectedRVS.')
    else:
      rvsValue = self.selectedPpf(random(),discardedElems)
    return rvsValue

class Uniform(BoostDistribution):
  """
    Uniform univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Uniform, cls).getInputSpecification()
    inputSpecification.description = r"""Classical uniform distribution. The probability density function for the uniform
      distribution is given by $f(x)=\frac{1}{b-a}$ for $a \le x \le b$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html} for more details.
      """
    return inputSpecification

  def __init__(self, lowerBound=None, upperBound=None):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.range = 0.0
    self.type = 'Uniform'
    self.distType = 'Continuous'
    self.compatibleQuadrature.append('Legendre')
    self.compatibleQuadrature.append('ClenshawCurtis')
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature = 'Legendre'
    self.preferredPolynomials = 'Legendre'
    if upperBound is not None:
      self.upperBound = upperBound
      self.upperBoundUsed = True
    if lowerBound is not None:
      self.lowerBound = lowerBound
      self.lowerBoundUsed = True
    if self.lowerBoundUsed and self.upperBoundUsed:
      self.range = self.upperBound - self.lowerBound

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    #self.lowerBound   = pdict.pop('lowerBound'  )
    #self.upperBound   = pdict.pop('upperBound'   )
    self.range        = pdict.pop('range')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['xMin'] = self.lowerBound
    retDict['xMax'] = self.upperBound
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    if not self.upperBoundUsed or not self.lowerBoundUsed:
      self.raiseAnError(IOError,'the Uniform distribution needs both upperBound and lowerBound attributes. Got upperBound? '+ str(self.upperBoundUsed) + '. Got lowerBound? '+str(self.lowerBoundUsed))
    self.range = self.upperBound - self.lowerBound
    self.initializeDistribution()

  def stdProbabilityNorm(self):
    """Returns the factor to scale error norm by so that norm(probability)=1.
    @ In, None, None
    @ Out float, norm
    """
    return 0.5

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['range'] = self.range
    return paramDict
    # no other additional parameters required

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.convertToDistrDict['Legendre']       = self.convertLegendreToUniform
    self.convertToQuadDict ['Legendre']       = self.convertUniformToLegendre
    self.measureNormDict   ['Legendre']       = self.stdProbabilityNorm
    self.convertToDistrDict['ClenshawCurtis'] = self.convertLegendreToUniform
    self.convertToQuadDict ['ClenshawCurtis'] = self.convertUniformToLegendre
    self.measureNormDict   ['ClenshawCurtis'] = self.stdProbabilityNorm
    self._distribution = Distributions1D.BasicUniformDistribution(self.lowerBound,self.lowerBound+self.range)

  def convertUniformToLegendre(self,y):
    """Converts from distribution domain to standard Legendre [-1,1].
      @ In, y, float/array of floats, points to convert
      @ Out float/array of floats, converted points
    """
    return (y-self.untruncatedMean())/(self.range/2.)

  def convertLegendreToUniform(self,x):
    """Converts from standard Legendre [-1,1] to distribution domain.
      @ In, y, float/array of floats, points to convert
      @ Out float/array of floats, converted points
    """
    return self.range/2.*x+self.untruncatedMean()

DistributionsCollection.addSub(Uniform.getInputSpecification())

class Normal(BoostDistribution):
  """
    Normal univariate distribution
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Normal, cls).getInputSpecification()
    inputSpecification.description = r"""Classical Gaussian normal distribution. The probability density function for the normal
      distribution is given by $f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ for $-\infty\leq x \leq \infty$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("mean",
        descr=r"""the distribution mean or expected value. For a standard normal distribution,
          this is zero. """,
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("sigma",
        descr=r"""the standard deviation (stdv) of this distribution. For a standard normal distribution,
          this is one. The variance of this distribution is the square of this value, $\sigma^2$.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self, mean=0.0, sigma=1.0):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.mean  = mean
    self.sigma = sigma
    self.hasInfiniteBound = True
    self.type = 'Normal'
    self.distType = 'Continuous'
    self.compatibleQuadrature.append('Hermite')
    self.compatibleQuadrature.append('CDF')
    #THESE get set in initializeDistribution, since it depends on truncation
    #self.preferredQuadrature  = 'Hermite'
    #self.preferredPolynomials = 'Hermite'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.mean  = pdict.pop('mean' )
    self.sigma = pdict.pop('sigma')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['mu'] = self.mean
    retDict['sigma'] = self.sigma
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    meanFind = paramInput.findFirst('mean' )
    if meanFind is not None:
      self.mean  = meanFind.value
    else:
      self.raiseAnError(IOError,'mean value needed for normal distribution')
    sigmaFind = paramInput.findFirst('sigma')
    if sigmaFind is not None:
      self.sigma = sigmaFind.value
    else:
      self.raiseAnError(IOError,'sigma value needed for normal distribution')
    self.initializeDistribution() #FIXME no other distros have this...needed?

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['mean' ] = self.mean
    paramDict['sigma'] = self.sigma
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.convertToDistrDict['Hermite'] = self.convertHermiteToNormal
    self.convertToQuadDict ['Hermite'] = self.convertNormalToHermite
    self.measureNormDict   ['Hermite'] = self.stdProbabilityNorm
    if (not self.upperBoundUsed) and (not self.lowerBoundUsed):
      self._distribution = Distributions1D.BasicNormalDistribution(self.mean, self.sigma)
      self.lowerBound = -sys.float_info.max
      self.upperBound =  sys.float_info.max
      self.preferredQuadrature  = 'Hermite'
      self.preferredPolynomials = 'Hermite'
    else:
      self.preferredQuadrature  = 'CDF'
      self.preferredPolynomials = 'Legendre'
      if self.lowerBoundUsed == False:
        a = -sys.float_info.max
        self.lowerBound = a
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:
        b = self.upperBound
      self._distribution = Distributions1D.BasicNormalDistribution(self.mean, self.sigma, a, b)

  def stdProbabilityNorm(self,std=False):
    """Returns the factor to scale error norm by so that norm(probability)=1.
    @ In, None, None
    @ Out float, norm
    """
    sv = str(scipy.__version__).split('.')
    if int(sv[0])==0 and int(sv[1])==15:
      self.raiseAWarning('SciPy 0.15 detected!  In this version, the normalization factor for normal distributions was modified.')
      self.raiseAWarning('Using modified value...')
      return 1.0/np.sqrt(np.pi/2.)
    else:
      return 1.0/np.sqrt(2.*np.pi)

  def convertNormalToHermite(self,y):
    """Converts from distribution domain to standard Hermite [-inf,inf].
    @ In, y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return (y-self.untruncatedMean())/(self.sigma)

  def convertHermiteToNormal(self,x):
    """Converts from standard Hermite [-inf,inf] to distribution domain.
    @ In, y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return self.sigma*x+self.untruncatedMean()

DistributionsCollection.addSub(Normal.getInputSpecification())

class Gamma(BoostDistribution):
  """
    Gamma univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Gamma, cls).getInputSpecification()
    inputSpecification.description = r"""Classical gamma distribution. The probability density function for the gamma
      distribution is given by $f(x,\alpha)=\frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}$
      for $x\geq 0,\alpha>0$, and where $\Gamma(\alpha)$ refers to the gamma function.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("low",
        descr=r"""the lower domain value of this distribution. Setting this to a nonzero value will shift the
          distribution to the left or right. \default{0.0} """,
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha",
        descr=r"""first shape parameter.
          See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html} for more details.  """,
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta",
        descr=r"""rate parameter, also inverse scale parameter.
          See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html} for more details. \default{1.0}
          """,
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self, low=0.0, alpha=0.0, beta=1.0):
    """
      Constructor
      @ In, low, float, lower domain boundary
      @ In, alpha, float, shape parameter
      @ In, beta, float, 1/scale or the inverse scale parameter
      @ Out, None
    """
    super().__init__()
    self.low = low
    self.alpha = alpha
    self.beta = beta
    self.type = 'Gamma'
    self.distType = distType.continuous
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('Laguerre')
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'Laguerre'
    self.preferredPolynomials = 'Laguerre'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.low   = pdict.pop('low'  )
    self.alpha = pdict.pop('alpha')
    self.beta  = pdict.pop('beta' )

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['k'] = self.alpha
    retDict['theta'] = 1.0/self.beta
    retDict['low'] = self.low
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    lowFind = paramInput.findFirst('low')
    if lowFind != None:
      self.low = lowFind.value
    alphaFind = paramInput.findFirst('alpha')
    if alphaFind != None:
      self.alpha = alphaFind.value
    else:
      self.raiseAnError(IOError,'alpha value needed for Gamma distribution')
    betaFind = paramInput.findFirst('beta')
    if betaFind != None:
      self.beta = betaFind.value
    # check if lower bound are set, otherwise default
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.low
    self.initializeDistribution() #TODO this exists in a couple classes; does it really need to be here and not in Simulation? - No. - Andrea

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['low'] = self.low
    paramDict['alpha'] = self.alpha
    paramDict['beta'] = self.beta
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.convertToDistrDict['Laguerre'] = self.convertLaguerreToGamma
    self.convertToQuadDict ['Laguerre'] = self.convertGammaToLaguerre
    self.measureNormDict   ['Laguerre'] = self.stdProbabilityNorm
    if (not self.upperBoundUsed):
      # and (not self.lowerBoundUsed):
      self._distribution = Distributions1D.BasicGammaDistribution(self.alpha,1.0/self.beta,self.low)
      #self.lowerBoundUsed = 0.0
      self.upperBound     = sys.float_info.max
      self.preferredQuadrature  = 'Laguerre'
      self.preferredPolynomials = 'Laguerre'
    else:
      self.preferredQuadrature  = 'CDF'
      self.preferredPolynomials = 'Legendre'
      if self.lowerBoundUsed == False:
        a = 0.0
        self.lowerBound = a
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:
        b = self.upperBound
      self._distribution = Distributions1D.BasicGammaDistribution(self.alpha,1.0/self.beta,self.low,a,b)

  def convertGammaToLaguerre(self,y):
    """Converts from distribution domain to standard Laguerre [0,inf].
    @ In, y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return (y-self.low)*(self.beta)

  def convertLaguerreToGamma(self,x):
    """Converts from standard Laguerre [0,inf] to distribution domain.
    @ In, y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return x/self.beta+self.low

  def stdProbabilityNorm(self):
    """Returns the factor to scale error norm by so that norm(probability)=1.
    @ In, None, None
    @ Out float, norm
    """
    return 1./factorial(self.alpha-1)

DistributionsCollection.addSub(Gamma.getInputSpecification())

class Beta(BoostDistribution):
  """
    Beta univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Beta, cls).getInputSpecification()
    inputSpecification.description = r"""Classical beta distribution. The probability density function for the beta
      distribution is given by $f(x;\alpha,\beta)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}$
      for $\alpha>0, \beta>0$ and $0 \leq x \leq 1$
      and $B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("low",
        descr=r"""the lower domain value of this distribution. Setting this to a nonzero value will shift the
          distribution to the left or right. \default{0.0} """,
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("high",
        descr=r"""the upper domain value of this distribution. Setting this to a nonzero value will shift the right side
          of the distribution to the left or right. \default{0.0} """,
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha",
        descr=r"""first shape parameter. If specified, \xmlNode{beta} must also be provided, and
          \xmlNode{peakFactor} cannot be specified.
          See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html} for more details. """,
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta",
        descr=r"""second shape parameter. If specified, \xmlNode{alpha} must also be provided, and
          \xmlNode{peakFactor} cannot be specified.
          See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html} for more details. """,
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("peakFactor",
        descr=r"""alternate shape parameter. If specified, neither \xmlNode{alpha} nor \xmlNode{beta} may be specified.
          """,
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.low = 0.0
    self.high = 1.0
    self.alpha = 0.0
    self.beta = 0.0
    self.type = 'Beta'
    self.distType = distType.continuous
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('Jacobi')
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'Jacobi'
    self.preferredPolynomials = 'Jacobi'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.low   = pdict.pop('low'  )
    self.high  = pdict.pop('high' )
    self.alpha = pdict.pop('alpha')
    self.beta  = pdict.pop('beta' )

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['alpha'] = self.alpha
    retDict['beta'] = self.beta
    retDict['scale'] = self.high-self.low
    retDict['low'] = self.low
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    lowFind = paramInput.findFirst('low')
    if lowFind != None:
      self.low = lowFind.value
    hiFind = paramInput.findFirst('high')
    if hiFind != None:
      self.high = hiFind.value
    alphaFind = paramInput.findFirst('alpha')
    betaFind = paramInput.findFirst('beta')
    peakFind = paramInput.findFirst('peakFactor')
    if alphaFind != None and betaFind != None and peakFind == None:
      self.alpha = alphaFind.value
      self.beta  = betaFind.value
    elif (alphaFind == None and betaFind == None) and peakFind != None:
      peakFactor = peakFind.value
      if not 0 <= peakFactor <= 1:
        self.raiseAnError(IOError,'peakFactor must be from 0 to 1, inclusive!')
      #this empirical formula is used to make it so factor->alpha: 0->1, 0.5~7.5, 1->99
      self.alpha = 0.5*23.818**(5.*peakFactor/3.) + 0.5
      self.beta = self.alpha
    else:
      self.raiseAnError(IOError,'Either provide (alpha and beta) or peakFactor!')
    # check if lower or upper bounds are set, otherwise default
    if not self.upperBoundUsed:
      self.upperBoundUsed = True
      self.upperBound     = self.high
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.low
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['low'  ] = self.low
    paramDict['high' ] = self.high
    paramDict['alpha'] = self.alpha
    paramDict['beta' ] = self.beta
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.convertToDistrDict['Jacobi'] = self.convertJacobiToBeta
    self.convertToQuadDict ['Jacobi'] = self.convertBetaToJacobi
    self.measureNormDict   ['Jacobi'] = self.stdProbabilityNorm
    #this "if" section can only be called if distribution not generated using readMoreXML
    if (not self.upperBoundUsed) and (not self.lowerBoundUsed):
      self._distribution = Distributions1D.BasicBetaDistribution(self.alpha,self.beta,self.high-self.low,self.low)
    else:
      if self.lowerBoundUsed == False:
        a = 0.0
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
      else:
        b = self.upperBound
      self._distribution = Distributions1D.BasicBetaDistribution(self.alpha,self.beta,self.high-self.low,self.low,a,b)
    self.preferredPolynomials = 'Jacobi'
    self.compatibleQuadrature.append('Jacobi')
    self.compatibleQuadrature.append('ClenshawCurtis')

  def convertBetaToJacobi(self,y):
    """
      Converts from distribution domain to standard Beta [0,1].
      @ In, y, float/array of floats, points to convert
      @ Out, convertBetaToJacobi, float/array of floats, converted points
    """
    u = 0.5*(self.high+self.low)
    s = 0.5*(self.high-self.low)
    return (y-u)/(s)

  def convertJacobiToBeta(self,x):
    """
      Converts from standard Jacobi [0,1] to distribution domain.
      @ In, y, float/array of floats, points to convert
      @ Out, convertJacobiToBeta, float/array of floats, converted points
    """
    u = 0.5*(self.high+self.low)
    s = 0.5*(self.high-self.low)
    return s*x+u

  def stdProbabilityNorm(self):
    """
      Returns the factor to scale error norm by so that norm(probability)=1.
      @ In, None
      @ Out, norm, float, norm
    """
    B = factorial(self.alpha-1)*factorial(self.beta-1)/factorial(self.alpha+self.beta-1)
    norm = 1.0/(2**(self.alpha+self.beta-1)*B)
    return norm

DistributionsCollection.addSub(Beta.getInputSpecification())

class Triangular(BoostDistribution):
  """
    Triangular univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Triangular, cls).getInputSpecification()
    inputSpecification.description = r"""classical triangular distribution. The probability density function for the
      triangular distribution is given by
      $f(x)=\frac{2(x-a)}{(b-a)(c-a))}$ for $a \le x < c$, and $\frac{2(b-x)}{(b-a)(b-c)}$ for $c< x \le b$, and 0 otherwise.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("min",
        descr=r"""lower domain boundary of this distribution, referred to as $a$ in equation form.""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("max",
        descr=r"""upper domain boundary of this distribution, referred to as $b$ in equation form.""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("apex",
        descr=r"""location of the peak of the distribution, referred to as $c$ in equation form.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.apex = 0.0   # peak location
    self.min  = None  # domain lower boundary
    self.max  = None  # domain upper boundary
    self.type = 'Triangular'
    self.distType = distType.continuous
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.apex = pdict.pop('apex')
    self.min  = pdict.pop('min' )
    self.max  = pdict.pop('max' )

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['xPeak'] = self.apex
    retDict['lowerBound'] = self.min
    retDict['upperBound'] = self.max
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    apexFind = paramInput.findFirst('apex')
    if apexFind != None:
      self.apex = apexFind.value
    else:
      self.raiseAnError(IOError,'apex value needed for Triangular  distribution')
    minFind = paramInput.findFirst('min')
    if minFind != None:
      self.min = minFind.value
    else:
      self.raiseAnError(IOError,'min value needed for Triangular distribution')
    maxFind = paramInput.findFirst('max')
    if maxFind != None:
      self.max = maxFind.value
    else:
      self.raiseAnError(IOError,'max value needed for Triangular distribution')
    # check if lower or upper bounds are set, otherwise default
    if not self.upperBoundUsed:
      self.upperBoundUsed = True
      self.upperBound     = self.max
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.min
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['apex' ] = self.apex
    paramDict['min'  ] = self.min
    paramDict['max'  ] = self.max
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False) or (self.min == self.lowerBound and self.max == self.upperBound):
      self._distribution = Distributions1D.BasicTriangularDistribution(self.apex,self.min,self.max)
    else:
      self.raiseAnError(IOError,'Truncated triangular not yet implemented')

DistributionsCollection.addSub(Triangular.getInputSpecification())

class Poisson(BoostDistribution):
  """
    Poisson univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Poisson, cls).getInputSpecification()
    inputSpecification.description = r"""classical Poisson discrete distribution. The probability mass function for the
      Poisson distribution is given by
      $f(k)=\frac{\mu^k e^{-\mu}}{k!}$ where $k$ is the number of occurances and $\mu$ is the mean.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("mu",
        descr=r"""mean rate of events/time""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.mu  = 0.0
    self.type = 'Poisson'
    self.hasInfiniteBound = True
    self.distType = distType.discrete
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.mu = pdict.pop('mu')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['mu'] = self.mu
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    muFind = paramInput.findFirst('mu')
    if muFind != None:
      self.mu = muFind.value
    else:
      self.raiseAnError(IOError,'mu value needed for poisson distribution')
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['mu'  ] = self.mu
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = Distributions1D.BasicPoissonDistribution(self.mu)
      self.lowerBound = 0.0
      self.upperBound = sys.float_info.max
    else:
      self.raiseAnError(IOError,'Truncated poisson not yet implemented')

DistributionsCollection.addSub(Poisson.getInputSpecification())

class Binomial(BoostDistribution):
  """
    Binomial univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Binomial, cls).getInputSpecification()
    inputSpecification.description = r"""classical binomial discrete distribution. The probability mass function for the
      binomial distribution is given by
      $f(k;n,p)=\binom{n}{k}p^k (1-p)^{n-k}$ where $k$ is the number of occurances, $p$ is the probabilty of occurance,
      and $n$ is the number of experiments.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("n",
        descr=r"""number of experiments or trials.""",
        contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("p",
        descr=r"""probability of occurance, often probability of a success.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.n       = 0.0
    self.p       = 0.0
    self.type     = 'Binomial'
    self.hasInfiniteBound = True
    self.distType = distType.discrete
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.n = pdict.pop('n')
    self.p = pdict.pop('p')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['n'] = self.n
    retDict['p'] = self.p
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    nFind = paramInput.findFirst('n')
    if nFind != None:
      self.n = nFind.value
    else:
      self.raiseAnError(IOError,'n value needed for Binomial distribution')
    pFind = paramInput.findFirst('p')
    if pFind != None:
      self.p = pFind.value
    else:
      self.raiseAnError(IOError,'p value needed for Binomial distribution')
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['n'  ] = self.n
    paramDict['p'  ] = self.p
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = Distributions1D.BasicBinomialDistribution(self.n,self.p)
    else:
      self.raiseAnError(IOError,'Truncated Binomial not yet implemented')

DistributionsCollection.addSub(Binomial.getInputSpecification())

class Bernoulli(BoostDistribution):
  """
    Bernoulli univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Bernoulli, cls).getInputSpecification()
    inputSpecification.description = r"""classical Bernoulli discrete distribution. The probability mass function for the
      Bernoulli distribution is given by
      $f(k;p)=p$ if $k=1$ and $f(k;p)=1-p$ if $k=0$, where $k$ is the possible outcomes and $p$ is the probabilty of occurance.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Bernoulli.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("p",
        descr=r"""probability of occurrance, or success.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.p        = 0.0
    self.type     = 'Bernoulli'
    self.distType = distType.discrete
    self.lowerBound = 0.0
    self.upperBound = 1.0
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.p = pdict.pop('p')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['p'] = self.p
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    pFind = paramInput.findFirst('p')
    if pFind != None:
      self.p = pFind.value
    else:
      self.raiseAnError(IOError,'p value needed for Bernoulli distribution')
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['p'] = self.p
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = Distributions1D.BasicBernoulliDistribution(self.p)
    else:
      self.raiseAnError(IOError,'Truncated Bernoulli not yet implemented')

DistributionsCollection.addSub(Bernoulli.getInputSpecification())

class Geometric(BoostDistribution):
  """
    Geometric univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Geometric, cls).getInputSpecification()
    inputSpecification.description = r"""classical geometric discrete distribution. The probability mass function for the
      geometric distribution is given by
      $f(k;p)=(1-p)^{k}p$, where $k$ is the possible outcomes and $p$ is the probabilty of occurance.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("p",
        descr=r"""probability of occurrance, or success.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.p        = 0.0
    self.type     = 'Geometric'
    self.distType = distType.discrete
    self.lowerBound = 0.0
    self.upperBound = 1.0
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.p = pdict.pop('p')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['p'] = self.p
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    pFind = paramInput.findFirst('p')
    if pFind != None:
      self.p = pFind.value
    else: self.raiseAnError(IOError,'p value needed for Geometric distribution')
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['p'] = self.p
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = Distributions1D.BasicGeometricDistribution(self.p)
    else:  self.raiseAnError(IOError,'Truncated Geometric not yet implemented')

DistributionsCollection.addSub(Geometric.getInputSpecification())

class Categorical(Distribution):
  """
    Class for the categorical distribution also called "generalized Bernoulli distribution"
    Note: this distribution can have only numerical (float) outcome; in the future we might want to include also the possibility to give symbolic outcome
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = InputData.parameterInputFactory(cls.__name__, ordered=True, baseNode=None)
    inputSpecification.description = r"""classical categorical discrete distribution, sometimes also referred to
      as a multinomial distribution. The categorical distribution
      describes the result of a random variable that can have $K$ possible outcome states, with each outcome potentially
      having a distinct probability. These states can be numbers as well as strings.
      """
    StatePartInput = InputData.parameterInputFactory("state",
        descr=r"""probability of this state's outcome""",
        contentType=InputTypes.FloatType)
    StatePartInput.addParam("outcome",
        InputTypes.FloatOrStringType,
        True,
        descr=r"""value of this state's outcome""")
    inputSpecification.addSub(StatePartInput, InputData.Quantity.one_to_infinity)
    inputSpecification.addSub(InputData.parameterInputFactory("rtol",
        contentType=InputTypes.FloatType,
        descr=r"""Relative tolerance used to identify close states in case of
            float/int states. Not used for string states!""",
        default=1e-6))
    ## Because we do not inherit from the base class, we need to manually
    ## add the name back in.
    inputSpecification.addParam("name", InputTypes.StringType, True,
        descr=r"""User-defined name to designate this entity in the RAVEN input file.""")

    return inputSpecification

  def __init__(self):
    """
      Function that initializes the categorical distribution
      @ In, None
      @ Out, none
    """
    super().__init__()
    self.mapping        = {}
    self.values         = set()
    self.type           = 'Categorical'
    self.dimensionality = 1
    self.distType       = distType.discrete
    self.isFloat        = True
    self.rtol = 1e-6

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    isFloats = []
    for child in paramInput.subparts:
      if child.getName() == "state":
        outcome = child.parameterValues["outcome"]
        value = child.value
        self.mapping[outcome] = value
        isFloats.append(utils.floatConversion(outcome) is not None)
        if outcome in self.values:
          self.raiseAnError(IOError,'Categorical distribution has identical outcomes')
        else:
          self.values.add(float(outcome) if isFloats[-1] else outcome)
      else:
        self.raiseAnError(IOError,'Invalid xml node for Categorical distribution; only "state" is allowed')
    if False in isFloats:
      self.isFloat = False
    else:
      self.rtol = paramInput.findNodesAndExtractValues(['rtol'])[0]['rtol']
    self.initializeDistribution()
    self.upperBoundUsed = True
    self.lowerBoundUsed = True

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Distribution.getInitParams(self)
    paramDict['mapping'] = self.mapping
    paramDict['values'] = self.values
    return paramDict

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.mapping = pdict.pop('mapping')
    self.values = pdict.pop('values')

  def initializeFromDict(self, inputDict):
    """
      Function that initializes the distribution provided a dictionary
      @ In, inputDict, dict, dictionary containing the np.arrays for xAxis and pAxis
      @ Out, None
    """
    isFloats = []
    for idx, val in enumerate(inputDict['outcome']):
      self.mapping[val] = inputDict['state'][idx]
      self.values.add(val)
      isFloats.append(utils.floatConversion(val) is not None)
    if False in isFloats:
      self.isFloat = False
    self.checkDistParams()

  def initializeDistribution(self):
    """
      Function that initializes the distribution
      @ In, None
      @ Out, None
    """
    self.checkDistParams()

    self.lowerBound = min(self.mapping.keys())
    self.upperBound = max(self.mapping.keys())

  def checkDistParams(self):
    """
      Function that checks that the sum of all state probabilities is equal to 1 and perform pdf value normalization
      @ In, None
      @ Out, None
    """
    # check all probability values are between 0.0 and 1.0
    for element, value in self.mapping.items():
      if value < 0.0:
        self.raiseAnError(IOError,f'Categorical distribution entry {element} cannot be negative; received: {value}')

    localSum = sum(self.mapping.values())
    if not mathUtils.compareFloats(localSum, 1., self.rtol):
      # courtesy warning
      self.raiseAWarning(f'Provided weights for Categorical distribution summed to {localSum}; normalizing to 1.')

    # Probability values normalization
    for key in self.mapping.keys():
      self.mapping[key] = self.mapping[key]/localSum

  def pdf(self,x):
    """
      Function that calculates the pdf value of x
      @ In, x, float/string, value to get the pdf at
      @ Out, pdfValue, float, requested pdf
    """
    if x in self.values:
      pdfValue = self.mapping[x]
    else:
      if self.isFloat:
        vals = sorted(list(self.values))
        idx = [idx for idx in range(len(vals)) if utils.isClose(vals[idx], x, relTolerance=self.rtol)]
        if not len(idx):
          self.raiseAnError(IOError,f'{self.type} distribution cannot compute pdf for {x} since the closest '
                            f'state {list(vals)[bisect.bisect(vals, x)]} is outside the acceptance interval given by the provided relative tolerance {self.rtol}!')
        idx = idx[0]
        val =  list(vals)[idx]
        pdfValue = self.mapping[val]
        if not utils.isClose(val, x, relTolerance=self.rtol):
          self.raiseAnError(IOError,f'{self.type} distribution cannot compute pdf for {x} since the closest '
                            f'state {val} is outside the acceptance interval given by the provided relative tolerance {self.rtol}!')
      else:
        self.raiseAnError(IOError,f'{self.type} distribution cannot compute pdf for {x} since the states are not floats/integers and the'
                            f'value {x} is not present in the list of states!!')
    return pdfValue

  def cdf(self,x):
    """
      Function to get the cdf value of x
      @ In, x, float/string, value to get the cdf at
      @ Out, cumulative, float, requested cdf
    """
    sortedMapping = sorted(self.mapping.items(), key=operator.itemgetter(0))
    if x == sortedMapping[-1][0]:
      return 1.0

    if x in self.values:
      cumulative=0.0
      for element in sortedMapping:
        cumulative += element[1]
        if x == ( float(element[0]) if self.isFloat else element[0] ):
          return cumulative
    else:
      if self.isFloat:
        cumulative=0.0
        for idx in range(len(sortedMapping)-1):
          cumulative += sortedMapping[idx][1]
          if x >= sortedMapping[idx][0] and x <= sortedMapping[idx+1][0]:
            return cumulative
      # if we reach this point we must error out
      self.raiseAnError(IOError,f'{self.type} distribution cannot calculate cdf for ' + str(x))

  def ppf(self,x):
    """
      Function that calculates the inverse of the cdf given 0 =< x =< 1
      @ In, x, float, value to get the ppf at
      @ Out, element[0], float/string, requested inverse cdf
    """
    if x > 1. or x < 0.:
      self.raiseAnError(IOError,f'{self.type} distribution cannot calculate ppf for', str(x), '! Valid value should within [0,1]!')
    sortedMapping = sorted(self.mapping.items(), key=operator.itemgetter(0))
    if x == 1.0:
      return float(sortedMapping[-1][0]) if self.isFloat else sortedMapping[-1][0]
    else:
      cumulative=0.0
      for element in sortedMapping:
        cumulative += element[1]
        if cumulative >= x:
          return float(element[0]) if self.isFloat else element[0]

  def rvs(self):
    """
      Return a random state of the categorical distribution
      @ In, None
      @ Out, rvsValue, float, the random state
    """
    rvsValue = self.ppf(random())
    return rvsValue

DistributionsCollection.addSub(Categorical.getInputSpecification())

class UniformDiscrete(Distribution):
  """
    Class for the uniform discrete distribution
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    BaseInputType = InputTypes.makeEnumType("base", "baseType", ["withReplacement","withoutReplacement"])

    specs = super(UniformDiscrete, cls).getInputSpecification()
    specs.description = r"""The UniformDiscrete distribution is a discrete distribution which describes a random variable
                            that can have $N$ values having equal probability value. This distribution allows the user to
                            choose two kinds of sampling strategies: with or without replacement.
                            In case the ``without replacement'' strategy is used, the distribution samples from the set of
                            specified $N$ values reduced by the previously sampled values. After, the sampler has generated
                            values for all variables, the distribution is resetted (i.e., the set of values that can be sampled
                            is returned to $N$). In case the ``with replacement'' strategy is used, the distribution samples
                            always from the complete set of specified $N$ values.
                            """

    np = InputData.parameterInputFactory('nPoints', contentType=InputTypes.IntegerType, printPriority=109,
    descr=r""" Number of points between lower and upper bound. """)
    specs.addSub(np)

    strategy = InputData.parameterInputFactory('strategy', BaseInputType, printPriority=109,
    descr=r""" Type of sampling strategy. """)
    specs.addSub(strategy)

    return specs

  def __init__(self):
    """
      Function that initializes the Uniform Discrete distribution
      @ In, None
      @ Out, none
    """
    super().__init__()
    self.type           = 'UniformDiscrete'
    self.dimensionality = 1
    self.distType       = distType.discrete
    self.memory         = True
    self.nPoints = None

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    if self.lowerBound is None:
      self.raiseAnError(IOError,'lowerBound value needed for UniformDiscrete distribution')

    if self.upperBound is None:
      self.raiseAnError(IOError,'upperBound value needed for UniformDiscrete distribution')

    strategy = paramInput.findFirst('strategy')
    if strategy is not None:
      self.strategy = strategy.value
    else:
      self.raiseAnError(IOError,'strategy specification needed for UniformDiscrete distribution')

    nPoints = paramInput.findFirst('nPoints')
    if nPoints is not None:
      self.nPoints = nPoints.value

    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Distribution.getInitParams(self)
    paramDict['strategy'] = self.strategy
    paramDict['nPoints'] = self.nPoints
    return paramDict

  def initializeDistribution(self):
    """
      Function that initializes the distribution
      @ In, None
      @ Out, None
    """
    if self.nPoints is None:
      self.xArray   = np.arange(self.lowerBound,self.upperBound+1)
    else:
      self.xArray   = np.linspace(self.lowerBound,self.upperBound,self.nPoints)

    # Here the actual calculation of discrete distribution parameters is performed
    self.pdfArray = 1.0/self.xArray.size * np.ones(self.xArray.size)
    paramsDict={}
    paramsDict['outcome'] = self.xArray
    paramsDict['state'] = self.pdfArray

    self.categoricalDist = Categorical()
    self.categoricalDist.initializeFromDict(paramsDict)
    initialPerm = randomUtils.randomPermutation(self.xArray.tolist(),self)
    self.pot = np.asarray(initialPerm)

  def initializeFromDict(self, inputDict):
    """
      Function that initializes the distribution provided a dictionary
      @ In, inputDict, dict, dictionary containing the np.arrays for xAxis and pAxis
      @ Out, None
    """
    self.strategy = inputDict['strategy']
    self.categoricalDist = Categorical()
    self.categoricalDist.initializeFromDict(inputDict)
    initialPerm = randomUtils.randomPermutation(inputDict['outcome'].tolist(),self)
    self.pot = np.asarray(initialPerm)

  def pdf(self,x):
    """
      Function that calculates the pdf value of x
      @ In, x, float/string, value to get the pdf at
      @ Out, pdfValue, float, requested pdf
    """
    return self.categoricalDist.pdf(x)

  def cdf(self,x):
    """
      Function to get the cdf value of x
      @ In, x, float/string, value to get the cdf at
      @ Out, cumulative, float, requested cdf
    """
    return self.categoricalDist.cdf(x)

  def ppf(self,x):
    """
      Function that calculates the inverse of the cdf given 0 =< x =< 1
      @ In, x, float, value to get the ppf at
      @ Out, element[0], float/string, requested inverse cdf
    """
    return self.categoricalDist.ppf(x)

  def rvs(self):
    """
      Return a random state of the distribution
      @ In, None
      @ Out, rvsValue, float, the random state
    """
    if self.strategy == 'withReplacement':
      return self.categoricalDist.rvs()
    else:
      if self.pot.size == 0:
        # re-initialize the distribution
        self.reset()
        self.raiseAWarning("The Uniform Discrete distribution " + str(self.name) + " has been internally reset outside the sampler.")
      rvsValue = self.pot[-1]
      self.pot = np.resize(self.pot, self.pot.size - 1)
    return rvsValue

  def selectedRvs(self,discardedElems):
    """
      Return a random state of the distribution without discardedElems
      @ In, discardedElems, np array, list of discarded elements
      @ Out, rvsValue, float, the random state
    """
    if self.nPoints is None:
      self.xArray   = np.arange(self.lowerBound,self.upperBound+1)
    else:
      self.xArray   = np.linspace(self.lowerBound,self.upperBound,self.nPoints)

    self.xArray = np.setdiff1d(self.xArray,discardedElems)

    self.pdfArray = 1/self.xArray.size * np.ones(self.xArray.size)
    paramsDict={}
    paramsDict['outcome'] = self.xArray
    paramsDict['state'] = self.pdfArray
    paramsDict['strategy'] = self.strategy

    self.tempUniformDiscrete = UniformDiscrete()
    self.tempUniformDiscrete.initializeFromDict(paramsDict)

    rvsValue = self.tempUniformDiscrete.rvs()
    return rvsValue

  def reset(self):
    """
      Reset the distribution
      @ In, None
      @ Out, None
    """
    newPerm = randomUtils.randomPermutation(self.xArray.tolist(),self)
    self.pot = np.asarray(newPerm)

DistributionsCollection.addSub(UniformDiscrete.getInputSpecification())

class MarkovCategorical(Categorical):
  """
    Class for the Markov categorical distribution based on "Markov Model"
    Note: this distribution can have only numerical (float) outcome; in the future we might want to include also the possibility to give symbolic outcome
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = InputData.parameterInputFactory(cls.__name__, ordered=True, baseNode=None)
    inputSpecification.description = r"""Markov categorical discrete distribution. Describes a random variable
      that can have $K$ possible outcomes, based on the steady state probilities provided by a Markov model.
      """

    StatePartInput = InputData.parameterInputFactory("state",
        descr=r"""probability of occurrance, or outcome 1.""",
        contentType=InputTypes.StringType)
    StatePartInput.addParam("outcome",
        InputTypes.FloatType,
        True,
        descr=r"""value of this outcome""")
    StatePartInput.addParam("index",
        InputTypes.IntegerType,
        True,
        descr=r"""indexes steady state probabilities corresponding to the transition matrix""")
    TransitionInput = InputData.parameterInputFactory("transition",
        descr=r"""transition matrix of the desired Markov model""",
        contentType=InputTypes.StringType)
    inputSpecification.addSub(StatePartInput, InputData.Quantity.one_to_infinity)
    inputSpecification.addSub(TransitionInput, InputData.Quantity.zero_to_one)
    inputSpecification.addSub(InputData.parameterInputFactory("workingDir",
        descr=r"""filesystem referential path""",
        contentType=InputTypes.StringType))
    ## Because we do not inherit from the base class, we need to manually
    ## add the name back in.
    inputSpecification.addParam("name", InputTypes.StringType, True)

    return inputSpecification

  def __init__(self):
    """
      Function that initializes the categorical distribution
      @ In, None
      @ Out, none
    """
    super().__init__()
    self.dimensionality = 1
    self.distType       = distType.discrete
    self.type           = 'MarkovCategorical'
    self.steadyStatePb  = None # variable containing the steady state probabilities of the Markov Model
    self.transition     = None # transition matrix of a continuous time Markov Model

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    workingDir = paramInput.findFirst('workingDir')
    if workingDir is not None:
      self.workingDir = workingDir.value
    else:
      self.workingDir = os.getcwd()

    for child in paramInput.subparts:
      if child.getName() == "state":
        outcome = child.parameterValues["outcome"]
        markovIndex = child.parameterValues["index"]
        self.mapping[outcome] = markovIndex
        if outcome in self.values:
          self.raiseAnError(IOError,'Markov Categorical distribution has identical outcomes')
        else:
          self.values.add(outcome)
      elif child.getName() == "transition":
        transition = [float(value) for value in child.value.split()]
        dim = int(np.sqrt(len(transition)))
        if dim == 1:
          self.raiseAnError(IOError, "The dimension of transition matrix should be greater than 1!")
        elif dim**2 != len(transition):
          self.raiseAnError(IOError, "The transition matrix is not a square matrix!")
        self.transition = np.asarray(transition).reshape((-1,dim))
    #Check the correctness of user inputs
    invalid = self.transition is None
    if invalid:
      self.raiseAnError(IOError, "Transition matrix is not provided, please use 'transition' node to provide the transition matrix!")
    if len(self.mapping.values()) != len(set(self.mapping.values())):
      self.raiseAnError(IOError, "The states of Markov Categorical distribution have identifcal indices!")

    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict['transition'] = self.transition
    paramDict['steadyStatePb'] = self.steadyStatePb
    return paramDict

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    super()._localSetState(pdict)
    self.transition = pdict.pop('transition')
    self.steadyStatePb = pdict.pop('steadyStatePb')

  def initializeDistribution(self):
    """
      Function that initializes the distribution and checks that the sum of all state probabilities is equal to 1
      @ In, None
      @ Out, None
    """
    self.steadyStatePb = self.computeSteadyStatePb(self.transition)
    for key, value in self.mapping.items():
      try:
        self.mapping[key] = self.steadyStatePb[value - 1]
      except IndexError:
        self.raiseAnError(IOError, "Index ",value, " for outcome ", key, " is out of bounds! Maximum index should be ", len(self.steadyStatePb))
    super().initializeDistribution()

  def computeSteadyStatePb(self, transition):
    """
      Function that compute the steady state probabilities for given transition matrix
      @ In, transition, numpy.array, transition matrix for Markov model
      @ Out, steadyStatePb, numpy.array, 1-D array of steady state probabilities
    """
    dim = transition.shape[0]
    perturbTransition = copy.copy(transition)
    perturbTransition[0] = 1
    q = np.zeros(dim)
    q[0] = 1
    steadyStatePb = np.dot(LA.inv(perturbTransition),q)

    return steadyStatePb

DistributionsCollection.addSub(MarkovCategorical.getInputSpecification())

class Logistic(BoostDistribution):
  """
    Logistic univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Logistic, cls).getInputSpecification()
    inputSpecification.description = r"""classical logistic distribution. The probability density function for the
      logistic distribution is given by
      $f(x)=\frac{1}{1+e^{-\frac{x-\lambda}{\sigma}}}$ for $-\infty \leq x < \infty$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("location",
        descr=r"""location parameter, referred to as $\lambda$ in equation form.""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("scale",
        descr=r"""scale parameter, referred to as $\sigma$ in equation form.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.location  = 0.0
    self.scale = 1.0
    self.type = 'Logistic'
    self.distType = distType.continuous
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.location = pdict.pop('location')
    self.scale    = pdict.pop('scale'   )

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['scale'] = self.scale
    retDict['location'] = self.location
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    locationFind = paramInput.findFirst('location')
    if locationFind != None:
      self.location = locationFind.value
    else:
      self.raiseAnError(IOError,'location value needed for Logistic distribution')
    scaleFind = paramInput.findFirst('scale')
    if scaleFind != None:
      self.scale = scaleFind.value
    else:
      self.raiseAnError(IOError,'scale value needed for Logistic distribution')
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['location'] = self.location
    paramDict['scale'   ] = self.scale
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = Distributions1D.BasicLogisticDistribution(self.location,self.scale)
    else:
      if self.lowerBoundUsed == False:
        a = -sys.float_info.max
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
      else:
        b = self.upperBound
      self._distribution = Distributions1D.BasicLogisticDistribution(self.location,self.scale,a,b)

DistributionsCollection.addSub(Logistic.getInputSpecification())

class Laplace(BoostDistribution):
  """
    Laplace univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Laplace, cls).getInputSpecification()
    inputSpecification.description = r"""classical Laplace distribution. The probability density function for the
      Laplace distribution is given by
      $f(x)=\frac{1}{2b}e^{-\frac{\left| x-\mu\right| }{b}}$ for $-\infty \leq x \leq \infty$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("location",
        descr=r"""location parameter, referred to as $\mu$ in equation form.""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("scale",
        descr=r"""scale parameter, referred to as $b$ in equation form.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.location  = 0.0
    self.scale = 1.0
    self.type = 'Laplace'
    self.distType = distType.continuous
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.location = pdict.pop('location')
    self.scale    = pdict.pop('scale'   )

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['scale'] = self.scale
    retDict['location'] = self.location
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    locationFind = paramInput.findFirst('location')
    if locationFind != None:
      self.location = locationFind.value
    else:
      self.raiseAnError(IOError,'location value needed for Laplace distribution')
    scaleFind = paramInput.findFirst('scale')
    if scaleFind != None:
      self.scale = scaleFind.value
    else:
      self.raiseAnError(IOError,'scale value needed for Laplace distribution')
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['location'] = self.location
    paramDict['scale'   ] = self.scale
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.lowerBoundUsed == False:
      self.lowerBound = -sys.float_info.max
    if self.upperBoundUsed == False:
      self.upperBound = sys.float_info.max
    self._distribution = Distributions1D.BasicLaplaceDistribution(self.location,self.scale,self.lowerBound,self.upperBound)

DistributionsCollection.addSub(Laplace.getInputSpecification())

class Exponential(BoostDistribution):
  """
    Exponential univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Exponential, cls).getInputSpecification()
    inputSpecification.description = r"""classical exponential distribution. The probability density function for the
      exponential distribution is given by
      $f(x)=\lambda e^{-\lambda x}$ for $0 \leq x \leq \infty$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("low",
        descr=r"""lower domain boundary for this distribution""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda",
        descr=r"""rate parameter for this distribution, shown as $\lambda$ in equation form""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.lambdaVar = 1.0
    self.low        = 0.0
    self.type = 'Exponential'
    self.distType = distType.continuous
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.lambdaVar = pdict.pop('lambda')
    self.low        = pdict.pop('low'   )

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['lambda'] = self.lambdaVar
    retDict['low'] = self.low
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    lambdaFind = paramInput.findFirst('lambda')
    if lambdaFind != None:
      self.lambdaVar = lambdaFind.value
    else:
      self.raiseAnError(IOError,'lambda value needed for Exponential distribution')
    low  = paramInput.findFirst('low')
    if low != None:
      self.low = low.value
    else:
      self.low = 0.0
    # check if lower bound is set, otherwise default
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.low
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['lambda'] = self.lambdaVar  # rate parameter
    paramDict['low'   ] = self.low        # lower domain boundary
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False):
      self._distribution = Distributions1D.BasicExponentialDistribution(self.lambdaVar,self.low)
      self.lowerBound = self.low
      self.upperBound = sys.float_info.max
    else:
      if self.lowerBoundUsed == False:
        self.lowerBound = self.low
      if self.upperBoundUsed == False:
        self.upperBound = sys.float_info.max
      self._distribution = Distributions1D.BasicExponentialDistribution(self.lambdaVar,self.low,self.lowerBound,self.upperBound)

  def convertDistrPointsToStd(self,y):
    """
      Convert Distribution point to Std Point
      @ In, y, float, the point that needs to be converted
      @ Out, converted, float, the converted point
    """
    quad=self.quadratureSet()
    if quad.type=='Laguerre':
      converted = (y-self.low)*(self.lambdaVar)
    else:
      converted = Distribution.convertDistrPointsToStd(self,y)
    return converted

  def convertStdPointsToDistr(self,x):
    """
      Convert Std Point to Distribution point
      @ In, x, float, the point that needs to be converted
      @ Out, converted, float, the converted point
    """
    quad=self.quadratureSet()
    if quad.type=='Laguerre':
      converted = x/self.lambdaVar+self.low
    else:
      converted = Distribution.convertStdPointsToDistr(self,x)
    return converted

DistributionsCollection.addSub(Exponential.getInputSpecification())

class LogNormal(BoostDistribution):
  """
    LogNormal univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(LogNormal, cls).getInputSpecification()
    inputSpecification.description = r"""log-normal distribution. The probability density function for the
      log-normal distribution is given by
      $f(x)=\frac{1}{x\sigma\sqrt{2\pi}}e^{-\frac{(\ln{x}-\mu)^2}{2\sigma^2}}$ for $0 \leq x \leq \infty$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("mean",
        descr=r"""mean of log of the distribution""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("sigma",
        descr=r"""standard deviation of the log of the distribution""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("low",
        descr=r"""distribution lower domain boundary""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.mean = 1.0
    self.sigma = 1.0
    self.low = 0.0
    self.type = 'LogNormal'
    self.distType = distType.continuous
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.mean  = pdict.pop('mean' )
    self.sigma = pdict.pop('sigma')

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['mu'] = self.mean
    retDict['sigma'] = self.sigma
    retDict['low'] = self.low
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    meanFind = paramInput.findFirst('mean')
    if meanFind != None:
      self.mean = meanFind.value
    else:
      self.raiseAnError(IOError,'mean value needed for LogNormal distribution')
    sigmaFind = paramInput.findFirst('sigma')
    if sigmaFind != None:
      self.sigma = sigmaFind.value
    else:
      self.raiseAnError(IOError,'sigma value needed for LogNormal distribution')
    lowFind = paramInput.findFirst('low')
    if lowFind != None:
      self.low = lowFind.value
    else:
      self.low = 0.0
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['mean' ] = self.mean
    paramDict['sigma'] = self.sigma
    paramDict['low'] = self.low
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = Distributions1D.BasicLogNormalDistribution(self.mean,self.sigma,self.low)
      self.lowerBound = 0.0
      self.upperBound =  sys.float_info.max
    else:
      if self.lowerBoundUsed == False:
        self.lowerBound = self.low
      if self.upperBoundUsed == False:
        self.upperBound = sys.float_info.max
      self._distribution = Distributions1D.BasicLogNormalDistribution(self.mean,self.sigma,self.low,self.lowerBound,self.upperBound)

DistributionsCollection.addSub(LogNormal.getInputSpecification())

class Weibull(BoostDistribution):
  """
    Weibull univariate distribution
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Weibull, cls).getInputSpecification()
    inputSpecification.description = r"""Weibull distribution. The probability density function for the
      Weibull distribution is given by
      $f(x)=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}$ for $0 \leq x \leq \infty$.
      See \url{https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html} for more details.
      """
    inputSpecification.addSub(InputData.parameterInputFactory("low",
        descr=r"""distribution lower domain boundary""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("k",
        descr=r"""shape parameter""",
        contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda",
        descr=r"""scale parameter""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.lambdaVar = 1.0
    self.k = 1.0
    self.type = 'Weibull'
    self.distType = distType.continuous
    self.low = 0.0
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.lambdaVar = pdict.pop('lambda')
    self.k          = pdict.pop('k'     )

  def getCrowDistDict(self):
    """
      Returns a dictionary of the keys and values that would be
      used to create the distribution for a Crow input file.
      @ In, None
      @ Out, retDict, dict, the dictionary of crow distributions
    """
    retDict = Distribution.getCrowDistDict(self)
    retDict['lambda'] = self.lambdaVar
    retDict['k'] = self.k
    retDict['low'] = self.low
    return retDict

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    lambdaFind = paramInput.findFirst('lambda')
    if lambdaFind != None:
      self.lambdaVar = lambdaFind.value
    else:
      self.raiseAnError(IOError,'lambda (scale) value needed for Weibull distribution')
    kFind = paramInput.findFirst('k')
    if kFind != None:
      self.k = kFind.value
    else:
      self.raiseAnError(IOError,'k (shape) value needed for Weibull distribution')
    lowFind = paramInput.findFirst('low')
    if lowFind != None:
      self.low = lowFind.value
    else:
      self.low = 0.0
    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = BoostDistribution.getInitParams(self)
    paramDict['lambda'] = self.lambdaVar
    paramDict['k'     ] = self.k
    paramDict['low'   ] = self.low
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False):
      self._distribution = Distributions1D.BasicWeibullDistribution(self.k,self.lambdaVar,self.low)
      self.lowerBound = self.low
      self.upperBound = sys.float_info.max
    else:
      if self.lowerBoundUsed == False:
        self.lowerBound = self.low
      if self.upperBoundUsed == False:
        self.upperBound = sys.float_info.max
      self._distribution = Distributions1D.BasicWeibullDistribution(self.k,self.lambdaVar,self.lowerBound,self.upperBound,self.low)

DistributionsCollection.addSub(Weibull.getInputSpecification())

class Custom1D(Distribution):
  """
    Custom1D univariate distribution which is initialized by a dataObject compatible .csv file
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Custom1D, cls).getInputSpecification()
    inputSpecification.description = r"""Custom user-defined distribution. This allows empirically-defined
        functions that are not currently defined in RAVEN to be used for probability weighting and sampling.
        The distribution is defined through empirical distribution in a CSV file."""
    inputSpecification.addSub(InputData.parameterInputFactory("workingDir",
        descr=r"""relative working directory that contains the input file""",
        contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("functionType",
        descr=r"""type of initialization values specifid in the input file (pdf or cdf)""",
        contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("dataFilename",
        descr=r"""name of file to be used to initialize the distribution""",
        contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("functionID",
        descr=r"""ID of the function associated to the variableID in the input file""",
        contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("variableID",
        descr=r"""ID of the variable in the input file""",
        contentType=InputTypes.StringType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dataFilename    = None
    self.functionType    = None
    self.type            = 'Custom1D'
    self.functionID      = None
    self.variableID      = None
    self.dimensionality  = 1
    self.distType        = distType.continuous
    # Scipy.interpolate.UnivariateSpline is used
    self.k               = 4 # Degree of the smoothing spline, Must be <=5
    self.s               = 0 # Positive smoothing factor used to choose the number of knots
                             # Default 0, indicates spline will interpolate through all data points

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    #BoostDistribution._handleInput(self, paramInput)
    workingDir = paramInput.findFirst('workingDir')
    if workingDir != None:
      self.workingDir = workingDir.value

    self.functionType = paramInput.findFirst('functionType').value.lower()
    if self.functionType == None:
      self.raiseAnError(IOError,' functionType parameter is needed for custom1Ddistribution distribution')
    if not self.functionType in ['cdf','pdf']:
      self.raiseAnError(IOError,' wrong functionType parameter specified for custom1Ddistribution distribution (pdf or cdf)')

    dataFilename = paramInput.findFirst('dataFilename')
    if dataFilename != None:
      self.dataFilename = os.path.join(self.workingDir,dataFilename.value)
    else:
      self.raiseAnError(IOError,'<dataFilename> parameter needed for custom1Ddistribution distribution')

    self.functionID = paramInput.findFirst('functionID').value
    if self.functionID == None:
      self.raiseAnError(IOError,' functionID parameter is needed for custom1Ddistribution distribution')

    self.variableID = paramInput.findFirst('variableID').value
    if self.variableID == None:
      self.raiseAnError(IOError,' variableID parameter is needed for custom1Ddistribution distribution')

    self.initializeDistribution()


  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """

    f = open(self.dataFilename, 'r')
    reader = csv.reader(f)
    headers = next(reader)
    indexFunctionID = headers.index(self.functionID)
    indexVariableID = headers.index(self.variableID)
    f.close()
    rawData = np.genfromtxt(self.dataFilename, delimiter="," , skip_header=1, usecols=(indexVariableID,indexFunctionID))

    self.data = rawData[rawData[:,0].argsort()]
    self.lowerBound = self.data[0,0]
    self.upperBound = self.data[-1,0]

    if self.functionType == 'cdf':
      self.cdfFunc = UnivariateSpline(self.data[:,0], self.data[:,1], k=self.k, s=self.s)
      self.pdfFunc = self.cdfFunc.derivative()
      self.invCDF  = UnivariateSpline(self.data[:,1], self.data[:,0], k=self.k, s=self.s)
    else:
      self.pdfFunc = UnivariateSpline(self.data[:,0], self.data[:,1], k=self.k, s=self.s)
      cdfValues = np.zeros(self.data[:,0].size)
      for i in range(self.data[:,0].size):
        cdfValues[i] = self.pdfFunc.integral(self.data[0][0],self.data[i,0])
      self.invCDF = UnivariateSpline(cdfValues, self.data[:,0] , k=self.k, s=self.s)

    # Note that self.invCDF is creating a new spline where I switch its term.
    # Instead of doing spline(x,f(x)) I am creating its inverse spline(f(x),x)
    # This can be done if f(x) is monothonic increasing with x (which is true for cdf)

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict['workingDir'] = self.workingDir
    paramDict['dataFilename'] = self.dataFilename
    paramDict['functionID'] = self.functionID
    paramDict['functionType'] = self.functionType
    paramDict['variableID'] = self.variableID
    paramDict['k'] = self.k
    paramDict['s'] = self.s
    return paramDict

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.workingDir = pdict.pop('workingDir')
    self.dataFilename = pdict.pop('dataFilename')
    self.functionID = pdict.pop('functionID')
    self.functionType = pdict.pop('functionType')
    self.variableID = pdict.pop('variableID')
    self.k = pdict.pop('k')
    self.s = pdict.pop('s')

  def pdf(self,x):
    """
      Function that calculates the pdf value of x
      @ In, x, scalar , coordinates to get the pdf at
      @ Out, pdfValue, scalar, requested pdf
    """
    pdfValue = self.pdfFunc(x)
    return pdfValue

  def cdf(self,x):
    """
      Function that calculates the cdf value of x
      @ In, x, scalar , coordinates to get the cdf at
      @ Out, pdfValue, scalar, requested pdf
    """
    if self.functionType == 'cdf':
      cdfValue = self.cdfFunc(x)
    else:
      cdfValue = self.pdfFunc.integral(self.data[0][0],x)
    return cdfValue

  def ppf(self,x):
    """
      Return the ppf of given coordinate
      @ In, x, float, the x coordinates
      @ Out, ppfValue, float, ppf values
    """
    ppfValue = self.invCDF(x)
    return ppfValue

  def rvs(self):
    """
      Return a random state of the custom1D distribution
      @ In, None
      @ Out, rvsValue, float/string, the random state
    """
    rvsValue = self.ppf(random())
    return rvsValue

DistributionsCollection.addSub(Custom1D.getInputSpecification())

class LogUniform(Distribution):
  """
    Log Uniform univariate distribution
    If x~LogUnif(a,b) then log(x)~Unif(log(a),log(b))
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(LogUniform, cls).getInputSpecification()
    inputSpecification.description = r"""Log-Uniform distribution. This distribution is associated
        to a variable $x'$ such that $x = e^{x'}$. Sometimes known as the Reciprocal distribution. The
        probability density function of this distribution is
        $f(x)=\frac{1}{x(\ln{b}-\ln{a})}$ for $a \le x \le b$."""

    BaseInputType = InputTypes.makeEnumType("base", "baseType", ["natural","decimal"])
    inputSpecification.addSub(InputData.parameterInputFactory("base",
        BaseInputType,
        descr=r"""exponent base for scaling the underlying uniform distribution, either "decimal" (base 10) or
        "natural" base ($e$)."""))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.base = None

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.minVal = min(math.exp(self.upperBound),math.exp(self.lowerBound))
    self.maxVal = max(math.exp(self.upperBound),math.exp(self.lowerBound))

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    self.base = paramInput.findFirst('base').value
    if self.base not in ['natural','decimal']:
      self.raiseAnError(IOError,' base parameter is needed for LogUniform distribution (either natural or decimal)')

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict['base'] = self.base
    return paramDict

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.base = pdict.pop('base')

  def pdf(self,x):
    """
      Function that calculates the pdf value of x
      @ In, x, float , coordinates to get the pdf at
      @ Out, pdfValue, float, requested pdf
    """
    if self.base == 'natural':
      pdfValue = 1./(self.upperBound-self.lowerBound) * 1./x
    else:
      pdfValue = 1./(self.upperBound-self.lowerBound) * 1./x * 1./math.log(10.)
    return pdfValue

  def cdf(self,x):
    """
      Function that calculates the cdf value of x
      @ In, x, float , coordinates to get the cdf at
      @ Out, pdfValue, float, requested pdf
    """
    if self.base == 'natural':
      cdfValue = (math.log(x)-self.lowerBound)/(self.upperBound-self.lowerBound)
    else:
      cdfValue = (math.log10(x)-self.lowerBound)/(self.upperBound-self.lowerBound)
    return cdfValue

  def ppf(self,x):
    """
      Return the ppf of given coordinate
      @ In, x, float, the x coordinates
      @ Out, ppfValue, float, ppf values
    """
    if self.base == 'natural':
      ppfValue = math.exp((self.upperBound-self.lowerBound)*x + self.lowerBound)
    else:
      ppfValue = 10.**((self.upperBound-self.lowerBound)*x + self.lowerBound)
    return ppfValue

  def rvs(self):
    """
      Return a random value
      @ In, None
      @ Out, rvsValue, float, the random value
    """
    rvsValue = self.ppf(random())
    return rvsValue

DistributionsCollection.addSub(LogUniform.getInputSpecification())

class NDimensionalDistributions(Distribution):
  """
    General base class for NDimensional distributions
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(NDimensionalDistributions, cls).getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("workingDir",
        descr=r"""relative working directory that contains data files""",
        contentType=InputTypes.StringType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.dataFilename = None
    self.functionType = None
    self.type = 'NDimensionalDistributions'
    self.dimensionality  = None

    self.RNGInitDisc = 5
    self.RNGtolerance = 0.2

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    workingDir = paramInput.findFirst('workingDir')
    if workingDir != None:
      self.workingDir = workingDir.value
    else:
      self.workingDir = os.getcwd()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict['functionType'] = self.functionType
    paramDict['dataFilename'] = self.dataFilename
    paramDict['workingDir'] = self.workingDir
    return paramDict

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    self.functionType = pdict.pop('functionType')
    self.dataFilename = pdict.pop('dataFilename')
    self.workingDir = pdict.pop('workingDir')

  #######
  def updateRNGParam(self, dictParam):
    """
      Updated parameters of RNG
      @ In, dictParam, dict, dictionary of initialization parameters
      @ Out, None
    """
    for key in dictParam:
      if key == 'tolerance':
        self.RNGtolerance = dictParam['tolerance']
      elif key == 'initialGridDisc':
        self.RNGInitDisc  = dictParam['initialGridDisc']
    self._distribution.updateRNGparameter(self.RNGtolerance,self.RNGInitDisc)
  ######

  def getDimensionality(self):
    """
      Function return the dimensionality of the distribution
      @ In, None
      @ Out, dimensionality, int, the dimensionality of the distribution
    """
    dimensionality = self._distribution.returnDimensionality()
    return dimensionality

  def returnLowerBound(self, dimension):
    """
      Function that return the lower bound of the distribution for a particular dimension
      @ In, dimension, int, dimension considered
      @ Out, value, float, lower bound of the distribution
    """
    value = self._distribution.returnLowerBound(dimension)
    return value

  def returnUpperBound(self, dimension):
    """
      Function that return the upper bound of the distribution for a particular dimension
      @ In, dimension, int, dimension considered
      @ Out, value, float, upper bound of the distribution
    """
    value = self._distribution.returnUpperBound(dimension)
    return value

  def marginalDistribution(self, x, variable):
    """
      Compute the cdf marginal distribution
      @ In, x, float, the coordinate for at which the inverse marginal distribution needs to be computed
      @ In, variable, int, the variable id dimension coordinate (e.g. 0 => 1st coordinate, 1 => 2nd coordinate)
      @ Out, marginalDistribution, float, the marginal cdf value at coordinate x
    """
    return self._distribution.marginal(x, variable)

DistributionsCollection.addSub(NDimensionalDistributions.getInputSpecification())

class NDInverseWeight(NDimensionalDistributions):
  """
    NDInverseWeight multi-variate distribution (inverse weight interpolation)
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(NDInverseWeight, cls).getInputSpecification()
    inputSpecification.description = r"""Custom user-defined N-dimensional distribution. This allows empirically-defined
        custom distributions that are not currently defined in RAVEN to be used for probability weighting and sampling.
        The combined distribution is defined through empirical distribution in a CSV file."""
    DataFilenameParameterInput = InputData.parameterInputFactory("dataFilename",
        descr=r"""name of file to be used to initialize the distribution""",
        contentType=InputTypes.StringType)
    DataFilenameParameterInput.addParam("type",
        InputTypes.StringType,
        True,
        descr=r"""type of initialization values specifid in the input file (PDF or CDF)""")
    inputSpecification.addSub(DataFilenameParameterInput)

    inputSpecification.addSub(InputData.parameterInputFactory("p",
        descr=r"""power parameter. Greater values of p assign greater influence to values closest to
          interpolation points.""",
        contentType=InputTypes.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.p = None
    self.type = 'NDInverseWeight'

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    pFind = paramInput.findFirst('p')
    if pFind != None:
      self.p = pFind.value
    else:
      self.raiseAnError(IOError,'Minkowski distance parameter <p> not found in NDInverseWeight distribution')

    dataFilename = paramInput.findFirst('dataFilename')
    if dataFilename != None:
      self.dataFilename = os.path.join(self.workingDir,dataFilename.value)
    else:
      self.raiseAnError(IOError,'<dataFilename> parameter needed for MultiDimensional Distributions!!!!')

    functionType = dataFilename.parameterValues['type']
    if functionType != None:
      self.functionType = functionType
    else:
      self.raiseAnError(IOError,'<functionType> parameter needed for MultiDimensional Distributions!!!!')

    self.initializeDistribution()

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict['p'] = self.p
    return paramDict

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    super()._localSetState(pdict)
    self.p = pdict.pop('p')

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.functionType == 'CDF':
      self._distribution = CrowDistribution1D.BasicMultiDimensionalInverseWeight(str(self.dataFilename), self.p,True)
    else:
      self._distribution = CrowDistribution1D.BasicMultiDimensionalInverseWeight(str(self.dataFilename), self.p,False)
    self.dimensionality = self._distribution.returnDimensionality()
    self.lowerBound = [self.returnLowerBound(dim) for dim in range(self.dimensionality)]
    self.upperBound = [self.returnUpperBound(dim) for dim in range(self.dimensionality)]

  def cdf(self,x):
    """
      calculate the cdf value for given coordinate x
      @ In, x, list, list of variable coordinate
      @ Out, cdfValue, float, cdf value
    """
    cdfValue = self._distribution.cdf(numpyToCxxVector(x))
    return cdfValue

  def ppf(self,x):
    """
      Return the ppf of given coordinate
      @ In, x, np.array, the x coordinates
      @ Out, ppfValue, np.array, ppf values
    """
    ppfValue = self._distribution.inverseCdf(x,random())
    return ppfValue

  def pdf(self,x):
    """
      Function that calculates the pdf value of x
      @ In, x, np.array , coordinates to get the pdf at
      @ Out, pdfValue, np.array, requested pdf
    """
    pdfValue = self._distribution.pdf(numpyToCxxVector(x))
    return pdfValue

  def cellIntegral(self,x,dx):
    """
      Compute the integral of N-D cell in this distribution
      @ In, x, np.array, x coordinates
      @ In, dx, np.array, discretization passes
      @ Out, integralReturn, float, the integral
    """
    coordinate = numpyToCxxVector(x)
    dxs = numpyToCxxVector(dx)
    integralReturn = self._distribution.cellIntegral(coordinate,dxs)
    return integralReturn

  def inverseMarginalDistribution(self, x, variable):
    """
      Compute the inverse of the Marginal distribution
      @ In, x, float, the cdf for at which the inverse marginal distribution needs to be computed
      @ In, variable, int, the variable id dimension coordinate (e.g. 0 => 1st coordinate, 1 => 2nd coordinate)
      @ Out, inverseMarginal, float, the marginal inverse cdf value at coordinate x
    """
    if (x>=0.0) and (x<=1.0):
      inverseMarginal = self._distribution.inverseMarginal(min(1.-sys.float_info.epsilon,
                                                           max(sys.float_info.epsilon,x)),
                                                           variable)
    else:
      self.raiseAnError(ValueError,'NDInverseWeight: inverseMarginalDistribution(x) with x outside [0.0,1.0]')
    return inverseMarginal

  def untruncatedCdfComplement(self, x):
    """
      Function to get the untruncated  cdf complement at a provided coordinate
      @ In, x, float, value to get the untruncated  cdf complement  at
      @ Out, float, requested untruncated  cdf complement
    """
    self.raiseAnError(NotImplementedError,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    """
      Function to get the untruncated  Hazard  at a provided coordinate
      @ In, x, float, value to get the untruncated  Hazard   at
      @ Out, float, requested untruncated  Hazard
    """
    self.raiseAnError(NotImplementedError,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    """
      Function to get the untruncated  Mean
      @ In, None
      @ Out, float, requested Mean
    """
    self.raiseAnError(NotImplementedError,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    """
      Function to get the untruncated  Median
      @ In, None
      @ Out, float, requested Median
    """
    self.raiseAnError(NotImplementedError,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    """
      Function to get the untruncated  Mode
      @ In, None
      @ Out, untrMode, float, requested Mode
    """
    self.raiseAnError(NotImplementedError,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    """
      Return the random coordinate
      @ In, args, dict, arguments (for future usage)
      @ Out, rvsValue, np.array, the random coordinate
    """
    rvsValue = self._distribution.inverseCdf(random(),random())
    return rvsValue

DistributionsCollection.addSub(NDInverseWeight.getInputSpecification())

class NDCartesianSpline(NDimensionalDistributions):
  """
    NDCartesianSpline multi-variate distribution (cubic spline interpolation)
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(NDCartesianSpline, cls).getInputSpecification()
    inputSpecification.description = r"""describes a N-dimensional distribution given a set of points regularly
        distributed on a Cartesian grid. These points sample the PDF of the original distribution. Distributed
        values (PDF or CDF) are calculated using the ND Spline interpolation scheme."""

    DataFilenameParameterInput = InputData.parameterInputFactory("dataFilename",
        descr=r"""name of file to be used to initialize the distribution""",
        contentType=InputTypes.StringType)
    DataFilenameParameterInput.addParam("type",
        InputTypes.StringType,
        True,
        descr=r"""type of initialization values specifid in the input file (PDF or CDF)""")
    inputSpecification.addSub(DataFilenameParameterInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type = 'NDCartesianSpline'

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    dataFilename = paramInput.findFirst('dataFilename')
    if dataFilename != None:
      self.dataFilename = os.path.join(self.workingDir,dataFilename.value)
    else:
      self.raiseAnError(IOError,'<dataFilename> parameter needed for MultiDimensional Distributions!!!!')

    functionType = dataFilename.parameterValues['type']
    if functionType != None:
      self.functionType = functionType
    else:
      self.raiseAnError(IOError,'<functionType> parameter needed for MultiDimensional Distributions!!!!')

    self.initializeDistribution()

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.raiseAMessage('initialize Distribution')
    if self.functionType == 'CDF':
      self._distribution = CrowDistribution1D.BasicMultiDimensionalCartesianSpline(str(self.dataFilename),True)
    else:
      self._distribution = CrowDistribution1D.BasicMultiDimensionalCartesianSpline(str(self.dataFilename),False)
    self.dimensionality = self._distribution.returnDimensionality()
    self.lowerBound = [self.returnLowerBound(dim) for dim in range(self.dimensionality)]
    self.upperBound = [self.returnUpperBound(dim) for dim in range(self.dimensionality)]

  def cdf(self,x):
    """
      calculate the cdf value for given coordinate x
      @ In, x, list, list of variable coordinate
      @ Out, cdfValue, float, cdf value
    """
    cdfValue = self._distribution.cdf(numpyToCxxVector(x))
    return cdfValue

  def ppf(self,x):
    """
      Return the ppf of given coordinate
      @ In, x, np.array, the x coordinates
      @ Out, ppfValue, np.array, ppf values
    """
    ppfValue = self._distribution.inverseCdf(x,random())
    return ppfValue

  def pdf(self,x):
    """
      Function that calculates the pdf value of x
      @ In, x, np.array , coordinates to get the pdf at
      @ Out, pdfValue, np.array, requested pdf
    """
    pdfValue = self._distribution.pdf(numpyToCxxVector(x))
    return pdfValue

  def cellIntegral(self,x,dx):
    """
      Compute the integral of N-D cell in this distribution
      @ In, x, np.array, x coordinates
      @ In, dx, np.array, discretization passes
      @ Out, integralReturn, float, the integral
    """
    coordinate = numpyToCxxVector(x)
    dxs = numpyToCxxVector(dx)
    integralReturn = self._distribution.cellIntegral(coordinate,dxs)
    return integralReturn

  def inverseMarginalDistribution(self, x, variable):
    """
      Compute the inverse of the Margina distribution
      @ In, x, float, the coordinate for at which the inverse marginal distribution needs to be computed
      @ In, variable, int, the variable id dimension coordinate (e.g. 0 => 1st coordinate, 1 => 2nd coordinate)
      @ Out, inverseMarginal, float, the marginal cdf value at coordinate x
    """
    if (x>=0.0) and (x<=1.0):
      inverseMarginal = self._distribution.inverseMarginal(min(1.-sys.float_info.epsilon,
                                                           max(sys.float_info.epsilon,x)),
                                                           variable)
    else:
      self.raiseAnError(ValueError,'NDCartesianSpline: inverseMarginalDistribution(x) with x ' +str(x)+' outside [0.0,1.0]')
    return inverseMarginal

  def untruncatedCdfComplement(self, x):
    """
      Function to get the untruncated  cdf complement at a provided coordinate
      @ In, x, float, value to get the untruncated  cdf complement  at
      @ Out, float, requested untruncated  cdf complement
    """
    self.raiseAnError(NotImplementedError,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    """
      Function to get the untruncated  Hazard  at a provided coordinate
      @ In, x, float, value to get the untruncated  Hazard   at
      @ Out, float, requested untruncated  Hazard
    """
    self.raiseAnError(NotImplementedError,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    """
      Function to get the untruncated  Mean
      @ In, None
      @ Out, float, requested Mean
    """
    self.raiseAnError(NotImplementedError,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    """
      Function to get the untruncated  Median
      @ In, None
      @ Out, float, requested Median
    """
    self.raiseAnError(NotImplementedError,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    """
      Function to get the untruncated  Mode
      @ In, None
      @ Out, untrMode, float, requested Mode
    """
    self.raiseAnError(NotImplementedError,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    """
      Return the random coordinate
      @ In, args, dict, arguments (for future usage)
      @ Out, rvsValue, np.array, the random coordinate
    """
    rvsValue = self._distribution.inverseCdf(random(),random())
    return rvsValue

DistributionsCollection.addSub(NDCartesianSpline.getInputSpecification())

class MultivariateNormal(NDimensionalDistributions):
  """
    MultivariateNormal multi-variate distribution (analytic)
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Overridden method to get a reference to the class that specifies the input data for
      the MultivariateNormal class, since it must add its own custom parameter.
      @ In, None
      @ Out, inputSpecification, InputData.ParameterInput, class to use for input data.
    """
    inputSpecification = super(MultivariateNormal, cls).getInputSpecification()
    inputSpecification.description = r"""describes a N-dimensional multivariate Gaussian normal distribution.
        This generalizes the univariate normal distribution to higher dimensions. The multivariate normal
        distribution is defined by an N-dimensional random vector $\widehat{x}$ and is defined by a multidimensional
        mean $\widehat{\mu}$ and standard deviation $\Sigma$. The probability density function for this distribution is
        $f(\widehat{x})=\frac{1}{\sqrt{(2\pi)^k}\left|\Sigma\right| } e^{-\frac{1}{2}(\widehat{x}-\widehat{\mu})^T \Sigma^{-1}(\widehat{x}-\widehat{\mu})}$."""

    MuListParameterInput = InputData.parameterInputFactory("mu",
        descr=r"""list of mean values for each dimension of the distribution""",
        contentType=InputTypes.StringType)

    CovarianceListParameterInput = InputData.parameterInputFactory("covariance",
        descr=r"""list of covariance values in the covariance matrix. These are specified based on the \xmlAttr{type} parameter""",
        contentType=InputTypes.StringType)
    CovarianceListParameterInput.addParam("type",
        InputTypes.StringType,
        False,
        descr=r"""type of covariance. \xmlString{abs} indicates a normal covariance matrix, while \xmlString{rel}
        indicates a relative covarience matrics.
        For \xmlNode{transformation}, \xmlString{pca} can be combined with both types, and
        \xmlString{spline} only accepts \xmlString{abs}\default{abs}""")

    TransformationParameterInput = InputData.parameterInputFactory("transformation",
        descr=r"""enables input parameter transformation using principle component analysis (PCA). If enabled,
        PCA is used on the input covariance matrix with truncation to the specified rank.""")
    RankParameterInput = InputData.parameterInputFactory("rank",
        descr=r"""desired dimensionality reduction for covariance matrix""",
        contentType=InputTypes.IntegerType)
    TransformationParameterInput.addSub(RankParameterInput)

    inputSpecification.addSub(MuListParameterInput)
    inputSpecification.addSub(CovarianceListParameterInput)
    inputSpecification.addSub(TransformationParameterInput)

    MultivariateMethodType = InputTypes.makeEnumType("multivariateMethod","multivariateMethodType",["pca","spline"])
    inputSpecification.addParam("method",
        MultivariateMethodType,
        True)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type = 'MultivariateNormal'
    self.distType = distType.continuous
    self.mu  = None
    self.covariance = None
    self.covarianceType = 'abs'  # abs: absolute covariance, rel: relative covariance matrix
    self.method = 'pca'          # pca: using pca method to compute the pdf, and inverseCdf, another option is 'spline', i.e. using
                                 # cartesian spline method to compute the pdf, cdf, inverseCdf, ...
    self.transformMatrix = None  # np.array stores the transform matrix
    self.dimension = None        # the dimension of given problem
    self.rank = None             # the effective rank for the PCA analysis
    self.transformation = False       # flag for input reduction analysis

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    if paramInput.parameterValues['method'] == 'pca':
      self.method = 'pca'
    elif paramInput.parameterValues['method'] == 'spline':
      self.method = 'spline'
    else:
      self.raiseAnError(IOError,'The method attribute for the MultivariateNormal Distribution is not correct, choose "pca" or "spline"')
    for child in paramInput.subparts:
      if child.getName() == 'mu':
        mu = [float(value) for value in child.value.split()]
        self.dimension = len(mu)
      elif child.getName() == 'covariance':
        covariance = [float(value) for value in child.value.split()]
        if 'type' in child.parameterValues:
          self.covarianceType = child.parameterValues['type']
      elif child.getName() == 'transformation':
        self.transformation = True
        for childChild in child.subparts:
          if childChild.getName() == 'rank':
            self.rank = childChild.value

    if self.rank == None:
      self.rank = self.dimension
    self.mu = mu
    self.covariance = covariance
    #check square covariance
    rt = np.sqrt(len(self.covariance))
    covDim = int(rt)
    if covDim != rt:
      self.raiseAnError(IOError,'Covariance matrix is not square!  Contains %i entries.' %len(self.covariance))
    #sanity check on dimensionality
    if covDim != len(self.mu):
      self.raiseAnError(IOError,'Invalid dimensions! Covariance has %i entries (%i x %i), but mu has %i entries!' %(len(self.covariance),covDim,covDim,len(self.mu)))
    self.initializeDistribution()

  def _localSetState(self,pdict):
    """
      Set the pickling state (local)
      @ In, pdict, dict, the namespace state
      @ Out, None
    """
    super()._localSetState(pdict)
    self.method = pdict.pop('method')
    self.dimension = pdict.pop('dimension')
    self.rank = pdict.pop('rank')
    self.mu = pdict.pop('mu')
    self.covariance = pdict.pop('covariance')

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict['method'] = self.method
    paramDict['dimension'] = self.dimension
    paramDict['rank'] = self.rank
    paramDict['mu'] = self.mu
    paramDict['covariance'] = self.covariance
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.raiseAMessage('initialize distribution')
    if self.method == 'spline':
      if self.covarianceType != 'abs':
        self.raiseAnError(IOError,'covariance with type ' + self.covariance + ' is not implemented for ' + self.method + ' method')
      mu = numpyToCxxVector(self.mu)
      covariance = numpyToCxxVector(self.covariance)
      self._distribution = CrowDistribution1D.BasicMultivariateNormal(covariance, mu)
    elif self.method == 'pca':
      self._distribution = DistributionsND.MultivariateNormalPCA(self.covariance, self.mu, self.covarianceType, self.rank)
    if self.transformation:
      self.lowerBound = [-sys.float_info.max]*self.rank
      self.upperBound = [sys.float_info.max]*self.rank
    else:
      self.lowerBound = [self.returnLowerBound(dim) for dim in range(self.dimension)]
      self.upperBound = [self.returnUpperBound(dim) for dim in range(self.dimension)]

  def cdf(self,x):
    """
      calculate the cdf value for given coordinate x
      @ In, x, List, list of variable coordinate
      @ Out, cdfValue, float, cdf value
    """
    if self.method == 'spline':
      cdfValue = self._distribution.cdf(numpyToCxxVector(x))
    else:
      self.raiseAnError(NotImplementedError,'cdf not yet implemented for ' + self.method + ' method')
    return cdfValue

  def transformationMatrix(self,index=None):
    """
      Return the transformation matrix from Crow
      @ In, None
      @ In, index, list, optional, input coordinate index, list values for the index of the latent variables
      @ Out, L, np.array, the transformation matrix
    """
    if self.method == 'pca':
      L = self._distribution.getTransformationMatrix(index)
    else:
      self.raiseAnError(NotImplementedError,' transformationMatrix is not yet implemented for ' + self.method + ' method')
    return L

  def inverseTransformationMatrix(self,index=None):
    """
      Return the inverse transformation matrix from Crow
      @ In, None
      @ In, index, list, optional, input coordinate index, list values for the index of the original variables
      @ Out, L, np.array, the inverse transformation matrix
    """
    if self.method == 'pca':
      L = self._distribution.getInverseTransformationMatrix(index)
    else:
      self.raiseAnError(NotImplementedError,' inverse transformationMatrix is not yet implemented for ' + self.method + ' method')
    return L

  def returnSingularValues(self,index=None):
    """
      Return the singular values from Crow
      @ In, None
      @ In, index, list, optional, input coordinate index, list values for the index of the input variables
      @ Out, singularValues, np.array, the singular values vector
    """
    if self.method == 'pca':
      singularValues = self._distribution.getSingularValues(index)
    else:
      self.raiseAnError(NotImplementedError,' returnSingularValues is not available for ' + self.method + ' method')
    return singularValues

  def pcaInverseTransform(self,x,index=None):
    """
      Transform latent parameters back to models' parameters
      @ In, x, list, input coordinate, list values for the latent variables
      @ In, index, list, optional, input coordinate index, list values for the index of the latent variables
      @ Out, values, list, return the values of manifest variables with type of list
    """
    if self.method == 'pca':
      if len(x) > self.rank:
        self.raiseAnError(IOError,'The dimension of the latent variables defined in <Samples> is larger than the rank defined in <Distributions>')
      values = self._distribution.coordinateInverseTransformed(x, index)
    else:
      self.raiseAnError(NotImplementedError,'ppfTransformedSpace not yet implemented for ' + self.method + ' method')
    return values

  def ppf(self,x):
    """
      Return the ppf of given coordinate
      @ In, x, np.array, the x coordinates
      @ Out, ppfValue, np.array, ppf values
    """
    if self.method == 'spline':
      ppfValue = self._distribution.inverseCdf(numpyToCxxVector(x), random())
    else:
      self.raiseAnError(NotImplementedError,'ppf is not yet implemented for ' + self.method + ' method')
    return ppfValue

  def pdf(self,x):
    """
      Return the pdf of given coordinate
      @ In, x, np.array, the x coordinates
      @ Out, pdfValue, np.array, pdf values
    """
    if self.transformation:
      pdfValue = self.pdfInTransformedSpace(x)
    elif self.method == 'pca':
      pdfValue = self._distribution.pdf(x)
    else:
      pdfValue = self._distribution.pdf(numpyToCxxVector(x))
    return pdfValue

  def logPdf(self,x):
    """
      Function to get the log pdf at a provided coordinate
      @ In, x, np.array, the x coordinates
      @ Out, logPdf, np.array, requested log pdf
    """
    logPdf = np.log(self.pdf(x))
    return logPdf

  def pdfInTransformedSpace(self,x):
    """
      Return the pdf of given coordinate in the transformed space
      @ In, x, np.array, the x coordinates
      @ Out, pdfInTransformedSpace, np.array, pdf values in the transformed space
    """
    if self.method == 'pca':
      pdfInTransformedSpace = self._distribution.pdfInTransformedSpace(x)
    else:
      self.raiseAnError(NotImplementedError,'ppfTransformedSpace not yet implemented for ' + self.method + ' method')
    return pdfInTransformedSpace

  def cellIntegral(self,x,dx):
    """
      Compute the integral of N-D cell in this distribution
      @ In, x, np.array, x coordinates
      @ In, dx, np.array, discretization passes
      @ Out, integralReturn, float, the integral
    """
    if self.method == 'pca':
      if self.transformation:
        self.raiseAWarning("The ProbabilityWeighted is computed on the reduced transformed space")
      else:
        self.raiseAWarning("The ProbabilityWeighted is computed on the full transformed space")
      integralReturn = self._distribution.cellProbabilityWeight(x, dx)
    elif self.method == 'spline':
      coordinate = numpyToCxxVector(x)
      dxs = numpyToCxxVector(dx)
      integralReturn = self._distribution.cellIntegral(coordinate,dxs)
    else:
      self.raiseAnError(NotImplementedError,'cellIntegral not yet implemented for ' + self.method + ' method')
    return integralReturn

  def marginalCdf(self, x):
    """
      Calculate the marginal distribution for given coordinate x
      @ In, x, float, the coordinate for given marginal distribution
      @ Out, marginalCdfForPCA, float, the marginal cdf value at coordinate x
    """
    if self.method == 'pca':
      marginalCdfForPCA = self._distribution.marginalCdfForPCA(x)
    else:
      self.raiseAnError(NotImplementedError,'marginalCdf  not yet implemented for ' + self.method + ' method')
    return marginalCdfForPCA

  def inverseMarginalDistribution(self, x, variable):
    """
      Compute the inverse of the Margina distribution
      @ In, x, float, the coordinate for at which the inverse marginal distribution needs to be computed
      @ In, variable, int, the variable id dimension coordinate (e.g. 0 => 1st coordinate, 1 => 2nd coordinate)
      @ Out, inverseMarginal, float, the marginal cdf value at coordinate x
    """
    if (x >= 0.0) and (x <= 1.0):
      if self.method == 'pca':
        inverseMarginal = self._distribution.inverseMarginalForPCA(min(1.-sys.float_info.epsilon,
                                                                   max(sys.float_info.epsilon,
                                                                   x)))  # TODO can probably remove min/max and just use x
      elif self.method == 'spline':
        inverseMarginal=  self._distribution.inverseMarginal(min(1.-sys.float_info.epsilon,
                                                             max(sys.float_info.epsilon,x)),
                                                             variable)
    else:
      self.raiseAnError(ValueError,'NDInverseWeight: inverseMarginalDistribution(x) with x ' +str(x)+' outside [0.0,1.0]')
    return inverseMarginal

  def untruncatedCdfComplement(self, x):
    """
      Function to get the untruncated  cdf complement at a provided coordinate
      @ In, x, float, value to get the untruncated  cdf complement  at
      @ Out, float, requested untruncated  cdf complement
    """
    self.raiseAnError(NotImplementedError,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    """
      Function to get the untruncated  Hazard  at a provided coordinate
      @ In, x, float, value to get the untruncated  Hazard   at
      @ Out, float, requested untruncated  Hazard
    """
    self.raiseAnError(NotImplementedError,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self, x):
    """
      Function to get the untruncated  Mean
      @ In, x, float, the value
      @ Out, float, requested Mean
    """
    self.raiseAnError(NotImplementedError,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self, x):
    """
      Function to get the untruncated  Median
      @ In, x, float, the value
      @ Out, float, requested Median
    """
    self.raiseAnError(NotImplementedError,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self, x):
    """
      Function to get the untruncated  Mode
      @ In, x, float, the value
      @ Out, untrMode, float, requested Mode
    """
    self.raiseAnError(NotImplementedError,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self, *args):
    """
      Return the random coordinate
      @ In, args, dict, arguments (for future usage)
      @ Out, rvsValue, np.array, the random coordinate
    """
    if self.method == 'spline':
      rvsValue = self._distribution.inverseCdf(random(),random())
    # if no transformation, then return the coordinate for the original input parameters
    # if there is a transformation, then return the coordinate in the reduced space
    elif self.method == 'pca':
      rands = random(self.rank)
      # use marginal CDF (unit normal) to
      rands = self._distribution.inverseMarginalForPCA(rands)
      if self.transformation:
        rvsValue = rands
      else:
        rvsValue = self._distribution.coordinateInverseTransformed(rands)
    else:
      self.raiseAnError(NotImplementedError,'rvs is not yet implemented for ' + self.method + ' method')
    return rvsValue

def numpyToCxxVector(x):
  """
    Utility function for converting a numpy array into a C++ vector swig object.

    @ In, x, np.ndarray, the 1d numpy array to convert
    @ Out, xCxx, C++ vector, the converted vector
  """
  x = np.atleast_1d(x)
  if x.ndim > 1:
    raise ValueError('x must be 1d, not {}d'.format(x.ndim))

  xCxx = CrowDistribution1D.vectord_cxx(len(x))
  for i in range(len(x)):
    xCxx[i] = x[i]

  return xCxx

DistributionsCollection.addSub(MultivariateNormal.getInputSpecification())

factory = EntityFactory('Distribution', returnInputParameter=True)
factory.registerAllSubtypes(Distribution)
factory.unregisterSubtype('BoostDistribution')

def returnInputParameter():
  """
    Function returns the InputParameterClass that can be used to parse the
    whole collection.
    @ Out, DistributionsCollection, DistributionsCollection, class for parsing.
  """
  return DistributionsCollection()
