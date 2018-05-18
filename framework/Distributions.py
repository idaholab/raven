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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
#from __builtin__ import None
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
import scipy
from math import gamma
import os
import operator
from collections import OrderedDict
import csv
from scipy.interpolate import UnivariateSpline
import math as math
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils import utils
from utils.randomUtils import random
distribution1D = utils.findCrowModule('distribution1D')
from utils import InputData
from utils import mathUtils
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
                              'LogNormal':'LogNormalDistribution',
                              'Weibull':'WeibullDistribution',
                              'NDInverseWeight': 'NDInverseWeightDistribution',
                              'NDCartesianSpline': 'NDCartesianSplineDistribution',
                              'MultivariateNormal' : 'MultivariateNormalDistribution',
                              'Laplace' : 'LaplaceDistribution',
                              'Geometric' : 'GeometricDistribution',
                              'LogUniform' : 'LogUniformDistribution',
}

class DistributionsCollection(InputData.ParameterInput):
  """
    Class for reading in a collection of distributions
  """

DistributionsCollection.createClass("Distributions")


class Distribution(BaseType):
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
    inputSpecification = super(Distribution, cls).getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory('upperBound', contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory('lowerBound', contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)
    self.upperBoundUsed       = False  # True if the distribution is right truncated
    self.lowerBoundUsed       = False  # True if the distribution is left truncated
    self.hasInfiniteBound     = False  # True if the untruncated distribution has bounds of +- system max
    self.upperBound           = 0.0  # Right bound
    self.lowerBound           = 0.0  # Left bound
    self.__adjustmentType     = '' # this describe how the re-normalization to preserve the probability should be done for truncated distributions
    self.dimensionality       = None # Dimensionality of the distribution (1D or ND)
    self.disttype             = None # distribution type (continuous or discrete)
    self.printTag             = 'DISTRIBUTIONS'
    self.preferredPolynomials = None  # best polynomial for probability-weighted norm of error
    self.preferredQuadrature  = None  # best quadrature for probability-weighted norm of error
    self.compatibleQuadrature = [] #list of compatible quadratures
    self.convertToDistrDict   = {} #dict of methods keyed on quadrature types to convert points from quadrature measure and domain to distribution measure and domain
    self.convertToQuadDict    = {} #dict of methods keyed on quadrature types to convert points from distribution measure and domain to quadrature measure and domain
    self.measureNormDict     = {} #dict of methods keyed on quadrature types to provide scalar adjustment for measure transformation (from quad to distr)
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
    self.messageHandler   = pdict.pop('messageHandler'  )
    self._localSetState(pdict)
    self.initializeDistribution()

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
    paramDict['messageHandler'  ] = self.messageHandler
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

  def getDisttype(self):
    """
      Function return distribution type
      @ In, None
      @ Out, disttype, string,  ('Continuous' or 'Discrete')
    """
    return self.disttype

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
    Distribution.__init__(self)
    self.dimensionality  = 1
    self.disttype        = 'Continuous'

  def cdf(self,x):
    """
      Function to get the cdf at a provided coordinate
      @ In, x, float, value to get the cdf at
      @ Out, retunrCdf, float, requested cdf
    """
    retunrCdf = self._distribution.cdf(x)
    return retunrCdf

  def ppf(self,x):
    """
      Function to get the inverse cdf at a provided coordinate
      @ In, x, float, value to get the inverse cdf at
      @ Out, retunrPpf, float, requested inverse cdf
    """
    retunrPpf = self._distribution.inverseCdf(x)
    return retunrPpf

  def pdf(self,x):
    """
      Function to get the pdf at a provided coordinate
      @ In, x, float, value to get the pdf at
      @ Out, returnPdf, float, requested pdf
    """
    returnPdf = self._distribution.pdf(x)
    return returnPdf

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


  def rvs(self,*args):
    """
      Function to get random numbers
      @ In, args, dict, args
      @ Out, rvsValue, float or list, requested random number or numbers
    """
    if len(args) == 0:
      rvsValue = self.ppf(random())
    else:
      rvsValue = [self.rvs() for _ in range(args[0])]
    return rvsValue

class Uniform(BoostDistribution):
  """
    Uniform univariate distribution
  """

  def __init__(self, lowerBound = None, upperBound = None):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.range = 0.0
    self.type = 'Uniform'
    self.disttype = 'Continuous'
    self.compatibleQuadrature.append('Legendre')
    self.compatibleQuadrature.append('ClenshawCurtis')
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature = 'Legendre'
    self.preferredPolynomials = 'Legendre'
    if upperBound is not None:
      self.upperBound = upperBound
      self.upperBoundUsed = True
      print("upperBound", self.upperBound)
    if lowerBound is not None:
      self.lowerBound = lowerBound
      self.lowerBoundUsed = True
      print("lowerBound", self.lowerBound)
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
    BoostDistribution._handleInput(self, paramInput)
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
    self._distribution = distribution1D.BasicUniformDistribution(self.lowerBound,self.lowerBound+self.range)

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
    inputSpecification.addSub(InputData.parameterInputFactory("mean", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("sigma", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self, mean=0.0, sigma=1.0):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.mean  = mean
    self.sigma = sigma
    self.hasInfiniteBound = True
    self.type = 'Normal'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicNormalDistribution(self.mean,
                                                                  self.sigma)
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
      self._distribution = distribution1D.BasicNormalDistribution(self.mean,
                                                                  self.sigma,
                                                                  a,b)

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
    inputSpecification.addSub(InputData.parameterInputFactory("low", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.low = 0.0
    self.alpha = 0.0
    self.beta = 1.0
    self.type = 'Gamma'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicGammaDistribution(self.alpha,1.0/self.beta,self.low)
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
      self._distribution = distribution1D.BasicGammaDistribution(self.alpha,1.0/self.beta,self.low,a,b)

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
    #return self.beta**self.alpha/factorial(self.alpha-1.)
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
    inputSpecification.addSub(InputData.parameterInputFactory("low", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("high", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("peakFactor", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.low = 0.0
    self.high = 1.0
    self.alpha = 0.0
    self.beta = 0.0
    self.type = 'Beta'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.high-self.low,self.low)
    else:
      if self.lowerBoundUsed == False:
        a = 0.0
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
      else:
        b = self.upperBound
      self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.high-self.low,a,b,self.low)
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
    inputSpecification.addSub(InputData.parameterInputFactory("apex", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("min", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("max", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.apex = 0.0
    self.min  = 0.0
    self.max  = 0.0
    self.type = 'Triangular'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
    apexFind = paramInput.findFirst('apex')
    if apexFind != None:
      self.apex = apexFind.value
    else:
      self.raiseAnError(IOError,'apex value needed for normal distribution')
    minFind = paramInput.findFirst('min')
    if minFind != None:
      self.min = minFind.value
    else:
      self.raiseAnError(IOError,'min value needed for normal distribution')
    maxFind = paramInput.findFirst('max')
    if maxFind != None:
      self.max = maxFind.value
    else:
      self.raiseAnError(IOError,'max value needed for normal distribution')
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
      self._distribution = distribution1D.BasicTriangularDistribution(self.apex,self.min,self.max)
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
    inputSpecification.addSub(InputData.parameterInputFactory("mu", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.mu  = 0.0
    self.type = 'Poisson'
    self.hasInfiniteBound = True
    self.disttype = 'Discrete'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicPoissonDistribution(self.mu)
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
    inputSpecification.addSub(InputData.parameterInputFactory("n", contentType=InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("p", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.n       = 0.0
    self.p       = 0.0
    self.type     = 'Binomial'
    self.hasInfiniteBound = True
    self.disttype = 'Discrete'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicBinomialDistribution(self.n,self.p)
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
    inputSpecification.addSub(InputData.parameterInputFactory("p", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.p        = 0.0
    self.type     = 'Bernoulli'
    self.disttype = 'Discrete'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicBernoulliDistribution(self.p)
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
    inputSpecification.addSub(InputData.parameterInputFactory("p", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.p        = 0.0
    self.type     = 'Geometric'
    self.disttype = 'Discrete'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicGeometricDistribution(self.p)
    else:  self.raiseAnError(IOError,'Truncated Geometric not yet implemented')

DistributionsCollection.addSub(Geometric.getInputSpecification())

class Categorical(Distribution):
  """
    Class for the categorical distribution also called " generalized Bernoulli distribution"
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

    StatePartInput = InputData.parameterInputFactory("state", contentType=InputData.FloatType)
    StatePartInput.addParam("outcome", InputData.FloatType, True)
    inputSpecification.addSub(StatePartInput, InputData.Quantity.one_to_infinity)

    ## Because we do not inherit from the base class, we need to manually
    ## add the name back in.
    inputSpecification.addParam("name", InputData.StringType, True)

    return inputSpecification

  def __init__(self):
    """
      Function that initializes the categorical distribution
      @ In, None
      @ Out, none
    """
    Distribution.__init__(self)
    self.mapping        = {}
    self.values         = set()
    self.type           = 'Categorical'
    self.dimensionality = 1
    self.disttype       = 'Discrete'

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == "state":
        outcome = child.parameterValues["outcome"]
        value = child.value
        self.mapping[outcome] = value
        if float(outcome) in self.values:
          self.raiseAnError(IOError,'Categorical distribution has identical outcomes')
        else:
          self.values.add(float(outcome))
      else:
        self.raiseAnError(IOError,'Invalid xml node for Categorical distribution; only "state" is allowed')
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
    paramDict['mapping'] = self.mapping
    paramDict['values'] = self.values
    return paramDict

  def initializeDistribution(self):
    """
      Function that initializes the distribution and checks that the sum of all state probabilities is equal to 1
      @ In, None
      @ Out, None
    """
    totPsum = 0.0
    for element in self.mapping:
      totPsum += self.mapping[element]
    if not mathUtils.compareFloats(totPsum,1.0):
      self.raiseAnError(IOError,'Categorical distribution cannot be initialized: sum of probabilities is '+repr(totPsum)+', not 1.0')

  def pdf(self,x):
    """
      Function that calculates the pdf value of x
      @ In, x, float/string, value to get the pdf at
      @ Out, pdfValue, float, requested pdf
    """
    if x in self.values:
      pdfValue =  self.mapping[x]
    else:
      self.raiseAnError(IOError,'Categorical distribution cannot calculate pdf for ' + str(x))
    return pdfValue

  def cdf(self,x):
    """
      Function to get the cdf value of x
      @ In, x, float/string, value to get the cdf at
      @ Out, cumulative, float, requested cdf
    """
    sortedMapping = sorted(self.mapping.items(), key=operator.itemgetter(0))
    if x in self.values:
      cumulative=0.0
      for element in sortedMapping:
        cumulative += element[1]
        if x == float(element[0]):
          return cumulative
    else:
      self.raiseAnError(IOError,'Categorical distribution cannot calculate cdf for ' + str(x))

  def ppf(self,x):
    """
      Function that calculates the inverse of the cdf given 0 =< x =< 1
      @ In, x, float, value to get the ppf at
      @ Out, element[0], float/string, requested inverse cdf
    """
    sortedMapping = sorted(self.mapping.items(), key=operator.itemgetter(0))
    cumulative=0.0
    for element in sortedMapping:
      cumulative += element[1]
      if cumulative >= x:
        return float(element[0])

  def rvs(self):
    """
      Return a random state of the categorical distribution
      @ In, None
      @ Out, rvsValue, float/string, the random state
    """
    rvsValue = self.ppf(random())
    return rvsValue

DistributionsCollection.addSub(Categorical.getInputSpecification())

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
    inputSpecification.addSub(InputData.parameterInputFactory("location", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("scale", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.location  = 0.0
    self.scale = 1.0
    self.type = 'Logistic'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicLogisticDistribution(self.location,self.scale)
    else:
      if self.lowerBoundUsed == False:
        a = -sys.float_info.max
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
      else:
        b = self.upperBound
      self._distribution = distribution1D.BasicLogisticDistribution(self.location,self.scale,a,b)

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
    inputSpecification.addSub(InputData.parameterInputFactory("location", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("scale", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.location  = 0.0
    self.scale = 1.0
    self.type = 'Laplace'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
      a = -sys.float_info.max
    else:
      a = self.lowerBound
    if self.upperBoundUsed == False:
      b = sys.float_info.max
    else:
      b = self.upperBound
    self._distribution = distribution1D.BasicLaplaceDistribution(self.location,self.scale,a,b)

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
    inputSpecification.addSub(InputData.parameterInputFactory("low", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.lambdaVar = 1.0
    self.low        = 0.0
    self.type = 'Exponential'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
    paramDict['lambda'] = self.lambdaVar
    paramDict['low'   ] = self.low
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False):
      self._distribution = distribution1D.BasicExponentialDistribution(self.lambdaVar,self.low)
      self.lowerBound = 0.0
      self.upperBound = sys.float_info.max
    else:
      if self.lowerBoundUsed == False:
        a = self.low
        self.lowerBound = a
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:
        b = self.upperBound
      self._distribution = distribution1D.BasicExponentialDistribution(self.lambdaVar,a,b,self.low)

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
    inputSpecification.addSub(InputData.parameterInputFactory("mean", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("sigma", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("low", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.mean = 1.0
    self.sigma = 1.0
    self.low = 0.0
    self.type = 'LogNormal'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
      self._distribution = distribution1D.BasicLogNormalDistribution(self.mean,self.sigma,self.low)
      self.lowerBound = -sys.float_info.max
      self.upperBound =  sys.float_info.max
    else:
      if self.lowerBoundUsed == False:
        a = self.low #-sys.float_info.max
        self.lowerBound = a
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:
        b = self.upperBound
      self._distribution = distribution1D.BasicLogNormalDistribution(self.mean,self.sigma,self.low,a,b)

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
    inputSpecification.addSub(InputData.parameterInputFactory("low", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("k", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BoostDistribution.__init__(self)
    self.lambdaVar = 1.0
    self.k = 1.0
    self.type = 'Weibull'
    self.disttype = 'Continuous'
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
    BoostDistribution._handleInput(self, paramInput)
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
    # check if lower  bound is set, otherwise default
    #if not self.lowerBoundUsed:
    #  self.lowerBoundUsed = True
    #  # lower bound = 0 since no location parameter available
    #  self.lowerBound     = 0.0
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
      # or self.lowerBound == 0.0:
      self._distribution = distribution1D.BasicWeibullDistribution(self.k,self.lambdaVar,self.low)
    else:
      if self.lowerBoundUsed == False:
        a = self.low
        self.lowerBound = a
      else:
        a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:
        b = self.upperBound
      self._distribution = distribution1D.BasicWeibullDistribution(self.k,self.lambdaVar,a,b,self.low)

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
    inputSpecification.addSub(InputData.parameterInputFactory("workingDir", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("functionType", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("dataFilename", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("functionID", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("variableID", contentType=InputData.StringType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Distribution.__init__(self)
    self.dataFilename    = None
    self.functionType    = None
    self.type            = 'Custom1D'
    self.functionID      = None
    self.variableID      = None
    self.dimensionality  = 1
    self.disttype        = 'Continuous'

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

    f = open(self.dataFilename, 'rb')
    reader = csv.reader(f)
    headers = reader.next()
    indexFunctionID = headers.index(self.functionID)
    indexVariableID = headers.index(self.variableID)
    f.close()
    rawData = np.genfromtxt(self.dataFilename, delimiter="," , skip_header=1, usecols=(indexVariableID,indexFunctionID))

    self.data = rawData[rawData[:,0].argsort()]

    if self.functionType == 'cdf':
      self.cdfFunc = UnivariateSpline(self.data[:,0], self.data[:,1], k=4, s=0)
      self.pdfFunc = self.cdfFunc.derivative()
      self.invCDF  = UnivariateSpline(self.data[:,1], self.data[:,0], k=4, s=0)
    else:
      # self.functionType == 'pdf'
      self.pdfFunc = UnivariateSpline(self.data[:,0], self.data[:,1], k=4, s=0)
      cdfValues = np.zeros(self.data[:,0].size)
      for i in range(self.data[:,0].size):
        cdfValues[i] = self.pdfFunc.integral(self.data[0][0],self.data[i,0])
      self.invCDF = UnivariateSpline(cdfValues, self.data[:,0] , k=4, s=0)

    # Note that self.invCDF is creating a new spline where I switch its term.
    # Instead of doing spline(x,f(x)) I am creating its inverse spline(f(x),x)
    # This can be done if f(x) is monothonic increasing with x (which is true for cdf)
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

    BaseInputType = InputData.makeEnumType("base", "baseType", ["natural","decimal"])

    inputSpecification.addSub(InputData.parameterInputFactory("lowerBound", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("upperBound", contentType=InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("base"      , BaseInputType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Distribution.__init__(self)
    self.upperBound = None
    self.lowerBound = None
    self.base       = None

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
    self.lowerBound = paramInput.findFirst('lowerBound').value
    if self.lowerBound is None:
      self.raiseAnError(IOError,' lowerBound parameter is needed for LogUniform distribution')

    self.upperBound = paramInput.findFirst('upperBound').value
    if self.upperBound is None:
      self.raiseAnError(IOError,' upperBound parameter is needed for LogUniform distribution')

    if self.upperBound < self.lowerBound:
      self.raiseAnError(IOError,' LogUniform distribution: the upperBound parameter is lower than the lowerBound parameter')

    self.base = paramInput.findFirst('base').value
    if self.base not in ['natural','decimal']:
      self.raiseAnError(IOError,' base parameter is needed for LogUniform distribution (either natural or decimal)')

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
      print(x,pdfValue)
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
    inputSpecification.addSub(InputData.parameterInputFactory("workingDir", contentType=InputData.StringType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Distribution.__init__(self)
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
    Distribution._handleInput(self, paramInput)
    workingDir = paramInput.findFirst('workingDir')
    if workingDir != None:
      self.workingDir = workingDir.value


  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Distribution.getInitParams(self)
    paramDict['functionType'] = self.functionType
    paramDict['dataFilename'] = self.dataFilename
    return paramDict

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


    DataFilenameParameterInput = InputData.parameterInputFactory("dataFilename", contentType=InputData.StringType)
    DataFilenameParameterInput.addParam("type", InputData.StringType, True)
    inputSpecification.addSub(DataFilenameParameterInput)

    inputSpecification.addSub(InputData.parameterInputFactory("p", contentType=InputData.FloatType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    NDimensionalDistributions.__init__(self)
    self.p  = None
    self.type = 'NDInverseWeight'

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    NDimensionalDistributions._handleInput(self, paramInput)
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
    paramDict = NDimensionalDistributions.getInitParams(self)
    paramDict['p'] = self.p
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    if self.functionType == 'CDF':
      self._distribution = distribution1D.BasicMultiDimensionalInverseWeight(str(self.dataFilename), self.p,True)
    else:
      self._distribution = distribution1D.BasicMultiDimensionalInverseWeight(str(self.dataFilename), self.p,False)

  def cdf(self,x):
    """
      calculate the cdf value for given coordinate x
      @ In, x, list, list of variable coordinate
      @ Out, cdfValue, float, cdf value
    """
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    cdfValue = self._distribution.cdf(coordinate)
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
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    pdfValue = self._distribution.pdf(coordinate)
    return pdfValue

  def cellIntegral(self,x,dx):
    """
      Compute the integral of N-D cell in this distribution
      @ In, x, np.array, x coordinates
      @ In, dx, np.array, discretization passes
      @ Out, integralReturn, float, the integral
    """
    coordinate = distribution1D.vectord_cxx(len(x))
    dxs        = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
      dxs[i]=dx[i]
    integralReturn = self._distribution.cellIntegral(coordinate,dxs)
    return integralReturn

  def inverseMarginalDistribution(self, x, variable):
    """
      Compute the inverse of the Margina distribution
      @ In, x, float, the coordinate for at which the inverse marginal distribution needs to be computed
      @ In, variable, string, the variable id
      @ Out, inverseMarginal, float, the marginal cdf value at coordinate x
    """
    if (x>0.0) and (x<1.0):
      inverseMarginal = self._distribution.inverseMarginal(x, variable)
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

    DataFilenameParameterInput = InputData.parameterInputFactory("dataFilename", contentType=InputData.StringType)
    DataFilenameParameterInput.addParam("type", InputData.StringType, True)
    inputSpecification.addSub(DataFilenameParameterInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    NDimensionalDistributions.__init__(self)
    self.type = 'NDCartesianSpline'

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    NDimensionalDistributions._handleInput(self, paramInput)
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
    paramDict = NDimensionalDistributions.getInitParams(self)
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.raiseAMessage('initialize Distribution')
    if self.functionType == 'CDF':
      self._distribution = distribution1D.BasicMultiDimensionalCartesianSpline(str(self.dataFilename),True)
    else:
      self._distribution = distribution1D.BasicMultiDimensionalCartesianSpline(str(self.dataFilename),False)

  def cdf(self,x):
    """
      calculate the cdf value for given coordinate x
      @ In, x, list, list of variable coordinate
      @ Out, cdfValue, float, cdf value
    """
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    cdfValue = self._distribution.cdf(coordinate)
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
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    pdfValue = self._distribution.pdf(coordinate)
    return pdfValue

  def cellIntegral(self,x,dx):
    """
      Compute the integral of N-D cell in this distribution
      @ In, x, np.array, x coordinates
      @ In, dx, np.array, discretization passes
      @ Out, integralReturn, float, the integral
    """
    coordinate = distribution1D.vectord_cxx(len(x))
    dxs        = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
      dxs[i]=dx[i]
    integralReturn = self._distribution.cellIntegral(coordinate,dxs)
    return integralReturn

  def inverseMarginalDistribution(self, x, variable):
    """
      Compute the inverse of the Margina distribution
      @ In, x, float, the coordinate for at which the inverse marginal distribution needs to be computed
      @ In, variable, string, the variable id
      @ Out, inverseMarginal, float, the marginal cdf value at coordinate x
    """
    if (x>=0.0) and (x<=1.0):
      inverseMarginal = self._distribution.inverseMarginal(x, variable)
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

    MuListParameterInput = InputData.parameterInputFactory("mu", contentType=InputData.StringType)

    CovarianceListParameterInput = InputData.parameterInputFactory("covariance", contentType=InputData.StringType)
    CovarianceListParameterInput.addParam("type", InputData.StringType, False)

    TransformationParameterInput = InputData.parameterInputFactory("transformation")
    RankParameterInput = InputData.parameterInputFactory("rank", contentType=InputData.IntegerType)
    TransformationParameterInput.addSub(RankParameterInput)

    inputSpecification.addSub(MuListParameterInput)
    inputSpecification.addSub(CovarianceListParameterInput)
    inputSpecification.addSub(TransformationParameterInput)

    MultivariateMethodType = InputData.makeEnumType("multivariateMethod","multivariateMethodType",["pca","spline"])
    inputSpecification.addParam("method", MultivariateMethodType, True)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    NDimensionalDistributions.__init__(self)
    self.type = 'MultivariateNormal'
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

  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = NDimensionalDistributions.getInitParams(self)
    return paramDict

  def initializeDistribution(self):
    """
      Method to initialize the distribution
      @ In, None
      @ Out, None
    """
    self.raiseAMessage('initialize distribution')
    mu = distribution1D.vectord_cxx(len(self.mu))
    for i in range(len(self.mu)):
      mu[i] = self.mu[i]
    covariance = distribution1D.vectord_cxx(len(self.covariance))
    for i in range(len(self.covariance)):
      covariance[i] = self.covariance[i]
    if self.method == 'spline':
      if self.covarianceType != 'abs':
        self.raiseAnError(IOError,'covariance with type ' + self.covariance + ' is not implemented for ' + self.method + ' method')
      self._distribution = distribution1D.BasicMultivariateNormal(covariance, mu)
    elif self.method == 'pca':
      self._distribution = distribution1D.BasicMultivariateNormal(covariance, mu, str(self.covarianceType), self.rank)

  def cdf(self,x):
    """
      calculate the cdf value for given coordinate x
      @ In, x, List, list of variable coordinate
      @ Out, cdfValue, float, cdf value
    """
    if self.method == 'spline':
      coordinate = distribution1D.vectord_cxx(len(x))
      for i in range(len(x)):
        coordinate[i] = x[i]
      cdfValue = self._distribution.cdf(coordinate)
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
      if index is not None:
        coordinateIndex = distribution1D.vectori_cxx(len(index))
        for i in range(len(index)):
          coordinateIndex[i] = index[i]
          matrixDim = self._distribution.getTransformationMatrixDimensions(coordinateIndex)
          transformation = self._distribution.getTransformationMatrix(coordinateIndex)
      else:
        matrixDim = self._distribution.getTransformationMatrixDimensions()
        transformation = self._distribution.getTransformationMatrix()
      row = matrixDim[0]
      column = matrixDim[1]
      # convert 1D vector to 2D array
      L = np.atleast_1d(transformation).reshape(row,column)
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
      if index is not None:
        coordinateIndex = distribution1D.vectori_cxx(len(index))
        for i in range(len(index)):
          coordinateIndex[i] = index[i]
          matrixDim = self._distribution.getInverseTransformationMatrixDimensions(coordinateIndex)
          inverseTransformation = self._distribution.getInverseTransformationMatrix(coordinateIndex)
      else:
        matrixDim = self._distribution.getInverseTransformationMatrixDimensions()
        inverseTransformation = self._distribution.getInverseTransformationMatrix()
      row = matrixDim[0]
      column = matrixDim[1]
      # convert 1D vector to 2D array
      L = np.atleast_1d(inverseTransformation).reshape(row,column)
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
      if index is not None:
        coordinateIndex = distribution1D.vectori_cxx(len(index))
        for i in range(len(index)):
          coordinateIndex[i] = index[i]
        singularValues = self._distribution.getSingularValues(coordinateIndex)
      else:
        singularValues = self._distribution.getSingularValues()
      singularValues = np.atleast_1d(singularValues).tolist()
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
        self.raiseAnError(IOError,'The dimension of the latent variables defined in <Samples> is large than the rank defined in <Distributions>')
      coordinate = distribution1D.vectord_cxx(len(x))
      for i in range(len(x)):
        coordinate[i] = x[i]
      if index is not None:
        coordinateIndex = distribution1D.vectori_cxx(len(index))
        for i in range(len(index)):
          coordinateIndex[i] = index[i]
        originalCoordinate = self._distribution.coordinateInverseTransformed(coordinate,coordinateIndex)
      else:
        originalCoordinate = self._distribution.coordinateInverseTransformed(coordinate)
      values = np.atleast_1d(originalCoordinate).tolist()
    else:
      self.raiseAnError(NotImplementedError,'ppfTransformedSpace not yet implemented for ' + self.method + ' method')
    return values

  def coordinateInTransformedSpace(self):
    """
      Return the coordinate in the transformed space
      @ In, None
      @ Out, coordinateInTransformedSpace, np.array, coordinates
    """
    if self.method == 'pca':
      coordinateInTransformedSpace = self._distribution.coordinateInTransformedSpace(self.rank)
    else:
      self.raiseAnError(NotImplementedError,'ppfTransformedSpace not yet implemented for ' + self.method + ' method')
    return coordinateInTransformedSpace

  def ppf(self,x):
    """
      Return the ppf of given coordinate
      @ In, x, np.array, the x coordinates
      @ Out, ppfValue, np.array, ppf values
    """
    if self.method == 'spline':
      ppfValue = self._distribution.inverseCdf(x,random())
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
    else:
      coordinate = distribution1D.vectord_cxx(len(x))
      for i in range(len(x)):
        coordinate[i] = x[i]
      pdfValue = self._distribution.pdf(coordinate)
    return pdfValue

  def pdfInTransformedSpace(self,x):
    """
      Return the pdf of given coordinate in the transformed space
      @ In, x, np.array, the x coordinates
      @ Out, pdfInTransformedSpace, np.array, pdf values in the transformed space
    """
    if self.method == 'pca':
      coordinate = distribution1D.vectord_cxx(len(x))
      for i in range(len(x)):
        coordinate[i] = x[i]
      pdfInTransformedSpace = self._distribution.pdfInTransformedSpace(coordinate)
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
    coordinate = distribution1D.vectord_cxx(len(x))
    dxs        = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
      dxs[i]=dx[i]
    if self.method == 'pca':
      if self.transformation:
        self.raiseAWarning("The ProbabilityWeighted is computed on the reduced transformed space")
      else:
        self.raiseAWarning("The ProbabilityWeighted is computed on the full transformed space")
      integralReturn = self._distribution.cellProbabilityWeight(coordinate,dxs)
    elif self.method == 'spline':
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
      @ In, variable, string, the variable id
      @ Out, inverseMarginal, float, the marginal cdf value at coordinate x
    """
    if (x > 0.0) and (x < 1.0):
      if self.method == 'pca':
        inverseMarginal = self._distribution.inverseMarginalForPCA(x)
      elif self.method == 'spline':
        inverseMarginal=  self._distribution.inverseMarginal(x, variable)
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

  def rvs(self,*args):
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
      if self.transformation:
        rvsValue = self._distribution.coordinateInTransformedSpace(self.rank)
      else:
        coordinate = self._distribution.coordinateInTransformedSpace(self.rank)
        rvsValue = self._distribution.coordinateInverseTransformed(coordinate)
    else:
      self.raiseAnError(NotImplementedError,'rvs is not yet implemented for ' + self.method + ' method')
    return rvsValue

DistributionsCollection.addSub(MultivariateNormal.getInputSpecification())

__base                                = 'Distribution'
__interFaceDict                       = {}
__interFaceDict['Uniform'           ] = Uniform
__interFaceDict['Normal'            ] = Normal
__interFaceDict['Gamma'             ] = Gamma
__interFaceDict['Beta'              ] = Beta
__interFaceDict['Triangular'        ] = Triangular
__interFaceDict['Poisson'           ] = Poisson
__interFaceDict['Binomial'          ] = Binomial
__interFaceDict['Bernoulli'         ] = Bernoulli
__interFaceDict['Categorical'       ] = Categorical
__interFaceDict['Logistic'          ] = Logistic
__interFaceDict['Exponential'       ] = Exponential
__interFaceDict['LogNormal'         ] = LogNormal
__interFaceDict['Weibull'           ] = Weibull
__interFaceDict['Custom1D'          ] = Custom1D
__interFaceDict['NDInverseWeight'   ] = NDInverseWeight
__interFaceDict['NDCartesianSpline' ] = NDCartesianSpline
__interFaceDict['MultivariateNormal'] = MultivariateNormal
__interFaceDict['Laplace'           ] = Laplace
__interFaceDict['Geometric'         ] = Geometric
__interFaceDict['LogUniform'        ] = LogUniform
__knownTypes                          = __interFaceDict.keys()

def knownTypes():
  """
    Return the known types
    @ In, None
    @ Out, __knownTypes, list, the known types
  """
  return __knownTypes

def returnInstance(Type,caller):
  """
    Function interface for creating an instance to a database specialized class (for example, HDF5)
    @ In, Type, string, class type
    @ In, caller, instance, the caller instance
    @ Out, returnInstance, instance, instance of the class
    Note: Interface function
  """
  try:
    return __interFaceDict[Type]()
  except KeyError:
    caller.raiseAnError(NameError,'not known '+__base+' type '+Type)

def returnInputParameter():
  """
    Function returns the InputParameterClass that can be used to parse the
    whole collection.
    @ Out, DistributionsCollection, DistributionsCollection, class for parsing.
  """
  return DistributionsCollection()
