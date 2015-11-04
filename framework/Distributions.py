"""
Created on Mar 7, 2013

@author: crisr
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
import scipy
#from scipy.misc import factorial
from math import gamma
import os
import operator
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
import utils
distribution1D = utils.find_distribution1D()
#Internal Modules End--------------------------------------------------------------------------------

def factorial(x):
  return gamma(x+1)

stochasticEnv = distribution1D.DistributionContainer.Instance()

"""
 Mapping between internal framework and Crow distribution name
"""
_FrameworkToCrowDistNames = {'Uniform':'UniformDistribution',
                              'Normal':'NormalDistribution',
                              'Gamma':'GammaDistribution',
                              'Beta':'BetaDistribution',
                              'Triangular':'TriangularDistribution',
                              'Poisson':'PoissonDistribution',
                              'Binomial':'BinomialDistribution',
                              'Bernoulli':'BernoulliDistribution',
                              'Logistic':'LogisticDistribution',
                              'Exponential':'ExponentialDistribution',
                              'LogNormal':'LogNormalDistribution',
                              'Weibull':'WeibullDistribution',
                              'NDInverseWeight': 'NDInverseWeightDistribution',
                              'NDCartesianSpline': 'NDCartesianSplineDistribution',
                              'MultivariateNormal' : 'MultivariateNormalDistribution'}


class Distribution(BaseType):
  """
  a general class containing the distributions
  """
  def __init__(self):
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
    pdict={}
    self.addInitParams(pdict)
    pdict['type']=self.type
    return pdict

  def __setstate__(self,pdict):
    self.__init__()
    self.upperBoundUsed   = pdict.pop('upperBoundUsed'  )
    self.lowerBoundUsed   = pdict.pop('lowerBoundUsed'  )
    self.hasInfiniteBound = pdict.pop('hasInfiniteBound')
    self.upperBound       = pdict.pop('upperBound'      )
    self.lowerBound       = pdict.pop('lowerBound'      )
    self.__adjustmentType = pdict.pop('adjustmentType'  )
    self.dimensionality   = pdict.pop('dimensionality'  )
    self.type             = pdict.pop('type')
    self._localSetState(pdict)
    self.initializeDistribution()

  def _readMoreXML(self,xmlNode):
    """
    Readmore xml, see BaseType.py explaination.
    """
    if xmlNode.find('upperBound') !=None:
      self.upperBound = float(xmlNode.find('upperBound').text)
      self.upperBoundUsed = True
    if xmlNode.find('lowerBound')!=None:
      self.lowerBound = float(xmlNode.find('lowerBound').text)
      self.lowerBoundUsed = True
    if xmlNode.find('adjustment') !=None: self.__adjustment = xmlNode.find('adjustment').text
    else: self.__adjustment = 'scaling'

  def getCrowDistDict(self):
    """
    Returns a dictionary of the keys and values that would be
    used to create the distribution for a Crow input file.
    """
    retDict = {}
    retDict['type'] = _FrameworkToCrowDistNames[self.type]
    if self.lowerBoundUsed:
      retDict['xMin'] = self.lowerBound
    if self.upperBoundUsed:
      retDict['xMax'] = self.upperBound
    return retDict

  def addInitParams(self,tempDict):
    """
    Function to get the input params that belong to this class
    @ In, tempDict, temporary dictionary
    """
    tempDict['upperBoundUsed'  ] = self.upperBoundUsed
    tempDict['lowerBoundUsed'  ] = self.lowerBoundUsed
    tempDict['hasInfiniteBound'] = self.hasInfiniteBound
    tempDict['upperBound'      ] = self.upperBound
    tempDict['lowerBound'      ] = self.lowerBound
    tempDict['adjustmentType'  ] = self.__adjustmentType
    tempDict['dimensionality'  ] = self.dimensionality

  def rvsWithinCDFbounds(self,LowerBound,upperBound):
    """
    Function to get a random number from a truncated distribution
    @ In, LowerBound, float -> lower bound
    @ In, upperBound, float -> upper bound
    @ In,           , float -> random number
    """
    point = float(np.random.rand(1))*(upperBound-LowerBound)+LowerBound
    return self._distribution.InverseCdf(point)

  def rvsWithinbounds(self,LowerBound,upperBound):
    """
    Function to get a random number from a truncated distribution
    @ In, LowerBound, float -> lower bound
    @ In, upperBound, float -> upper bound
    @ Out,          , float -> random number
    """
    CDFupper = self._distribution.Cdf(upperBound)
    CDFlower = self._distribution.Cdf(LowerBound)
    return self.rvsWithinCDFbounds(CDFlower,CDFupper)

  def convertToDistr(self,qtype,pts):
    """Converts points from the quadrature "qtype" standard domain to the distribution domain.
    @ In qtype, string, type of quadrature to convert from
    @ In pts, array of floats, points to convert
    @ Out, array of floats, converted points
    """
    return self.convertToDistrDict[qtype](pts)

  def convertToQuad(self,qtype,pts):
    """Converts points from the distribution domain to the quadrature "qtype" standard domain.
    @ In qtype, string, type of quadrature to convert to
    @ In pts, array of floats, points to convert
    @ Out, array of floats, converted points
    """
    return self.convertToQuadDict[qtype](pts)

  def measureNorm(self,qtype):
    """Provides the integral/jacobian conversion factor between the distribution domain and the quadrature domain.
    @ In qtype, string, type of quadrature to convert to
    @ Out, float, conversion factor
    """
    return self.measureNormDict[qtype]()

  def _convertDistrPointsToCdf(self,pts):
    """Converts points in the distribution domain to [0,1].
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    """
    try: return self.cdf(pts.real)
    except TypeError: return list(self.cdf(x) for x in pts)

  def _convertCdfPointsToDistr(self,pts):
    """Converts points in [0,1] to the distribution domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    """
    try: return self.ppf(pts.real)
    except TypeError: return list(self.ppf(x) for x in pts)

  def _convertCdfPointsToStd(self,pts):
    """Converts points in [0,1] to [-1,1], the uniform distribution's STANDARD domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    """
    try: return 2.0*pts.real-1.0
    except TypeError: return list(2.0*x-1.0 for x in pts)

  def _convertStdPointsToCdf(self,pts):
    """Converts points in [-1,1] to [0,1] (CDF domain).
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    """
    try: return 0.5*(pts.real+1.0)
    except TypeError: return list(0.5*(x+1.0) for x in pts)

  def CDFconvertToQuad(self,pts):
    """Converts all the way from distribution domain to [-1,1] quadrature domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    """
    return self._convertCdfPointsToStd(self._convertDistrPointsToCdf(pts))

  def CDFconvertToDistr(self,pts):
    """Converts all the way from [-1,1] quadrature domain to distribution domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    """
    return self._convertCdfPointsToDistr(self._convertStdPointsToCdf(pts))

  def CDFMeasureNorm(self):
    """Integral norm/jacobian for [-1,1] Legendre quadrature.
    @ In None, None
    @ Out float, normalization factor
    """
    return 1.0/2.0;

  def getDimensionality(self):
    """
    Function return the dimensionality of the distribution
    @ In, None, None
    @ Out, integer
    """
    return self.dimensionality

  def getDisttype(self):
    """
    Function return distribution type
    @ In, None, None
    @ Out, String ('Continuous' or 'Discrete')
    """
    return self.disttype


def random():
  """
  Function to get a random number <1<
  @ In, None, None
  @ Out, float, random number
  """
  return stochasticEnv.random()

def randomSeed(value):
  """
  Function to get a random seed
  @ In, None, None
  @ Out, integer, random seed
  """
  return stochasticEnv.seedRandom(value)

def randomIntegers(low,high,caller):
  """
  Function to get a random integer
  @ In, low, integer -> low boundary
  @ In, high, integer -> upper boundary
  @ Out, integer, random int
  """
  intRange = high-low
  rawNum = low + random()*intRange
  rawInt = int(round(rawNum))
  if rawInt < low or rawInt > high:
    caller.raiseAMessage("Random int out of range")
    rawInt = max(low,min(rawInt,high))
  return rawInt

def randomPermutation(l,caller):
  """
  Function to get a random permutation
  @ In, l, list -> list to be permuted
  @ Out, list, randomly permuted list
  """
  newList = []
  oldList = l[:]
  while len(oldList) > 0: newList.append(oldList.pop(randomIntegers(0,len(oldList)-1,caller)))
  return newList

class BoostDistribution(Distribution):
  """
  Base distribution class based on boost
  """
  def __init__(self):
    Distribution.__init__(self)
    self.dimensionality  = 1
    self.disttype        = 'Continuous'

  def cdf(self,x):
    """
    Function to get the cdf at a provided coordinate
    @ In, x, float -> value to get the cdf at
    @ Out, float, requested cdf
    """

    return self._distribution.Cdf(x)

  def ppf(self,x):
    """
    Function to get the inverse cdf at a provided coordinate
    @ In, x, float -> value to get the inverse cdf at
    @ Out, float, requested inverse cdf
    """
    return self._distribution.InverseCdf(x)

  def pdf(self,x):
    """
    Function to get the pdf at a provided coordinate
    @ In, x, float -> value to get the pdf at
    @ Out, float, requested pdf
   """
#     value = 0.0
#     for i in str(x).strip().split(','):
#       value +=  self._distribution.Pdf(float(i))
#
#     return value
    return self._distribution.Pdf(x)


  def untruncatedCdfComplement(self, x):
    """
    Function to get the untruncated  cdf complement at a provided coordinate
    @ In, x, float -> value to get the untruncated  cdf complement  at
    @ Out, float, requested untruncated  cdf complement
    """
    return self._distribution.untrCdfComplement(x)

  def untruncatedHazard(self, x):
    """
    Function to get the untruncated  Hazard  at a provided coordinate
    @ In, x, float -> value to get the untruncated  Hazard   at
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
    @ Out, float, requested Mode
    """

    return self._distribution.untrMode()


  def rvs(self,*args):
    """
    Function to get random numbers
    @ In, args, dictionary, args
    @ Out, float or list, requested random number or numbers
    """

    if len(args) == 0: return self.ppf(random())
    else             : return [self.rvs() for _ in range(args[0])]


class Uniform(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.range = 0.0
    self.type = 'Uniform'
    self.disttype = 'Continuous'
    self.compatibleQuadrature.append('Legendre')
    self.compatibleQuadrature.append('ClenshawCurtis')
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature = 'Legendre'
    self.preferredPolynomials = 'Legendre'

  def _localSetState(self,pdict):
    #self.lowerBound   = pdict.pop('lowerBound'  )
    #self.upperBound   = pdict.pop('upperBound'   )
    self.range        = pdict.pop('range')

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['xMin'] = self.lowerBound
    retDict['xMax'] = self.upperBound
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self,xmlNode)
    if not self.upperBoundUsed or not self.lowerBoundUsed:
      self.raiseAnError(IOError,'the Uniform distribution needs both upperBound and lowerBound attributes. Got upperBound? '+ str(self.upperBoundUsed) + '. Got lowerBound? '+str(self.lowerBoundUsed))
    self.range = self.upperBound - self.lowerBound
    self.initializeDistribution()

  def stdProbabilityNorm(self):
    """Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    """
    return 0.5

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self,tempDict)
    tempDict['range'] = self.range
    # no other additional parameters required

  def initializeDistribution(self):
    self.convertToDistrDict['Legendre']       = self.convertLegendreToUniform
    self.convertToQuadDict ['Legendre']       = self.convertUniformToLegendre
    self.measureNormDict   ['Legendre']       = self.stdProbabilityNorm
    self.convertToDistrDict['ClenshawCurtis'] = self.convertLegendreToUniform
    self.convertToQuadDict ['ClenshawCurtis'] = self.convertUniformToLegendre
    self.measureNormDict   ['ClenshawCurtis'] = self.stdProbabilityNorm
    self._distribution = distribution1D.BasicUniformDistribution(self.lowerBound,self.lowerBound+self.range)

  def convertUniformToLegendre(self,y):
    """Converts from distribution domain to standard Legendre [-1,1].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return (y-self.untruncatedMean())/(self.range/2.)

  def convertLegendreToUniform(self,x):
    """Converts from standard Legendre [-1,1] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return self.range/2.*x+self.untruncatedMean()



class Normal(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mean  = 0.0
    self.sigma = 0.0
    self.hasInfiniteBound = True
    self.type = 'Normal'
    self.disttype = 'Continuous'
    self.compatibleQuadrature.append('Hermite')
    self.compatibleQuadrature.append('CDF')
    #THESE get set in initializeDistribution, since it depends on truncation
    #self.preferredQuadrature  = 'Hermite'
    #self.preferredPolynomials = 'Hermite'

  def _localSetState(self,pdict):
    self.mean  = pdict.pop('mean' )
    self.sigma = pdict.pop('sigma')

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['mu'] = self.mean
    retDict['sigma'] = self.sigma
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    meanFind = xmlNode.find('mean' )
    if meanFind != None: self.mean  = float(meanFind.text)
    else: self.raiseAnError(IOError,'mean value needed for normal distribution')
    sigmaFind = xmlNode.find('sigma')
    if sigmaFind != None: self.sigma = float(sigmaFind.text)
    else: self.raiseAnError(IOError,'sigma value needed for normal distribution')
    self.initializeDistribution() #FIXME no other distros have this...needed?

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma

  def initializeDistribution(self):
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
      else:a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:b = self.upperBound
      self._distribution = distribution1D.BasicNormalDistribution(self.mean,
                                                                  self.sigma,
                                                                  a,b)

  def stdProbabilityNorm(self,std=False):
    """Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    """
    sv = str(scipy.__version__).split('.')
    if int(sv[0])==0 and int(sv[1])<15:
      return 1.0/np.sqrt(2.*np.pi)
    else:
      return 1.0/np.sqrt(np.pi/2.)

  def convertNormalToHermite(self,y):
    """Converts from distribution domain to standard Hermite [-inf,inf].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return (y-self.untruncatedMean())/(self.sigma)

  def convertHermiteToNormal(self,x):
    """Converts from standard Hermite [-inf,inf] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return self.sigma*x+self.untruncatedMean()

class Gamma(BoostDistribution):
  def __init__(self):
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
    self.low   = pdict.pop('low'  )
    self.alpha = pdict.pop('alpha')
    self.beta  = pdict.pop('beta' )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['k'] = self.alpha
    retDict['theta'] = 1.0/self.beta
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self,xmlNode)
    lowFind = xmlNode.find('low')
    if lowFind != None: self.low = float(lowFind.text)
    alphaFind = xmlNode.find('alpha')
    if alphaFind != None: self.alpha = float(alphaFind.text)
    else: self.raiseAnError(IOError,'alpha value needed for Gamma distribution')
    betaFind = xmlNode.find('beta')
    if betaFind != None: self.beta = float(betaFind.text)
    # check if lower bound are set, otherwise default
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.low
    self.initializeDistribution() #TODO this exists in a couple classes; does it really need to be here and not in Simulation? - No. - Andrea

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta

  def initializeDistribution(self):
    self.convertToDistrDict['Laguerre'] = self.convertLaguerreToGamma
    self.convertToQuadDict ['Laguerre'] = self.convertGammaToLaguerre
    self.measureNormDict   ['Laguerre'] = self.stdProbabilityNorm
    if (not self.upperBoundUsed): # and (not self.lowerBoundUsed):
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
      else:a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:b = self.upperBound
      self._distribution = distribution1D.BasicGammaDistribution(self.alpha,1.0/self.beta,self.low,a,b)

  def convertGammaToLaguerre(self,y):
    """Converts from distribution domain to standard Laguerre [0,inf].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return (y-self.low)*(self.beta)

  def convertLaguerreToGamma(self,x):
    """Converts from standard Laguerre [0,inf] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    return x/self.beta+self.low

  def stdProbabilityNorm(self):
    """Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    """
    #return self.beta**self.alpha/factorial(self.alpha-1.)
    return 1./factorial(self.alpha-1)


class Beta(BoostDistribution):
  def __init__(self):
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
    self.low   = pdict.pop('low'  )
    self.high  = pdict.pop('high' )
    self.alpha = pdict.pop('alpha')
    self.beta  = pdict.pop('beta' )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['alpha'] = self.alpha
    retDict['beta'] = self.beta
    retDict['scale'] = self.high-self.low
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self,xmlNode)
    lowFind = xmlNode.find('low')
    if lowFind != None: self.low = float(lowFind.text)
    hiFind = xmlNode.find('high')
    if hiFind != None: self.high = float(hiFind.text)
    alphaFind = xmlNode.find('alpha')
    betaFind = xmlNode.find('beta')
    peakFind = xmlNode.find('peakFactor')
    if alphaFind != None and betaFind != None and peakFind == None:
      self.alpha = float(alphaFind.text)
      self.beta  = float(betaFind.text)
    elif (alphaFind == None and betaFind == None) and peakFind != None:
      peakFactor = float(peakFind.text)
      if not 0 <= peakFactor <= 1: self.raiseAnError(IOError,'peakFactor must be from 0 to 1, inclusive!')
      #this empirical formula is used to make it so factor->alpha: 0->1, 0.5~7.5, 1->99
      self.alpha = 0.5*23.818**(5.*peakFactor/3.) + 0.5
      self.beta = self.alpha
    else: self.raiseAnError(IOError,'Either provide (alpha and beta) or peakFactor!')
    # check if lower or upper bounds are set, otherwise default
    if not self.upperBoundUsed:
      self.upperBoundUsed = True
      self.upperBound     = self.high
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.low
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self,tempDict)
    tempDict['low'  ] = self.low
    tempDict['high' ] = self.high
    tempDict['alpha'] = self.alpha
    tempDict['beta' ] = self.beta

  def initializeDistribution(self):
    self.convertToDistrDict['Jacobi'] = self.convertJacobiToBeta
    self.convertToQuadDict ['Jacobi'] = self.convertBetaToJacobi
    self.measureNormDict   ['Jacobi'] = self.stdProbabilityNorm
    #this "if" section can only be called if distribution not generated using readMoreXML
    if (not self.upperBoundUsed) and (not self.lowerBoundUsed):
      self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.high-self.low,self.low)
    else:
      if self.lowerBoundUsed == False: a = 0.0
      else:a = self.lowerBound
      if self.upperBoundUsed == False: b = sys.float_info.max
      else:b = self.upperBound
      self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.high-self.low,a,b,self.low)
    self.preferredPolynomials = 'Jacobi'
    self.compatibleQuadrature.append('Jacobi')
    self.compatibleQuadrature.append('ClenshawCurtis')

  def convertBetaToJacobi(self,y):
    """Converts from distribution domain to standard Beta [0,1].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    u = 0.5*(self.high+self.low)
    s = 0.5*(self.high-self.low)
    return (y-u)/(s)

  def convertJacobiToBeta(self,x):
    """Converts from standard Jacobi [0,1] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    """
    u = 0.5*(self.high+self.low)
    s = 0.5*(self.high-self.low)
    return s*x+u

  def stdProbabilityNorm(self):
    """Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    """
    B = factorial(self.alpha-1)*factorial(self.beta-1)/factorial(self.alpha+self.beta-1)
    norm = 1.0/(2**(self.alpha+self.beta-1)*B)
    return norm



class Triangular(BoostDistribution):
  def __init__(self):
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
    self.apex = pdict.pop('apex')
    self.min  = pdict.pop('min' )
    self.max  = pdict.pop('max' )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['xPeak'] = self.apex
    retDict['lowerBound'] = self.min
    retDict['upperBound'] = self.max
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    apexFind = xmlNode.find('apex')
    if apexFind != None: self.apex = float(apexFind.text)
    else: self.raiseAnError(IOError,'apex value needed for normal distribution')
    minFind = xmlNode.find('min')
    if minFind != None: self.min = float(minFind.text)
    else: self.raiseAnError(IOError,'min value needed for normal distribution')
    maxFind = xmlNode.find('max')
    if maxFind != None: self.max = float(maxFind.text)
    else: self.raiseAnError(IOError,'max value needed for normal distribution')
    # check if lower or upper bounds are set, otherwise default
    if not self.upperBoundUsed:
      self.upperBoundUsed = True
      self.upperBound     = self.max
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.min
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['apex' ] = self.apex
    tempDict['min'  ] = self.min
    tempDict['max'  ] = self.max

  def initializeDistribution(self):
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False) or (self.min == self.lowerBound and self.max == self.upperBound):
      self._distribution = distribution1D.BasicTriangularDistribution(self.apex,self.min,self.max)
    else:
      self.raiseAnError(IOError,'Truncated triangular not yet implemented')



class Poisson(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mu  = 0.0
    self.type = 'Poisson'
    self.hasInfiniteBound = True
    self.disttype = 'Discrete'
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    self.mu = pdict.pop('mu')


  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['mu'] = self.mu
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    muFind = xmlNode.find('mu')
    if muFind != None: self.mu = float(muFind.text)
    else: self.raiseAnError(IOError,'mu value needed for poisson distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['mu'  ] = self.mu

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicPoissonDistribution(self.mu)
      self.lowerBound = 0.0
      self.upperBound = sys.float_info.max
    else:
      self.raiseAnError(IOError,'Truncated poisson not yet implemented')


class Binomial(BoostDistribution):
  def __init__(self):
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
    self.n = pdict.pop('n')
    self.p = pdict.pop('p')

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['n'] = self.n
    retDict['p'] = self.p
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    nFind = xmlNode.find('n')
    if nFind != None: self.n = float(nFind.text)
    else: self.raiseAnError(IOError,'n value needed for Binomial distribution')
    pFind = xmlNode.find('p')
    if pFind != None: self.p = float(pFind.text)
    else: self.raiseAnError(IOError,'p value needed for Binomial distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['n'  ] = self.n
    tempDict['p'  ] = self.p

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicBinomialDistribution(self.n,self.p)
    else: self.raiseAnError(IOError,'Truncated Binomial not yet implemented')
#
#
#
class Bernoulli(BoostDistribution):
  def __init__(self):
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
    self.p = pdict.pop('p')

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['p'] = self.p
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    pFind = xmlNode.find('p')
    if pFind != None: self.p = float(pFind.text)
    else: self.raiseAnError(IOError,'p value needed for Bernoulli distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['p'  ] = self.p

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicBernoulliDistribution(self.p)
    else:  self.raiseAnError(IOError,'Truncated Bernoulli not yet implemented')


class Categorical(Distribution):
  """
  Class for the categorical distribution also called " generalized Bernoulli distribution"
  Note: this distribution can have only numerical (float) outcome; in the future we might want to include also the possibility to give symbolic outcome
  """

  def __init__(self):
    """
    Function that initializes the categorical distribution
    @ In, None
    @ Out, none
   """
    Distribution.__init__(self)
    self.mapping = {}
    self.values = set()
    self.type     = 'Categorical'
    self.disttype = 'Discrete'

  def _readMoreXML(self,xmlNode):
    """
    Function that retrive data to initialize the categorical distribution from the xmlNode
    @ In, None
    @ Out, None
   """
    Distribution._readMoreXML(self, xmlNode)

    for child in xmlNode:
      if child.tag == "state":
        outcome = child.attrib['outcome']
        self.mapping[outcome] = float(child.text)
        if float(outcome) in self.values:
          self.raiseAnError(IOError,'Categorical distribution has identical outcomes')
        else:
          self.values.add(float(outcome))
      else:
        self.raiseAnError(IOError,'Invalide xml node for Categorical distribution; only "state" is allowed')

    self.initializeDistribution()

  def addInitParams(self,tempDict):
    """
    Function to get the input params that belong to this class
    @ In, tempDict, temporary dictionary
    @ Out, tempDict, temporary dictionary
    """
    Distribution.addInitParams(self, tempDict)
    tempDict['mapping'] = self.mapping
    tempDict['values'] = self.values

  def initializeDistribution(self):
    """
    Function that initializes the distribution and checks that the sum of all state probabilities is equal to 1
    @ In, None
    @ Out, None
    """
    totPsum = 0.0
    for element in self.mapping:
      totPsum += self.mapping[str(element)]
    if totPsum!=1.0: self.raiseAnError(IOError,'Categorical distribution cannot be initialized: sum of probabilities is not 1.0')

  def pdf(self,x):
    """
    Function that calculates the pdf value of x
    @ In, x, float/string -> value to get the pdf at
    @ Out, float, requested pdf
    """
    if x in self.values:
      return self.mapping[str(x)]
    else: self.raiseAnError(IOError,'Categorical distribution cannot calculate pdf for ' + str(x))

  def cdf(self,x):
    """
    Function to get the cdf value of x
    @ In, x, float/string -> value to get the pdf at
    @ Out, float, requested cdf
    """
    sortedMapping = sorted(self.mapping.items(), key=operator.itemgetter(0))
    if x in self.values:
      cumulative=0.0
      for element in sortedMapping:
        cumulative += element[1]
        if x == float(element[0]):
          return cumulative
    else: self.raiseAnError(IOError,'Categorical distribution cannot calculate cdf for ' + str(x))

  def ppf(self,x):
    """
    Function that calculates the inverse of the cdf given 0 =< x =< 1
    @ In, x, float -> value to get the pdf at
    @ Out, float/string, requested inverse cdf
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
    @ Out, float/string
   """
    return self.ppf(random())

class Logistic(BoostDistribution):
  def __init__(self):
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
    self.location = pdict.pop('location')
    self.scale    = pdict.pop('scale'   )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['scale'] = self.scale
    retDict['location'] = self.location
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    locationFind = xmlNode.find('location')
    if locationFind != None: self.location = float(locationFind.text)
    else: self.raiseAnError(IOError,'location value needed for Logistic distribution')
    scaleFind = xmlNode.find('scale')
    if scaleFind != None: self.scale = float(scaleFind.text)
    else: self.raiseAnError(IOError,'scale value needed for Logistic distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['location'] = self.location
    tempDict['scale'   ] = self.scale

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicLogisticDistribution(self.location,self.scale)
    else:
      if self.lowerBoundUsed == False: a = -sys.float_info.max
      else:a = self.lowerBound
      if self.upperBoundUsed == False: b = sys.float_info.max
      else:b = self.upperBound
      self._distribution = distribution1D.BasicLogisticDistribution(self.location,self.scale,a,b)


class Exponential(BoostDistribution):
  def __init__(self):
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
    self.lambdaVar = pdict.pop('lambda')
    self.low        = pdict.pop('low'   )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['lambda'] = self.lambdaVar
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    lambdaFind = xmlNode.find('lambda')
    if lambdaFind != None: self.lambdaVar = float(lambdaFind.text)
    else: self.raiseAnError(IOError,'lambda value needed for Exponential distribution')
    low  = xmlNode.find('low')
    if low != None: self.low = float(low.text)
    else: self.low = 0.0
    # check if lower bound is set, otherwise default
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.low
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['lambda'] = self.lambdaVar
    tempDict['low'] = self.low

  def initializeDistribution(self):
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False):
      self._distribution = distribution1D.BasicExponentialDistribution(self.lambdaVar,self.low)
      self.lowerBound = 0.0
      self.upperBound = sys.float_info.max
    else:
      if self.lowerBoundUsed == False:
        a = self.low
        self.lowerBound = a
      else:a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:b = self.upperBound
      self._distribution = distribution1D.BasicExponentialDistribution(self.lambdaVar,a,b,self.low)

  def convertDistrPointsToStd(self,y):
    quad=self.quadratureSet()
    if quad.type=='Laguerre':
      return (y-self.low)*(self.lambdaVar)
    else:
      return Distribution.convertDistrPointsToStd(self,y)

  def convertStdPointsToDistr(self,x):
    quad=self.quadratureSet()
    if quad.type=='Laguerre':
      return x/self.lambdaVar+self.low
    else:
      return Distribution.convertStdPointsToDistr(self,x)


class LogNormal(BoostDistribution):
  def __init__(self):
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
    self.mean  = pdict.pop('mean' )
    self.sigma = pdict.pop('sigma')

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['mu'] = self.mean
    retDict['sigma'] = self.sigma
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    meanFind = xmlNode.find('mean')
    if meanFind != None: self.mean = float(meanFind.text)
    else: self.raiseAnError(IOError,'mean value needed for LogNormal distribution')
    sigmaFind = xmlNode.find('sigma')
    if sigmaFind != None: self.sigma = float(sigmaFind.text)
    else: self.raiseAnError(IOError,'sigma value needed for LogNormal distribution')
    lowFind = xmlNode.find('low')
    if lowFind != None: self.low = float(lowFind.text)
    else: self.low = 0.0
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma
    tempDict['low'] = self.low

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicLogNormalDistribution(self.mean,self.sigma,self.low)
      self.lowerBound = -sys.float_info.max
      self.upperBound =  sys.float_info.max
    else:
      if self.lowerBoundUsed == False:
        a = self.low #-sys.float_info.max
        self.lowerBound = a
      else:a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:b = self.upperBound
      self._distribution = distribution1D.BasicLogNormalDistribution(self.mean,self.sigma,self.low,a,b)


class Weibull(BoostDistribution):
  def __init__(self):
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
    self.lambdaVar = pdict.pop('lambda')
    self.k          = pdict.pop('k'     )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['lambda'] = self.lambdaVar
    retDict['k'] = self.k
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    lambdaFind = xmlNode.find('lambda')
    if lambdaFind != None: self.lambdaVar = float(lambdaFind.text)
    else: self.raiseAnError(IOError,'lambda (scale) value needed for Weibull distribution')
    kFind = xmlNode.find('k')
    if kFind != None: self.k = float(kFind.text)
    else: self.raiseAnError(IOError,'k (shape) value needed for Weibull distribution')
    lowFind = xmlNode.find('low')
    if lowFind != None: self.low = float(lowFind.text)
    else: self.low = 0.0
    # check if lower  bound is set, otherwise default
    #if not self.lowerBoundUsed:
    #  self.lowerBoundUsed = True
    #  # lower bound = 0 since no location parameter available
    #  self.lowerBound     = 0.0
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['lambda'] = self.lambdaVar
    tempDict['k'     ] = self.k
    tempDict['low'   ] = self.low

  def initializeDistribution(self):
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False): # or self.lowerBound == 0.0:
      self._distribution = distribution1D.BasicWeibullDistribution(self.k,self.lambdaVar,self.low)
    else:
      if self.lowerBoundUsed == False:
        a = self.low
        self.lowerBound = a
      else:a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:b = self.upperBound
      self._distribution = distribution1D.BasicWeibullDistribution(self.k,self.lambdaVar,a,b,self.low)



class NDimensionalDistributions(Distribution):

  def __init__(self):
    Distribution.__init__(self)
    self.dataFilename = None
    self.functionType = None
    self.type = 'NDimensionalDistributions'
    self.dimensionality  = None

    self.RNGInitDisc = 10
    self.RNGtolerance = 0.1

  def _readMoreXML(self,xmlNode):
    Distribution._readMoreXML(self, xmlNode)
    workingDir = xmlNode.find('workingDir')
    if workingDir != None: self.workingDir = workingDir.text

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['functionType'] = self.functionType
    tempDict['dataFilename'] = self.dataFilename

  #######
  def updateRNGParam(self, dictParam):
    self.RNGtolerance = 0.1
    self.RNGInitDisc  = 10
    for key in dictParam:
      if key == 'tolerance':
        self.RNGtolerance = dictParam['tolerance']
      elif key == 'initialGridDisc':
        self.RNGInitDisc  = dictParam['initialGridDisc']
    self._distribution.updateRNGparameter(self.RNGtolerance,self.RNGInitDisc)
  ######

  def getDimensionality(self):
    return  self._distribution.returnDimensionality()
  
  def returnLowerBound(self, dimension):
    return self._distribution.returnLowerBound(dimension)
  
  def returnUpperBound(self, dimension):
    return self._distribution.returnUpperBound(dimension)

class NDInverseWeight(NDimensionalDistributions):

  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.p  = None
    self.type = 'NDInverseWeight'

  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)
    pFind = xmlNode.find('p')
    if pFind != None: self.p = float(pFind.text)
    else: self.raiseAnError(IOError,'Minkowski distance parameter <p> not found in NDInverseWeight distribution')

    dataFilename = xmlNode.find('dataFilename')
    if dataFilename != None: self.dataFilename = os.path.join(self.workingDir,dataFilename.text)
    else: self.raiseAnError(IOError,'<dataFilename> parameter needed for MultiDimensional Distributions!!!!')

    functionType = dataFilename.attrib['type']
    if functionType != None: self.functionType = functionType
    else: self.raiseAnError(IOError,'<functionType> parameter needed for MultiDimensional Distributions!!!!')

    self.initializeDistribution()

  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)
    tempDict['p'] = self.p

  def initializeDistribution(self):
    if self.functionType == 'CDF':
      self._distribution = distribution1D.BasicMultiDimensionalInverseWeight(str(self.dataFilename), self.p,True)
    else:
      self._distribution = distribution1D.BasicMultiDimensionalInverseWeight(str(self.dataFilename), self.p,False)

  def cdf(self,x):
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    return self._distribution.Cdf(coordinate)

  def ppf(self,x):
    return self._distribution.InverseCdf(x,random())

  def pdf(self,x):
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    return self._distribution.Pdf(coordinate)

  def cellIntegral(self,x,dx):
    coordinate = distribution1D.vectord_cxx(len(x))
    dxs        = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
      dxs[i]=dx[i]
    return self._distribution.cellIntegral(coordinate,dxs)

  def inverseMarginalDistribution (self, x, variable):
    if (x>0.0) and (x<1.0):
      return self._distribution.inverseMarginal(x, variable)
    else:
      self.raiseAnError(ValueError,'NDInverseWeight: inverseMarginalDistribution(x) with x outside [0.0,1.0]')

  def untruncatedCdfComplement(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    self.raiseAnError(NotImplementedError,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    self.raiseAnError(NotImplementedError,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    self.raiseAnError(NotImplementedError,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    return self._distribution.InverseCdf(random(),random())


class NDCartesianSpline(NDimensionalDistributions):
  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.type = 'NDCartesianSpline'

  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)

    dataFilename = xmlNode.find('dataFilename')
    if dataFilename != None: self.dataFilename = os.path.join(self.workingDir,dataFilename.text)
    else: self.raiseAnError(IOError,'<dataFilename> parameter needed for MultiDimensional Distributions!!!!')

    functionType = dataFilename.attrib['type']
    if functionType != None: self.functionType = functionType
    else: self.raiseAnError(IOError,'<functionType> parameter needed for MultiDimensional Distributions!!!!')

    self.initializeDistribution()

  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)

  def initializeDistribution(self):
    self.raiseAMessage('====== BasicMultiDimensional NDCartesianSpline initialize Distribution ======')
    if self.functionType == 'CDF':
      self._distribution = distribution1D.BasicMultiDimensionalCartesianSpline(str(self.dataFilename),True)
    else:
      self._distribution = distribution1D.BasicMultiDimensionalCartesianSpline(str(self.dataFilename),False)

  def cdf(self,x):
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    return self._distribution.Cdf(coordinate)

  def ppf(self,x):
    return self._distribution.InverseCdf(x,random())

  def pdf(self,x):
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    return self._distribution.Pdf(coordinate)

  def cellIntegral(self,x,dx):
    coordinate = distribution1D.vectord_cxx(len(x))
    dxs        = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
      dxs[i]=dx[i]
    return self._distribution.cellIntegral(coordinate,dxs)

  def inverseMarginalDistribution (self, x, variable):
    if (x>=0.0) and (x<=1.0):
      return self._distribution.inverseMarginal(x, variable)
    else:
      self.raiseAnError(ValueError,'NDCartesianSpline: inverseMarginalDistribution(x) with x ' +str(x)+' outside [0.0,1.0]')

  def untruncatedCdfComplement(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    self.raiseAnError(NotImplementedError,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    self.raiseAnError(NotImplementedError,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    self.raiseAnError(NotImplementedError,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    return self._distribution.InverseCdf(random(),random())


class MultivariateNormal(NDimensionalDistributions):

  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.type = 'MultivariateNormal'
    self.mu  = None
    self.covariance = None

  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)
    mu = xmlNode.find('mu')
    if mu != None: self.mu = [float(i) for i in mu.text.split()]
    else: self.raiseAnError(IOError,'<mu> parameter needed for MultivariateNormal Distributions!!!!')
    covariance = xmlNode.find('covariance')
    if covariance != None: self.covariance = [float(i) for i in covariance.text.split()]
    else: self.raiseAnError(IOError,'<covariance> parameter needed for MultivariateNormal Distributions!!!!')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)

  def initializeDistribution(self):
    self.raiseAMessage('====== BasicMultiDimensional MultivariateNormal initialize distribution ======')
    mu = distribution1D.vectord_cxx(len(self.mu))
    for i in range(len(self.mu)):
      mu[i] = self.mu[i]
    covariance = distribution1D.vectord_cxx(len(self.covariance))
    for i in range(len(self.covariance)):
      covariance[i] = self.covariance[i]
    self._distribution = distribution1D.BasicMultivariateNormal(covariance, mu)

  def cdf(self,x):
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    return self._distribution.Cdf(coordinate)

  def ppf(self,x):
    return self._distribution.InverseCdf(x,random())

  def pdf(self,x):
    coordinate = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
    return self._distribution.Pdf(coordinate)

  def cellIntegral(self,x,dx):
    coordinate = distribution1D.vectord_cxx(len(x))
    dxs        = distribution1D.vectord_cxx(len(x))
    for i in range(len(x)):
      coordinate[i] = x[i]
      dxs[i]=dx[i]
    return self._distribution.cellIntegral(coordinate,dxs)

  def inverseMarginalDistribution (self, x, variable):
    if (x>0.0) and (x<1.0):
      return self._distribution.inverseMarginal(x, variable)
    else:
      self.raiseAnError(ValueError,'NDInverseWeight: inverseMarginalDistribution(x) with x ' +str(x)+' outside [0.0,1.0]')

  def untruncatedCdfComplement(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self, x):
    self.raiseAnError(NotImplementedError,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    return self._distribution.InverseCdf(random(),random())


__base                        = 'Distribution'
__interFaceDict               = {}
__interFaceDict['Uniform'          ] = Uniform
__interFaceDict['Normal'           ] = Normal
__interFaceDict['Gamma'            ] = Gamma
__interFaceDict['Beta'             ] = Beta
__interFaceDict['Triangular'       ] = Triangular
__interFaceDict['Poisson'          ] = Poisson
__interFaceDict['Binomial'         ] = Binomial
__interFaceDict['Bernoulli'        ] = Bernoulli
__interFaceDict['Categorical'      ] = Categorical
__interFaceDict['Logistic'         ] = Logistic
__interFaceDict['Exponential'      ] = Exponential
__interFaceDict['LogNormal'        ] = LogNormal
__interFaceDict['Weibull'          ] = Weibull
__interFaceDict['NDInverseWeight'  ] = NDInverseWeight
__interFaceDict['NDCartesianSpline'] = NDCartesianSpline
__interFaceDict['MultivariateNormal'] = MultivariateNormal
__knownTypes                  = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  try: return __interFaceDict[Type]()
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
