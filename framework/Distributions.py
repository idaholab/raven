'''
Created on Mar 7, 2013

@author: crisr
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import xml.etree.ElementTree as ET #used for creating Beta in Normal distribution
import copy
import numpy as np
import scipy
import scipy.special as polys
#from scipy.misc import factorial
from math import gamma
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
import utils
distribution1D = utils.find_distribution1D()
#Internal Modules End--------------------------------------------------------------------------------

def factorial(x):
  return gamma(x+1)

stochasticEnv = distribution1D.DistributionContainer.Instance()

'''
 Mapping between internal framework and Crow distribution name
'''
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
  '''
  a general class containing the distributions
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.upperBoundUsed       = False  # True if the distribution is right truncated
    self.lowerBoundUsed       = False  # True if the distribution is left truncated
    self.hasInfiniteBound     = False  # True if the untruncated distribution has bounds of +- system max
    self.upperBound           = 0.0  # Right bound
    self.lowerBound           = 0.0  # Left bound
    self.__adjustmentType     = '' # this describe how the re-normalization to preserve the probability should be done for truncated distributions
    self.dimensionality       = None # Dimensionality of the distribution (1D or ND)
    self.printTag             = utils.returnPrintTag('DISTRIBUTIONS')
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
    '''
    Readmore xml, see BaseType.py explaination.
    '''
    if xmlNode.find('upperBound') !=None:
      self.upperBound = float(xmlNode.find('upperBound').text)
      self.upperBoundUsed = True
    if xmlNode.find('lowerBound')!=None:
      self.lowerBound = float(xmlNode.find('lowerBound').text)
      self.lowerBoundUsed = True
    if xmlNode.find('adjustment') !=None: self.__adjustment = xmlNode.find('adjustment').text
    else: self.__adjustment = 'scaling'

  def getCrowDistDict(self):
    '''
    Returns a dictionary of the keys and values that would be
    used to create the distribution for a Crow input file.
    '''
    retDict = {}
    retDict['type'] = _FrameworkToCrowDistNames[self.type]
    if self.lowerBoundUsed:
      retDict['xMin'] = self.lowerBound
    if self.upperBoundUsed:
      retDict['xMax'] = self.upperBound
    return retDict

  def addInitParams(self,tempDict):
    '''
    Function to get the input params that belong to this class
    @ In, tempDict, temporary dictionary
    '''
    tempDict['upperBoundUsed'  ] = self.upperBoundUsed
    tempDict['lowerBoundUsed'  ] = self.lowerBoundUsed
    tempDict['hasInfiniteBound'] = self.hasInfiniteBound
    tempDict['upperBound'      ] = self.upperBound
    tempDict['lowerBound'      ] = self.lowerBound
    tempDict['adjustmentType'  ] = self.__adjustmentType
    tempDict['dimensionality'  ] = self.dimensionality

  def rvsWithinCDFbounds(self,LowerBound,upperBound):
    '''
    Function to get a random number from a truncated distribution
    @ In, LowerBound, float -> lower bound
    @ In, upperBound, float -> upper bound
    @ In,           , float -> random number
    '''
    point = float(np.random.rand(1))*(upperBound-LowerBound)+LowerBound
    return self._distribution.InverseCdf(point)

  def rvsWithinbounds(self,LowerBound,upperBound):
    '''
    Function to get a random number from a truncated distribution
    @ In, LowerBound, float -> lower bound
    @ In, upperBound, float -> upper bound
    @ Out,          , float -> random number
    '''
    CDFupper = self._distribution.Cdf(upperBound)
    CDFlower = self._distribution.Cdf(LowerBound)
    return self.rvsWithinCDFbounds(CDFlower,CDFupper)

  def convertToDistr(self,qtype,pts):
    '''Converts points from the quadrature "qtype" standard domain to the distribution domain.
    @ In qtype, string, type of quadrature to convert from
    @ In pts, array of floats, points to convert
    @ Out, array of floats, converted points
    '''
    return self.convertToDistrDict[qtype](pts)

  def convertToQuad(self,qtype,pts):
    '''Converts points from the distribution domain to the quadrature "qtype" standard domain.
    @ In qtype, string, type of quadrature to convert to
    @ In pts, array of floats, points to convert
    @ Out, array of floats, converted points
    '''
    return self.convertToQuadDict[qtype](pts)

  def measureNorm(self,qtype):
    '''Provides the integral/jacobian conversion factor between the distribution domain and the quadrature domain.
    @ In qtype, string, type of quadrature to convert to
    @ Out, float, conversion factor
    '''
    return self.measureNormDict[qtype]()

  def _convertDistrPointsToCdf(self,pts):
    '''Converts points in the distribution domain to [0,1].
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    '''
    try: return self.cdf(pts.real)
    except TypeError: return list(self.cdf(x) for x in pts)

  def _convertCdfPointsToDistr(self,pts):
    '''Converts points in [0,1] to the distribution domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    '''
    try: return self.ppf(pts.real)
    except TypeError: return list(self.ppf(x) for x in pts)

  def _convertCdfPointsToStd(self,pts):
    '''Converts points in [0,1] to [-1,1], the uniform distribution's STANDARD domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    '''
    try: return 2.0*pts.real-1.0
    except TypeError: return list(2.0*x-1.0 for x in pts)

  def _convertStdPointsToCdf(self,pts):
    '''Converts points in [-1,1] to [0,1] (CDF domain).
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    '''
    try: return 0.5*(pts.real+1.0)
    except TypeError: return list(0.5*(x+1.0) for x in pts)

  def CDFconvertToQuad(self,pts):
    '''Converts all the way from distribution domain to [-1,1] quadrature domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    '''
    return self._convertCdfPointsToStd(self._convertDistrPointsToCdf(pts))

  def CDFconvertToDistr(self,pts):
    '''Converts all the way from [-1,1] quadrature domain to distribution domain.
    @ In pts, array of floats, points to convert
    @ Out, float/array of floats, converted points
    '''
    return self._convertCdfPointsToDistr(self._convertStdPointsToCdf(pts))

  def CDFMeasureNorm(self):
    '''Integral norm/jacobian for [-1,1] Legendre quadrature.
    @ In None, None
    @ Out float, normalization factor
    '''
    return 1.0/2.0;


  def getDimensionality(self):
    return self.dimensionality


def random():
  '''
  Function to get a random number <1<
  @ In, None, None
  @ Out, float, random number
  '''
  return stochasticEnv.random()

def randomSeed(value):
  '''
  Function to get a random seed
  @ In, None, None
  @ Out, integer, random seed
  '''
  return stochasticEnv.seedRandom(value)

def randomIntegers(low,high):
  '''
  Function to get a random integer
  @ In, low, integer -> low boundary
  @ In, high, integer -> upper boundary
  @ Out, integer, random int
  '''
  int_range = high-low
  raw_num = low + random()*int_range
  raw_int = int(round(raw_num))
  if raw_int < low or raw_int > high:
    utils.raiseAMessage('DISTRIBUTIONS',"Random int out of range")
    raw_int = max(low,min(raw_int,high))
  return raw_int

def randomPermutation(l):
  '''
  Function to get a random permutation
  @ In, l, list -> list to be permuted
  @ Out, list, randomly permuted list
  '''
  new_list = []
  old_list = l[:]
  while len(old_list) > 0:
    new_list.append(old_list.pop(randomIntegers(0,len(old_list)-1)))
  return new_list

class BoostDistribution(Distribution):
  '''
  Base distribution class based on boost
  '''
  def __init__(self):
    Distribution.__init__(self)
    self.dimensionality  = 1
    self.disttype        = 'Continuous'

  def cdf(self,x):
    '''
    Function to get the cdf at a provided coordinate
    @ In, x, float -> value to get the cdf at
    @ Out, flaot, requested cdf
    '''
    return self._distribution.Cdf(x)

  def ppf(self,x):
    '''
    Function to get the inverse cdf at a provided coordinate
    @ In, x, float -> value to get the inverse cdf at
    @ Out, flaot, requested inverse cdf
    '''
    return self._distribution.InverseCdf(x)

  def pdf(self,x):
    '''
    Function to get the pdf at a provided coordinate
    @ In, x, float -> value to get the pdf at
    @ Out, flaot, requested pdf
    '''
    return self._distribution.Pdf(x)

  def untruncatedCdfComplement(self, x):
    '''
    Function to get the untruncated  cdf complement at a provided coordinate
    @ In, x, float -> value to get the untruncated  cdf complement  at
    @ Out, flaot, requested untruncated  cdf complement
    '''
    return self._distribution.untrCdfComplement(x)

  def untruncatedHazard(self, x):
    '''
    Function to get the untruncated  Hazard  at a provided coordinate
    @ In, x, float -> value to get the untruncated  Hazard   at
    @ Out, flaot, requested untruncated  Hazard
    '''
    return self._distribution.untrHazard(x)

  def untruncatedMean(self):
    '''
    Function to get the untruncated  Mean
    @ In, None
    @ Out, flaot, requested Mean
    '''
    return self._distribution.untrMean()

  def untruncatedMedian(self):
    '''
    Function to get the untruncated  Median
    @ In, None
    @ Out, flaot, requested Median
    '''
    return self._distribution.untrMedian()

  def untruncatedMode(self):
    '''
    Function to get the untruncated  Mode
    @ In, None
    @ Out, flaot, requested Mode
    '''
    return self._distribution.untrMode()


  def rvs(self,*args):
    '''
    Function to get random numbers
    @ In, args, dictionary, args
    @ Out, flaot or list, requested random number or numbers
    '''
    if len(args) == 0: return self.ppf(random())
    else             : return [self.rvs() for _ in range(args[0])]


class Uniform(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.range = 0.0
    self.type = 'Uniform'
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
      utils.raiseAnError(IOError,self,'the Uniform distribution needs both upperBound and lowerBound attributes. Got upperBound? '+ str(self.upperBoundUsed) + '. Got lowerBound? '+str(self.lowerBoundUsed))
    self.range = self.upperBound - self.lowerBound
    self.initializeDistribution()

  def stdProbabilityNorm(self):
    '''Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    '''
    return 0.5

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
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
    '''Converts from distribution domain to standard Legendre [-1,1].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    return (y-self.untruncatedMean())/(self.range/2.)

  def convertLegendreToUniform(self,x):
    '''Converts from standard Legendre [-1,1] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    return self.range/2.*x+self.untruncatedMean()



class Normal(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mean  = 0.0
    self.sigma = 0.0
    self.hasInfiniteBound = True
    self.type = 'Normal'
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
    mean_find = xmlNode.find('mean' )
    if mean_find != None: self.mean  = float(mean_find.text)
    else: utils.raiseAnError(IOError,self,'mean value needed for normal distribution')
    sigma_find = xmlNode.find('sigma')
    if sigma_find != None: self.sigma = float(sigma_find.text)
    else: utils.raiseAnError(IOError,self,'sigma value needed for normal distribution')
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
    '''Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    '''
    sv = str(scipy.__version__).split('.')
    if int(sv[0])==0 and int(sv[1])<15:
      return 1.0/np.sqrt(2.*np.pi)
    else:
      return 1.0/np.sqrt(np.pi/2.)

  def convertNormalToHermite(self,y):
    '''Converts from distribution domain to standard Hermite [-inf,inf].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    return (y-self.untruncatedMean())/(self.sigma)

  def convertHermiteToNormal(self,x):
    '''Converts from standard Hermite [-inf,inf] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    return self.sigma*x+self.untruncatedMean()



class Gamma(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.low = 0.0
    self.alpha = 0.0
    self.beta = 1.0
    self.type = 'Gamma'
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
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    alpha_find = xmlNode.find('alpha')
    if alpha_find != None: self.alpha = float(alpha_find.text)
    else: utils.raiseAnError(IOError,self,'alpha value needed for Gamma distribution')
    beta_find = xmlNode.find('beta')
    if beta_find != None: self.beta = float(beta_find.text)
    else: self.beta=1.0
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
    '''Converts from distribution domain to standard Laguerre [0,inf].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    return (y-self.low)*(self.beta)

  def convertLaguerreToGamma(self,x):
    '''Converts from standard Laguerre [0,inf] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    return x/self.beta+self.low

  def stdProbabilityNorm(self):
    '''Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    '''
    #return self.beta**self.alpha/factorial(self.alpha-1.)
    return 1./factorial(self.alpha-1)


class Beta(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.low = 0.0
    self.hi = 1.0
    self.alpha = 0.0
    self.beta = 0.0
    self.type = 'Beta'
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('Jacobi')
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'Jacobi'
    self.preferredPolynomials = 'Jacobi'

  def _localSetState(self,pdict):
    self.low   = pdict.pop('low'  )
    self.hi    = pdict.pop('hi'   )
    self.alpha = pdict.pop('alpha')
    self.beta  = pdict.pop('beta' )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['alpha'] = self.alpha
    retDict['beta'] = self.beta
    retDict['scale'] = self.hi-self.low
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self,xmlNode)
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    hi_find = xmlNode.find('hi')
    #high_find = xmlNode.find('high')
    if hi_find != None: self.hi = float(hi_find.text)
    #elif high_find != None: self.hi = float(high_find.text)
    else:
        if xmlNode.find('high') != None: self.hi = float(xmlNode.find('high').text)
    alpha_find = xmlNode.find('alpha')
    beta_find = xmlNode.find('beta')
    peak_find = xmlNode.find('peakFactor')
    if alpha_find != None and beta_find != None and peak_find == None:
      self.alpha = float(alpha_find.text)
      self.beta  = float(beta_find.text)
    elif (alpha_find == None and beta_find == None) and peak_find != None:
      peakFactor = float(peak_find.text)
      if not 0 <= peakFactor <= 1: utils.raiseAnError(IOError,self,'peakFactor must be from 0 to 1, inclusive!')
      #this empirical formula is used to make it so factor->alpha: 0->1, 0.5~7.5, 1->99
      self.alpha = 0.5*23.818**(5.*peakFactor/3.) + 0.5
      self.beta = self.alpha
    else:
      utils.raiseAnError(IOError,self,'Either provide (alpha and beta) or peakFactor!')
    # check if lower or upper bounds are set, otherwise default
    if not self.upperBoundUsed:
      self.upperBoundUsed = True
      self.upperBound     = self.hi
    if not self.lowerBoundUsed:
      self.lowerBoundUsed = True
      self.lowerBound     = self.low
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta

  def initializeDistribution(self):
    self.convertToDistrDict['Jacobi'] = self.convertJacobiToBeta
    self.convertToQuadDict ['Jacobi'] = self.convertBetaToJacobi
    self.measureNormDict   ['Jacobi'] = self.stdProbabilityNorm
    #this "if" section can only be called if distribution not generated using readMoreXML
    if (not self.upperBoundUsed) and (not self.lowerBoundUsed):
      self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.hi-self.low,self.low)
    else:
      if self.lowerBoundUsed == False: a = 0.0
      else:a = self.lowerBound
      if self.upperBoundUsed == False: b = sys.float_info.max
      else:b = self.upperBound
      self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.hi-self.low,a,b,self.low)
    self.preferredPolynomials = 'Jacobi'
    self.compatibleQuadrature.append('Jacobi')
    self.compatibleQuadrature.append('ClenshawCurtis')

  def convertBetaToJacobi(self,y):
    '''Converts from distribution domain to standard Beta [0,1].
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    u = 0.5*(self.hi+self.low)
    s = 0.5*(self.hi-self.low)
    return (y-u)/(s)

  def convertJacobiToBeta(self,x):
    '''Converts from standard Jacobi [0,1] to distribution domain.
    @ In y, float/array of floats, points to convert
    @ Out float/array of floats, converted points
    '''
    u = 0.5*(self.hi+self.low)
    s = 0.5*(self.hi-self.low)
    return s*x+u

  def stdProbabilityNorm(self):
    '''Returns the factor to scale error norm by so that norm(probability)=1.
    @ In None, None
    @ Out float, norm
    '''
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
    apex_find = xmlNode.find('apex')
    if apex_find != None: self.apex = float(apex_find.text)
    else: utils.raiseAnError(IOError,self,'apex value needed for normal distribution')
    min_find = xmlNode.find('min')
    if min_find != None: self.min = float(min_find.text)
    else: utils.raiseAnError(IOError,self,'min value needed for normal distribution')
    max_find = xmlNode.find('max')
    if max_find != None: self.max = float(max_find.text)
    else: utils.raiseAnError(IOError,self,'max value needed for normal distribution')
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
      utils.raiseAnError(IOError,self,'Truncated triangular not yet implemented')



class Poisson(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mu  = 0.0
    self.type = 'Poisson'
    self.hasInfiniteBound = True
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
    mu_find = xmlNode.find('mu')
    if mu_find != None: self.mu = float(mu_find.text)
    else: utils.raiseAnError(IOError,self,'mu value needed for poisson distribution')
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
      utils.raiseAnError(IOError,self,'Truncated poisson not yet implemented')


class Binomial(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.n       = 0.0
    self.p       = 0.0
    self.type     = 'Binomial'
    self.hasInfiniteBound = True
    self.disttype = 'Descrete'
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
    n_find = xmlNode.find('n')
    if n_find != None: self.n = float(n_find.text)
    else: utils.raiseAnError(IOError,self,'n value needed for Binomial distribution')
    p_find = xmlNode.find('p')
    if p_find != None: self.p = float(p_find.text)
    else: utils.raiseAnError(IOError,self,'p value needed for Binomial distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['n'  ] = self.n
    tempDict['p'  ] = self.p

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicBinomialDistribution(self.n,self.p)
    else: utils.raiseAnError(IOError,self,'Truncated Binomial not yet implemented')
#
#
#
class Bernoulli(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.p        = 0.0
    self.type     = 'Bernoulli'
    self.disttype = 'Descrete'
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
    p_find = xmlNode.find('p')
    if p_find != None: self.p = float(p_find.text)
    else: utils.raiseAnError(IOError,self,'p value needed for Bernoulli distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['p'  ] = self.p

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicBernoulliDistribution(self.p)
    else:  utils.raiseAnError(IOError,self,'Truncated Bernoulli not yet implemented')

  def cdf(self,x):
    if x <= 0.5: return self._distribution.Cdf(self.lowerBound)
    else       : return self._distribution.Cdf(self.upperBound)

  def pdf(self,x):
    if x <= 0.5: return self._distribution.Pdf(self.lowerBound)
    else       : return self._distribution.Pdf(self.upperBound)

  def untruncatedCdfComplement(self, x):
    if x <= 0.5: return self._distribution.untrCdfComplement(self.lowerBound)
    else       : return self._distribution.untrCdfComplement(self.upperBound)

  def untruncatedHazard(self, x):
    if x <= 0.5: return self._distribution.untrHazard(self.lowerBound)
    else       : return self._distribution.untrHazard(self.upperBound)
#
#
#
class Logistic(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.location  = 0.0
    self.scale = 1.0
    self.type = 'Logistic'
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
    location_find = xmlNode.find('location')
    if location_find != None: self.location = float(location_find.text)
    else: utils.raiseAnError(IOError,self,'location value needed for Logistic distribution')
    scale_find = xmlNode.find('scale')
    if scale_find != None: self.scale = float(scale_find.text)
    else: utils.raiseAnError(IOError,self,'scale value needed for Logistic distribution')
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
    self.lambda_var = 1.0
    self.low        = 0.0
    self.type = 'Exponential'
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    self.lambda_var = pdict.pop('lambda')
    self.low        = pdict.pop('low'   )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['lambda'] = self.lambda_var
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    lambda_find = xmlNode.find('lambda')
    if lambda_find != None: self.lambda_var = float(lambda_find.text)
    else: utils.raiseAnError(IOError,self,'lambda value needed for Exponential distribution')
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
    tempDict['lambda'] = self.lambda_var
    tempDict['low'] = self.low

  def initializeDistribution(self):
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False):
      self._distribution = distribution1D.BasicExponentialDistribution(self.lambda_var,self.low)
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
      self._distribution = distribution1D.BasicExponentialDistribution(self.lambda_var,a,b,self.low)

  def convertDistrPointsToStd(self,y):
    quad=self.quadratureSet()
    if quad.type=='Laguerre':
      return (y-self.low)*(self.lambda_var)
    else:
      return Distribution.convertDistrPointsToStd(self,y)

  def convertStdPointsToDistr(self,x):
    quad=self.quadratureSet()
    if quad.type=='Laguerre':
      return x/self.lambda_var+self.low
    else:
      return Distribution.convertStdPointsToDistr(self,x)


class LogNormal(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mean = 1.0
    self.sigma = 1.0
    self.low = 0.0
    self.type = 'LogNormal'
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
    mean_find = xmlNode.find('mean')
    if mean_find != None: self.mean = float(mean_find.text)
    else: utils.raiseAnError(IOError,self,'mean value needed for LogNormal distribution')
    sigma_find = xmlNode.find('sigma')
    if sigma_find != None: self.sigma = float(sigma_find.text)
    else: utils.raiseAnError(IOError,self,'sigma value needed for LogNormal distribution')
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
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
    self.lambda_var = 1.0
    self.k = 1.0
    self.type = 'Weibull'
    self.low = 0.0
    self.hasInfiniteBound = True
    self.compatibleQuadrature.append('CDF')
    self.preferredQuadrature  = 'CDF'
    self.preferredPolynomials = 'CDF'

  def _localSetState(self,pdict):
    self.lambda_var = pdict.pop('lambda')
    self.k          = pdict.pop('k'     )

  def getCrowDistDict(self):
    retDict = Distribution.getCrowDistDict(self)
    retDict['lambda'] = self.lambda_var
    retDict['k'] = self.k
    retDict['low'] = self.low
    return retDict

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    lambda_find = xmlNode.find('lambda')
    if lambda_find != None: self.lambda_var = float(lambda_find.text)
    else: utils.raiseAnError(IOError,self,'lambda (scale) value needed for Weibull distribution')
    k_find = xmlNode.find('k')
    if k_find != None: self.k = float(k_find.text)
    else: utils.raiseAnError(IOError,self,'k (shape) value needed for Weibull distribution')
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    else: self.low = 0.0
    # check if lower  bound is set, otherwise default
    #if not self.lowerBoundUsed:
    #  self.lowerBoundUsed = True
    #  # lower bound = 0 since no location parameter available
    #  self.lowerBound     = 0.0
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['lambda'] = self.lambda_var
    tempDict['k'     ] = self.k
    tempDict['low'   ] = self.low

  def initializeDistribution(self):
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False): # or self.lowerBound == 0.0:
      self._distribution = distribution1D.BasicWeibullDistribution(self.k,self.lambda_var,self.low)
    else:
      if self.lowerBoundUsed == False:
        a = self.low
        self.lowerBound = a
      else:a = self.lowerBound
      if self.upperBoundUsed == False:
        b = sys.float_info.max
        self.upperBound = b
      else:b = self.upperBound
      self._distribution = distribution1D.BasicWeibullDistribution(self.k,self.lambda_var,a,b,self.low)



class NDimensionalDistributions(Distribution):

  def __init__(self):
    Distribution.__init__(self)
    self.data_filename = None
    self.function_type = None
    self.type = 'NDimensionalDistributions'
    self.dimensionality  = None

    self.RNGInitDisc = 10
    self.RNGtolerance = 0.1

  def _readMoreXML(self,xmlNode):
    Distribution._readMoreXML(self, xmlNode)
    working_dir = xmlNode.find('working_dir')
    if working_dir != None: self.working_dir = working_dir.text
    '''
    data_filename = xmlNode.find('data_filename')
    if data_filename != None: self.data_filename = self.working_dir+data_filename.text
    else: raisea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> <data_filename> parameter needed for MultiDimensional Distributions!!!!')

    function_type = xmlNode.find('function_type')
    if not function_type: self.function_type = 'CDF'
    else:
      self.function_type = function_type.upper()
      if self.function_type not in ['CDF','PDF']:  raisea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> <function_type> parameter needs to be either CDF or PDF in MultiDimensional Distributions!!!!')
    '''
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['function_type'] = self.function_type
    tempDict['data_filename'] = self.data_filename

  #######
  def updateRNGParam(self, dictParam):
    self.RNGtolerance = 0.1
    self.RNGInitDisc  = 10
    for key in dictParam:
      if key == 'tolerance':
        self.RNGtolerance = dictParam['tolerance']
      elif key == 'initial_grid_disc':
        self.RNGInitDisc  = dictParam['initial_grid_disc']
    self._distribution.updateRNGparameter(self.RNGtolerance,self.RNGInitDisc)
  ######

  def getDimensionality(self):
    return  self._distribution.returnDimensionality()

class NDInverseWeight(NDimensionalDistributions):

  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.p  = None
    self.type = 'NDInverseWeight'

  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)
    p_find = xmlNode.find('p')
    if p_find != None: self.p = float(p_find.text)
    else: utils.raiseAnError(IOError,self,'Minkowski distance parameter <p> not found in NDInverseWeight distribution')

    data_filename = xmlNode.find('data_filename')
    if data_filename != None: self.data_filename = os.path.join(self.working_dir,data_filename.text)
    else: utils.raiseAnError(IOError,self,'<data_filename> parameter needed for MultiDimensional Distributions!!!!')

    function_type = data_filename.attrib['type']
    if function_type != None: self.function_type = function_type
    else: utils.raiseAnError(IOError,self,'<function_type> parameter needed for MultiDimensional Distributions!!!!')

    self.initializeDistribution()

  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)
    tempDict['p'] = self.p

  def initializeDistribution(self):
    if self.function_type == 'CDF':
      self._distribution = distribution1D.BasicMultiDimensionalInverseWeight(str(self.data_filename), self.p,True)
    else:
      self._distribution = distribution1D.BasicMultiDimensionalInverseWeight(str(self.data_filename), self.p,False)

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
      utils.raiseAnError(ValueError,self,'NDInverseWeight: inverseMarginalDistribution(x) with x outside [0.0,1.0]')

  def untruncatedCdfComplement(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    return self._distribution.InverseCdf(random(),random())


class NDCartesianSpline(NDimensionalDistributions):
  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.type = 'NDCartesianSpline'

  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)

    data_filename = xmlNode.find('data_filename')
    if data_filename != None: self.data_filename = os.path.join(self.working_dir,data_filename.text)
    else: utils.raiseAnError(IOError,self,'<data_filename> parameter needed for MultiDimensional Distributions!!!!')

    function_type = data_filename.attrib['type']
    if function_type != None: self.function_type = function_type
    else: utils.raiseAnError(IOError,self,'<function_type> parameter needed for MultiDimensional Distributions!!!!')

    self.initializeDistribution()

  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)

  def initializeDistribution(self):
    utils.raiseAMessage(self,'====== BasicMultiDimensional NDCartesianSpline initialize Distribution ======')
    if self.function_type == 'CDF':
      self._distribution = distribution1D.BasicMultiDimensionalCartesianSpline(str(self.data_filename),True)
    else:
      self._distribution = distribution1D.BasicMultiDimensionalCartesianSpline(str(self.data_filename),False)

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
      utils.raiseAnError(ValueError,self,'NDInverseWeight: inverseMarginalDistribution(x) with x outside [0.0,1.0]')

  def untruncatedCdfComplement(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    return self._distribution.InverseCdf(random(),random())

class NDScatteredMS(NDimensionalDistributions):
  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.p  = None
    self.precision = None
    self.type = 'NDScatteredMS'

  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)
    p_find = xmlNode.find('p')
    if p_find != None: self.p = float(p_find.text)
    else: utils.raiseAnError(IOError,self,'Minkowski distance parameter <p> not found in NDScatteredMS distribution')
    precision_find = xmlNode.find('precision')
    if precision_find != None: self.precision = int(precision_find.text)
    else: utils.raiseAnError(IOError,self,'precision parameter <precision> not found in NDScatteredMS distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)
    tempDict['p'] = self.p
    tempDict['precision'] = self.precision

  def initializeDistribution(self):
    #NDimensionalDistributions.initializeDistribution()
    self._distribution = distribution1D.BasicMultiDimensionalScatteredMS(self.p,self.precision)

  def cdf(self,x):
    return self._distribution.Cdf(x)

  def ppf(self,x):
    return self._distribution.InverseCdf(x)

  def pdf(self,x):
    return self._distribution.Pdf(x)

  def untruncatedCdfComplement(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    utils.raiseAnError(NotImplementedError,self,'rvs not yet implemented for ' + self.type)


class MultivariateNormal(NDimensionalDistributions):

  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.type = 'MultivariateNormal'
    self.mu  = None

  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)

    data_filename = xmlNode.find('data_filename')
    if data_filename != None: self.data_filename = self.working_dir+data_filename.text
    else: utils.raiseAnError(IOError,self,'<data_filename> parameter needed for MultivariateNormal Distributions!!!!')

    mu = xmlNode.find('mu')
    if data_filename != None: self.mu = [float(i) for i in mu.text.split()]
    else: utils.raiseAnError(IOError,self,'<mu> parameter needed for MultivariateNormal Distributions!!!!')

    self.initializeDistribution()

  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)

  def initializeDistribution(self):
    utils.raiseAMessage(self,'====== BasicMultiDimensional MultivariateNormal initialize distribution ======')
    mu = distribution1D.vectord_cxx(len(self.mu))
    for i in range(len(self.mu)):
      mu[i] = self.mu[i]
    self._distribution = distribution1D.BasicMultivariateNormal(str(self.data_filename), mu)

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
      utils.raiseAnError(ValueError,self,'NDInverseWeight: inverseMarginalDistribution(x) with x outside [0.0,1.0]')

  def untruncatedCdfComplement(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self, x):
    utils.raiseAnError(NotImplementedError,self,'untruncatedMode not yet implemented for ' + self.type)

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
__interFaceDict['Logistic'         ] = Logistic
__interFaceDict['Exponential'      ] = Exponential
__interFaceDict['LogNormal'        ] = LogNormal
__interFaceDict['Weibull'          ] = Weibull
__interFaceDict['NDInverseWeight'  ] = NDInverseWeight
__interFaceDict['NDScatteredMS'    ] = NDScatteredMS
__interFaceDict['NDCartesianSpline'] = NDCartesianSpline
__interFaceDict['MultivariateNormal'] = MultivariateNormal
__knownTypes                  = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: utils.raiseAnError(NameError,'DISTRIBUTIONS','not known '+__base+' type '+Type)
