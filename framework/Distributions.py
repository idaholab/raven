'''
Created on Mar 7, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import scipy.stats.distributions  as dist
import numpy as np
from BaseType import BaseType
import scipy.special as polys
from scipy.misc import factorial
import Quadrature
if sys.version_info.major > 2:
  import distribution1D
else:
  import distribution1Dpy2 as distribution1D

stochasticEnv = distribution1D.DistributionContainer.Instance()

class Distribution(BaseType):
  ''' 
  a general class containing the distributions
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.upperBoundUsed = False  #True if the distribution is right truncated
    self.lowerBoundUsed = False  #True if the distribution is left truncated
    self.upperBound       = 0.0  #Right bound
    self.lowerBound       = 0.0  #Left bound
    self.__adjustmentType   = ''   #this describe how the re-normalization to preserve the probability should be done for truncated distributions
    self.bestQuad         = None #the quadrature that best integrate the distribution
    
  def readMoreXML(self,xmlNode):
    #this is part of the stochastic collocation sampler not of the distribution!!!!!!
    #FIXME
    try:
      QuadType = xmlNode.attrib['ExpansionQuadrature']
      self.bestQuad = Quadrature.returnInstance(QuadType)
    except KeyError:
      pass
    if xmlNode.find('upperBound') !=None:
      self.upperBound = float(xmlNode.find('upperBound').text)
      self.upperBoundUsed = True
    if xmlNode.find('lowerBound')!=None:
      self.lowerBound = float(xmlNode.find('lowerBound').text)
      self.lowerBoundUsed = True
    if xmlNode.find('adjustment') !=None:
      self.__adjustment = xmlNode.find('adjustment').text
    else:
      self.__adjustment = 'scaling'

  def addInitParams(self,tempDict):
    tempDict['upperBoundUsed'] = self.upperBoundUsed
    tempDict['lowerBoundUsed'] = self.lowerBoundUsed
    tempDict['upperBound'    ] = self.upperBound
    tempDict['lowerBound'    ] = self.lowerBound
    tempDict['adjustmentType'] = self.__adjustmentType
    tempDict['bestQuad'      ] = self.bestQuad

  def rvsWithinCDFbounds(self,LowerBound,upperBound):
    point = np.random.rand(1)*(upperBound-LowerBound)+LowerBound
    return self._distribution.ppf(point)

  def rvsWithinbounds(self,LowerBound,upperBound):
    CDFupper = self._distribution.cdf(upperBound)
    CDFlower = self._distribution.cdf(LowerBound)
    return self.rvsWithinCDFbounds(CDFlower,CDFupper)

  def setQuad(self,quad,exp_order):
    self.__distQuad=quad
    self.__exp_order=exp_order

  def quad(self):
    try: return self.__distQuad
    except AttributeError: raise IOError ('No quadrature has been set for this distr. yet.')

  def polyOrder(self):
    try: return self.__exp_order
    except AttributeError: raise IOError ('Quadrature has not been set for this distr. yet.')

class SKDistribution(Distribution):

  def cdf(self,*args):
    """Cumulative Distribution Function"""
    return self._distribution.cdf(*args)

  def ppf(self,*args):
    """Percent Point Function (Inverse of CDF)"""
    return self._distribution.ppf(*args)

  def rvs(self,*args):
    """Random Variates"""
    return self._distribution.rvs(*args)

class BoostDistribution(Distribution):
  def cdf(self,x):
    return self._distribution.Cdf(x)

  def ppf(self,x):
    return self._distribution.RandomNumberGenerator(x)

  def rvs(self,*args):
    if len(args) == 0:
      return self.ppf(stochasticEnv.random())
    else:
      return [self.rvs() for x in range(args[0])]


#==============================================================\
#    Distributions convenient for stochastic collocation
#==============================================================\

class Uniform(BoostDistribution):
  def __init__(self):
    Distribution.__init__(self)
    self.low = 0.0
    self.hi = 0.0
    self.type = 'Uniform'
    self.bestQuad = Quadrature.Legendre

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    else: raise Exception('low value needed for uniform distribution')
    hi_find = xmlNode.find('hi')
    if hi_find != None: self.hi = float(hi_find.text)
    else: raise Exception('hi value needed for uniform distribution')
#    self.initializeDistribution() this call is done by the sampler each time a new step start
    self.range=self.hi-self.low
    #assign associated polynomial types
    self.polynomial = polys.legendre
    #define functions locally, then point to them
    def norm(n):
      '''Returns normalization constant for polynomial type, given the poly ordeir'''
      return np.sqrt((2.*n+1.)/2.)

    def standardToActualPoint(x): #standard -> actual
      '''Given a [-1,1] point, converts to parameter value.'''
      return x*self.range/2.+self._distribution.mean()

    def actualToStandardPoint(x): #actual -> standard
      '''Given a parameter value, converts to [-1,1] point.'''
      return (x-self._distribution.mean())/(self.range/2.)

    def standardToActualWeight(x): #standard -> actual
      '''Given normal quadrature weight, returns adjusted weight.'''
      return x/(self.range/2.)

    def probNorm(x): #normalizes probability if total != 1
      '''Returns the poly factor to scale by so that sum(probability)=1.'''
      return self.range

    # point to functions
    self.poly_norm = norm
    self.actual_point = standardToActualPoint
    self.std_point = actualToStandardPoint
    self.actual_weight = standardToActualWeight
    self.probability_norm = probNorm



  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    # no other additional parameters required

  def initializeDistribution(self):
    #self._distribution = dist.uniform(loc=self.low,scale=self.range)
    self._distribution = distribution1D.BasicUniformDistribution(self.low,self.low+self.range)


class Normal(BoostDistribution):
  def __init__(self):
    Distribution.__init__(self)
    self.mean  = 0.0
    self.sigma = 0.0
    self.type = 'Normal'
    self.bestQuad = Quadrature.StatHermite

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    mean_find = xmlNode.find('mean' )
    if mean_find != None: self.mean  = float(mean_find.text)
    else: raise Exception('mean value needed for normal distribution')
    sigma_find = xmlNode.find('sigma')
    if sigma_find != None: self.sigma = float(sigma_find.text)
    else: raise Exception('sigma value needed for normal distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma

  def initializeDistribution(self):
    print("initialize",self)
    if (not self.upperBoundUsed) and (not self.lowerBoundUsed):
      #self._distribution = dist.norm(loc=self.mean,scale=self.sigma)
      self._distribution = distribution1D.BasicNormalDistribution(self.mean,
                                                                  self.sigma)
      self.polynomial = polys.hermitenorm
      def norm(n):
        return (np.sqrt(np.sqrt(2.*np.pi)*factorial(n)))**(-1)

      def standardToActualPoint(x): #standard -> actual
        return x*self.sigma**2/2.+self._distribution.mean()

      def actualToStandardPoint(x): #actual -> standard
        return (x-self._distribution.mean())/(self.sigma**2/2.)

      def standardToActualWeight(x): #standard -> actual
        return x/(self.sigma**2/2.)

      def probNorm(x): #normalizes if total prob. != 1
        return 1.0

      self.poly_norm = norm
      self.actual_point = standardToActualPoint
      self.std_point = actualToStandardPoint
      self.actual_weight = standardToActualWeight
      self.probability_norm = probNorm
    else:
      #print("truncnorm")
      #FIXME special case distribution for stoch collocation
      if self.lowerBoundUsed == False: a = -sys.float_info[max]
      else:a = self.lowerBound
      #else:a = (self.lowerBound - self.mean) / self.sigma
      if self.upperBoundUsed == False: b = sys.float_info[max]
      else:b = self.upperBound
      #else:b = (self.upperBound - self.mean) / self.sigma
      self._distribution = distribution1D.BasicNormalDistribution(self.mean,
                                                                  self.sigma,
                                                                  a,b)
      #self._distribution = dist.truncnorm(a,b,loc=self.mean,scale=self.sigma)
    
class Gamma(BoostDistribution):
  def __init__(self):
    Distribution.__init__(self)
    self.low = 0.0
    self.alpha = 0.0
    self.beta = 1.0
    self.type = 'Gamma'
    self.bestQuad = Quadrature.Laguerre

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    else: raise Exception('low value needed for Gamma distribution')
    alpha_find = xmlNode.find('alpha')
    if alpha_find != None: self.alpha = float(alpha_find.text)
    else: raise Exception('alpha value needed for Gamma distribution')
    beta_find = xmlNode.find('beta')
    if beta_find != None: self.beta = float(beta_find.text)
    else: self.beta=1.0
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta

  def initializeDistribution(self):
    #self._distribution = dist.gamma(self.alpha,loc=self.low,scale=self.beta)
    self._distribution = distribution1D.BasicGammaDistribution(self.alpha,1.0/self.beta,self.low)
    self.polynomial = polys.genlaguerre
    def norm(n):
      return np.sqrt(factorial(n)/polys.gamma(n+self.alpha+1.0))

    def standardToActualPoint(x): #standard -> actual
      return x/self.alpha+self.alpha+self.low #TODO these correct? no beta used

    def actualToStandardPoint(x): #actual -> standard
      return (x-self.low-self.alpha)*self.alpha

    def standardToActualWeight(x): #standard -> actual
      return x

    def probNorm(x): #normalizes probability if total != 1
      return 1.0

    self.poly_norm=norm
    self.actual_point = standardToActualPoint
    self.std_point = actualToStandardPoint
    self.actual_weight = standardToActualWeight
    self.probability_norm = probNorm

class Beta(BoostDistribution):
  def __init__(self):
    Distribution.__init__(self)
    self.low = 0.0
    self.hi = 0.0
    self.alpha = 0.0
    self.beta = 0.0
    self.type = 'Beta'
    self.bestQuad = Quadrature.Jacobi
    # TODO default to specific Beta distro?

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    else: raise Exception('low value needed for Gamma distribution')
    hi_find = xmlNode.find('hi')
    if hi_find != None: self.hi = float(hi_find.text)
    else: raise Exception('hi value needed for Gamma distribution')
    alpha_find = xmlNode.find('alpha')
    if alpha_find != None: self.alpha = float(alpha_find.text)
    else: raise Exception('alpha value needed for Gamma distribution')
    beta_find = xmlNode.find('beta')
    if beta_find != None: self.beta = float(beta_find.text)
    else: raise Exception('beta value needed for Gamma distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta

  def initializeDistribution(self):
    #self._distribution = dist.beta(self.alpha,self.beta,scale=self.hi-self.low)
    self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.hi-self.low)

#==========================================================\
#    other distributions
#==========================================================\


# Add polynomials, shifting, zero-to-one to these!
class Triangular(BoostDistribution):
  def __init__(self):
    Distribution.__init__(self)
    self.apex = 0.0
    self.min  = 0.0
    self.max  = 0.0
    self.type = 'Triangular'
    self.bestQuad = None

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    apex_find = xmlNode.find('apex')
    if apex_find != None: self.apex = float(apex_find.text)
    else: raise Exception('apex value needed for normal distribution')
    min_find = xmlNode.find('min')
    if min_find != None: self.min = float(min_find.text)
    else: raise Exception('min value needed for normal distribution')
    max_find = xmlNode.find('max')
    if max_find != None: self.max = float(max_find.text)
    else: raise Exception('max value needed for normal distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['apex' ] = self.apex
    tempDict['min'  ] = self.min
    tempDict['max'  ] = self.max

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      #c = (self.apex-self.min)/(self.max-self.min)
      #self._distribution = dist.triang(c,loc=self.min,scale=(self.max-self.min))
      self._distribution = distribution1D.BasicTriangularDistribution(self.apex,self.min,self.max)
    else:
      raise IOError ('Truncated triangular not yet implemented')
    
    
class Poisson(BoostDistribution):
  def __init__(self):
    Distribution.__init__(self)
    self.mu  = 0.0
    self.type = 'Poisson'
    
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    mu_find = xmlNode.find('mu')
    if mu_find != None: self.mu = float(mu_find.text)
    else: raise Exception('mu value needed for poisson distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['mu'  ] = self.mu
    
  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      #self._distribution = dist.poisson(self.mu)
      self._distribution = distribution1D.BasicPoissonDistribution(self.mu)
    else:
      raise IOError ('Truncated poisson not yet implemented')    
    
    
class Binomial(BoostDistribution):
  def __init__(self):
    Distribution.__init__(self)
    self.n  = 0.0
    self.p  = 0.0
    self.type = 'Binomial'
    
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    n_find = xmlNode.find('n')
    if n_find != None: self.n = float(n_find.text)
    else: raise Exception('n value needed for Binomial distribution')
    p_find = xmlNode.find('p')
    if p_find != None: self.p = float(p_find.text)
    else: raise Exception('p value needed for Binomial distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['n'  ] = self.n
    tempDict['p'  ] = self.p
    
  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      #self._distribution = dist.binom(n=self.n,p=self.p)
      self._distribution = distribution1D.BasicBinomialDistribution(self.n,self.p)
    else:
      raise IOError ('Truncated Binomial not yet implemented')   



__base                        = 'Distribution'
__interFaceDict               = {}
__interFaceDict['Uniform'   ] = Uniform
__interFaceDict['Normal'    ] = Normal
__interFaceDict['Gamma'     ] = Gamma
__interFaceDict['Beta'      ] = Beta
__interFaceDict['Triangular'] = Triangular
__interFaceDict['Poisson'   ] = Poisson
__interFaceDict['Binomial'  ] = Binomial
__knownTypes                  = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  

