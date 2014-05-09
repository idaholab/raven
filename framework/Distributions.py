'''
Created on Mar 7, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import numpy as np
from BaseType import BaseType
import scipy.special as polys
from scipy.misc import factorial
if sys.version_info.major > 2:
  import distribution1Dpy3 as distribution1D
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
    self.dimensionality   = None #Dimensionality of the distribution (1D or ND)
    
  def _readMoreXML(self,xmlNode):
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
    tempDict['dimensionality'] = self.dimensionality

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

def random():
  return stochasticEnv.random()

def randomSeed(value):
  return stochasticEnv.seedRandom(value)

def randomIntegers(low,high):
  int_range = high-low
  raw_num = low + random()*int_range
  raw_int = int(round(raw_num))
  if raw_int < low or raw_int > high:
    print("Random int out of range")
    raw_int = max(low,min(raw_int,high))
  return raw_int
  
def randomPermutation(l):
  new_list = []
  old_list = l[:]
  while len(old_list) > 0:
    new_list.append(old_list.pop(randomIntegers(0,len(old_list)-1)))
  return new_list

class BoostDistribution(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.dimensionality  = '1D'
  
  def cdf(self,x):
    return self._distribution.Cdf(x)

  def ppf(self,x):
    return self._distribution.InverseCdf(x)

  def pdf(self,x):
    return self._distribution.Pdf(x)
    
  def untruncatedCdfComplement(self, x):
    return self._distribution.untrCdfComplement(x)

  def untruncatedHazard(self, x):
    return self._distribution.untrHazard(x)

  def untruncatedMean(self):
    return self._distribution.untrMean()

  def untruncatedMedian(self):
    return self._distribution.untrMedian()

  def untruncatedMode(self):
    return self._distribution.untrMode()

  def rvs(self,*args):
    if len(args) == 0:
      return self.ppf(random())
    else:
      return [self.rvs() for _ in range(args[0])]
#==============================================================\
#    Distributions convenient for stochastic collocation
#==============================================================\

class Uniform(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.low = 0.0
    self.hi = 0.0
    self.type = 'Uniform'

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self,xmlNode)
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    else: raise Exception('low value needed for uniform distribution')
    hi_find = xmlNode.find('hi')
    high_find = xmlNode.find('high')
    if hi_find != None: self.hi = float(hi_find.text)
    elif high_find != None: self.hi = float(high_find.text)
    else: raise Exception('hi or high value needed for uniform distribution')
#    self.initializeDistribution() this call is done by the sampler each time a new step start
    self.range=self.hi-self.low
    # check if lower or upper bounds are set, otherwise default 
    if not self.upperBoundUsed: 
      self.upperBoundUsed = True
      self.upperBound     = self.hi
    if not self.lowerBoundUsed: 
      self.lowerBoundUsed = True
      self.lowerBound     = self.low 
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
    BoostDistribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    # no other additional parameters required

  def initializeDistribution(self):
    #self._distribution = dist.uniform(loc=self.low,scale=self.range)
    self._distribution = distribution1D.BasicUniformDistribution(self.low,self.low+self.range)

  # def cdf(self,x):
  #   value = super(Uniform,self).cdf(x) 
  #   print("Uniform CDF",x,value)
  #   return value

  # def ppf(self,x):
  #   value = super(Uniform,self).ppf(x)
  #   print("Uniform PPF",x,value)
  #   return value

  # def rvs(self,*args):
  #   value = super(Uniform,self).rvs(*args)
  #   print("Uniform RVS",value)
  #   return value


class Normal(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mean  = 0.0
    self.sigma = 0.0
    self.type = 'Normal'

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    mean_find = xmlNode.find('mean' )
    if mean_find != None: self.mean  = float(mean_find.text)
    else: raise Exception('mean value needed for normal distribution')
    sigma_find = xmlNode.find('sigma')
    if sigma_find != None: self.sigma = float(sigma_find.text)
    else: raise Exception('sigma value needed for normal distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma

  def initializeDistribution(self):
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

      def probNorm(_): #normalizes if total prob. != 1
        return 1.0

      self.poly_norm = norm
      self.actual_point = standardToActualPoint
      self.std_point = actualToStandardPoint
      self.actual_weight = standardToActualWeight
      self.probability_norm = probNorm
    else:
      print('FIXME: this should be removed.... :special case distribution for stochastic colocation')
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
    BoostDistribution.__init__(self)
    self.low = 0.0
    self.alpha = 0.0
    self.beta = 1.0
    self.type = 'Gamma'

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self,xmlNode)
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    else: raise Exception('low value needed for Gamma distribution')
    alpha_find = xmlNode.find('alpha')
    if alpha_find != None: self.alpha = float(alpha_find.text)
    else: raise Exception('alpha value needed for Gamma distribution')
    beta_find = xmlNode.find('beta')
    if beta_find != None: self.beta = float(beta_find.text)
    else: self.beta=1.0
    # check if lower bound are set, otherwise default 
    if not self.lowerBoundUsed: 
      self.lowerBoundUsed = True
      self.lowerBound     = self.low 
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self,tempDict)
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

    def probNorm(_): #normalizes probability if total != 1
      return 1.0

    self.poly_norm=norm
    self.actual_point = standardToActualPoint
    self.std_point = actualToStandardPoint
    self.actual_weight = standardToActualWeight
    self.probability_norm = probNorm

class Beta(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.low = 0.0
    self.hi = 0.0
    self.alpha = 0.0
    self.beta = 0.0
    self.type = 'Beta'
    print('FIXME: # TODO default to specific Beta distro?')
    # TODO default to specific Beta distro?

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self,xmlNode)
    low_find = xmlNode.find('low')
    if low_find != None: self.low = float(low_find.text)
    else: raise Exception('low value needed for Gamma distribution')
    hi_find = xmlNode.find('hi')
    high_find = xmlNode.find('high')
    if hi_find != None: self.hi = float(hi_find.text)
    elif high_find != None: self.hi = float(high_find.text)
    else: raise Exception('hi or high value needed for Gamma distribution')
    alpha_find = xmlNode.find('alpha')
    if alpha_find != None: self.alpha = float(alpha_find.text)
    else: raise Exception('alpha value needed for Gamma distribution')
    beta_find = xmlNode.find('beta')
    if beta_find != None: self.beta = float(beta_find.text)
    else: raise Exception('beta value needed for Gamma distribution')
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
    #self._distribution = dist.beta(self.alpha,self.beta,scale=self.hi-self.low)
    self._distribution = distribution1D.BasicBetaDistribution(self.alpha,self.beta,self.hi-self.low)

#==========================================================\
#    other distributions
#==========================================================\


# Add polynomials, shifting, zero-to-one to these!
class Triangular(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.apex = 0.0
    self.min  = 0.0
    self.max  = 0.0
    self.type = 'Triangular'

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    apex_find = xmlNode.find('apex')
    if apex_find != None: self.apex = float(apex_find.text)
    else: raise Exception('apex value needed for normal distribution')
    min_find = xmlNode.find('min')
    if min_find != None: self.min = float(min_find.text)
    else: raise Exception('min value needed for normal distribution')
    max_find = xmlNode.find('max')
    if max_find != None: self.max = float(max_find.text)
    else: raise Exception('max value needed for normal distribution')
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
      raise IOError ('Truncated triangular not yet implemented')
    
    
class Poisson(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mu  = 0.0
    self.type = 'Poisson'
    
  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    mu_find = xmlNode.find('mu')
    if mu_find != None: self.mu = float(mu_find.text)
    else: raise Exception('mu value needed for poisson distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['mu'  ] = self.mu
    
  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicPoissonDistribution(self.mu)
    else:
      raise IOError ('Truncated poisson not yet implemented')    
    
    
class Binomial(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.n  = 0.0
    self.p  = 0.0
    self.type = 'Binomial'
    
  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    n_find = xmlNode.find('n')
    if n_find != None: self.n = float(n_find.text)
    else: raise Exception('n value needed for Binomial distribution')
    p_find = xmlNode.find('p')
    if p_find != None: self.p = float(p_find.text)
    else: raise Exception('p value needed for Binomial distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['n'  ] = self.n
    tempDict['p'  ] = self.p
    
  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      #self._distribution = dist.binom(n=self.n,p=self.p)
      self._distribution = distribution1D.BasicBinomialDistribution(self.n,self.p)
    else:
      raise IOError ('Truncated Binomial not yet implemented')   

class Bernoulli(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.p  = 0.0
    self.type = 'Bernoulli'
    
  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    p_find = xmlNode.find('p')
    if p_find != None: self.p = float(p_find.text)
    else: raise Exception('p value needed for Bernoulli distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['p'  ] = self.p
    
  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicBernoulliDistribution(self.p)
    else:
      raise IOError ('Truncated Bernoulli not yet implemented')

class Logistic(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.location  = 0.0
    self.scale = 1.0
    self.type = 'Logistic'
    
  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    location_find = xmlNode.find('location')
    if location_find != None: self.location = float(location_find.text)
    else: raise Exception('location value needed for Logistic distribution')
    scale_find = xmlNode.find('scale')
    if scale_find != None: self.scale = float(scale_find.text)
    else: raise Exception('scale value needed for Logistic distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['location'] = self.location
    tempDict['scale'   ] = self.scale
    
  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicLogisticDistribution(self.location,self.scale)
    else:
      raise IOError ('Truncated Logistic not yet implemented')

class Exponential(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.lambda_var = 1.0
    self.type = 'Exponential'

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    lambda_find = xmlNode.find('lambda')
    if lambda_find != None: self.lambda_var = float(lambda_find.text)
    else: raise Exception('lambda value needed for Exponential distribution')
    # check if lower bound is set, otherwise default 
    if not self.lowerBoundUsed: 
      self.lowerBoundUsed = True
      self.lowerBound     = 0.0
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['lambda'] = self.lambda_var
    
  def initializeDistribution(self):
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False) or (self.lowerBound == 0.0):
      self._distribution = distribution1D.BasicExponentialDistribution(self.lambda_var)
    else:
      raise IOError ('Truncated Exponential not yet implemented')
    
class LogNormal(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.mean = 1.0
    self.sigma = 1.0
    self.type = 'LogNormal'

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    mean_find = xmlNode.find('mean')
    if mean_find != None: self.mean = float(mean_find.text)
    else: raise Exception('mean value needed for LogNormal distribution')
    sigma_find = xmlNode.find('sigma')
    if sigma_find != None: self.sigma = float(sigma_find.text)
    else: raise Exception('sigma value needed for LogNormal distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma
    
  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self._distribution = distribution1D.BasicLogNormalDistribution(self.mean,self.sigma)
    else:
      raise IOError ('Truncated LogNormal not yet implemented')
    
class Weibull(BoostDistribution):
  def __init__(self):
    BoostDistribution.__init__(self)
    self.lambda_var = 1.0
    self.k = 1.0
    self.type = 'Weibull'

  def _readMoreXML(self,xmlNode):
    BoostDistribution._readMoreXML(self, xmlNode)
    lambda_find = xmlNode.find('lambda')
    if lambda_find != None: self.lambda_var = float(lambda_find.text)
    else: raise Exception('lambda (scale) value needed for Weibull distribution')
    k_find = xmlNode.find('k')
    if k_find != None: self.k = float(k_find.text)
    else: raise Exception('k (shape) value needed for Weibull distribution')
    # check if lower  bound is set, otherwise default 
    if not self.lowerBoundUsed: 
      self.lowerBoundUsed = True
      # lower bound = 0 since no location parameter available
      self.lowerBound     = 0.0
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    BoostDistribution.addInitParams(self, tempDict)
    tempDict['lambda'] = self.lambda_var
    tempDict['k'     ] = self.k
    
  def initializeDistribution(self):
    if (self.lowerBoundUsed == False and self.upperBoundUsed == False) or self.lowerBound == 0.0:
      self._distribution = distribution1D.BasicWeibullDistribution(self.k,self.lambda_var)
    else:
      raise IOError ('Truncated Weibull not yet implemented')

class NDimensionalDistributions(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.data_filename = None
    self.function_type = None
    self.type = 'NDimensionalDistributions'
    self.dimensionality  = 'ND'
  def _readMoreXML(self,xmlNode):
    Distribution._readMoreXML(self, xmlNode)
    data_filename = xmlNode.find('data_filename')
    if data_filename != None: self.data_filename = data_filename.text
    else: raise Exception('<data_filename> parameter needed for MultiDimensional Distributions!!!!')
    function_type = xmlNode.find('function_type')
    if not function_type: self.function_type = 'CDF'
    else:
      self.function_type = function_type.upper()
      if self.function_type not in ['CDF','PDF']:  raise Exception('<function_type> parameter needs to be either CDF or PDF in MultiDimensional Distributions!!!!')
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['function_type'] = self.function_type
    tempDict['data_filename'] = self.data_filename
    
class NDInverseWeight(NDimensionalDistributions):
  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.p  = None
    self.type = 'NDInverseWeight'
    
  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)
    self.p = xmlNode.find('p')
    if self.p != None: self.p = float(self.p)
    else: raise Exception('Minkowski distance parameter <p> not found in NDInverseWeight distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)
    tempDict['p'] = self.p
    
  def initializeDistribution(self):
    NDimensionalDistributions.initializeDistribution()
    self._distribution = distribution1D.BasicMultiDimensionalInverseWeight(self.p)

  def cdf(self,x):
    return self._distribution.Cdf(x)

  def ppf(self,x):
    return self._distribution.InverseCdf(x)

  def pdf(self,x):
    return self._distribution.Pdf(x)
    
  def untruncatedCdfComplement(self, x):
    raise NotImplementedError('untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    raise NotImplementedError('untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    raise NotImplementedError('untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    raise NotImplementedError('untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    raise NotImplementedError('untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    raise NotImplementedError('rvs not yet implemented for ' + self.type)

class NDScatteredMS(NDimensionalDistributions):
  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.p  = None
    self.precision = None
    self.type = 'NDScatteredMS'
    
  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)
    self.p = xmlNode.find('p')
    if self.p != None: self.p = float(self.p)
    else: raise Exception('Minkowski distance parameter <p> not found in NDScatteredMS distribution')
    self.precision = xmlNode.find('precision')
    if self.precision != None: self.precision = float(self.precision)
    else: raise Exception('precision parameter <precision> not found in NDScatteredMS distribution')
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)
    tempDict['p'] = self.p
    tempDict['precision'] = self.precision
    
  def initializeDistribution(self):
    NDimensionalDistributions.initializeDistribution()
    self._distribution = distribution1D.BasicMultiDimensionalScatteredMS(self.p,self.precision)

  def cdf(self,x):
    return self._distribution.Cdf(x)

  def ppf(self,x):
    return self._distribution.InverseCdf(x)

  def pdf(self,x):
    return self._distribution.Pdf(x)
    
  def untruncatedCdfComplement(self, x):
    raise NotImplementedError('untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    raise NotImplementedError('untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    raise NotImplementedError('untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    raise NotImplementedError('untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    raise NotImplementedError('untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    raise NotImplementedError('rvs not yet implemented for ' + self.type)

class NDCartesianSpline(NDimensionalDistributions):
  def __init__(self):
    NDimensionalDistributions.__init__(self)
    self.type = 'NDCartesianSpline'
    
  def _readMoreXML(self,xmlNode):
    NDimensionalDistributions._readMoreXML(self, xmlNode)
    self.initializeDistribution()
    
  def addInitParams(self,tempDict):
    NDimensionalDistributions.addInitParams(self, tempDict)
    
  def initializeDistribution(self):
    NDimensionalDistributions.initializeDistribution()
    self._distribution = distribution1D.BasicMultiDimensionalCartesianSpline()

  def cdf(self,x):
    return self._distribution.Cdf(x)

  def ppf(self,x):
    return self._distribution.InverseCdf(x)

  def pdf(self,x):
    return self._distribution.Pdf(x)
    
  def untruncatedCdfComplement(self, x):
    raise NotImplementedError('untruncatedCdfComplement not yet implemented for ' + self.type)

  def untruncatedHazard(self, x):
    raise NotImplementedError('untruncatedHazard not yet implemented for ' + self.type)

  def untruncatedMean(self):
    raise NotImplementedError('untruncatedMean not yet implemented for ' + self.type)

  def untruncatedMedian(self):
    raise NotImplementedError('untruncatedMedian not yet implemented for ' + self.type)

  def untruncatedMode(self):
    raise NotImplementedError('untruncatedMode not yet implemented for ' + self.type)

  def rvs(self,*args):
    raise NotImplementedError('rvs not yet implemented for ' + self.type)


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
__knownTypes                  = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)  

