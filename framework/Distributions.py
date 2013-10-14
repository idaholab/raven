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
    self.adjustmentType   = ''   #this describe how the re-normalization to preserve the probability should be done for truncated distributions
    self.bestQuad         = None #the quadrature that best integrate the distribution
    
  def readMoreXML(self,xmlNode):
    #this is part of the stochastic collocation sampler not of the distribution!!!!!!
    #FIXME
    try:
      QuadType = xmlNode.attrib['ExpansionQuadrature']
      self.bestQuad = Quadrature.returnInstance(QuadType)
    except:
      pass
    if xmlNode.find('upperBound') !=None:
      self.upperBound = float(xmlNode.find('upperBound').text)
      self.upperBoundUsed = True
    if xmlNode.find('lowerBound')!=None:
      self.lowerBound = float(xmlNode.find('lowerBound').text)
      self.lowerBoundUsed = True
    if xmlNode.find('adjustment') !=None:
      self.adjustment = xmlNode.find('adjustment').text
    else:
      self.adjustment = 'scaling'

  def addInitParams(self,tempDict):
    tempDict['upperBoundUsed'] = self.upperBoundUsed
    tempDict['lowerBoundUsed'] = self.lowerBoundUsed
    tempDict['upperBound'    ] = self.upperBound
    tempDict['lowerBound'    ] = self.lowerBound
    tempDict['adjustmentType'] = self.adjustmentType
    tempDict['bestQuad'      ] = self.bestQuad

  def rvsWithinCDFbounds(self,LowerBound,upperBound):
    point = np.random.rand(1)*(upperBound-LowerBound)+LowerBound
    return self.distribution.ppf(point)

  def rvsWithinbounds(self,LowerBound,upperBound):
    CDFupper = self.distribution.cdf(upperBound)
    CDFlower = self.distribution.cdf(LowerBound)
    return self.rvsWithinCDFbounds(CDFlower,CDFupper)

  def setQuad(self,quad,exp_order):
    self.distQuad=quad
    self.exp_order=exp_order

  def quad(self):
    try: return self.distQuad
    except: raise IOError ('No quadrature has been set for this distr. yet.')

  def polyOrder(self):
    try: return self.exp_order
    except: raise IOError ('Quadrature has not been set for this distr. yet.')

#==============================================================\
#    Distributions convenient for stochastic collocation
#==============================================================\

class Uniform(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.low = 0.0
    self.hi = 0.0
    self.type = 'Uniform'
    self.bestQuad = Quadrature.Legendre

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    try: self.low = float(xmlNode.find('low').text)
    except: raise Exception('low value needed for uniform distribution')
    try: self.hi = float(xmlNode.find('hi').text)
    except: raise Exception('hi value needed for uniform distribution')
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
      return x*self.range/2.+self.distribution.mean()

    def actualToStandardPoint(x): #actual -> standard
      '''Given a parameter value, converts to [-1,1] point.'''
      return (x-self.distribution.mean())/(self.range/2.)

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
    self.distribution = dist.uniform(loc=self.low,scale=self.range)



class Normal(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.mean  = 0.0
    self.sigma = 0.0
    self.type = 'Normal'
    self.bestQuad = Quadrature.StatHermite

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.mean  = float(xmlNode.find('mean' ).text)
    except: raise Exception('mean value needed for normal distribution')
    try: self.sigma = float(xmlNode.find('sigma').text)
    except: raise Exception('sigma value needed for normal distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma

  def initializeDistribution(self):
    if (not self.upperBoundUsed) and (not self.lowerBoundUsed):
      self.distribution = dist.norm(loc=self.mean,scale=self.sigma)
      self.polynomial = polys.hermitenorm
      def norm(n):
        return (np.sqrt(np.sqrt(2.*np.pi)*factorial(n)))**(-1)

      def standardToActualPoint(x): #standard -> actual
        return x*self.sigma**2/2.+self.distribution.mean()

      def actualToStandardPoint(x): #actual -> standard
        return (x-self.distribution.mean())/(self.sigma**2/2.)

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
      #FIXME special case distribution for stoch collocation
      if self.lowerBoundUsed == False: a = -sys.float_info[max]
      else:a = (self.lowerBound - self.mean) / self.sigma
      if self.upperBoundUsed == False: b = sys.float_info[max]
      else:b = (self.upperBound - self.mean) / self.sigma
      self.distribution = dist.truncnorm(a,b,loc=self.mean,scale=self.sigma)
    
class Gamma(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.low = 0.0
    self.alpha = 0.0
    self.beta = 1.0
    self.type = 'Gamma'
    self.bestQuad = Quadrature.Laguerre

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    try: self.low = float(xmlNode.find('low').text)
    except: raise Exception('low value needed for Gamma distribution')
    try: self.alpha = float(xmlNode.find('alpha').text)
    except: raise Exception('alpha value needed for Gamma distribution')
    try: self.beta = float(xmlNode.find('beta').text)
    except: self.beta=1.0
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta

  def initializeDistribution(self):
    self.distribution = dist.gamma(self.alpha,loc=self.low,scale=self.beta)
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

class Beta(Distribution):
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
    try: self.low = float(xmlNode.find('low').text)
    except: raise Exception('low value needed for Gamma distribution')
    try: self.hi = float(xmlNode.find('hi').text)
    except: raise Exception('hi value needed for Gamma distribution')
    try: self.alpha = float(xmlNode.find('alpha').text)
    except: raise Exception('alpha value needed for Gamma distribution')
    try: self.beta = float(xmlNode.find('beta').text)
    except: raise Exception('beta value needed for Gamma distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta

  def initializeDistribution(self):
    self.distribution = dist.beta(self.alpha,self.beta,scale=self.hi-self.low)

#==========================================================\
#    other distributions
#==========================================================\


# Add polynomials, shifting, zero-to-one to these!
class Custom(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.apex = 0.0
    self.min  = 0.0
    self.max  = 0.0
    self.type = 'Triangular'
    self.bestQuad = None

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.apex = float(xmlNode.find('apex').text)
    except: raise Exception('apex value needed for normal distribution')
    try: self.min = float(xmlNode.find('min').text)
    except: raise Exception('min value needed for normal distribution')
    try: self.max = float(xmlNode.find('max').text)
    except: raise Exception('max value needed for normal distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['apex' ] = self.apex
    tempDict['min'  ] = self.min
    tempDict['max'  ] = self.max

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      c = (self.apex-self.min)/(self.max-self.min)
      self.distribution = dist.triang(c,loc=self.min,scale=(self.max-self.min))
    else:
      raise IOError ('Truncated triangular not yet implemented')

#
#
#
class Triangular(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.apex = 0.0
    self.min  = 0.0
    self.max  = 0.0
    self.type = 'Triangular'
    self.bestQuad = None

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.apex = float(xmlNode.find('apex').text)
    except: raise Exception('apex value needed for normal distribution')
    try: self.min = float(xmlNode.find('min').text)
    except: raise Exception('min value needed for normal distribution')
    try: self.max = float(xmlNode.find('max').text)
    except: raise Exception('max value needed for normal distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['apex' ] = self.apex
    tempDict['min'  ] = self.min
    tempDict['max'  ] = self.max

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      c = (self.apex-self.min)/(self.max-self.min)
      self.distribution = dist.triang(c,loc=self.min,scale=(self.max-self.min))
    else:
      raise IOError ('Truncated triangular not yet implemented')

def returnInstance(Type):
  base = 'Distribution'
  InterfaceDict = {}
  InterfaceDict['Uniform'  ]  = Uniform
  InterfaceDict['Normal'   ]  = Normal
  InterfaceDict['Gamma'    ]  = Gamma
  InterfaceDict['Beta'     ]  = Beta
  InterfaceDict['Triangular'] = Triangular
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
