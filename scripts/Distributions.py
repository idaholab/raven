'''
Created on Mar 7, 2013

@author: crisr
'''
import sys
import xml.etree.ElementTree as ET
import scipy.stats.distributions  as dist
import scipy.special as polys
import numpy as np
from BaseType import BaseType
from scipy.misc import factorial

#import Quadrature as quads

class Distribution(BaseType):
  ''' 
  a general class containing the distributions
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.upperBoundUsed = False
    self.lowerBoundUsed = False
    self.upperBound       = 0.0
    self.lowerBound       = 0.0
    self.adjustmentType   = ''
    
  def readMoreXML(self,xmlNode):
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
  def rvsWithinCDFbounds(self,LowerBound,upperBound):
    point = np.random.rand(1)*(upperBound-LowerBound)+LowerBound
    return self.distribution.ppt(point)
  def rvsWithinbounds(self,LowerBound,upperBound):
    CDFupper = self.distribution.cdf(upperBound)
    CDFlower = self.distribution.cdf(LowerBound)
    return self.rvsWithinCDFbounds(CDFlower,CDFupper)
  def setQuad(self,quad,exp_order):
    self.distQuad=quad
    self.exp_order=exp_order
  def quad(self):
    try: return self.distQuad
    except: raise IOError ('No quadrature has been set for this distribution yet.')
  def polyOrder(self):
    try: return self.exp_order
    except: raise IOError ('Quadrature has not been set for this distribution yet.')


#==========================================================\
#    Distributions convenient for stochastic collocation
#==========================================================\

class Uniform(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.low = 0.0
    self.hi = 0.0
    self.type = 'Uniform'
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    try: self.low = float(xmlNode.find('low').text)
    except: raise 'low value needed for uniform distribution'
    try: self.hi = float(xmlNode.find('hi').text)
    except: raise 'hi value needed for uniform distribution'
    self.inDistr()
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    # no other additional parameters required
  def inDistr(self):
    self.range=self.hi-self.low
    self.distribution = dist.uniform(loc=self.low,scale=self.range)
    self.polynomial = polys.legendre
    #define functions locally, then point to them
    def norm(n):
      return np.sqrt((2.*n+1.)/2.) #TODO this is polynomial specific, but poly is set in quadrature!
    def standardToActualPoint(x): #standard -> actual
      return x*self.range/2.+self.distribution.mean()
    def actualToStandardPoint(x): #actual -> standard
      return (x-self.distribution.mean())/(self.range/2.)
    def standardToActualWeight(x): #standard -> actual
      return x/(self.range/2.)
    def probNorm(x): #normalizes probability if total != 1
      return self.range
    self.poly_norm = norm
    self.actual_point = standardToActualPoint
    self.std_point = actualToStandardPoint
    self.actual_weight = standardToActualWeight
    self.probability_norm = probNorm


class Normal(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.mean  = 0.0
    self.sigma = 0.0
    self.type = 'Normal'
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.mean  = float(xmlNode.find('mean' ).text)
    except: raise 'mean value needed for normal distribution'
    try: self.sigma = float(xmlNode.find('sigma').text)
    except: raise 'sigma value needed for normal distribution'
    self.inDistr()
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma
  def inDistr(self):
    if self.upperBoundUsed == False and self.lowerBoundUsed == False:
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
      def probNorm(x): #normalizes probability if total != 1
        return 1.0
      self.poly_norm=norm
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
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    try: self.low = float(xmlNode.find('low').text)
    except: raise 'low value needed for Gamma distribution'
    try: self.alpha = float(xmlNode.find('alpha').text)
    except: raise 'alpha value needed for Gamma distribution'
    try: self.beta = float(xmlNode.find('beta').text)
    except: self.beta=1.0
    self.inDistr()
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta
  def inDistr(self):
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
    # TODO default to specific Beta distro?
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self,xmlNode)
    try: self.low = float(xmlNode.find('low').text)
    except: raise 'low value needed for Gamma distribution'
    try: self.hi = float(xmlNode.find('hi').text)
    except: raise 'hi value needed for Gamma distribution'
    try: self.alpha = float(xmlNode.find('alpha').text)
    except: raise 'alpha value needed for Gamma distribution'
    try: self.beta = float(xmlNode.find('beta').text)
    except: raise 'beta value needed for Gamma distribution'
    self.inDistr()
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    tempDict['alpha'] = self.alpha
    tempDict['beta'] = self.beta
  def inDistr(self):
    self.distribution = dist.beta(self.alpha,self.beta,scale=self.hi-self.low)



#==========================================================\
#    other distributions
#==========================================================\


# Add polynomials, shifting, zero-to-one to these!


class Triangular(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.apex = 0.0
    self.min  = 0.0
    self.max  = 0.0
    self.type = 'Triangular'
  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.apex = float(xmlNode.find('apex').text)
    except: raise 'apex value needed for normal distribution'
    try: self.min = float(xmlNode.find('min').text)
    except: raise 'min value needed for normal distribution'
    try: self.max = float(xmlNode.find('max').text)
    except: raise 'max value needed for normal distribution'
    self.inDistr()
  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['apex' ] = self.apex
    tempDict['min'  ] = self.min
    tempDict['max'  ] = self.max
  def inDistr(self):
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
