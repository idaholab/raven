'''
Created on Mar 7, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import xml.etree.ElementTree as ET
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
    self.upperBoundUsed = False
    self.lowerBoundUsed = False
    self.upperBound       = 0.0
    self.lowerBound       = 0.0
    self.adjustmentType   = ''
    self.bestQuad = None

  def readMoreXML(self,xmlNode):
    try:
      self.bestQuad = Quadrature.returnInstance(xmlNode.attrib['ExpansionQuadrature'])
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
    self.inDistr()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self,tempDict)
    tempDict['low'] = self.low
    tempDict['hi'] = self.hi
    # no other additional parameters required

  def inDistr(self):
    self.range=self.hi-self.low
    self.distribution = dist.uniform(loc=self.low,scale=self.range)
    #assign associated polynomial types
    self.polynomial = polys.legendre
    #define functions locally, then point to them
    def norm(n):
      '''Returns normalization constant for polynomial type, given the poly ordeir'''
      return np.sqrt((2.*n+1.)/2.)

#    def standardToActualWeight(x): #standard -> actual
#      '''Given normal quadrature weight, returns adjusted weight.'''
#      print ('StA wt for',self,'w_st =',x,'is',x/(self.range/2.))
#      return x/np.sqrt((self.range/2.))

    def probNorm(x): #normalizes probability if total != 1
      '''Returns the poly factor to scale by so that sum(probability)=1.'''
      #print ('probNorm for',self,'is',self.range)
      return 1.0/self.range
#from here up to .......
    def standardToActualPoint(x): #standard -> actual
      '''Given a [-1,1] point, converts to parameter value.'''
      return x*self.range/2.+self.distribution.mean()

    def actualToStandardPoint(x): #actual -> standard
      '''Given a parameter value, converts to [-1,1] point.'''
      return (x-self.distribution.mean())/(self.range/2.)

    def getMeGaussPoint(pointIndex):
      '''for a given index of the quadrature it return the point value in the standard system'''
      return standardToActualPoint(self.distQuad.quad_pts[pointIndex])

    def actualWeights(pointIndex):
      '''returns the weights on the actual reference system'''
      return self.distQuad.weights[pointIndex]*np.sqrt(self.range/2.)

    def evNormPolyonInterp(order,coord):
      return self.distQuad.evNormPoly(order,actualToStandardPoint(coord))/np.sqrt(self.range/2.)

    def evNormPolyonGauss(order,coord):
      return self.distQuad.evNormPoly(order,actualToStandardPoint(coord))

    def measure():
      return 1./(self.range)

    self.gaussPoint         = getMeGaussPoint
    self.actualWeights      = actualWeights
    self.evNormPolyonGauss  = evNormPolyonGauss
    self.evNormPolyonInterp = evNormPolyonInterp
    self.std_Point          = actualToStandardPoint
    self.actual_Point       = standardToActualPoint
    self.measure            = measure


#here, this are the only function used.......
#
#    # point to functions
#    self.poly_norm = norm
#    self.probability_norm = probNorm




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
    self.inDistr()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['mean' ] = self.mean
    tempDict['sigma'] = self.sigma

  def inDistr(self):
    if (not self.upperBoundUsed) and (not self.lowerBoundUsed):
      self.distribution = dist.norm(loc=self.mean,scale=self.sigma)
      self.polynomial = polys.hermitenorm
#      def norm(n):
#        return (np.sqrt(np.sqrt(2.*np.pi)*factorial(n)))**(-1)
#
#      def standardToActualWeight(x): #standard -> actual
#        return x/(self.sigma**2)
#
#      def probNorm(x): #normalizes if total prob. != 1
#        return 1.0/(np.sqrt(2*np.pi)*self.sigma)

#from here up to .......
      def standardToActualPoint(x): #standard -> actual
        print('standard '+str(x))
        print('Actual '+str(x*self.sigma*np.sqrt(2.)+self.distribution.mean()))
        return x*self.sigma/np.sqrt(2.)+self.distribution.mean()

      def actualToStandardPoint(x): #actual -> standard
        return (x-self.distribution.mean())/(self.sigma)*np.sqrt(2.)

      def getMeGaussPoint(pointIndex):
        return standardToActualPoint(self.distQuad.quad_pts[pointIndex])


      def actualWeights(pointIndex):
        print('self.distQuad.weights[pointIndex]/np.sqrt(2.*np.pi*self.sigma*np.sqrt(2.)) '+str(self.distQuad.weights[pointIndex]/np.sqrt(2.*np.pi*self.sigma*np.sqrt(2.))))
        return self.distQuad.weights[pointIndex]*np.sqrt(self.sigma/np.sqrt(2))

      def evNormPolyonGauss(order,coord):
        a=np.exp((actualToStandardPoint(coord)/2.)**2)
        return self.distQuad.evNormPoly(order,actualToStandardPoint(coord))*a

      def evNormPolyonInterp(order,coord):
        a=np.exp(  -(coord-self.distribution.mean())**2/((self.sigma)**2)/2. )*np.sqrt(np.sqrt(2)/self.sigma)
        return self.distQuad.evNormPoly(order,actualToStandardPoint(coord))*a

      def measure():
        return 1./self.sigma/2./np.sqrt(np.pi)


      self.gaussPoint          = getMeGaussPoint
      self.actualWeights       = actualWeights
      self.evNormPolyonGauss   = evNormPolyonGauss
      self.evNormPolyonInterp  = evNormPolyonInterp
      self.std_Point           = actualToStandardPoint
      self.actual_Point        = standardToActualPoint
      self.measure             = measure

#here, this are the only function used.......

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

class Poisson(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.mu  = 0.0
    self.type = 'Poisson'

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.mu = float(xmlNode.find('mu').text)
    except: raise Exception('mu value needed for poisson distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['mu'  ] = self.mu

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self.distribution = dist.poisson(self.mu)
    else:
      raise IOError ('Truncated poisson not yet implemented')
  def inDistr(self):
    pass

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

class Binomial(Distribution):
  def __init__(self):
    Distribution.__init__(self)
    self.n  = 0.0
    self.p  = 0.0
    self.type = 'Binomial'

  def readMoreXML(self,xmlNode):
    Distribution.readMoreXML(self, xmlNode)
    try: self.n = float(xmlNode.find('n').text)
    except: raise Exception('n value needed for Binomial distribution')
    try: self.p = float(xmlNode.find('p').text)
    except: raise Exception('p value needed for Binomial distribution')
    self.initializeDistribution()

  def addInitParams(self,tempDict):
    Distribution.addInitParams(self, tempDict)
    tempDict['n'  ] = self.n
    tempDict['p'  ] = self.p

  def initializeDistribution(self):
    if self.lowerBoundUsed == False and self.upperBoundUsed == False:
      self.distribution = dist.binom(n=self.n,p=self.p)
    else:
      raise IOError ('Truncated Binomial not yet implemented')
  def inDistr(self):
    pass


def returnInstance(Type):
  base = 'Distribution'
  InterfaceDict = {}
  InterfaceDict['Uniform'  ]  = Uniform
  InterfaceDict['Normal'   ]  = Normal
  InterfaceDict['Gamma'    ]  = Gamma
  InterfaceDict['Beta'     ]  = Beta
  InterfaceDict['Triangular'] = Triangular
  InterfaceDict['Poisson'] = Poisson
  InterfaceDict['Binomial'] = Binomial
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
