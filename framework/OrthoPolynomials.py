'''
Created on Nov 24, 2014

@author: talbpw
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import scipy.special as polys
from scipy.misc import factorial
import xml.etree.ElementTree as ET
#External Modules End--------------------------------------------------------------------------------

#Internal Modules
from BaseClasses import BaseType
from utils import returnPrintTag, returnPrintPostTag, find_distribution1D
import Distributions
#Internal Modules End--------------------------------------------------------------------------------

class OrthogonalPolynomial(object):
  '''Provides polynomial generators for stochastic collocation.'''
  def __init__(self):
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    self.debug = True
    self.params=[]

  def initialize(self):
    self._poly = None #self.monomial #orthogonal polynomial constructor function  
    self._evPoly = None #self.evalMonomial #orthogonal polynomial constructor function  

  def __getitem__(self,order):
    '''Returns the polynomial with order 'order', as poly[2]'''
    return self._poly(order,*self.params) * self.norm(order)

  def __call__(self,order,pt):
    '''Returns the polynomial of order 'order' evaluated at 'pt'.
       Has to be overwritten if parameters are required.'''
    inps=self.params+[self.pointMod(pt)]
    return self._evPoly(order,*inps) * self.norm(order)

  def _readMoreXML(self,xmlNode):
    if self.debug:print('Quadrature: need to fix _readMoreXML')
    self._localReadMoreXML(xmlNode)
    return

  def _localReadMoreXML(self,xmlNode):
    self.params=[]

  def norm(self,order):
    '''Normalization constant for polynomials so that integrating two of them
       w.r.t. the weight factor produces the kroenecker delta.'''
    return 1

  def pointMod(self,pt):
    '''Some scipy polys are orthogonal w.r.t. slightly different weights.
       This change of variable function fixes orthogonality to what we want.'''
    return pt

  def stdPointMod(self,x):
    return x

  def monomial(self,order): #these are default, but not orthogonal at all.
    coeffs=[1]+[0]*(order-1)
    return np.poly1d(coeffs)

  def evalMonomial(self,order,pt):
    return self.monomial(order)(pt)

  def setMeasures(self,quadSet):
    #make a uniform distribution to use the quantile (ppf) function for cdf,CC case
    if quadSet.type.startswith('CDF'):# in ['CDF','ClenshawCurtis']:
      self.__distr=self.makeDistribution()
      self.pointMod = self.cdfPoint
    else:
      raise IOError('OrthoPolynomials: No implementation for',quadSet,'quadrature and',self.type,'polynomials.')

  def _getDistr(self):
    return self.__distr

  def samePoint(self,x):
    return x

  def cdfPoint(self,x):
    '''ppf() converts to from [0,1] to distribution range,
       0.5(x+1) converts from [-1,1] to [0,1],
       sqrt(2) fixes scipy Legendre polynomial weighting'''
    return self.__distr.ppf(0.5*(x+1.))#*self.scipyNorm()

  def scipyNorm(self):
    return 1.

class Legendre(OrthogonalPolynomial):
  def initialize(self,quad):
    self.printTag = 'LEGENDRE-ORTHOPOLY'
    self._poly = polys.legendre
    self._evPoly = polys.eval_legendre
    self.setMeasures(quad)

  def setMeasures(self,quad):
    if quad.type in ['Legendre','ClenshawCurtis']:
      self.pointMod = self.stdPointMod
    elif quad.type=='ClenshawCurtis':
      self.pointMod = self.stdPointMod
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    uniformElement = ET.Element("uniform")
    element = ET.Element("low",{})
    element.text = "-1"
    uniformElement.append(element)
    element = ET.Element("hi",{})
    element.text = "1"
    uniformElement.append(element)
    uniform = Distributions.Uniform()
    uniform._readMoreXML(uniformElement)
    uniform.initializeDistribution()
    return uniform

  def stdPointMod(self,x):
    return x#self.scipyNorm()

  def scipyNorm(self):
    return np.sqrt(2)

  def norm(self,n):
    return np.sqrt((2.*n+1.))#/2.)
    #OLD NOTE the first 2 is included because scipy legendre poly1d is orthogonal
    #over [-1,1] with with weight function 1:
    #http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.legendre.html#scipy.special.legendre 



class Hermite(OrthogonalPolynomial):
  def initialize(self,quad):
    self.printTag = 'HERMITE-ORTHOPOLY'
    self._poly = polys.hermitenorm
    self._evPoly = polys.eval_hermitenorm
    self.setMeasures(quad)

  def setMeasures(self,quad):
    if quad.type=='Hermite':
      self.pointMod = self.stdPointMod
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    normalElement = ET.Element("normal")
    element = ET.Element("mean",{})
    element.text = "0"
    normalElement.append(element)
    element = ET.Element("sigma",{})
    element.text = "1"
    normalElement.append(element)
    normal = Distributions.Normal()
    normal._readMoreXML(normalElement)
    normal.initializeDistribution()
    return normal

  def norm(self,n):
    #cof = 1/(2.*np.sqrt(2))
    return 1.0/np.sqrt(factorial(n))#/(np.sqrt(np.sqrt(2.*np.pi)*factorial(n)))



class Laguerre(OrthogonalPolynomial):
  def initialize(self,quad):
    self.printTag = 'LAGUERRE-ORTHOPOLY'
    self._poly = polys.genlaguerre
    self._evPoly = polys.eval_genlaguerre
    self.params=quad.params
    self.setMeasures(quad)

  def setMeasures(self,quad):
    if quad.type=='Laguerre':
      self.pointMod = self.stdPointMod
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    gammaElement = ET.Element("gamma")
    element = ET.Element("low",{})
    element.text = "0"
    gammaElement.append(element)
    element = ET.Element("alpha",{})
    element.text = "%s" %(self.params[0]+1)
    gammaElement.append(element)
    gamma = Distributions.Gamma()
    gamma._readMoreXML(gammaElement)
    gamma.initializeDistribution()
    return gamma

  def norm(self,order):
    return np.sqrt(factorial(order)/factorial(order+self.params[0]))



class Jacobi(OrthogonalPolynomial):
  def initialize(self,quad):
    self.printTag = 'JACOBI-ORTHOPOLY'
    self._poly = polys.jacobi
    self._evPoly = polys.eval_jacobi
    self.params=quad.params
    self.setMeasures(quad)

#  def _localReadMoreXML(self,xmlNode):
#    self.params = []
#    if xmlNode.find('alpha') != None:
#      alpha=float(xmlNode.find('alpha').text)
#    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Jacobi polynmials require alpha keyword; not found.')
#    if xmlNode.find('beta') != None:
#      beta=float(xmlNode.find('beta').text)
#    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Jacobi polynomials require beta keyword; not found.')
#    self.params = [beta-1,alpha-1]

  def setMeasures(self,quad):
    if quad.type=='Jacobi':
      self.pointMod = self.stdPointMod
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    jacobiElement = ET.Element("jacobi")
    element = ET.Element("alpha",{})
    element.text = "%s" %self.params[1]+1
    jacobiElement.append(element)
    element = ET.Element("beta",{})
    element.text = "%s" %self.params[0]+1
    jacobiElement.append(element)

  def norm(self,n):
    a=self.params[0]#+1
    b=self.params[1]#+1
    coeff=1.
    #THESE THREE AS ARE work for uniform, stashing a copy of them
    #coeff*=np.sqrt((2.*n+a+b+1.) /2**(a+b+1))
    #coeff*=np.sqrt(factorial(n)*factorial(n+a+b)/(factorial(n+a)*factorial(n+b)))#=1 for a=b=0
    #coeff*=np.sqrt(2)
    #print('DEBUG n =',n)
    #print('  DEBUG 1',(2.*n+a+b+1.) /2**(a+b+1))
    #print('  DEBUG 2',factorial(n)*factorial(n+a+b)/(factorial(n+a)*factorial(n+b)))
    coeff*=np.sqrt((2.*n+a+b+1.) /2**(a+b+1))
    coeff*=np.sqrt(factorial(n)*factorial(n+a+b)/(factorial(n+a)*factorial(n+b)))
    coeff*=np.sqrt(2)

    #wtf
    cof2 = 1
    cof2 *= 2.**(a+b)/(a+b+1.)
    cof2 *= factorial(a)*factorial(b)/factorial(a+b)

    coeff*=np.sqrt(cof2)

    #coeff*=np.sqrt(16./15.) #a=5,b=2
    #a+=1
    #b+=1
    #coeff*=np.sqrt(2**(a+b-2)*factorial(a)*factorial(b)/factorial(a+b+1))

    #coeff*=np.sqrt(factorial(a+b-1)/(factorial(a-1)*factorial(b-1)))
    #coeff*=np.sqrt(1./(2.*2**(a+b-2)))

    #print('DEBUG poly norm',n,coeff)
    #print('DEBUG norm',factorial(a+b-1)/(factorial(a-1)*factorial(b-1)))
    return coeff



'''
 Interface Dictionary (factory) (private)
'''
__base = 'OrthoPolynomial'
__interFaceDict = {}
__interFaceDict['Legendre'] = Legendre
__interFaceDict['Hermite'] = Hermite
__interFaceDict['Laguerre'] = Laguerre
__interFaceDict['Jacobi'] = Jacobi
#__interFaceDict['Lagrange'] = Lagrange TODO
__knownTypes = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  '''
    function used to generate a Filter class
    @ In, Type : Filter type
    @ Out,Instance of the Specialized Filter class
  '''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
