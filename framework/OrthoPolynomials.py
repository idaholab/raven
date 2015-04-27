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
#from scipy.misc import factorial
from math import gamma
import xml.etree.ElementTree as ET
#External Modules End--------------------------------------------------------------------------------

#Internal Modules
from BaseClasses import BaseType
import Distributions
import Quadratures
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class OrthogonalPolynomial(MessageHandler.MessageUser):
  '''Provides polynomial generators and evaluators for stochastic collocation.'''
  def __init__(self):
    self.type    = self.__class__.__name__
    self.name    = self.__class__.__name__
    self._poly   = None #tool for generating orthopoly1d objects
    self._evPoly = None #tool for evaluating 1d polynomials at (order,point)
    self.params  = [] #additional parameters needed for polynomial (alpha, beta, etc)
    self.messageHandler = None

  def initialize(self,quad,messageHandler):
    self.messageHandler = messageHandler

  def __getitem__(self,order):
    '''Returns the polynomial with order 'order';
       for example poly[2] returns the orthonormal 2nd-order polynomial object.
    @ In order, int, order of polynomial to return
    @ Out orthopoly1d object, requested polynomial
    '''
    return self._poly(order,*self.params) * self.norm(order)

  def __call__(self,order,pt):
    '''Returns the polynomial of order 'order' evaluated at 'pt'.
       Has to be overwritten if parameters are required.
    @ In order, int, order at which polynomial should be evaluated
    @ In pt, float, value at which polynomial should be evaluated
    @ Out, float, evaluation of polynomial
    '''
    inps=self.params+[self.pointMod(pt)]
    return self._evPoly(order,*inps) * self.norm(order)

  def __getstate__(self):
    '''Pickle dump method.
    @ In, None, None
    @ Out, Quadrature instance, defining quad for polynomial
    '''
    return self.quad,self.messageHandler

  def __setstate__(self,items):
    '''Pickle load method.
    @ In, quad, Quadrature instance
    @ Out, None, None
    '''
    self.__init__()
    self.initialize(*items)#quad,messageHandler)

  def __eq__(self,other):
    '''
    Equality method.
    @ In other, object, object to compare equivalence
    @ Out boolean, truth of matching equality
    '''
    return self._poly==other._poly and self._evPoly==other._evPoly and self.params==other.params

  def __ne__(self,other):
    '''
    Inequality method.
    @ In other, object, object to compare equivalence
    @ Out boolean, truth of matching inequality
    '''
    return not self.__eq__(other)

  def norm(self,order):
    '''Normalization constant for polynomials so that integrating two of them
       w.r.t. the weight factor produces the kroenecker delta. Default is 1.
    @ In order, int, polynomial order to get norm of
    @ Out, float, value of poly norm
    '''
    return 1

  def pointMod(self,pt):
    '''Some polys are orthonormal w.r.t. slightly different weights.
       This change of variable function fixes orthonormality to what we want.
    @ In pt, float, point to modify
    @ Out, float, modified point
    '''
    return pt

  def stdPointMod(self,x):
    '''Provides a default for inheriting classes.  This is the pointMod that
       should be used with the 'default' choices.
    @ In x, float, point to modify
    @ Out, float, modified point
    '''
    return x

  def setMeasures(self,quad):
    '''If you got here, it means the inheriting orthopoly object doesn't have a
       specific implementation for the quadSet given.  Here we catch the universal
       options.
    @ In quad, Quadrature object, quadrature that will make coeffs for these polys
    @ Out, None, None
    '''
    if quad.type.startswith('CDF'): #covers CDFLegendre and CDFClenshawCurtis
      self.__distr=self.makeDistribution()
      self.pointMod = self.cdfPoint
      self.quad = quad
    else:
      self.raiseAnError(IOError,'No implementation for '+quad.type+' quadrature and',self.type,'polynomials.')

  def _getDistr(self):
    '''Returns the private distribution used for the CDF-version quadratures; for debugging.
    @ In None, None
    @ Out Disribution object, standardized associated distribution
    '''
    return self.__distr

  def cdfPoint(self,x):
    '''ppf() converts to from [0,1] to distribution range,
       0.5(x+1) converts from [-1,1] to [0,1].
    @ In x, float, point
    @ Out, float, converted point
    '''
    return self.__distr.ppf(0.5*(x+1.))

  def scipyNorm(self):
    '''Some functions are slightly different in scipy; this is for fixing that.
    @ In None, None
    @ Out, float, required norm
    '''
    return 1.

  def makeDistribution(self):
    ''' Used to make standardized distribution for this poly type.
    @ In None, None
    @ Out None, None
    '''
    pass


class Legendre(OrthogonalPolynomial):
  def initialize(self,quad,messageHandler):
    OrthogonalPolynomial.initialize(self,quad,messageHandler)
    self.printTag = 'LEGENDRE-ORTHOPOLY'
    self._poly    = polys.legendre
    self._evPoly  = polys.eval_legendre
    self.setMeasures(quad)

  def setMeasures(self,quad):
    if quad.type in ['Legendre','ClenshawCurtis']:
      self.pointMod = self.stdPointMod
      self.quad = quad
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    uniformElement = ET.Element("uniform")
    element = ET.Element("lowerBound",{})
    element.text = "-1"
    uniformElement.append(element)
    element = ET.Element("upperBound",{})
    element.text = "1"
    uniformElement.append(element)
    uniform = Distributions.Uniform()
    uniform._readMoreXML(uniformElement)
    uniform.initializeDistribution()
    return uniform

  def stdPointMod(self,x):
    return x

  def scipyNorm(self):
    return np.sqrt(2)

  def norm(self,n):
    return np.sqrt((2.*n+1.))


class Hermite(OrthogonalPolynomial):
  def initialize(self,quad,messageHandler):
    OrthogonalPolynomial.initialize(self,quad,messageHandler)
    self.printTag = 'HERMITE-ORTHOPOLY'
    self._poly    = polys.hermitenorm
    self._evPoly  = polys.eval_hermitenorm
    self.setMeasures(quad)

  def setMeasures(self,quad):
    if quad.type=='Hermite':
      self.pointMod = self.stdPointMod
      self.quad = quad
    else:
      OrthogonalPolynomial.setMeasures(self,quad.type)

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
    #if n==0:return 1
    return 1.0/np.sqrt(gamma(1.0+n))



class Laguerre(OrthogonalPolynomial):
  def initialize(self,quad,messageHandler):
    OrthogonalPolynomial.initialize(self,quad,messageHandler)
    self.printTag = 'LAGUERRE-ORTHOPOLY'
    self._poly    = polys.genlaguerre
    self._evPoly  = polys.eval_genlaguerre
    self.params   = quad.params
    self.setMeasures(quad)

  def setMeasures(self,quad):
    if quad.type=='Laguerre':
      self.pointMod = self.stdPointMod
      self.quad = quad
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
    #if order==0 and self.params[0]==0: return 1
    #if order==0: return np.sqrt(1.0/gamma(1.0+self.params[0]))
    return np.sqrt(gamma(1.0+order)/gamma(1.0+order+self.params[0]))



class Jacobi(OrthogonalPolynomial):
  def initialize(self,quad,messageHandler):
    OrthogonalPolynomial.initialize(self,quad,messageHandler)
    self.printTag = 'JACOBI-ORTHOPOLY'
    self._poly    = polys.jacobi
    self._evPoly  = polys.eval_jacobi
    self.params   = quad.params
    self.setMeasures(quad)

  def setMeasures(self,quad):
    if quad.type=='Jacobi':
      self.pointMod = self.stdPointMod
      self.quad = quad
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
#    coeff=1.
#    coeff*=np.sqrt((2.*n+a+b+1.) /2**(a+b+1))
#    coeff*=np.sqrt(gamma(1.0+n)*gamma(1.0+n+a+b)/(gamma(1.0+n+a)*gamma(1.0+n+b)))
#    coeff*=np.sqrt(2)
#    #not sure why I need this factor, but it corrects all cases I tested
#    #FIXME it might be wrong for n>1, though, it occurs to me...
#    cof2 = 1
#    cof2 *= 2.**(a+b)/(a+b+1.)
#    cof2 *= gamma(1.0+a)*gamma(1.0+b)/gamma(1.0+a+b)
#    coeff*=np.sqrt(cof2)
    ###speedup attempt###
#    coeff=(2.0*n+a+b+1.0)*\
#          gamma(1.0+n)*gamma(1.0+n+a+b)/(gamma(1.0+n+a)*gamma(1.0+n+b))*gamma(1.0+a)*gamma(1.0+b)/gamma(1.0+a+b+1.0)
    ###speedup attempt 2###
    coeff=(2.0*n+a+b+1.0)*\
          gamma(n+1)*gamma(n+a+b+1)/(gamma(n+a+1)*gamma(n+b+1))*\
          gamma(a+1)*gamma(b+1)/gamma(a+b+2.0)
    return np.sqrt(coeff)



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

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  '''
    function used to generate a Filter class
    @ In, Type : Filter type
    @ Out,Instance of the Specialized Filter class
  '''
  if Type in knownTypes(): return __interFaceDict[Type]()
  else: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
