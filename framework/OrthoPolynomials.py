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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules
from BaseClasses import BaseType
from utils import returnPrintTag, returnPrintPostTag, find_distribution1D
#Internal Modules End--------------------------------------------------------------------------------

def getCollocationSet(distr,polytype,order):
  polytype = polytype.strip().lower()
  if   polytype=='legendre': return Legendre(distr,order)
  elif polytype=='hermite' : return Hermite (distr,order)
  elif polytype=='laguerre': return Laguerre(distr,order)
  elif polytype=='jacobi'  : return Jacobi  (distr,order)
  else: raise IOError('no collocation set found for listed type:',polytype)
  #TODO more options to come


class OrthogonalPolynomial(BaseType):
  '''Provides polynomial generators for stochastic collocation.'''
  def __init__(self):
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    self.debug = False

    self._poly = None #orthogonal polynomial constructor function  

  def initialize(self):
    pass

  def __call__(self,order):
    return self._poly(order,*self.params)*self.norm(order)

  def norm(self,order):
    return 1;


class Legendre(OrthogonalPolynomial):
  def initalize(self):
    self._poly = polys.legendre

  def norm(self,n):
    return np.sqrt((2.*n+1.)/2.)


class Hermite(OrthogonalPolynomial):
  def initialize(self):
    self._poly = polys.hermitenorm

  def norm(self,n):
    return (np.sqrt(np.sqrt(2.*np.pi)*factorial(n)))**(-1)


class Laguerre(OrthogonalPolynomial): #TODO tests and such
  def initialize(self):
    self._poly = polys.genlaguerre

  def norm(self,order):
    return np.sqrt(factorial(order)/polys.gamma(order+self.distr.alpha+1.0))



class Jacobi(OrthogonalPolynomial):
  def initialize(self):
    self._poly = polys.genlaguerre

  def norm(self,order):
    a=self.alpha
    b=self.beta
    coeff1=1./no.sqrt(2.**(a+b+1)/(2.*order+a+b+1.))
    coeff2=1./np.sqrt(factorial(order+a)*factorial(order+b)/(factorial(order)*factorial(order+a+b)))
    return coeff1*coeff2



class Irregular(CollocationSet):
  #TODO FIX ME
  '''This covers all the collocation sets that don't fit in the regular 4.
     It uses the CDF and inverse CDF to map onto U[0,1].'''
  def setPolynomial(self,quadType=None):
    self._polynomial = polys.legendre
    self._opt_quad = quads.p_roots
    self.setQuadrature(quadType)

  #def polyNorm(self,n):
  #  return np.sqrt((2.*n+1.)/2.)

  def cdfDomainToInputDomainPoint(self,x):
    try: return self.distr.ppf(x)
    except TypeError:
      sln=[]
      for xs in x:
        sln.append(self.distr.ppf(xs))
      return sln
  def InputDomainToCdfDomainPoint(self,x):
    try: return self.distr.cdf(x)
    except TypeError:
      sln=[]
      for xs in x:
        sln.append(self.distr.cdf(xs))
      return sln

  def cdfDomainToBasisDomainPoint(self,x):
    return 2.0*x-1.0
  def BasisDomainToCdfDomainPoint(self,x):
    return 0.5*(x+1.0)

  def stdToActPoint(self,x):
    a=self.BasisDomainToCdfDomainPoint(x)
    b=self.cdfDomainToInputDomainPoint(a)
    return self.cdfDomainToInputDomainPoint(self.BasisDomainToCdfDomainPoint(x))
  def actToStdPoint(self,x):
    return self.cdfDomainToBasisDomainPoint(self.InputDomainToCdfDomainPoint(x))


