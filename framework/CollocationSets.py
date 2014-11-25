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
import scipy.special.orthogonal as quads
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
  else: raise IOError('no quadrature found for listed quadrature type:',polytype)
  #TODO more options to come


class CollocationSet(object): #TODO make compatible to base class
  '''Provides polynomial and quadrature generators for stochastic collocation.'''
  def __init__(self,distr,order):
    self.distr = distr
    self.max_poly_order = order
    self.setPolynomial()

  def generatePtsAndWts(self,order,std=False):
    '''Generates points and weights for quadrature, and translates them
       into distribution domain space.'''
    pts,wts = self._quad(order)
    if not std:
      pts = self.stdToActPoint(pts)
     # wts = self.stdToActWeight(wts) #wrong approach?
    return pts,wts

  def evaluatePolynomial(self,order,x):
    '''Evaluates the order-degree polynomial corresponding to the quadrature set.'''
    return self._polynomial(order)(x)*self.polyNorm(order)


class Legendre(CollocationSet):
  def setPolynomial(self):
    self._polynomial = polys.legendre
    self._quad = quads.p_roots

  def polyNorm(self,n):
    return np.sqrt((2.*n+1.)/2.)

  def stdToActPoint(self,x):
    return x*self.distr.range/2.+self.distr.untruncatedMean()
  
  def stdToActWeight(self,x):
    return x/(self.distr.range/2.)

  def actToStdPoint(self,x):
    return (x-self.distr.untruncatedMean())/(self.distr.range/2.)



class Hermite(CollocationSet):
  def setPolynomial(self):
    self._polynomial = polys.hermitenorm
    self._quad = quads.he_roots

  def polyNorm(self,n):
    #return 1./np.sqrt(np.sqrt(2.*np.pi)*factorial(n))
    return (np.sqrt(np.sqrt(2.*np.pi)*factorial(n)))**(-1) #old

  def stdToActPoint(self,x):
    return x*self.distr.sigma+self.distr.untruncatedMean()#/self.distr.sigma
    #return x*self.distr.sigma**2/2.+self.distr.untruncatedMean() #old
  
  def stdToActWeight(self,x):
    return x/(self.distr.sigma)
    #return x/(self.distr.sigma**2/2.) #old

  def actToStdPoint(self,x):
    return (x-self.distr.untruncatedMean())/(self.distr.sigma)
    #return (x-self.distr.untruncatedMean())/(self.distr.sigma**2/2.) #old


class Laguerre(CollocationSet):
  def setPolynomial(self):
    self._polynomial = polys.genlaguerre
    self._quad = quads.la_roots

  def polyNorm(self,order):
    return np.sqrt(factorial(order)/polys.gamma(order+self.alpha+1.0))

  def stdToActPoint(self,x):
    return x/self.alpha+self.alpha+self.low
  
  def stdToActWeight(self,x):
    return x #really?

  def actToStdPoint(self,x):
    return  (x-self.low-self.alpha)*self.alpha

  def generatePtsAndWts(self,order):
    '''Overwritten to use alpha factor'''
    pts,wts = self._quad(order,self.alpha)
    pts = self.stdToActPoint(pts)
    wts = self.stdToActWeight(pts)
    return pts,wts

  def evaluatePolynomial(self,order,x):
    '''Overwritten to use alpha factor'''
    return self._polynomial(order,self.alpha)(x)*self.polyNorm(order)



class Jacobi(CollocationSet):
  def setPolynomial(self):
    self._polynomial = polys.genlaguerre
    self._quad = quads.j_roots

  def polyNorm(self,order):
    a=self.alpha
    b=self.beta
    coeff1=1./no.sqrt(2.**(a+b+1)/(2.*order+a+b+1.))
    coeff2=1./np.sqrt(factorial(order+a)*factorial(order+b)/(factorial(order)*factorial(order+a+b)))
    return coeff1*coeff2

  def stdToActPoint(self,x):
    return 1 #TODO
  
  def stdToActWeight(self,x):
    return 1 #TODO

  def actToStdPoint(self,x):
    return 1 #TODO

  def generatePtsAndWts(self,order):
    '''Overwritten to use alpha, beta factors'''
    pts,wts = self._quad(order,self.alpha,self.beta)
    pts = self.stdToActPoint(pts)
    wts = self.stdToActWeight(pts)
    return pts,wts

  def evaluatePolynomial(self,order,x):
    '''Overwritten to use alpha, beta factors'''
    return self._polynomial(order,self.alpha,self.beta)(x)*self.polyNorm(order)

