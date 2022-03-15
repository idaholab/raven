# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on Nov 24, 2014

@author: talbpw
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import scipy.special as polys
#from scipy.misc import factorial
from math import gamma
import xml.etree.ElementTree as ET
#External Modules End--------------------------------------------------------------------------------

#Internal Modules
from .EntityFactoryBase import EntityFactory
from .BaseClasses import MessageUser
from . import Distributions
#Internal Modules End--------------------------------------------------------------------------------

class OrthogonalPolynomial(MessageUser):
  """
    Provides polynomial generators and evaluators for stochastic collocation.
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type    = self.__class__.__name__
    self.name    = self.__class__.__name__
    self._poly   = None #tool for generating orthopoly1d objects
    self._evPoly = None #tool for evaluating 1d polynomials at (order,point)
    self.params  = [] #additional parameters needed for polynomial (alpha, beta, etc)

  def initialize(self,quad):
    """
      Initializes object with items not set in __init__
      @ In, quad, string, quadrature object name
      @ Out, None
    """
    pass

  def __getitem__(self,order,var=None):
    """
      Returns the polynomial with order 'order';
      for example poly[2] returns the orthonormal 2nd-order polynomial object.
      @ In, order, int, order of polynomial to return
      @ In, var, str, optional, name of variable to be used in return (default 'x')
      @ Out, __getitem__, orthopoly1d object, requested polynomial
    """
    if var==None:
      return self._poly(order,*self.params) * self.norm(order)
    else:
      return self._poly(order,*self.params,variable=var) * self.norm(order)

  def __call__(self,order,pt):
    """
      Returns the polynomial of order 'order' evaluated at 'pt'.
      Has to be overwritten if parameters are required.
      @ In, order, int, order at which polynomial should be evaluated
      @ In, pt, float, value at which polynomial should be evaluated
      @ Out, __call__, float, evaluation of polynomial
    """
    inps=self.params+[self.pointMod(pt)]
    return self._evPoly(order,*inps) * self.norm(order)

  def __getstate__(self):
    """
      Pickle dump method.
      @ In, None
      @ Out, state, tuple, (Quadrature instance, message handler) - defining quad for polynomial
    """
    state = [self.quad]
    return state

  def __setstate__(self,items):
    """
      Pickle load method.
      @ In, items, dict, objects required to restore
      @ Out, None
    """
    self.__init__()
    self.initialize(*items)

  def __eq__(self,other):
    """
      Equality method.
      @ In, other, object, object to compare equivalence
      @ Out, __eq__, bool, truth of matching equality
    """
    return self._poly==other._poly and self._evPoly==other._evPoly and self.params==other.params

  def __ne__(self,other):
    """
      Inequality method.
      @ In, other, object, object to compare equivalence
      @ Out, __ne__, bool, truth of matching inequality
    """
    return not self.__eq__(other)

  def norm(self,order):
    """
      Normalization constant for polynomials so that integrating two of them
      w.r.t. the weight factor produces the kroenecker delta. Default is 1.
      @ In, order, int, polynomial order to get norm of
      @ Out, norm, float, value of poly norm
    """
    return 1

  def pointMod(self,pt):
    """
      Some polys are orthonormal w.r.t. slightly different weights.
      This change of variable function fixes orthonormality to what we want.
      @ In, pt, float, point to modify
      @ Out, pt, float, modified point
    """
    return pt

  def stdPointMod(self,x):
    """
      Provides a default for inheriting classes.  This is the pointMod that
      should be used with the 'default' choices.
      @ In, x, float, point to modify
      @ Out, x, float, modified point
    """
    return x

  def setMeasures(self,quad):
    """
      If you got here, it means the inheriting orthopoly object doesn't have a
      specific implementation for the quadSet given.  Here we catch the universal
      options.
      @ In, quad, Quadrature object, quadrature that will make coeffs for these polys
      @ Out, None
    """
    if quad.type.startswith('CDF'):
      #covers CDFLegendre and CDFClenshawCurtis
      self.__distr=self.makeDistribution()
      self.pointMod = self.cdfPoint
      self.quad = quad
    else:
      self.raiseAnError(IOError,'No implementation for '+quad.type+' quadrature and',self.type,'polynomials.')

  def _getDistr(self):
    """
      Returns the private distribution used for the CDF-version quadratures; for debugging.
      @ In, None
      @ Out, Disribution object, standardized associated distribution
    """
    return self.__distr

  def cdfPoint(self,x):
    """
      ppf() converts to from [0,1] to distribution range,
      0.5(x+1) converts from [-1,1] to [0,1].
      @ In, x, float, point
      @ Out, cdfPoint, float, converted point
    """
    return self.__distr.ppf(0.5*(x+1.))

  def scipyNorm(self):
    """
      Some functions are slightly different in scipy; this is for fixing that.
      @ In, None
      @ Out, scipyNorm, float, required norm
    """
    return 1.

  def makeDistribution(self):
    """
      Used to make standardized distribution for this poly type.
      @ In, None
      @ Out, None
    """
    pass


class Legendre(OrthogonalPolynomial):
  """
    Provides polynomial Legendre generators and evaluators for stochastic collocation.
  """
  def initialize(self, quad):
    """
      Initializes object with items not set in __init__
      @ In, quad, string, quadrature object name
      @ Out, None
    """
    OrthogonalPolynomial.initialize(self, quad)
    self.printTag = 'LEGENDRE-ORTHOPOLY'
    self._poly    = polys.legendre
    self._evPoly  = polys.eval_legendre
    self.setMeasures(quad)

  def setMeasures(self,quad):
    """
      Implements specific measures for given quadSet.
      @ In, quad, Quadrature object, quadrature that will make coeffs for these polys
      @ Out, None
    """
    if quad.type in ['Legendre','ClenshawCurtis']:
      self.pointMod = self.stdPointMod
      self.quad = quad
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    """
      Used to make standardized distribution for this poly type.
      @ In, None
      @ Out, None
    """
    uniform = Distributions.Uniform(-1.0, 1.0)
    uniform.initializeDistribution()
    return uniform

  def stdPointMod(self,x):
    """
      Provides a default for inheriting classes.  This is the pointMod that
      should be used with the 'default' choices.
      @ In, x, float, point to modify
      @ Out, x, float, modified point
    """
    return x

  def scipyNorm(self):
    """
      Some functions are slightly different in scipy; this is for fixing that.
      @ In, None
      @ Out, scipyNorm, float, required norm
    """
    return np.sqrt(2)

  def norm(self,n):
    """
      Normalization constant for polynomials so that integrating two of them
      w.r.t. the weight factor produces the kroenecker delta. Default is 1.
      @ In, n, int, polynomial order to get norm of
      @ Out, norm, float, value of poly norm
    """
    return np.sqrt((2.*n+1.))


class Hermite(OrthogonalPolynomial):
  """
    Provides polynomial Hermite generators and evaluators for stochastic collocation.
  """
  def initialize(self, quad):
    """
      Initializes object with items not set in __init__
      @ In, quad, string, quadrature object name
      @ Out, None
    """
    OrthogonalPolynomial.initialize(self, quad)
    self.printTag = 'HERMITE-ORTHOPOLY'
    self._poly    = polys.hermitenorm
    self._evPoly  = polys.eval_hermitenorm
    self.setMeasures(quad)

  def setMeasures(self,quad):
    """
      Implements specific measures for given quadSet.
      @ In, quad, Quadrature object, quadrature that will make coeffs for these polys
      @ Out, None
    """
    if quad.type=='Hermite':
      self.pointMod = self.stdPointMod
      self.quad = quad
    else:
      OrthogonalPolynomial.setMeasures(self,quad.type)

  def makeDistribution(self):
    """
      Used to make standardized distribution for this poly type.
      @ In, None
      @ Out, normal, Distribution, the normal distribution
    """
    normal = Distributions.Normal(0.0, 1.0)
    normal.initializeDistribution()
    return normal

  def norm(self,n):
    """
      Normalization constant for polynomials so that integrating two of them
      w.r.t. the weight factor produces the kroenecker delta. Default is 1.
      @ In, n, int, polynomial order to get norm of
      @ Out, norm, float, value of poly norm
    """
    return 1.0/np.sqrt(gamma(1.0+n))



class Laguerre(OrthogonalPolynomial):
  """
    Provides polynomial Laguerre generators and evaluators for stochastic collocation.
  """
  def initialize(self, quad):
    """
      Initializes object with items not set in __init__
      @ In, quad, string, quadrature object name
      @ Out, None
    """
    OrthogonalPolynomial.initialize(self, quad)
    self.printTag = 'LAGUERRE-ORTHOPOLY'
    self._poly    = polys.genlaguerre
    self._evPoly  = polys.eval_genlaguerre
    self.params   = quad.params
    self.setMeasures(quad)

  def setMeasures(self,quad):
    """
      Implements specific measures for given quadSet.
      @ In, quad, Quadrature object, quadrature that will make coeffs for these polys
      @ Out, None
    """
    if quad.type=='Laguerre':
      self.pointMod = self.stdPointMod
      self.quad = quad
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    """
      Used to make standardized distribution for this poly type.
      @ In, None
      @ Out, gamma, Distribution, the gamma distribution
    """
    gamma = Distributions.Gamma(0.0,self.params[0]+1)
    gamma.initializeDistribution()
    return gamma

  def norm(self,order):
    """
      Normalization constant for polynomials so that integrating two of them
      w.r.t. the weight factor produces the kroenecker delta. Default is 1.
      @ In, order, int, polynomial order to get norm of
      @ Out, norm, float, value of poly norm
    """
    return np.sqrt(gamma(1.0+order)*gamma(1.0+self.params[0])/gamma(1.0+order+self.params[0]))



class Jacobi(OrthogonalPolynomial):
  """
    Provides polynomial Jacobi generators and evaluators for stochastic collocation.
  """
  def initialize(self, quad):
    """
      Initializes object with items not set in __init__
      @ In, quad, string, quadrature object name
      @ Out, None
    """
    OrthogonalPolynomial.initialize(self, quad)
    self.printTag = 'JACOBI-ORTHOPOLY'
    self._poly    = polys.jacobi
    self._evPoly  = polys.eval_jacobi
    self.params   = quad.params
    self.setMeasures(quad)

  def setMeasures(self,quad):
    """
      Implements specific measures for given quadSet.
      @ In, quad, Quadrature object, quadrature that will make coeffs for these polys
      @ Out, None
    """
    if quad.type=='Jacobi':
      self.pointMod = self.stdPointMod
      self.quad = quad
    else:
      OrthogonalPolynomial.setMeasures(self,quad)

  def makeDistribution(self):
    """
      Used to make standardized distribution for this poly type.
      @ In, None
      @ Out, jacobiElement, Distribution, jacobi distribution
    """
    jacobiElement = ET.Element("jacobi")
    element = ET.Element("alpha",{})
    element.text = "%s" %self.params[1]+1
    jacobiElement.append(element)
    element = ET.Element("beta",{})
    element.text = "%s" %self.params[0]+1
    jacobiElement.append(element)

  def norm(self,n):
    """
      Normalization constant for polynomials so that integrating two of them
      w.r.t. the weight factor produces the kroenecker delta. Default is 1.
      @ In, n, int, polynomial order to get norm of
      @ Out, norm, float, value of poly norm
    """
    a=self.params[0]
    b=self.params[1]
    ###speedup attempt 2###
    coeff=(2.0*n+a+b+1.0)*\
          gamma(n+1)*gamma(n+a+b+1)/(gamma(n+a+1)*gamma(n+b+1))*\
          gamma(a+1)*gamma(b+1)/gamma(a+b+2.0)
    return np.sqrt(coeff)


factory = EntityFactory('OrthoPolynomial')
factory.registerAllSubtypes(OrthogonalPolynomial)
