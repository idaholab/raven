'''
Created Feb 10, 2013

@author: talbpaul
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
import sys
import scipy.special.orthogonal as orth
import scipy.stats as spst
import scipy.special as sps
from scipy.misc import factorial
from itertools import product #all possible combinations from given sets



#============================================================================\
class Quadrature():
  '''
  Base class for developing quadrature of any dimension.
  '''
  def __init__(self,order=2):
    self.type = '' #string of quadrature type
    self.order = order #quadrature order
    self.quad_pts = np.zeros(self.order) #points in quadrature
    self.weights = np.zeros(self.order) #weights in quadrature
    self.setQuad() #sets up points and weights
    self.setDist() #sets up associated distribution

  def setQuad(self):
    raise IOError('Quadrature.setQuad must be overwritten by individual quadrature!')
  def setDist(self):
    raise IOError('Quadrature.setDist must be overwritten by individual quadrature!')

  def resetQuad(self,order):
    '''Sets quadrature with new order.'''
    #TODO in lobatto quad, allow for just adding points
    self.order=order
    self.setQuad()

  def integrate(self,func,mult=1.0):
    '''Integrates a given function using the quadrature points and weights.'''
    #overwritten in the multiquad class, this is for single quads
    result=0
    for n in range(len(self.quad_pts)):
      result+=self.weights[n]*func(self.quad_pts[n])
    return result*mult

  def integrateArray(self,arr,mult=1.0):
    '''Uses weights to integrate given array of values, 
       assuming they are on quad_pts.'''
    # same for individual quads or multiquad
    assert(len(arr) == len(self.weights))
    result=sum(arr*self.weights)
    return result*mult

  def keepReal(self,arr):
    '''Sometimes the quad points come out complex with 0j in the complex plain.
       This method checks to make sure it's zero, then gets rid of it.'''
    return np.real_if_close(arr)


#============================================================================\
class Legendre(Quadrature):
  def __init__(self,order=4,a=-1,b=1):
    self.a = a #lower bound, default -1
    self.b = b #upper bound, default  1
    Quadrature.__init__(self,order=order)

  def setQuad(self):
    self.type='Legendre'
    self.quad_pts,self.weights = orth.p_roots(self.order) #points and weights from scipy
    print ('Weights for quad',self,'are',self.weights)
    print ('Points for quad',self,'are',self.quad_pts)
    self.quad_pts = self.keepReal(self.quad_pts)
    self.keepReal = self.keepReal(self.weights)

  def setDist(self):
  #  self.dist=spst.uniform(-1,1) #associated distribution
    self.poly=sps.legendre #associated polynomials

  def evNormPoly(self,o,x):
    '''Evaluates normalized associated polynomials of order o at point x.'''
    return sps.eval_legendre(o,x)*np.sqrt((2.*o+1.)/2.)



class ShiftLegendre(Quadrature):
  def setQuad(self):
    self.type='ShiftedLegendre'
    self.quad_pts,self.weights=orth.ps_roots(self.order)
    self.quad_pts = self.keepReal(self.quad_pts)
    self.keepReal = self.keepReal(self.weights)

  def setDist(self):
  #  self.dist=spst.uniform(0,1)
    self.poly=sps.sh_legendre

  def evNormPoly(self,o,x):
    return sps.eval_sh_legendre(o,x)*np.sqrt(2.*o+1.)


#============================================================================\
class Hermite(Quadrature):
  def setQuad(self):
    self.type='Hermite'
    self.quad_pts,self.weights = orth.h_roots(self.order)
    self.quad_pts = self.keepReal(self.quad_pts)
    self.keepReal = self.keepReal(self.weights)

  def setDist(self):
  #  self.dist=spst.norm() #FIXME is this true?? exp(-x^2/<<2>>)
    self.poly=sps.hermite

  def evNormPoly(self,o,x):
    return sps.eval_hermitenorm(o,x)/np.sqrt(np.sqrt(np.pi)*2.**o*factorial(o))




class StatHermite(Quadrature):
  def setQuad(self):
    self.type='StatisticianHermite'
    self.quad_pts,self.weights = orth.he_roots(self.order)
    self.quad_pts = self.keepReal(self.quad_pts)
    self.keepReal = self.keepReal(self.weights)

  def setDist(self):
  #  self.dist=spst.norm()
    self.poly=sps.hermitenorm

  def evNormPoly(self,o,x):
    try:
      o=np.real(o)
      x=np.real(x)
    except:
      print(self,'.evNormPoly tried to convert to real but it failed.  Moving on.')
    return sps.eval_hermitenorm(o,x)/np.sqrt(np.sqrt(2.*np.pi)*factorial(o))




#============================================================================\
class Laguerre(Quadrature):
  def __init__(self,alpha,order):
    self.alpha=alpha
    Quadrature.__init__(self,order=order)

  def setQuad(self):
    self.quadType='GenLaguerre'
    self.quad_pts,self.weights = orth.la_roots(self.order,self.alpha)
    self.quad_pts = self.keepReal(self.quad_pts)
    self.keepReal = self.keepReal(self.weights)

  def setDist(self):
  #  self.dist=spst.gamma(self.alpha) #shift from [a,inf] to [0,inf]?
    self.poly=sps.genlaguerre

  def evNormPoly(self,o,x):
    return sps.eval_genlaguerre(o,self.alpha,x)/\
        np.sqrt(sps.gamma(o+self.alpha+1.)/factorial(o))

#============================================================================\
class Jacobi(Quadrature):
  def __init__(self,alpha,beta,order=4):
    self.alpha=alpha
    self.beta=beta
    Quadrature.__init__(self,order)

  def setQuad(self):
    self.quadType='Jacobi'
    self.quad_pts,self.weights = orth.j_roots(self.order,self.alpha,self.beta)
    self.quad_pts = self.keepReal(self.quad_pts)
    self.keepReal = self.keepReal(self.weights)

  def setDist(self):
  #  self.dist=spst.beta(self.alpha,self.beta)
    self.poly=sps.jacobi

  def evNormPoly(self,o,x):
    a=self.alpha
    b=self.beta
    gam=sps.gamma
    return sps.eval_jacobi(o,self.alpha,self.beta,x)/\
        np.sqrt(2**(a+b+1.)/(2.*o+a+b+1.))/\
        np.sqrt(gam(o+a+1.)*gam(o+b+1.)/(factorial(o)*gam(o+a+b+a+1.)))




#============================================================================\
class MultiQuad(Quadrature):
  '''
  Combines two or more quadratures to create ordinates, weights, integrate.
  '''
  def __init__(self,quads):
    self.quads=quads #dict of quadratures, indexed on var names
    self.type='multi-' #will be populated with names later
    self.totOrder=np.product([q.order for q in self.quads.values()]) #total quadrature order

    #these are all indexed on individual quadratures
    self.order={} #dict of orders of quadratures on quad
    self.dict_quads={}

    #these are indexed on possible combinations
    self.quad_pts={} #np.zeros(self.totOrder,dtype=tuple) #placeholder for quad_pt ordered tuples
    self.weights={} #np.zeros_like(self.quad_pts) #placeholder for weights associated with quad_pt tuples

    #lookup dictionaries
    self.indx_quad_pt = {} #index to quad_pt tuples
    self.indx_weight={} #index to weights
    self.quad_pt_index={} #quad_pts to indexes
    self.quad_pt_weight={} #quad_pts to weights
    #Quadrature.__init__(self)

    for i,q in enumerate(quads.values()):
      self.dict_quads[q]=i
      self.order[q]=q.order
      self.type+=q.type+'('+str(self.order[q])+')'
      self.type+='-' #seperator for quad type name
    self.type=self.type[:-1]
    print('Quads are:',self.type)

    #use itertools.product to get all possible quad_pt tuples from individual quad lists
    self.indices=list(product(*[range(len(quad.quad_pts)) for quad in self.quads.values()]))
    self.quad_pts=list(product(*[quad.quad_pts for quad in self.quads.values()]))
    weights=list(product(*[quad.weights for quad in self.quads.values()]))
    #multiply weights together instead of storing each seperately -> no need for separate
    self.weights = list(np.product(w) for w in weights)

    #make set of dictionaries
    self.indx_quad_pt=dict(zip(self.indices,self.quad_pts))
    self.indx_weight=dict(zip(self.indices,self.weights))
    self.quad_pt_index=dict(zip(self.quad_pts,self.indices))
    self.quad_pt_weight=dict(zip(self.quad_pts,self.weights))


  def integrate(self,func,mult=1.0):
    '''Integrates given function using inputs from nD quadrature.'''
    result=0
    for n in range(len(self.quad_pts)):
      result+=self.weights[n]*func(*self.quad_pts[n])
    return result*mult




#utilities
def part_ndenum(arr,lvl):
  '''Enumerates a multi-D array at a particular depth of dimension.
     For example, let arr be a 4D hypercube array of arrays of arrays of arrays.
     part_ndenum(arr,2) would enumerate over all the 2D arrays in all the possible 
     1st and 2nd dimension combinations.'''
  try:
    arr=np.array(arr)
    lvl=int(lvl)
  except ValueError: raise IOError('Inputs to Quadrature.part_ndenum must be array,integer!')
  assert lvl<=len(arr.shape),'Requested enumeration level > array dimension!'
  idx=np.ndindex(arr.shape[:lvl])
  for i in idx:
    yield i,arr[i]


def returnInstance(Type):
  base = 'Quadrature'
  InterfaceDict = {}
  InterfaceDict['Legendre'       ] = Legendre
  InterfaceDict['MultiQuad'      ] = MultiQuad
  InterfaceDict['Jacobi'         ] = Jacobi
  InterfaceDict['Laguerre'       ] = Laguerre
  InterfaceDict['StatHermite'    ] = StatHermite
  InterfaceDict['Hermite'        ] = Hermite
  InterfaceDict['ShiftLegendre'  ] = ShiftLegendre  
  print(Type)
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
