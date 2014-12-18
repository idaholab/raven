'''
Created on Dec 2, 2014

@author: talbpw
'''
#for future compatibility with Python 3-----------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3-------------------------------------------------

#External Modules---------------------------------------------------------------------
import numpy as np
import scipy.special.orthogonal as quads
from scipy.fftpack import ifft
from scipy.misc import factorial
from itertools import product
from collections import OrderedDict as OrdDict
#External Modules End-----------------------------------------------------------------

#Internal Modules
from BaseClasses import BaseType
from utils import returnPrintTag, returnPrintPostTag, find_distribution1D
#Internal Modules End-----------------------------------------------------------------


class SparseQuad(BaseType):
  '''Base class to produce sparse-grid multiple-dimension quadrature.
     Requires: dimension N, max polynomial level L, quadrature generation rules for each dimension, distributions?
  '''
  def __init__(self):
    self.c = [] #array of coefficient terms for smaller tensor grid entries

  def initialize(self, indexSet, quadRule, distrList):
    self.indexSet = np.array(indexSet[:])
    self.quadRule = quadRule
    self.distrList = distrList
    self.N= len(distrList.keys())
    maxPoly = 0
    for distr in self.distrList.values(): #TODO dict keys or values?  How are they stored?  Or list?
      maxPoly = max(maxPoly,distr.maxPolyOrder())
    #self.serialMakeCoeffs()
    #we can cheat if it's tensor product index set
    if indexSet.type=='Tensor Product':
      self.c=[1]#np.zeros(len(self.indexSet))
      self.indexSet=[self.indexSet[-1]]
    else:
      self.smarterMakeCoeffs()
      survive = np.nonzero(self.c!=0)
      self.c=self.c[survive]
      self.indexSet=self.indexSet[survive]
    self.SG=OrdDict() #keys on points, values on weights
    for j,cof in enumerate(self.c):
      idx = self.indexSet[j]
      m = self.quadRule(idx)+1
      new = self.tensorGrid(m,idx)
      for i in range(len(new[0])):
        newpt=tuple(new[0][i])
        newwt=new[1][i]*self.c[j]
        if newpt in self.SG.keys(): #possible point duplication
          self.SG[newpt]+=newwt
        else:
          self.SG[newpt] = newwt

  def __getitem__(self,n):
    return self.points(n),self.weights(n)

  def __len__(self):
    return len(self.weights())

  def __repr__(self):
    msg='SparseQuad:\n'
    for p in range(len(self)):
      msg+='    '+str(self[p])+'\n'
    return msg

  def _extrema(self):
    import matplotlib.pyplot as plt
    #find lowest pt
    points = self.point()
    low= np.ones(len(points[0]))*1e300
    hi = np.ones(len(points[0]))*(-1e300)
    for pt in pts:
      for i,p in enumerate(pt):
        low[i]=min(low[i],p)
        hi[i] =max(hi[i] ,p)
    return low,hi

  def _xy(self):
    return zip(*self.points())

  def _pointKey(self):
    #return self.distrList.keys()
    return list(d.type for d in self.distrList.values())

  def points(self,n=None):
    if n==None:
      return self.SG.keys()
    else:
      return self.SG.keys()[n]

  def weights(self,n=None):
    if n==None:
      return self.SG.values()
    else:
      return self.SG.values()[n]
  
  def serialMakeCoeffs(self):
    '''Brute force method to create coefficients for each index set in the sparse grid approximation.
      This particular implementation is faster for 2 dimensions, but slower for
      more than 2 dimensions, than the smarterMakeCeoffs.'''
    print('WARNING: serialMakeCoeffs may be broken.  smarterMakeCoeffs is better.')
    self.c=np.zeros(len(self.indexSet))
    jIter = product([0,1],repeat=self.N) #all possible combinations in the sum
    for jx in jIter: #from here down goes in the paralellized bit
      for i,ix in enumerate(self.indexSet):
        ix = np.array(ix)
        comb = tuple(jx+ix)
        if comb in self.indexSet:
          self.c[i]+=(-1)**sum(jx)

  def smarterMakeCoeffs(self):
    '''Somewhat optimized method to create coefficients for each index set in the sparse grid approximation.
       This particular implementation is faster for any more than 2 dimensions in comparison with the
       serialMakeCoeffs method.'''
    N=len(self.indexSet)
    #iSet = np.array(self.indexSet) #slower than looping
    iSet = self.indexSet[:]
    #for i,st in enumerate(iSet):
    #  iSet[i]=np.array(st)
    self.c=np.ones(N)
    for i in range(N): #could be parallelized from here
      idx = iSet[i]
      for j in range(i+1,N):
        jdx = iSet[j]
        d = jdx-idx
        if all(np.logical_and(d>=0,d<=1)):
          self.c[i]+=(-1)**sum(d)

  def tensorGrid(self,m,idx):
    pointLists=[]
    weightLists=[]
    for n,distr in enumerate(self.distrList.values()):
      mn = m[n]
      pts,wts=distr.quadratureSet()(mn)
      pointLists.append(pts)
      weightLists.append(wts)
    points = list(product(*pointLists))
    weights= list(product(*weightLists))
    for k,wtset in enumerate(weights):
      weights[k]=np.product(wtset)
    return points,weights

class QuadratureSet(BaseType):
  '''Base class to produce standard quadrature points and weights.
     Points and weights are obtained as

     myQuad = Legendre()
     pts,wts = myQuad(n)
  '''
  def __init__(self):
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    self.debug = False

  def __call__(self,order):
    '''Defines operations to return correct pts, wts'''
    pts,wts = self.rule(order,*self.params)
    pts = np.around(pts,decimals=15) #helps with nesting, might not be desirable
    if self.debug: print(self.printTag,'TODO this could probably be optimized further')
    return pts,wts

  def whatDoINeed(self):
    needDict = self._localWhatDoINeed()
    for val in self.assemblerObjects.values():
      for value in val:
        if value[0] not in needDict.keys(): needDict[value[0]] = []
        needDict[value[0]].append((value[1],value[2]))
    return needDict

  def _localWhatDoINeed(self):
    return {}

  def generateAssembler(self,initDict):
    self._localGenerateAssembler(initDict)

  def _localGenerateAssembler(self,initDict):
    pass

  def initialize(self,*params):
    self.rule   = None   # Function as rule(n) that takes an integer order and returns pts, wts
    self.params = params # list of additional parameters necessary for some quadratures

  def _readMoreXML(self,xmlNode):
    if self.debug:print('Quadrature: need to fix _readMoreXML')
    self._localReadMoreXML(xmlNode)
    return
    try:
      self.type = xmlNode.tag
      self.name = xmlNode.attrib['name']
      self.printTag = self.type.ljust(25)
      if 'debug' in xmlNode.attrib.keys(): self.debug = bool(xmlNode.attrib['debug'])
      #TODO assembler stuff
      if self.debug:print('TODO Quadrature needs to implement assembler stuff in readXML')
    except: pass

  def _localReadMoreXML(self,xmlNode):
    pass



class Legendre(QuadratureSet):
  def initialize(self):
    self.rule   = quads.p_roots
    self.params = []
    self.pointRule = GaussQuadRule

class CDF(Legendre): #added just for name distinguish; equiv to Legendre
  pass #TODO why don't I want this to be ClenshawCurtis by default?

class Hermite(QuadratureSet):
  def initialize(self):
    self.rule   = quads.he_roots
    self.params = []
    self.pointRule = GaussQuadRule


class Laguerre(QuadratureSet):
  def initialize(self):
    self.rule   = quads.la_roots
    self.pointRule = GaussQuadRule

  def _localReadMoreXML(self,xmlNode):
    self.params=[]
    if xmlNode.find('alpha') != None:
      alpha = float(xmlNode.find('alpha').text)
    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Laguerre quadrature requires alpha keyword; not found.')
    self.params = [alpha-1]


class Jacobi(QuadratureSet):
  def initialize(self):
    self.rule   = quads.j_roots
    self.pointRule = GaussQuadRule

  def _localReadMoreXML(self,xmlNode):
    self.params = []
    if xmlNode.find('alpha') != None:
      alpha=float(xmlNode.find('alpha').text)
    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Jacobi quadrature requires alpha keyword; not found.')
    if xmlNode.find('beta') != None:
      beta=float(xmlNode.find('beta').text)
    else: raise IOError(self.printTag+': '+returnPrintPostTag('ERROR')+'->Jacobi quadrature requires beta keyword; not found.')
    self.params = [beta-1,alpha-1]
    #NOTE this looks totally backward, BUT it is right!
    #The Jacobi measure switches the exponent naming convention
    #for Beta distribution, it's  x^(alpha-1) * (1-x)^(beta-1)
    #for Jacobi measure, it's (1+x)^alpha * (1-x)^beta


class ClenshawCurtis(QuadratureSet):
  def initialize(self):
    self.rule = self.cc_roots
    self.params = []
    self.pointRule = CCQuadRule

  def cc_roots(self,o):
    '''Computes Clenshaw Curtis nodes and weights for given order n=2^o+1'''
    n1=o #assures nested -> don't assume!
    if o==1:
      return np.array([np.array([0]),np.array([2])])
    else:
      n = n1-1
      C = np.zeros((n1,2))
      k = 2*(1+np.arange(np.floor(n/2)))
      C[::2,0] = 2/np.hstack((1,1-k*k))
      C[1,1]=-n
      V = np.vstack((C,np.flipud(C[1:n,:])))
      F = np.real(ifft(V,n=None,axis=0))
      x = F[0:n1,1]
      w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
    return x,w


def CCQuadRule(i):
  try: return np.array(list((0 if p==0 else 2**p) for p in i))
  except TypeError: return 0 if i==0 else 2**i


def GaussQuadRule(i):
  return i


'''
 Interface Dictionary (factory) (private)
'''
__base = 'QuadratureSet'
__interFaceDict = {}
__interFaceDict['Legendre'] = Legendre
__interFaceDict['Cdf'] = CDF
__interFaceDict['Hermite'] = Hermite
__interFaceDict['Laguerre'] = Laguerre
__interFaceDict['Jacobi'] = Jacobi
__interFaceDict['ClenshawCurtis'] = ClenshawCurtis
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

