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
Created on Dec 2, 2014

@author: talbpw
"""
#for future compatibility with Python 3-----------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------------

#External Modules---------------------------------------------------------------------
import numpy as np
import scipy.special.orthogonal as quads
import scipy.fftpack as fftpack
import itertools
import collections
import operator
#External Modules End-----------------------------------------------------------------

#Internal Modules
from .EntityFactoryBase import EntityFactory
from .BaseClasses import MessageUser
from .Decorators.Parallelization import Parallel
#Internal Modules End-----------------------------------------------------------------


class SparseGrid(MessageUser):
  """
    Base class to produce sparse-grid multiple-dimension quadrature.
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type           = 'BaseSparseQuad'
    self.printTag       = 'BaseSparseQuad'
    self.c              = []                                                      # array of coefficient terms for component tensor grid entries
    self.oldsg          = []                                                      # storage space for re-ordered versions of sparse grid
    self.indexSet       = None                                                    # IndexSet object
    self.distDict       = None                                                    # dict{varName: Distribution object}
    self.quadDict       = None                                                    # dict{varName: Quadrature object}
    self.polyDict       = None                                                    # dict{varName: OrthoPolynomial object}
    self.varNames       = []                                                      # array of names, in order of distDict.keys()
    self.N              = None                                                    # dimensionality of input space
    self.SG             = None                                                    # dict{ (point,point,point): weight}

  def initialize(self, varNames, indexSet, distDict, quadDict, handler):
    """
      Initializes sparse quad to be functional. At the end of this method, all points and weights should be set.
      @ In, varNames, list, the ordered list of grid dimension names
      @ In, indexSet, IndexSet object, index set
      @ In, distDict, dict{varName,Distribution object}, distributions
      @ In, quadDict, dict{varName,Quadrature object}, quadratures
      @ In, handler, JobHandler, parallel processing tool
      @ Out, None
    """
    self.origIndexSet   = indexSet
    self.indexSet       = np.array(indexSet[:])
    self.distDict       = distDict
    self.quadDict       = quadDict
    self.varNames       = varNames
    self.N              = len(self.varNames)
    self.SG             = collections.OrderedDict() #keys on points, values on weights
    #add methods to construct grid here

  ##### OVERWRITTEN BUILTINS #####
  def __getitem__(self,n):
    """
      Returns the point and weight for entry 'n'.
      @ In, n, int, index of desired components
      @ Out, __getitem__, tuple, points and weight at index n
    """
    return self.points(n),self.weights(n)

  def __len__(self):
    """
      Returns cardinality of sparse grid.
      @ In, None
      @ Out, __len__, int, size of sparse grid
    """
    return len(self.weights())

  def __repr__(self):
    """
      Slightly more human-readable version of printout.
      @ In, None
      @ Out, msg, string, list of points and weights
    """
    msg='SparseQuad: (point) | weight\n'
    for p in range(len(self)):
      msg+='    ('
      pt,wt = self[p]
      for i in pt:
        if i<0:
          msg+='%1.9f,' %i
        else:
          msg+=' %1.9f,' %i
      msg=msg[:-1]+') | %1.9f'%wt+'\n'
    return msg

  def __csv__(self):
    """
      Slightly more human-readable version of printout.
      @ In, None
      @ Out, msg, string, list of points and weights
    """
    msg=''
    for _ in range(len(self[0][0])):
      msg+='pt,'
    msg+='wt\n'
    for p in range(len(self)):
      pt,wt = self[p]
      for i in pt:
        msg+='%1.9f,' %i
      msg+='%1.9f\n' %wt
    return msg

  def __getstate__(self):
    """
      Determines picklable items
      @ In, None
      @ Out, pdict, dict, points and weights
    """
    pdict = self.getInitParams()
    return pdict

  def __setstate__(self,pdict):
    """
      Determines how to load from picklable items
      @ In, pdict, dict, points and weights
      @ Out, None
    """
    self.__init__()
    self.indexSet = pdict.pop('indexSet')
    self.distDict = pdict.pop('distDict')
    self.quadDict = pdict.pop('quadDict') # it was missing. Andrea
    self.varNames = pdict.pop('names')
    points        = pdict.pop('points')
    weights       = pdict.pop('weights')
    self.__initFromPoints(points,weights)

  def __eq__(self,other):
    """
      Checks equivalency between sparsequads
      @ In, other, object, object to compare to
      @ Out, __eq__, bool, equivalency
    """
    if not isinstance(other,self.__class__):
      return False
    if len(self.SG)!=len(other.SG):
      return False
    for pt,wt in self.SG.items():
      if wt != other.SG[pt]:
        return False
    return True

  def __hash__(self):
    """
      Returns a hash for this quadrature
      @ Out, __hash__, int, hash value
    """
    hashes = 0
    for var in self.varNames:
      hashes += hash(var)
    for pt,wt in self.SG.items():
      hashes += hash(pt)+hash(wt)
    return hashes

  def __ne__(self,other):
    """
      Checks inequivalency between sparsequads
      @ In, other, object, object to compare to
      @ Out, __ne__, bool, inequivalency
    """
    return not self.__eq__(other)

  ##### PRIVATE MEMBERS #####
  def __initFromPoints(self,pts,wts):
    """
      Initializes sparse grid from pt, wt arrays
      @ In, pts, array(tuple(float)), points for grid
      @ In, wts, array(float), weights for grid
      @ Out, None
    """
    newSG=collections.OrderedDict()
    for p,pt in enumerate(pts):
      newSG[pt]=wts[p]
    self.SG=newSG

  def __initFromDict(self,ndict):
    """
      Initializes sparse grid from dictionary
      @ In, ndict, {tuple(float): float}, {point: weight}
      @ Out, None
    """
    self.SG=ndict.copy()

  ##### PROTECTED MEMBERS #####
  def _remap(self,newNames):
    """
      Reorders data in the sparse grid.  For instance,
      original:       { (a1,b1,c1): w1,
                        (a2,b2,c2): w2,...}
      remap([a,c,b]): { (a1,c1,b1): w1,
                        (a2,c2,b2): w2,...}
      @ In, newNames, tuple(str), list of dimension names
      @ Out, None
    """
    #TODO optimize me!~~
    oldNames = self.varNames[:]
    self.raiseADebug('REMAPPING SPARSE GRID from '+str(oldNames)+' to '+str(newNames))
    #check consistency
    self.raiseADebug('old: '+str(oldNames)+' | new: '+str(newNames))
    if len(oldNames)!=len(newNames):
      self.raiseAnError(KeyError,'Remap mismatch! Dimensions are not the same!')
    for name in oldNames:
      if name not in newNames:
        self.raiseAnError(KeyError,'Remap mismatch! '+name+' not found in original variables!')
    wts = list(self.weights())
    #split by columns (dim) instead of rows (points)
    oldlists = list(self._xy())
    #stash point lists by name
    oldDict = {}
    for n,name in enumerate(oldNames):
      oldDict[name]=oldlists[n]
    #make new lists
    newlists = list(oldDict[name] for name in newNames)
    #sort new list
    newptwt = list( list(pt)+[wts[p]] for p,pt in enumerate(zip(*newlists)))
    newptwt.sort(key=operator.itemgetter(*range(len(newptwt[0]))))
    #recompile as ordered dict
    newSG=collections.OrderedDict()
    for combo in newptwt:
      newSG[tuple(combo[:-1])]=combo[-1] #weight is last entry
    self.SG = newSG
    self.varNames = newNames

  def _xy(self):
    """
      Returns reordered points.
       Points = [(a1,b1,...,z1),
                 (a2,b2,...,z2),
                 ...]
       Returns [(a1,a2,a3,...),
                (b1,b2,b3,...),
                ...,
                (z1,z2,z3,...)]
      @ In, None
      @ Out, _xy, array of tuples, points by dimension
    """
    return zip(*self.points())

  ##### PUBLIC MEMBERS #####
  def printOut(self):
    """
      Prints the existing quadrature points.
      @ In, None
      @ Out, None
    """
    self.raiseADebug('SparseQuad: (point) | weight')
    msg=''
    for p in range(len(self)):
      msg+='    ('
      pt,wt = self[p]
      for i in pt:
        if i<0:
          msg+='%1.9f,' %i
        else:
          msg+=' %1.9f,' %i
      msg=msg[:-1]+') | %1.9f'%wt
      self.raiseADebug(msg)
      msg=''

  def getInitParams(self):
    """
      Adds params required to initialize an instance of this object to a
      dictionary.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['indexSet']=self.indexSet
    paramDict['distDict']=self.distDict
    paramDict['quadDict']=self.quadDict
    paramDict['names'   ]=self.varNames
    paramDict['points'  ]=self.points()
    paramDict['weights' ]=self.weights()
    return paramDict

  def quadRule(self,idx):
    """
      Collects the cumulative effect of quadrature rules across the dimensions.i
      @ In, idx, tuple(int), index set point
      @ Out, tot, tuple(int), quadrature orders to use
    """
    tot=np.zeros(len(idx),dtype=np.int64)
    for i,ix in enumerate(idx):
      tot[i]=list(self.quadDict.values())[i].quadRule(ix)
    return tot

  def points(self,n=None):
    """
      Returns sparse grid points
      @ In, n, string, splice instruction
      @ Out, points, tuple(float) or tuple(tuple(float)), requested points
    """
    if n is None:
      return list(self.SG.keys())
    else:
      return list(self.SG.keys())[n]

  def weights(self,n=None):
    """
      Either returns the list of weights, or the weight indexed at n, or the weight corresponding to point n.
      @ In, n, string, optional, splice instruction
      @ Out, weights, float or tuple(float), requested weights
    """
    if n==None:
      return list(self.SG.values())
    else:
      try:
        return self.SG[tuple(n)]
      except TypeError:
        return list(self.SG.values())[n]

  @Parallel()
  def tensorGrid(self, m):
    """
      Creates a tensor itertools.product of quadrature points.
      @ In, m, list(int), number points
      @ Out, (points,weights), tuple(tuple(float),float), requisite points and weights
    """
    pointLists=[]
    weightLists=[]
    for n,var in enumerate(self.varNames):
      distr = self.distDict[var]
      quad = self.quadDict[var]
      mn = m[n]
      pts,wts=quad(mn)
      pts=pts.real
      wts=wts.real
      pts = distr.convertToDistr(quad.type,pts)
      pointLists.append(pts)
      weightLists.append(wts)
    points = list(itertools.product(*pointLists))
    weights= list(itertools.product(*weightLists))
    for k,wtset in enumerate(weights):
      weights[k]=np.product(wtset)
    return points,weights
#
#
#
#
class TensorGrid(SparseGrid):
  """
    Not really a sparse grid; this is the naive full grid.
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    SparseGrid.__init__(self)
    self.type     = 'TensorGrid'
    self.printTag = 'TensorGrid'

  def initialize(self, varNames, indexSet, distDict, quadDict, handler):
    """
      Initializes sparse quad to be functional.
      @ In, varNames, list, the ordered list of grid dimension names
      @ In, indexSet, IndexSet object, index set
      @ In, distDict, dict{varName,Distribution object}, distributions
      @ In, quadDict, dict{varName,Quadrature object}, quadratures
      @ In, handler, JobHandler, parallel processing tool
      @ Out, None
    """
    SparseGrid.initialize(self, varNames, indexSet, distDict, quadDict, handler)
    self.type           = 'BaseSparseQuad'
    self.printTag       = 'BaseSparseQuad'
    #find largest polynomial in each dimension
    largest = np.zeros(len(self.indexSet[0]))
    for idx in self.indexSet:
      for i in range(len(idx)):
        largest[i] = max(idx[i],largest[i])
    #construct tensor grid using largest in each dimension
    quadSizes = self.quadRule(largest)+1 #TODO give user access to this +1 rule
    points,weights = self.tensorGrid.original_function(self, quadSizes)
    for i,pt in enumerate(points):
      self.SG[pt] = weights[i]

#
#
#
#
class SmolyakSparseGrid(SparseGrid):
  """
    Uses Smolyak algorithm to construct reduced grids.
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    SparseGrid.__init__(self)
    self.type     = 'SmolyakSparseGrid'
    self.printTag = 'SmolyakSparseGrid'

  def initialize(self, varNames, indexSet, distDict, quadDict, handler):
    """
      Initializes sparse quad to be functional.
      @ In, varNames, list, the ordered list of grid dimension names
      @ In, indexSet, IndexSet object, index set
      @ In, distDict, dict{varName,Distribution object}, distributions
      @ In, quadDict, dict{varName,Quadrature object}, quadratures
      @ In, handler, JobHandler, parallel processing tool
      @ Out, None
    """
    SparseGrid.initialize(self, varNames, indexSet, distDict, quadDict, handler)
    #we know how this ends if it's tensor product index set
    if indexSet.type=='Tensor Product':
      self.c=[1]
      self.indexSet=[self.indexSet[-1]]
    else:
      if handler !=None:
        self.parallelMakeCoeffs(handler)
      else:
        self.smarterMakeCoeffs()
      survive = np.nonzero(self.c!=0)
      self.c=self.c[survive]
      self.indexSet=self.indexSet[survive]
    if handler!=None:
      self.parallelSparseQuadGen(handler)
    else:
      for j,cof in enumerate(self.c):
        idx = self.indexSet[j]
        m = self.quadRule(idx)+1
        new =   self.tensorGrid.original_function(self, m)
        for i in range(len(new[0])):
          newpt=tuple(new[0][i])
          newwt=new[1][i]*cof
          if newpt in self.SG.keys():
            self.SG[newpt]+=newwt
          else:
            self.SG[newpt] = newwt

  def parallelSparseQuadGen(self,handler):
    """
      Generates sparse quadrature points in parallel.
      @ In, handler, JobHandler, parallel processing tool
      @ Out, None
    """
    numRunsNeeded=len(self.c)
    j=-1
    prefix = 'sparseTensor_'
    while True:
      finishedJobs = handler.getFinished(jobIdentifier=prefix) #FIXME this is by far the most expensive line in this method
      #finishedJobs = handler.getFinished(prefix=prefix) #FIXME this is by far the most expensive line in this method
      for job in finishedJobs:
        if job.getReturnCode() == 0:
          new = job.getEvaluation()
          for i in range(len(new[0])):
            newpt = tuple(new[0][i])
            newwt = new[1][i]*float(str(job.identifier).replace(prefix, ""))
            if newpt in self.SG.keys():
              self.SG[newpt]+= newwt
            else:
              self.SG[newpt] = newwt
        else:
          self.raiseAMessage('Sparse quad generation (tensor) '+job.identifier+' failed...')
      if j<numRunsNeeded-1:
        for _ in range(min(numRunsNeeded-1-j,handler.availability())):
          j+=1
          cof=self.c[j]
          idx = self.indexSet[j]
          m=self.quadRule(idx)+1
          handler.addJob((self, m,),self.tensorGrid,prefix+str(cof))
      else:
        if handler.isFinished() and len(handler.getFinishedNoPop())==0:
          break #FIXME this is significantly the second-most expensive line in this method
      import time
      time.sleep(0.005)

  def smarterMakeCoeffs(self):
    """
      Somewhat optimized method to create coefficients for each index set in the sparse grid approximation.
      This particular implementation is faster for any more than 2 dimensions in comparison with the
      serialMakeCoeffs method.
      @ In, None
      @ Out, None
    """
    N=len(self.indexSet)
    iSet = self.indexSet[:]
    self.c=np.ones(N)
    for i in range(N):
      #could be parallelized from here
      idx = iSet[i]
      for j in range(i+1,N):
        jdx = iSet[j]
        d = jdx-idx
        if all(np.logical_and(d>=0,d<=1)):
          #TODO PROFILE ME -> it appears this is slightly faster than (-1)**sum(d)
          if sum(d) % 2 == 0:
            self.c[i] += 1
          else:
            self.c[i] -= 1
          #self.c[i]+=(-1)**sum(d)

  def parallelMakeCoeffs(self,handler):
    """
      Same thing as smarterMakeCoeffs, but in parallel.
      @ In, None
      @ Out, None
    """
    N=len(self.indexSet)
    self.c=np.zeros(N)
    i=-1
    prefix = 'sparseGrid_'
    while True:
      #finishedJobs = handler.getFinished(prefix=prefix)
      finishedJobs = handler.getFinished(jobIdentifier=prefix)
      for job in finishedJobs:
        if job.getReturnCode() == 0:
          self.c[int(str(job.identifier).replace(prefix, ""))]=job.getEvaluation()
        else:
          self.raiseAMessage('Sparse grid index '+job.identifier+' failed...')
      if i<N-1:
        #load new inputs, up to 100 at a time
        for k in range(min(handler.availability(),N-1-i)):
          i+=1
          handler.addJob((N,i,self.indexSet[i],self.indexSet[:]),makeSingleCoeff,prefix+str(i))
      else:
        if handler.isFinished() and len(handler.getFinishedNoPop())==0:
          break
      #TODO optimize this with a sleep time
#
#
#
#
#
#
#
class QuadratureSet(MessageUser):
  """
    Base class to produce standard quadrature points and weights.
     Points and weights are obtained as
     -------------------
     myQuad = Legendre()
     pts,wts = myQuad(n)
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    self.rule  = None #tool for generating points and weights for a given order
    self.params = [] #additional parameters for quadrature (alpha,beta, etc)

  def __call__(self,order):
    """
      Defines operations to return correct pts, wts
      @ In, order, int, order of desired quadrature
      @ Out, tuple(tuple(float),float) points and weight
    """
    pts,wts = self.rule(order,*self.params)
    pts = np.around(pts,decimals=15) #TODO helps with checking equivalence, might not be desirable
    return pts,wts

  def __eq__(self,other):
    """
      Checks equivalency of quad set
      @ In, other, object, object to compare to
      @ Out, bool, equivalency
    """
    return self.rule==other.rule and self.params==other.params

  def __ne__(self,other):
    """
      Checks inequivalency of quad set
      @ In, other, object, object to compare to
      @ Out, bool, inequivalency
    """
    return not self.__eq__(other)

  def initialize(self, distr):
    """
      Initializes specific settings for quadratures.  Must be overwritten.
      @ In, distr, Distribution object, distro represented by this quad
      @ Out, None
    """
    pass

  def quadRule(self,i):
    """
      Quadrature rule to use for order.  Defaults to Gauss, CC should set its own.
      @ In, i, int, quadrature level
      @ Out, quadRule, int, quadrature order
    """
    return GaussQuadRule(i)


class Legendre(QuadratureSet):
  """
    Legendre quadrature
  """
  def initialize(self, distr):
    """
      Initializes specific settings for quadratures.
      @ In, distr, Distribution object, distro represented by this quad
      @ Out, None
    """
    QuadratureSet.initialize(self, distr)
    self.rule   = quads.p_roots
    self.params = []
    self.pointRule = GaussQuadRule

class Hermite(QuadratureSet):
  """
    Hermite quadrature
  """
  def initialize(self, distr):
    """
      Initializes specific settings for quadratures.
      @ In, distr, Distribution object, distro represented by this quad
      @ Out, None
    """
    QuadratureSet.initialize(self, distr)
    self.rule   = quads.he_roots
    self.params = []
    self.pointRule = GaussQuadRule

class Laguerre(QuadratureSet):
  """
    Laguerre quadrature
  """
  def initialize(self, distr):
    """
      Initializes specific settings for quadratures.
      @ In, distr, Distribution object, distro represented by this quad
      @ Out, None
    """
    QuadratureSet.initialize(self, distr)
    self.rule   = quads.la_roots
    self.pointRule = GaussQuadRule
    if distr.type=='Gamma':
      self.params=[distr.alpha-1]
    else:
      self.raiseAnError(IOError,'No implementation for Laguerre quadrature on '+distr.type+' distribution!')

class Jacobi(QuadratureSet):
  """
    Jacobi quadrature
  """
  def initialize(self, distr):
    """
      Initializes specific settings for quadratures.
      @ In, distr, Distribution object, distro represented by this quad
      @ Out, None
    """
    QuadratureSet.initialize(self, distr)
    self.rule   = quads.j_roots
    self.pointRule = GaussQuadRule
    if distr.type=='Beta':
      self.params=[distr.beta-1,distr.alpha-1]
    #NOTE this looks totally backward, BUT it is right!
    #The Jacobi measure switches the exponent naming convention
    #for Beta distribution, it's  x^(alpha-1) * (1-x)^(beta-1)
    #for Jacobi measure, it's (1+x)^alpha * (1-x)^beta
    else:
      self.raiseAnError(IOError,'No implementation for Jacobi quadrature on '+distr.type+' distribution!')

class ClenshawCurtis(QuadratureSet):
  """
    ClenshawCurtis quadrature
  """
  def initialize(self, distr):
    """
      Initializes specific settings for quadratures.
      @ In, distr, Distribution object, distro represented by this quad
      @ Out, None
    """
    QuadratureSet.initialize(self, distr)
    self.rule = self.cc_roots
    self.params = []
    self.quadRule = CCQuadRule

  def cc_roots(self,o):
    """
      Computes Clenshaw Curtis nodes and weights for given order n=2^o+1
      @ In, o, int, level of quadrature to obtain
      @ Out, tuple(tuple(float),float), points and weights
    """
    #TODO FIXME a depreciation warning is being thrown in this prodedure
    n1=o
    if o==1:
      return np.array([np.array([0]),np.array([2])])
    else:
      n = n1-1
      C = np.zeros((n1,2))
      k = 2*(1+np.arange(np.floor(n/2)))
      C[::2,0] = 2/np.hstack((1,1-k*k))
      C[1,1]=-n
      V = np.vstack((C,np.flipud(C[1:n,:])))
      F = np.real(fftpack.ifft(V,n=None,axis=0))
      x = F[0:n1,1]
      w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
    return x,w


class CDFLegendre(Legendre):
  """
    Added just for name distinguish; equiv to Legendre
  """
  pass

class CDFClenshawCurtis(ClenshawCurtis):
  """
    Added just for name distinguish; equiv to ClenshawCurtis
  """
  pass


def CCQuadRule(i):
  """
    In order to get nested points, we need 2**i on Clenshaw-Curtis points instead of just i.
     For example, i=2 is not nested in i==1, but i==2**2 is.
    @ In, i, int, level desired
    @ Out, CCQuadRule, int, desired quad order
  """
  try:
    return np.array(list((0 if p==0 else 2**p) for p in i))
  except TypeError:
    return 0 if i==0 else 2**i


def GaussQuadRule(i):
  """
    We need no modification for Gauss rules, as we don't expect them to be nested.
    @ In, i, int, level desired
    @ Out, i, int, desired quad order
  """
  return i


#Debug module-level method
@Parallel()
def makeSingleCoeff(N,i,idx,iSet):
  """
    Batch-style algorithm to calculate a single coefficient
    @ In, N, int, required arguments
    @ In, i, int, required arguments
    @ In, idx, tuple(int), required arguments
    @ In, iSet, int, required arguments
    @ Out, c, float, coefficient for subtensor i
  """
  #N,i,idx,iSet = arglist
  c=1
  for j in range(i+1,N):
    jdx = iSet[j]
    d = jdx-idx
    if all(np.logical_and(d>=0,d<=1)):
      c += (-1)**sum(d)
  return c

class QuadFactory(EntityFactory):
  """
    Specific factory for this module
  """
  def returnInstance(self, Type, **kwargs):
    """
      Returns an instance pointer from this module.
      @ In, Type, string, requested object
      @ In, kwargs, dict, additional keyword arguments to constructor
      @ Out, returnInstance, instance, instance of the object
    """
    # some modification necessary to distinguish CDF on Legendre versus CDF on ClenshawCurtis
    if Type=='CDF':
      if kwargs['Subtype']=='Legendre':
        return self._registeredTypes['CDFLegendre']()
      elif kwargs['Subtype']=='ClenshawCurtis':
        return self._registeredTypes['CDFClenshawCurtis']()
    return self.returnClass(Type)()

factory = QuadFactory('QuadratureSet')
factory.registerAllSubtypes(QuadratureSet)
factory.registerType('smolyak', SmolyakSparseGrid)
factory.registerType('tensor', TensorGrid)
