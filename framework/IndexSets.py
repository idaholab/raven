from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
import itertools
from operator import itemgetter
import sys

import MessageHandler

class IndexSet(MessageHandler.MessageUser):
  """In stochastic collocation for generalised polynomial chaos, the Index Set
     is a set of all combinations of polynomial orders needed to represent the
     original model to a "level" L (maxPolyOrder).
  """
  def __init__(self):
    self.type          = 'IndexSet' #type of index set (Tensor Product, Total Degree, Hyperbolic Cross)
    self.printTag      = 'IndexSet' #type of index set (Tensor Product, Total Degree, Hyperbolic Cross)
    self.maxOrds       = None #maximum requested polynomial order requested for each distribution
    self.points        = []   #array of polynomial order tuples
    self.maxPolyOrder  = None #integer, maximum order polynomial to use in any one dimension -> misleading! Relative order for anisotropic case
    self.polyOrderList = []   #array of lists containing all the polynomial orders needed for each dimension
    self.impWeights    = []   #array of scalars for assigning importance weights to each dimension

  def __len__(self):
    """Returns number of entries in the index set.
    @ In , None  , None
    @ Out, int  , cardinality of index set
    """
    return len(self.points)

  def __getitem__(self,i=None):
    """Returns as if called on self.points.
    @ In , i     , string/int           , splice notation for array
    @ Out, array of tuples/tuple, requested points
    """
    if i==None: return np.array(self.points)
    else: return self.points[i]

  def __repr__(self):
    """Produces a more human-readable version of the index set.
    @ In, None, None
    @ Out, string, visual representation of index set
    """
    msg='IndexSet Printout:\n'
    if len(self.points[0])==2: #graphical block visualization
      left=0
      p=0
      while p<len(self.points)-1:
        pt = self.points[p]
        if pt[0]==left:
          msg+='  '+str(pt)
          p+=1
        else:
          msg+='\n'
          left+=1
    else: #just list them
      for pt in self.points:
        msg+='  '+str(pt)+'\n'
    return msg

  def __eq__(self,other):
    """Checks equivalency of index set
    @ In , other, object , object to compare to
    @ Out, boolean, equivalency
    """
    return self.type == other.type and \
           self.points == other.points and \
           (self.impWeights == other.impWeights).all()

  def __ne__(self,other):
    """Checks non-equivalency of index set
    @ In , other  , object , object to compare to
    @ Out, boolean, non-equivalency
    """
    return not self.__eq__(other)

  def _xy(self):
    """Returns reordered data.  Originally,
       Points = [(a1,b1,...,z1),
                 (a2,b2,...,z2),
                 ...]
       Returns [(a1,a2,a3,...),
                (b1,b2,b3,...),
                ...,
                (z1,z2,z3,...)]
    @ In , None  , None
    @ Out, array of tuples, points by dimension
    """
    return zip(*self.points)

  def initialize(self,distrList,impList,maxPolyOrder,msgHandler):
    """Initialize everything index set needs
    @ In , distrList   , dictionary of {varName:Distribution}, distribution access
    @ In , impList     , dictionary of {varName:float}, weights by dimension
    @ In , maxPolyOrder, int, relative maximum polynomial order to be used for index set
    @ Out, None        , None
    """
    numDim = len(distrList)
    #set up and normalize weights
    #  this algorithm assures higher weight means more importance,
    #  and end product is normalized so smallest is 1
    self.impWeights = np.array(list(impList[v] for v in distrList.keys()))
    self.impWeights/= np.max(self.impWeights)
    self.impWeights = 1.0/self.impWeights
    self.messageHandler=msgHandler
    #establish max orders
    self.maxOrder=maxPolyOrder
    self.polyOrderList=[]
    for distr in distrList.values():
      self.polyOrderList.append(range(self.maxOrder+1))

  def generateMultiIndex(self,N,rule,I=None,MI=None):
    """Recursive algorithm to build monotonically-increasing-order index set.
    @ In, N   , int            , dimension of the input space
    @ In, rule, function       , rule for type of index set (tensor product, total degree, etc)
    @ In, I   , array of scalar, single index point
    @ In, MI  , array of tuples, multiindex point collection
    @ Out, array of tuples, index set
    """
    L = self.maxOrder
    if I ==None: I =[]
    if MI==None: MI=[]
    if len(I)!=N:
      i=0
      while rule(I+[i]): #rule is defined by subclasses, limits number of index points by criteria
        MI = self.generateMultiIndex(N,rule,I+[i],MI)
        i+=1
    else:
      MI.append(tuple(I))
    return MI



class TensorProduct(IndexSet):
  """This Index Set requires only that the max poly order in the index point i is less than maxPolyOrder ( max(i)<=L )."""
  def initialize(self,distrList,impList,maxPolyOrder,messageHandler):
    IndexSet.initialize(self,distrList,impList,maxPolyOrder,messageHandler)
    self.type='Tensor Product'
    self.printTag='TensorProductIndexSet'
    target = sum(self.impWeights)/float(len(self.impWeights))*self.maxOrder
    def rule(i):
      big=0
      for j,p in enumerate(i):
        big=max(big,p*self.impWeights[j])
      return big <= target
    self.points = self.generateMultiIndex(len(distrList),rule)



class TotalDegree(IndexSet):
  """This Index Set requires the sum of poly orders in the index point is less than maxPolyOrder ( sum(i)<=L )."""
  def initialize(self,distrList,impList,maxPolyOrder,messageHandler):
    IndexSet.initialize(self,distrList,impList,maxPolyOrder,messageHandler)
    self.type='Total Degree'
    self.printTag='TotalDegreeIndexSet'
    #TODO if user has set max poly orders (levels), make it so you never use more
    #  - right now is only limited by the maximum overall level (and importance weight)
    target = sum(self.impWeights)/float(len(self.impWeights))*self.maxOrder
    def rule(i):
      tot=0
      for j,p in enumerate(i):
        tot+=p*self.impWeights[j]
      return tot<=target
    self.points = self.generateMultiIndex(len(distrList),rule)



class HyperbolicCross(IndexSet):
  """This Index Set requires the product of poly orders in the index point is less than maxPolyOrder ( prod(i+1)<=L+1 )."""
  def initialize(self,distrList,impList,maxPolyOrder,messageHandler):
    IndexSet.initialize(self,distrList,impList,maxPolyOrder,messageHandler)
    self.type='Hyperbolic Cross'
    self.printTag='HyperbolicCrossIndexSet'
    #TODO if user has set max poly orders (levels), make it so you never use more
    #  - right now is only limited by the maximum overall level (and importance weight)
    target = (self.maxOrder+1)**(sum(self.impWeights)/max(1,float(len(self.impWeights))))
    def rule(i):
      tot=1;
      for e,val in enumerate(i):
        tot*=(val+1)**self.impWeights[e]
      return tot<=target
    self.points = self.generateMultiIndex(len(distrList),rule)



class AdaptiveSet(IndexSet):
  def initialize(self,distrList,impList,maxPolyOrder,messageHandler):
    IndexSet.initialize(self,distrList,impList,maxPolyOrder,messageHandler)
    self.type     = 'Adaptive Index Set'
    self.printTag = self.type
    self.N        = len(distrList)
    self.points   = [tuple([0]*self.N)] #retained points in the index set
    self.rejects  = [] #list of tuples, rejected points in index set (too high order)
    self.shells   = [] #list of lists of tuples, retained points by adaptive layer 
    self.toTry    = [] #list of new viable points to try
    self.newestPoint = self.points[0] #tuple, new point to test adding

    self.shells.append([self.points[0]])
    self.raiseADebug('Variable key: '+str(distrList.keys()))

  def addPoint(self,maxPolyOrder=None):
    if len(self.toTry)<1:
      self.toTry = self.provideNextLayer(maxPolyOrder)
      if len(self.toTry)<1: raise MessageHandler.NoMoreSamplesNeeded
    self.newestPoint = self.toTry.pop()
    self.raiseADebug('Trying '+str(self.newestPoint))
    if self.newestPoint not in self.points:
      self.points.append(self.newestPoint)

  def provideNextLayer(self,maxPolyOrder=None):
    if len(self.shells[-1])==0: return []
    self.raiseADebug('Adding a layer...')
    #self.printOut()
    new=[]
    for oldpt in self.shells[-1]:
      #self.raiseADebug('    Expanding on '+str(oldpt))
      for i in range(self.N):
        newpt = np.array(oldpt)
        newpt[i]+=1
        if maxPolyOrder!=None and sum(newpt>maxPolyOrder)>0:
          self.raiseADebug("    Rejected point "+str(newpt)+" for too large polynomial order.")
        else:
          newpt=tuple(newpt)
          if newpt not in new and newpt not in self.rejects:
            new.append(newpt)
    self.shells.append([])
    return new

  def printOut(self):
    self.raiseADebug('    Index Set:')
    for l in self.shells:
      self.raiseADebug('        '+str(l))
    self.raiseADebug('    Rejects:')
    for r in self.rejects:
      self.raiseADebug('        '+str(r))

  def reject(self):
    self.raiseADebug('    Rejecting '+str(self.newestPoint))
    self.rejects.append(self.newestPoint)
    self.points.remove(self.newestPoint)

  def accept(self):
    self.raiseADebug('    Keeping '+str(self.newestPoint))
    if self.newestPoint not in self.shells[-1]:
      self.shells[-1].append(self.newestPoint)



"""
Interface Dictionary (factory) (private)
"""
__base = 'IndexSet'
__interFaceDict = {}
__interFaceDict['TensorProduct'  ] = TensorProduct
__interFaceDict['TotalDegree'    ] = TotalDegree
__interFaceDict['HyperbolicCross'] = HyperbolicCross
__interFaceDict['AdaptiveSet'    ] = AdaptiveSet
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  if Type in knownTypes(): return __interFaceDict[Type]()
  else: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
