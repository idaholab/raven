from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
import itertools
from operator import itemgetter
import sys
import operator

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
      while p<len(self.points):
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

  def print(self):
    self.raiseADebug('IndexSet Printout:')
    if len(self.points[0])==2: #graphical block visualization
      msg=''
      left=0
      p=0
      while p<len(self.points):
        pt = self.points[p]
        if pt[0]==left:
          msg+='  '+str(pt)
          p+=1
        else:
          self.raiseADebug(msg)
          msg=''
          left+=1
      self.raiseADebug(msg)
    else: #just list them
      for pt in self.points:
        self.raiseADebug('  '+str(pt))

  def order(self):
    """
      Orders the index set points in partially-increasing order.
      @ In, None
      @ Out, None
    """
    self.points.sort(key=operator.itemgetter(*range(len(self.points[0]))))

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
    #self.raiseADebug('TD points:')
    #self.print()



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



class Custom(IndexSet):
  """User-based index set point choices"""
  def initialize(self,distrList,impList,maxPolyOrder,messageHandler):
    """see base class"""
    IndexSet.initialize(self,distrList,impList,maxPolyOrder,messageHandler)
    self.type     = 'Custom'
    self.printTag = 'CustomIndexSet'
    self.N        = len(distrList)
    self.points   = []

  def setPoints(self,points):
    """
      Used to set the index set points manually.
      @ In, points, list of tuples to set points to
      @ Out, None
    """
    self.points=[]
    if len(points)>0:
      self.addPoints(points)

  def addPoints(self,points):
    """
      Adds points to existing index set. Reorders set on completion.
      @ In, points, either single tuple or list of tuples to add
      @ Out, None
    """
    if type(points)==list:
      for pt in points: self.points.append(pt)
    elif type(points)==tuple and len(points)==self.N:
      self.points.append(points)
    else: raiseAnError(ValueError,'Unexpected points to add to set:',points)
    self.order()



class AdaptiveSet(IndexSet):
  def initialize(self,distrList,impList,maxPolyOrder,messageHandler):
    IndexSet.initialize(self,distrList,impList,maxPolyOrder,messageHandler)
    self.type     = 'Adaptive Index Set'
    self.printTag = self.type
    self.N        = len(distrList)
    self.points   = [] #retained points in the index set
    firstpoint    = tuple([0]*self.N)
    self.active   = {firstpoint:None}
    self.SGs     = {firstpoint:None}
    self.history  = [] #list of tuples, index set point and its impact parameter

  def setSG(self,point,SG):
    if point in self.SGs.keys(): self.SGs[point]=SG
    else: self.raiseAnError(KeyError,'Tried to set sparse grid',SG,'for point',point,'but it is not in active set!')

  def setImpact(self,point,impact):
    if point in self.active.keys(): self.active[point]=impact
    else: self.raiseAnError(KeyError,'Tried to set impact',impact,'for point',point,'but it is not in active set!')

  def checkImpacts(self):
    for key,impact in self.active.items():
      if impact==None:return False
    return True
  
  def expand(self):
    #get the biggest helper
    pt = self.getBiggestImpact()
    impact = self.active[pt]
    msg=str(pt)+': '+str(impact)+' || '
    for apt,imp in self.active.items():
      msg+=str(apt)+': '+str(imp)+' | '
    self.history.append(msg)
    #make it permanent
    self.points.append(pt)
    self.newestPoint=pt
    self.order() #sort it as partially increasing
    #not an eligible bachelor anymore
    del self.active[pt]
    return pt,impact

  def getBiggestImpact(self):
    if not self.checkImpacts(): self.raiseAnError(ValueError,'Not all impacts have been set for active set!',self.active)
    if len(self.active)<1: self.raiseAnError(ValueError,'No active points in dictionary; search for forward points!')
    if len(self.active)==1: return self.active.keys()[0]
    mx = -1e300
    mxkey=None
    for key,val in self.active.items():
      mx = max(abs(val),mx)
      if abs(val)==mx: mxkey = key
    self.raiseADebug('  biggest impact:',mxkey,mx)
    if mxkey==None: return self.active.keys()[0]
    return mxkey

  def forward(self,pt,maxPoly=None):
    for i in range(self.N):
      newpt = list(pt)
      newpt[i]+=1
      if maxPoly != None:
        if newpt[i]>maxPoly:
          self.raiseADebug('Rejecting',tuple(newpt),'for too high polynomial')
          continue
      if tuple(newpt) in self.active.keys(): continue
      #self.raiseADebug('    considering adding',newpt)
      found=True
      for j in range(self.N):
        checkpt = newpt[:]
        if checkpt[j]==0:continue
        checkpt[j] -= 1
        #self.raiseADebug('        checking subordinate point',checkpt)
        if tuple(checkpt) not in self.points:
          found=False
          break
      if found:
        newpt=tuple(newpt)
        self.active[newpt]=None
        self.SGs   [newpt]=None

  def printOut(self):
    self.raiseADebug('    Accepted Points:')
    for p in self.points:
      self.raiseADebug('       ',p)#,'| %1.5e' %self.roms[p])
    self.raiseADebug('    Active Set | Impact:')
    for a,i in self.active.items():
      self.raiseADebug('       ',a,'|',i)

  def writeHistory(self):
    msg = '\n'.join(self.history)
    outFile = file('isethist.out','w')
    outFile.writelines(msg)
    outFile.close()

  def printHistory(self):
    self.raiseAMessage('Index Set Choice History:')
    for h in self.history:
      self.raiseAMessage('   ',h)


"""
Interface Dictionary (factory) (private)
"""
__base = 'IndexSet'
__interFaceDict = {}
__interFaceDict['TensorProduct'  ] = TensorProduct
__interFaceDict['TotalDegree'    ] = TotalDegree
__interFaceDict['HyperbolicCross'] = HyperbolicCross
__interFaceDict['Custom'         ] = Custom
__interFaceDict['AdaptiveSet'    ] = AdaptiveSet
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  if Type in knownTypes(): return __interFaceDict[Type]()
  else: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
