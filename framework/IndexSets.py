from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
from itertools import product

#TODO make preamble correct, inherit from BaseClass

class IndexSet(object):
  def __init__(self):
    self.impWeights = None #weights for anisotropic case
    self.type       = None #type of index set (Tensor Product, Total Degree, Hyperbolic Cross)
    self.maxOrds    = None #maximum requested polynomial order requested for each distribution

  def __len__(self):
    return len(self.points)

  def __getitem__(self,i=None):
    if i==None: return np.array(self.points)
    else: return self.points[i]

  def _extrema(self):
    low=np.ones(len(self.points[0]))*1e300
    hi =np.ones(len(self.points[0]))*(-1e300)
    for pt in self.points:
      for i,p in enumerate(pt):
        low[i]=min(low[i],p)
        hi[i] =max(hi[i],p)
    return low,hi

  def _xy(self):
    return zip(*self.points)

  def initialize(self,distrList):#,impWeights):
    numDim = len(distrList)
    #set up and normalize weights -> 
    impWeights = list(d.importanceWeight for d in distrList.values())
    #if impWeights==None:
    #  impWeights=np.ones(numDim)
    impWeights = np.array(impWeights)
    #this algorithm assures higher weight means more importance, and end product is normalized so smallest is 1
    impWeights=impWeights/np.max(impWeights)
    impWeights=1.0/impWeights
    self.impWeights = impWeights
    #establish max orders
    N = len(distrList.keys())
    #TODO make this input-able from user side
    #  - readMoreXML on distr, if maxPolyOrder not set, set it to the problem maxOrder or error out
    self.polyOrderList=[]
    self.maxOrder=0
    for distr in distrList.values():
      self.polyOrderList.append(range(distr.maxPolyOrder()+1))
      self.maxOrder = max(self.maxOrder,distr.maxPolyOrder())

  def generateMultiIndex(self,N,rule,I=None,MI=None):
    #recursive tool to build monotonically-increasing-order multi-index set
    L = self.maxOrder
    if I ==None: I =[]
    if MI==None: MI=[]
    if len(I)!=N:
      i=0
      while rule(I+[i]):
        MI = self.generateMultiIndex(N,rule,I+[i],MI)
        i+=1
    else:
      MI.append(tuple(I))
    return MI

class TensorProduct(IndexSet):
  def initialize(self,distrList):
    IndexSet.initialize(self,distrList)
    self.type='Tensor Product'
    self.points = list(product(*self.polyOrderList))

class TotalDegree(IndexSet):
  def initialize(self,distrList):
    IndexSet.initialize(self,distrList)
    self.type='Total Degree'
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
  def initialize(self,distrList):
    IndexSet.initialize(self,distrList)
    self.type='Hyperbolic Cross'
    #TODO if user has set max poly orders (levels), make it so you never use more
    #  - right now is only limited by the maximum overall level (and importance weight)
    target = (self.maxOrder+1)**(sum(self.impWeights)/max(1,float(len(self.impWeights))))
    def rule(i):
      tot=1;
      for e,val in enumerate(i):
        tot*=(val+1)**self.impWeights[e]
      return tot<=target
    self.points = self.generateMultiIndex(len(distrList),rule)


'''
Interface Dictionary (factory) (private)
'''
__base = 'IndexSet'
__interFaceDict = {}
__interFaceDict['TensorProduct'  ] = TensorProduct
__interFaceDict['TotalDegree'    ] = TotalDegree
__interFaceDict['HyperbolicCross'] = HyperbolicCross
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
