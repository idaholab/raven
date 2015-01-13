'''
Module for ROM models that require specific sampling sets, e.g. Stochastic Collocation
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import shutil
import numpy as np
from utils import metaclass_insert, returnPrintTag, returnPrintPostTag
import abc
import importlib
from operator import mul
from functools import reduce
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType, Assembler
import PostProcessors #import returnFilterInterface
import Samplers
import Models
import SupervisedLearning

import IndexSets
import Quadratures
import OrthoPolynomials

from CustomCommandExecuter import execCommand
#Internal Modules End--------------------------------------------------------------------------------

class SamplingROM(Models.ROM,Samplers.Grid,Assembler):
  def __init__(self):
    Samplers.Grid.__init__(self)
    Models.ROM.__init__(self)
    self.type = 'SamplingROM'

  def initialize(self,*args,**kwargs):
    what = kwargs['what']
    del kwargs['what']
    if what=='Sampler': Samplers.Grid.initialize(self,*args,**kwargs)
    elif what=='Model': Models.ROM.initialize(self,*args,**kwargs)
    else:
      raise TypeError('SamplingROM has no "initialize" for type '+what)

  def localInputAndChecks(self,xmlNode):
    Samplers.Grid.localInputAndChecks(self,xmlNode)

  def whatDoINeed(self):
    self._localWhatDoINeed()

  def generateAssembler(self,initDict):
    self._localGenerateAssembler(initDict)

class StochasticPolynomials(SamplingROM):
  def __init__(self):
    SamplingROM.__init__(self)
    self.type = 'StochasticPolynomials'
    self.printTag      = returnPrintTag('SAMPLINGROM-STOCHPOLYS')
    self.maxPolyOrder  = None  #L, the maximum polynomial expansion order to use
    self.indexSetType  = None  #TP, TD, or HC; the type of index set to use
    self.adaptive      = False #not yet implemented; adaptively samples index set and/or quadrature
    self.polyDict      = {}    # varName-indexed dict of polynomial types
    self.quadDict      = {}    # varName-indexed dict of quadrature types
    self.importanceDict= {}    # varName-indexed dict of importance weights 
    self.polyCoeffDict = {}    # polynomial index indexed dict of coefficient scalars
    self.readyToRun    = False # true when createModel is run
    self.lastOutput    = None  # pointer,used for checking if ready to pass info along
    self.ROM           = None  # pointer,used to hand info along to SVLengine
    #TODO add a way to check if you're going to try to train me -> cuz you can't.

  def _localWhatDoINeed(self): #Sampler side
    needDict = {}
    for value in self.assemblerObjects.values():
      if value[0] not in needDict.keys(): needDict[value[0]]=[]
      needDict[value[0]].append((value[1],value[2]))
    return needDict
    #also, I need a job handler
    pass # see Samplers->Adaptive, also BaseClasses.py

  def _localGenerateAssembler(self,initDict): #Sampler side
    for key, value in self.assemblerObjects.items():
      if key in 'TargetEvaluation': self.lastOutput = initDict[value[0]][value[2]]
      if key in 'ROM'             : self.ROM        = initDict[value[0]][value[2]]
      #FIXME jobHandler

  def _readMoreXML(self,xmlNode): #ROM and Sampler side
    #Models.ROM._readMoreXML(self,xmlNode)
    self.indexSetType =     xmlNode.attrib['indexSetType']  if 'indexSetType' in xmlNode.attrib.keys() else 'Tensor Product'
    self.maxPolyOrder = int(xmlNode.attrib['maxPolyOrder']) if 'maxPolyOrder' in xmlNode.attrib.keys() else 2
    self.doInParallel =     (xmlNode.attrib['parallel'].lower() in ['1','t','true','y','yes']) if 'parallel' in xmlNode.attrib.keys() else True
    #TODO add AdaptiveSP # self.adaptive     =(1 if xmlNode.attrib['adaptive'].lower() in ['true','t','1','y','yes'] else 0) if 'adaptive' in xmlNode.attrib.keys() else 0
    #send appropriate nodes to their corresponding hybrid parts
    for child in xmlNode:
      if child.tag=='Sampling':
        Samplers.Sampler._readMoreXML(self,child)
      elif child.tag=='ROM':
        Models.ROM._readMoreXML(self,child)
      xmlNode.remove(child)
      elif child.tag=='Assembler':
        pass
      else:
        raise IOError(self.printTag+' Unused tags in xmlNode:' +str(list(child.tag for child in xmlNode)))
      #TODO adaptive

  def localInputAndChecks(self,xmlNode): #Sampler side
    for child in xmlNode:
      importanceWeight = float(child.attrib['impWeight']) if 'impWeight' in child.attrib.keys() else 1
      if child.tag=='Distribution': varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable': varName = child.attrib['name']
      self.axisName.append(varName)
      quad_find = child.find('quadrature')
      if quad_find != None:
        #TODO more compatibility checking?
        quadType = quad_find.find('type').text if quad_find.find('type') != None else 'DEFAULT'
        polyType = quad_find.find('polynomials').text if quad_find.find('polynomials') != None else 'DEFAULT'
        quadSub = quad_find.find('subtype').text if quad_find.find('subtype') != None else None
        if quadType == 'CDF' and (quadSub!=None and quadSub not in ['Legendre','ClenshawCurtis']):
          raise IOError(self.printTag+' CDF only takes subtypes Legendre and ClenshawCurtis, not '+quadSub)
      else:
        quadType = 'DEFAULT'
        polyType = 'DEFAULT'
      self.gridInfo[varName] = [quadType,polyType,importanceWeight,quadSub]

  def localInitialize(self,handler=None): #Sampler side
    if handler != None and self.doInParallel == False: handler = None
    Samplers.Grid.localInitialize(self)
    for varName,dat in self.gridInfo.items():
      #dat[0] is the quadtype
      #dat[1] is the poly type
      #dat[2] is the importance weight
      #dat[3] is the optional subtype (CDF only)

      #FIXME alpha,beta for laguerre, jacobi
      #if dat[0] not in self.distDict[varName].compatibleQuadrature and dat[0]!='DEFAULT':
      #  raise IOError (self.printTag+' Incompatible quadrature <'+dat[0]+'> for distribution of '+varName+': '+distribution.type)
      if dat[0]=='DEFAULT': dat[0]=self.distDict[varName].preferredQuadrature
      quad = Quadratures.returnInstance(dat[0],dat[3]) #the user can optionally pass in a subtype for CDF quadrature (Legendre or ClenshawCurtis)
      quad.initialize()
      self.quadDict[varName]=quad

      if dat[1]=='DEFAULT': dat[1] = self.distDict[varName].preferredPolynomials
      poly = OrthoPolynomials.returnInstance(dat[1])
      poly.initialize()
      self.polyDict[varName] = poly
      #TODO how to check compatible polys?  Polys compatible with quadrature more than distribution, kind of both

      self.importanceDict[varName] = dat[2]

    self.norm = np.prod(list(self.distDict[v].measureNorm(self.quadDict[v].type) for v in self.distDict.keys()))

    self.indexSet = IndexSets.returnInstance(self.indexSetType)
    self.indexSet.initialize(self.distDict,self.importanceDict,self.maxPolyOrder)

    self.sparseGrid = Quadratures.SparseQuad()
    self.sparseGrid.initialize(self.indexSet,self.maxPolyOrder,self.distDict,self.quadDict,self.polyDict,handler)
    self.limit=len(self.sparseGrid)

  def localGenerateInput(self,model,myInput): #Sampler
    pts,weight = self.sparseGrid[self.counter-1]
    for v,varName in enumerate(self.axisName):
      self.values[varName] = pts[v]
      self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
    self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = weight
    self.inputInfo['SamplerType'] = 'Sparse Grid (SamplingROM)'

  def localFinalizeActualSampling(jobObject,model,myInput): #Sampler
    if len(self.lastOutput)==len(self.sparseGrid): #done sampling
      for target in self.ROM.SupervisedEngine.keys():
        self.ROM.SupervisedEngine[target].initialize(self.sparseGrid,self.distDict,self.quadDict)

  def createModel(self,SOLN): #FIXME specialize to subtype
    #TODO FIXME who can call this with SOLN keyed on points
    self.polyCoeffDict={}
    for i,idx in enumerate(self.sparseGrid.indexSet):
      idx=tuple(idx)
      self.polyCoeffDict[idx]=0
      for k,(pt,wt) in enumerate(self.sparseGrid): #i,pt,wt should be int,tuple,float
        stdPt = np.zeros(len(pt))
        for i,p in enumerate(pt):
          varName = self.distDict.keys()[i]
          stdPt[i] = self.distDict[varName].convertToQuad(self.quadDict[varName].type,p) #TODO FIXME does it need converting back?
        self.polyCoeffDict[idx]+=SOLN[pt]*self._multiDBasis(idx,stdPt)*wt
    self.SupervisedEngine[target].initialize(self.sparseGrid,self.distDict,self.quadDict)

  def _multiDBasis(self,orders,pts):
    tot=1
    for i,(o,p) in enumerate(zip(orders,pts)):
      tot*=self.polys.values()[i][o,p]
    return tot


class PolynomialExpansionEngine(SupervisedLearning.superVisedLearning):
  def initialize(self,sparseGrid,distDict,quadDict):
    self.polyCoeffDict = {}         #dict of polynomial index set coefficients
    self.sparseGrid    = sparseGrid #multi-D points and weights
    self.distDict      = distDict   #dict of distributions
    self.quadDict      = quadDict   #dict of quadratures

class GaussPolynomialEngine(PolynomialExpansionEngine):
  def __confidenceLocal__(self,edict):pass
  def __resetLocal__(self):pass
  def __returnCurrentSettingLocal__(self):pass
  def __returnInitialParametersLocal__(self):pass

  def __trainLocal__(self,featureVals,targetVals):
    self.polyCoeffDict={}
    for i,idx in enumerate(self.sparseGrid.indexSet):
      idx=tuple(idx)
      self.polyCoeffDict[idx]=0
      for k,(pt,wt) in enumerate(self.sparseGrid): #i,pt,wt should be int,tuple,float
        stdPt = np.zeros(len(pt))
        for i,p in enumerate(pt):
          varName = self.distDict.keys()[i]
          stdPt[i] = self.distDict[varName].convertToQuad(self.quadDict[varName].type,p) #TODO FIXME does it need converting back?
        self.polyCoeffDict[idx]+=SOLN[pt]*self.__multiDPolyBasisEval(idx,stdPt)*wt

  def __multiDPolyBasisEval(self,orders,pts):
    tot=1
    for i,(o,p) in enumerate(zip(orders,pts)):
      tot*=self.polys.values()[i][o,p]
    return tot

  def __evaluateLocal__(self,edict): #FIXME what's in edict?  Points keyed on....?
    tot=0
    stdPt = np.zeros(len(pt))
    for p,pt in enumerate(pts):
      varName = self.distDict.keys()[i]
      stdPt[i] = self.distDict[varName].convertToQuad(self.quads[varName].type(),p) #TODO FIXME does it need converting back?
    for idx,coeff in self.polyCoeffDict.items():
      tot+=coeff*self._multiDBasis(idx,stdPt)
    return tot


class GaussPolynomialEngine(PolynomialExpansionEngine):
  def __confidenceLocal__(self,edict):pass
  def __resetLocal__(self):pass
  def __returnCurrentSettingLocal__(self):pass
  def __returnInitialParametersLocal__(self):pass

  def __trainLocal__(self,featureVals,targetVals):
    pass #doesn't need special training

  def __evaluateLocal__(self,edict):
    tot=0
    for k,(pt,wt) in enumerate(self.sparseGrid):
      stdPt = np.zeros(len(pt))
      for i,p in enumerate(pt):
        varName = self.distDict.keys()[i]
        stdPt[i] = self.distDict[varName].convertToQuad(self.quadDict[varName].type,p)
      tot+=SOLN[pt]*self.__multiDLagrange(idx,stdPt)*wt #FIXME SOLN
    return tot

__base = "SamplingROM"

__interFaceDict = {}
__interFaceDict['StochasticPolynomials'] = StochasticPolynomials
__knownTypes = list(__interFaceDict.keys())

__engineDict = {}
__engineDict['GaussPolynomialEngine'] = GaussPolynomialEngine

def knownTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+'type '+Type)

Samplers.addKnownTypes(__interFaceDict)
Models.addKnownTypes(__interFaceDict)
SupervisedLearning.addToInterfaceDict(__engineDict)
