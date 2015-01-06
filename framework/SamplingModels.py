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

import IndexSets
import Quadratures
import OrthoPolynomials

from CustomCommandExecuter import execCommand
#Internal Modules End--------------------------------------------------------------------------------

class SamplingModel(Models.Dummy,Samplers.Grid):
  def __init__(self):
    Samplers.Grid.__init__(self)
    Models.Dummy.__init__(self)
    self.type = 'SamplingROM'

  def initialize(self,*args,**kwargs):
    Samplers.Grid.initialize(self,*args,**kwargs)
    #Models.Dummy.initialize(self,*args,**kwargs)
    try: Samplers.Grid.initialize(self,*args,**kwargs)
    except TypeError as te1:
      try:
        Models.Dummy.initialize(self,*args,**kwargs)
      except TypeError as te2:
        raise Exception(str(te1)+' | '+str(te2))
    #FIXME how else to figure out which to run?  This potentially masks errors in Samplers.Grid.

  def localInputAndChecks(self,xmlNode):
    Samplers.Grid.localInputAndChecks(self,xmlNode)

class StochasticPolynomials(SamplingModel):
  def __init__(self):
    SamplingModel.__init__(self)
    self.type = 'StochasticPolynomials'
    self.printTag      = returnPrintTag('SAMPLING ROM STOCHASTIC POLYS')
    self.maxPolyOrder  = None  #L, the maximum polynomial expansion order to use
    self.indexSetType  = None  #TP, TD, or HC; the type of index set to use
    self.adaptive      = False #not yet implemented; adaptively samples index set and/or quadrature
    self.polyDict      = {} # varName-indexed dict of polynomial types
    self.quadDict      = {} # varName-indexed dict of quadrature types
    self.importanceDict= {} # varName-indexed dict of importance weights 

  def _readMoreXML(self,xmlNode):
    Samplers.Sampler._readMoreXML(self,xmlNode)

  def localInputAndChecks(self,xmlNode):
    # sampling side #
    self.indexSetType =     xmlNode.attrib['indexSetType']  if 'indexSetType' in xmlNode.attrib.keys() else 'Tensor Product'
    self.maxPolyOrder = int(xmlNode.attrib['maxPolyOrder']) if 'maxPolyOrder' in xmlNode.attrib.keys() else 2
    self.doInParallel =     (xmlNode.attrib['parallel'].lower() in ['1','t','true','y','yes']) if 'parallel' in xmlNode.attrib.keys() else True
    #FIXME add AdaptiveSP # self.adaptive     =(1 if xmlNode.attrib['adaptive'].lower() in ['true','t','1','y','yes'] else 0) if 'adaptive' in xmlNode.attrib.keys() else 0
    for child in xmlNode:
      importanceWeight = float(child.attrib['impWeight']) if 'impWeight' in child.attrib.keys() else 1
      if child.tag=='Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
      self.axisName.append(varName) #TODO maybe gridInfo.keys() is the same as axisName?
      quad_find = child.find('quadrature')
      if quad_find != None:
        quadType = quad_find.find('type').text if quad_find.find('type') != None else 'DEFAULT'
        polyType = quad_find.find('polynomials').text if quad_find.find('polynomials') != None else 'DEFAULT'
        quadSub = quad_find.find('subtype').text if quad_find.find('subtype') != None else None
        if quadType == 'CDF' and (quadSub!=None and quadSub not in ['Legendre','ClenshawCurtis']):
          raise IOError(self.printTag+' CDF only takes subtypes Legendre and ClenshawCurtis, not '+quadSub)
      else:
        quadType = 'DEFAULT'
        polyType = 'DEFAULT'
      #print("DEBUG quad,poly type for "+varName+" is "+quadType,polyType)
      self.gridInfo[varName] = [quadType,polyType,importanceWeight,quadSub]
    SamplingModel.localInputAndChecks(self,xmlNode)

    # ROM side #


  def localInitialize(self,handler=None):
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

  def localGenerateInput(self,model,myInput):
    pts,weight = self.sparseGrid[self.counter-1]
    for v,varName in enumerate(self.axisName):
      self.values[varName] = pts[v]
      self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
    self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = weight
    self.inputInfo['SamplerType'] = 'Sparse Grid (SamplingROM)'
    if self.counter==len(self.sparseGrid):
      self.createModel()

  def createModel(self):
    self.polyCoeffDict={}
    for i,idx in enumerate(self.sparseGrid.indexSet):
      self.polyCoeffDict[idx]=0
      for k,(pt,wt) in enumerate(self.sparseGrid) #i,pt,wt should be int,tuple,float
        stdPt = np.zeros(len(pt))
        for i,p in enumerate(pt):
          varName = self.distDict.keys()[i]
          stdPt[i] = self.distDict[varName].convertToQuad(self.quads[varName].type(),p) #TODO FIXME does it need converting back?
        #TODO FIXME how to get solution out? Also, how to kick runs to get them going?
        self.polyCoeffDict[idx]+=SOLN[k]*self._multiDBasis(idx,stdPt)*wt

  def _multiDBasis(self,orders,pts):
    tot=1
    for i,(o,p) in enumerate(zip(orders,pts)):
      tot*=self.polys.values()[i][o,p]
    return tot

  def evaluateROM(self,pts):
    tot=0
    stdPt = np.zeros(len(pt))
    for p,pt in enumerate(pts):
      varName = self.distDict.keys()[i]
      stdPt[i] = self.distDict[varName].convertToQuad(self.quads[varName].type(),p) #TODO FIXME does it need converting back?
    for idx,coeff in self.polyCoeffDict.items():
      tot+=coeff*self._multiDBasis(idx,stdPt)
    return tot

  def run(self,Input,jobHandler):
    inRun = self._manipulateInput(Input[0])
    jobHandler.submitDict['Internal']((inRun),self.evaluateROM,str(Input[1]['prefix']),metadata=Input[1])

  #def collectOutput(self,finishedJob,output): use Dummy's version until I know better <-seems fitting.


__base = "SamplingModel"

__interFaceDict = {}
__interFaceDict['StochasticPolynomials'] = StochasticPolynomials
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  return __knownTypes

def returnInstance(Type):
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+'type '+Type)

Samplers.addKnownTypes(__interFaceDict)

