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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType, Assembler
import PostProcessors #import returnFilterInterface
from Samplers import Grid
import Models
import Quadrature
import OrthoPolynomials
from CustomCommandExecuter import execCommand
#Internal Modules End--------------------------------------------------------------------------------

class SamplingROM(ROM,Grid):
  def __init__(self):
    Grid.__init__(self)
    ROM.__init__(self)

  def localInputAndChecks(self,xmlNode):
    Grid.localInputAndChecks(self,xmlNode)

class StochasticPolynomials(SamplingROM):
  def __init__(self):
    SamplingROM.__init__(self)
    self.printTag    = returnPrintTag('SAMPLING ROM STOCHASTIC POLYS')
    self.maxPolyOrder= None  #L, the maximum polynomial expansion order to use
    self.indexSetType= None  #TP, TD, or HC; the type of index set to use
    self.adaptive    = False #not yet implemented; adaptively samples index set and/or quadrature

  def localInputAndChecks(self,xmlNode):
    compatible={} #keys are distro names, values are lists of quad/poly types
    SamplingROM.localInputAndChecks(self,xmlNode)
    self.indexSetType =     xmlNode.attrib['indexSetType']  if 'indexSetType' in xmlNode.attrib.keys() else 'Tensor Product'
    self.maxPolyOrder = int(xmlNode.attrib['maxPolyOrder']) if 'maxPolyOrder' in xmlNode.attrib.keys() else 2
    #FIXME self.adaptive     =(1 if xmlNode.attrib['adaptive'].lower() in ['true','t','1','y','yes'] else 0) if 'adaptive' in xmlNode.attrib.keys() else 0
    for child in xmlNode:
      importanceWeight = float(child.attrib['impWeight']) if 'impWeight' in child.attrib.keys() else 1
      elif child.tag=='Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
      for cchild in child:
        quad_find = xmlNode.find('quadrature')
        if quad_find != None:
          quadType = quad_find.find('type').text if quad_find.find('type') != None else 'DEFAULT' #TODO handle default cases in distribution
          poly_find = quad_find.find('polynomials').text if quad_find.find('polynomials') != None else 'DEFAULT'
      self.gridInfo[varName] = (quadType,polyType,importanceWeight)

  def localInitialize(self):
    Grid.localInitialize(self)
    for varName,dat in self.gridInfo.items():
      quad = Quadrature.returnInstance(dat[0])
      #FIXME alpha,beta for laguerre, jacobi
      if quad.type not in self.distDict[varName].compatibleQuadrature:
        raise IOError (self.printTag+' Incompatible quadrature <'+quad.type+'> for distribution of '+varName+': '+distribution.type)
      quad.initialize()

      poly = OrthoPolynomials.returnInstance(dat[1])
      poly.initialize()
      #TODO how to check compatible polys?  Polys compatible with quadrature more than distribution, kind of both

      self.distDict[varName].setQuadrature(quad)
      self.distDict[varName].setPolynomials(poly,self.maxPolyOrder)
    self.norm = np.prod(list(d.probabilityNorm() for d in self.distDict))

    self.indexSet = IndexSet.returnInstance(self.indexSetType)
    self.indexSet.initialize(self.distDict)

    self.sparseGrid = Quadrature.SparseQuad()
    self.sparseGrid.initialize(self.indexSet,self.distDict)

  def localGenerateInput(self,model,myInput):
    pts,weight = self.sparseGrid[counter-1]
    for v,varName in enumerate(self.axisName):
      self.values[varName] = pts[v]
      self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
    self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = weight
    self.inputInfo['SamplerType'] = 'Sparse Grid (SamplingROM)'
