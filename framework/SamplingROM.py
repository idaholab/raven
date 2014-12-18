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
    SamplingROM.localInputAndChecks(self,xmlNode)
    for name,attrib in child.attrib.iteritems():
      if   name=='maxPolyOrder': self.maxPolyOrder = int(attrib)
      elif name=='indexSetType': self.indexSetType = attrib
      elif name=='adaptive'    : self.adaptive = 1 if attrib.lower() in ['true','t','1','y','yes'] else 0
    for child in xmlNode:
      elif child.tag=='Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
      quadType = child.attrib['quadType'] #TODO what if keyError?  Try-catch here?
      polyType = child.attrib['polyType']
      self.gridInfo[varName] = (quadType,polyType)

  def localInitialize(self):
    Grid.localInitialize(self)
    for varName,dat in self.gridInfo.iteritems():
      quad = Quadrature.returnInstance(dat[0])
      poly = OrthoPolynomials.returnInstance(dat[1])
      #TODO how to access distributions from here?
      distDict[varName].setQuadrature(quad)
      distDict[varName].setPolynomials(poly,self.maxPolyOrder)

