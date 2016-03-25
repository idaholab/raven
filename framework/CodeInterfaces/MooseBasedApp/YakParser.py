from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
from macpath import split
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import os
import copy

class YakParser():
  '''import the Yak material properties input as xml tree, provide methods to add/change entries and print it back'''
  def __init__(self,inputFiles,aliasFile):
    self.properties = {} #storage for material properites.  Structure TBD.
    self.ngroups    = 0  #number of energy groups in problem

  def modifyInternalDictionary(self,**Kwargs):
    pass

  def writeNewInput(self,inFiles,origFiles):
    pass

  def parseMaterial(self,node):#,ng,ndnp,na,fis might need these
    ID = int(node.attrib['ID'])
    ao = int(node.attrib['NA']) #FIXME what is this??
    num_delay_group = 0 #TODO FIXME where do I get this?
    fissile = node.attrib['fissile'] == 'true'
    name = node.find('name').text
    self.readScalarXS(node,'TotalXS')
    self.readScalarXS(node,"ChiXS")
    self.readScalarXS(node,"KappaFissionXS")
    self.readScalarXS(node,"FissionXS")
    self.readScalarXS(node,"DNPlambda")
    self.readScalarXS(node,"DNFraction")
    self.readScalarXS(node,"DNSpectrum")
    self.readProfileXS(node,'ScatteringXS')
    self.readScalarXS(node,'DiffusionCoefficient')
    self.readScalarXS(node,"RemovalXS")
    self.readScalarXS(node,"AbsorptionXS")
    self.readScalarXS(node,"CaptureXS")
    self.readScalarXS(node,"NalphaXS")
    self.readScalarXS(node,"Flux")
    self.readScalarXS(node,"NeutronSpeed")

  def readScalarXS(self,node,XStype):
    pass

  def readProfileXS(self,node,XStype):
    pass
