"""
Created on Mar 8, 2013

@author: crisr
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from BaseClasses import BaseType
import utils


class Test(BaseType):
  """
  a genral class containing the distributions
  """
  def __init__(self):
    BaseType.__init__(self)
    self.toBeTested  = []
    self.tolerance = 0.0

  def _readMoreXML(self,xmlNode):
    #try:
    self.toBeTested = xmlNode.text.split(',')
    #except? risea IOError('not found variable list to be tested in tester '+self.name)
    #try:
    self.name = xmlNode.attrib['tolerance']
    #except? risea IOError('not found tolerance for tester '+self.name)

  def addInitParams(self,tempDict):
    tempDict['toBeTested'] = self.toBeTested
    tempDict['tolerance' ] = self.tolerance

  def reset(self):
    return

  def checkConvergence(self,inDictionary):  #if a ROM present ???
    return

  def getROM(self,ROM):
    return

  def getOutput(self,ROM):
    return

  def testOutput(self):
    return



class Sigma(Test):
  pass



class Integral(Test):
  pass

"""
 Interface Dictionary (factory) (private)
"""
__base                      = 'Data'
__interFaceDict             = {}
__interFaceDict['Sigma'   ] = Sigma
__interFaceDict['Integral'] = Integral
__knownTypes                = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  '''return one instance of Type'''
  try: return __interFaceDict[Type]()
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)

