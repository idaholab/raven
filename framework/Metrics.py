#External Modules------------------------------------------------------------------------------------
import os
import copy
import shutil
import math
import numpy as np
import abc
import importlib
import inspect
import atexit
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
import PostProcessors #import returnFilterInterface
import CustomCommandExecuter
import utils
import mathUtils
import TreeStructure
import Files

#Internal Modules End--------------------------------------------------------------------------------

class Metric(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  
  def __init__(self,runInfoDict):
    BaseType.__init__(self)
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    
  def initialize(self,inputDict):
    pass
  
  def _readMoreXML(self,xmlNode):
    pass    
  
  def readMoreXML(self,xmlNode):
    pass
    
  def distance(self,x,y,weights=None,paramDict=None):
    pass
  

class Minkowski(Metric):
  
  def initialize(self,inputDict):
    self.p = None
  
  def _readMoreXML(self,xmlNode):
    for child in xmlNode:
      if child.tag == 'p':
        self.p = float(child.text)
  
  def distance(self,x,y):    
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      value = 0
      for i in range(x.size):
        value += (x[i]-y[i])**self.p
      return math.sqrt(value)
    elif isinstance(x,dict) and isinstance(y,dict):
      if x.keys() == y.keys():
        value = 0
        for key in x.keys():
          if x[key].size == y[key].size:
            for i in range(x[key].size):
              value += (x[key][i]-y[key][i])**self.p 
            return math.sqrt(value)
          else:
            print('Metric Minkowski error: the length of the variable array ' + str(key) +' is not consistent among the two data sets')       
      else:
        print('Metric Minkowski error: the two data sets do not contain the same variables')
    else:
      print('Metric Minkowski error: the structures of the two data sets are different')


"""
 Factory......
"""
__base = 'metric'
__interFaceDict = {}
__interFaceDict['Minkowski'          ] = Minkowski
__knownTypes                      = list(__interFaceDict.keys())

#for classType in __interFaceDict.values():
#  classType.generateValidateDict()
#  classType.specializeValidateDict()

def addKnownTypes(newDict):
  for name,value in newDict.items():
    __interFaceDict[name]=value
    __knownTypes.append(name)

def knownTypes():
  return __knownTypes

needsRunInfo = True

def returnInstance(Type,runInfoDict,caller):
  """This function return an instance of the request model type"""
  try: return __interFaceDict[Type](runInfoDict)
  except KeyError: caller.raiseAnError(NameError,'METRICS','not known '+__base+' type '+Type)

def validate(className,role,what,caller):
  """This is the general interface for the validation of a model usage"""
  if className in __knownTypes: return __interFaceDict[className].localValidateMethod(role,what)
  else : caller.raiseAnError(IOError,'METRICS','the class '+str(className)+' it is not a registered model')
  
  