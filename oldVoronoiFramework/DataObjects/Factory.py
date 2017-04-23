"""
Created on September 16, 2015
@author: maljdp
extracted from alfoa (2/16/2013) DataObjects.py
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3-------------------------------------------

################################################################################
from DataObjects.Data import Data
from DataObjects.Point import Point
from DataObjects.PointSet import PointSet
from DataObjects.History import History
from DataObjects.HistorySet import HistorySet
## [ Add new class here ]
################################################################################
## Alternatively, to fully automate this file:
# from DataObjects import *
################################################################################

"""
 Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'Data'
__interFaceDict = {}

for classObj in eval(__base).__subclasses__():
  __interFaceDict[classObj.__name__] = classObj

def knownTypes():
  """
  Returns a list of strings that define the types of instantiable objects for
  this base factory.
  """
  return __interFaceDict.keys()

def returnInstance(Type,caller):
  """
  Attempts to create and return an instance of a particular type of object
  available to this factory.
  @ In, Type, string should be one of the knownTypes.
  @ In, caller, the object requesting the instance
                (used for error/debug messaging).
  @ Out, subclass object constructed with no arguments
  """
  try:
    return __interFaceDict[Type]()
  except KeyError:
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)

def returnClass(Type,caller):
  """
  Attempts to return a particular class type available to this factory.
  @ In, Type, string should be one of the knownTypes.
  @ In, caller, the object requesting the class
                (used for error/debug messaging).
  @ Out, reference to the subclass
  """
  try:
    return __interFaceDict[Type]
  except KeyError:
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)
