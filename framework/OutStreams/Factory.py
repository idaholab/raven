"""
Created on April 5, 2016
@author: maljdp
extracted from alfoa (11/14/2013) OutStreamManager.py
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3-------------------------------------------

################################################################################
from .OutStreamManager import OutStreamManager
from .OutStreamPlot import OutStreamPlot
from .OutStreamPrint import OutStreamPrint
## [ Add new class here ]
################################################################################
## Alternatively, to fully automate this file:
# from OutStreamManagers import *
################################################################################

"""
 Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'OutStreamManager'
__interFaceDict = {}

for classObj in eval(__base).__subclasses__():
  ## As long as these subclasses follow the pattern of starting with OutStream
  ## this will appropriately key them to a more user-friendly name without the
  ## need for them to redudantly prepend "X" as "OutStreamX"
  key = classObj.__name__.replace('OutStream','')
  __interFaceDict[key] = classObj

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
    print(eval(__base).__subclasses__())
    print(__interfaceDict.keys())
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
