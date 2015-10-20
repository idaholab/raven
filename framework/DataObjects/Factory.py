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

from Data import Data
from Point import Point
from PointSet import PointSet
from History import History
from HistorySet import HistorySet

"""
 Interface Dictionary (factory) (private)
"""
# Make sure you include your new class in the imports above, as long as that is
#  done, then this machinery will handle its appropriate generation for use
#  by the factory
__base = 'Data'
__interFaceDict = {}

for classObj in eval(__base).__subclasses__():
  __interFaceDict[classObj.__name__] = classObj

__knownTypes = __interFaceDict.keys()

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  try:
    return __interFaceDict[Type]()
  except KeyError:
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)

def returnClass(Type,caller):
  try:
    return __interFaceDict[Type]
  except KeyError:
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)
