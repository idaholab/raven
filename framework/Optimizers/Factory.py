"""
  Created on May 21, 2016
  @author: chenj
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3-------------------------------------------

################################################################################
from .Optimizer import Optimizer
from .GradientBasedOptimizer import GradientBasedOptimizer
from .SPSA import SPSA
from .FiniteDifference import FiniteDifference

# # Forward samplers
# from Samplers.ForwardSampler import ForwardSampler
# from Samplers.MonteCarlo import MonteCarlo
# from Samplers.Grid import Grid
# from Samplers.Stratified import Stratified
# from Samplers.FactorialDesign import FactorialDesign
# from Samplers.ResponseSurfaceDesign import ResponseSurfaceDesign
# from Samplers.Sobol import Sobol
# from Samplers.SparseGridCollocation import SparseGridCollocation
# from Samplers.EnsembleForward import EnsembleForwardSampler
# from Samplers.CustomSampler import CustomSampler
# 
# # Adaptive samplers
# from Samplers.AdaptiveSampler import AdaptiveSampler
# from Samplers.LimitSurfaceSearch import LimitSurfaceSearch
# from Samplers.AdaptiveSobol import AdaptiveSobol
# from Samplers.AdaptiveSparseGrid import AdaptiveSparseGrid
# # Dynamic Event Tree-based Samplers
# from Samplers.DynamicEventTree import DynamicEventTree
# from Samplers.AdaptiveDynamicEventTree import AdaptiveDET
## [ Add new class here ]


"""
 Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'Optimizer'
__interFaceDict = {}
__interFaceDict['GradientBasedOptimizer'        ] = GradientBasedOptimizer
__interFaceDict['SPSA'              ] = SPSA
__interFaceDict['FiniteDifference'        ] = FiniteDifference
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  """
    Returns a list of strings that define the types of instantiable objects for
    this base factory.
    @ In, None
    @ Out, knownTypes, list, the known types
  """
  return __interFaceDict.keys()

  
def returnInstance(Type,caller):
  """
    Attempts to create and return an instance of a particular type of object
    available to this factory.
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the instance
                  (used for error/debug messaging).
    @ Out, returnInstance, instance, subclass object constructed with no arguments
  """
  try:
    return __interFaceDict[Type]()
  except KeyError:
    print(knownTypes())
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)
  
def returnClass(Type,caller):
  """
    Attempts to return a particular class type available to this factory.
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the class
                  (used for error/debug messaging).
    @ Out, returnClass, class, reference to the subclass
  """
  try:
    return __interFaceDict[Type]
  except KeyError:
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)
