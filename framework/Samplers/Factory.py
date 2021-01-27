# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Created on May 16, 2016
  @author: alfoa
  extracted from alfoa (2/16/2013) Samplers.py
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------

################################################################################
from Samplers.Sampler import Sampler
# Forward samplers
from Samplers.ForwardSampler import ForwardSampler
from Samplers.MonteCarlo import MonteCarlo
from Samplers.Grid import Grid
from Samplers.Stratified import Stratified
from Samplers.FactorialDesign import FactorialDesign
from Samplers.ResponseSurfaceDesign import ResponseSurfaceDesign
from Samplers.Sobol import Sobol
from Samplers.SparseGridCollocation import SparseGridCollocation
from Samplers.EnsembleForward import EnsembleForward
from Samplers.CustomSampler import CustomSampler

# Adaptive samplers
from Samplers.AdaptiveSampler import AdaptiveSampler
from Samplers.LimitSurfaceSearch import LimitSurfaceSearch
from Samplers.AdaptiveSobol import AdaptiveSobol
from Samplers.AdaptiveSparseGrid import AdaptiveSparseGrid
from Samplers.AdaptiveMonteCarlo import AdaptiveMonteCarlo

# Dynamic Event Tree-based Samplers
from Samplers.DynamicEventTree import DynamicEventTree
from Samplers.AdaptiveDynamicEventTree import AdaptiveDynamicEventTree

# MCMC Samplers
from .MCMC import Metropolis
from .MCMC import AdaptiveMetropolis

## [ Add new class here ]
################################################################################
## Alternatively, to fully automate this file:
# from Samplers import *
################################################################################

"""
 Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'Sampler'
__interFaceDict = {}
__interFaceDict['MonteCarlo'              ] = MonteCarlo
__interFaceDict['Grid'                    ] = Grid
__interFaceDict['Stratified'              ] = Stratified
__interFaceDict['FactorialDesign'         ] = FactorialDesign
__interFaceDict['ResponseSurfaceDesign'   ] = ResponseSurfaceDesign
__interFaceDict['Sobol'                   ] = Sobol
__interFaceDict['SparseGridCollocation'   ] = SparseGridCollocation
__interFaceDict['CustomSampler'           ] = CustomSampler
__interFaceDict['EnsembleForward'         ] = EnsembleForward
__interFaceDict['LimitSurfaceSearch'      ] = LimitSurfaceSearch
__interFaceDict['AdaptiveSobol'           ] = AdaptiveSobol
__interFaceDict['AdaptiveSparseGrid'      ] = AdaptiveSparseGrid
__interFaceDict['DynamicEventTree'        ] = DynamicEventTree
__interFaceDict['AdaptiveDynamicEventTree'] = AdaptiveDynamicEventTree
__interFaceDict['AdaptiveMonteCarlo'      ] = AdaptiveMonteCarlo
__interFaceDict['Metropolis'              ] = Metropolis
__interFaceDict['AdaptiveMetropolis'      ] = AdaptiveMetropolis

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
