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
  Created on May 8, 2018
  @author: talbpaul
  extracted from SupervisedLearning.py
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3-------------------------------------------

################################################################################

# These lines ensure that we do not have to do something like:
# 'from Samplers.Sampler import Sampler' outside of this submodule
from SupervisedLearning.SupervisedLearning import SupervisedLearning
# Forward Samplers
from SupervisedLearning.ARMA               import ARMA
from SupervisedLearning.GaussPolynomialRom import GaussPolynomialRom
from SupervisedLearning.HDMRRom            import HDMRRom
from SupervisedLearning.MSR                import MSR
from SupervisedLearning.NDinterpoloatorRom import NDinterpoloatorRom
from SupervisedLearning.NDinvDistWeight    import NDinvDistWeight
from SupervisedLearning.NDsplineRom        import NDsplineRom
from SupervisedLearning.SciKitLearn        import SciKitLearn

## [ Add new class here ]
################################################################################


"""
 Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'superVisedLearning'
__interfaceDict                         = {}
__interfaceDict['NDspline'            ] = NDsplineRom
__interfaceDict['NDinvDistWeight'     ] = NDinvDistWeight
__interfaceDict['SciKitLearn'         ] = SciKitLearn
__interfaceDict['GaussPolynomialRom'  ] = GaussPolynomialRom
__interfaceDict['HDMRRom'             ] = HDMRRom
__interfaceDict['MSR'                 ] = MSR
__interfaceDict['ARMA'                ] = ARMA
__interfaceDict['pickledROM'          ] = pickledROM

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
