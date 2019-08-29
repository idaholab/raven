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
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
#End compatibility block for Python 3-------------------------------------------

################################################################################

from SupervisedLearning.SupervisedLearning import supervisedLearning
# Forward Samplers
from SupervisedLearning.ARMA               import ARMA
from SupervisedLearning.GaussPolynomialRom import GaussPolynomialRom
from SupervisedLearning.HDMRRom            import HDMRRom
from SupervisedLearning.MSR                import MSR
from SupervisedLearning.NDinterpolatorRom  import NDinterpolatorRom
from SupervisedLearning.NDinvDistWeight    import NDinvDistWeight
from SupervisedLearning.NDsplineRom        import NDsplineRom
from SupervisedLearning.SciKitLearn        import SciKitLearn
from SupervisedLearning.pickledROM         import pickledROM
from SupervisedLearning.PolyExponential    import PolyExponential
from SupervisedLearning.DynamicModeDecomposition import DynamicModeDecomposition
from .KerasClassifier import isTensorflowAvailable
if isTensorflowAvailable():
  from .KerasClassifier import KerasClassifier
  from SupervisedLearning.KerasMLPClassifier import KerasMLPClassifier
  from SupervisedLearning.KerasConvNetClassifier import KerasConvNetClassifier
  from SupervisedLearning.KerasLSTMClassifier import KerasLSTMClassifier
from SupervisedLearning.ROMCollection      import Collection, Segments, Clusters

## [ Add new class here ]
################################################################################


"""
 Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'supervisedLearning'
__interfaceDict                         = {}
__interfaceDict['NDspline'            ] = NDsplineRom
__interfaceDict['NDinvDistWeight'     ] = NDinvDistWeight
__interfaceDict['NDsplineRom'         ] = NDsplineRom
__interfaceDict['SciKitLearn'         ] = SciKitLearn
__interfaceDict['GaussPolynomialRom'  ] = GaussPolynomialRom
__interfaceDict['HDMRRom'             ] = HDMRRom
__interfaceDict['MSR'                 ] = MSR
__interfaceDict['ARMA'                ] = ARMA
__interfaceDict['pickledROM'          ] = pickledROM
__interfaceDict['PolyExponential'     ] = PolyExponential
__interfaceDict['DMD'                 ] = DynamicModeDecomposition
__interfaceDict['Segments'            ] = Segments
__interfaceDict['Clusters'            ] = Clusters
if isTensorflowAvailable():
  __interfaceDict['KerasMLPClassifier'    ] = KerasMLPClassifier
  __interfaceDict['KerasConvNetClassifier'] = KerasConvNetClassifier
  __interfaceDict['KerasLSTMClassifier'   ] = KerasLSTMClassifier

def knownTypes():
  """
    Returns a list of strings that define the types of instantiable objects for
    this base factory.
    @ In, None
    @ Out, knownTypes, list, the known types
  """
  return __interfaceDict.keys()

def returnInstance(Type,caller,**kwargs):
  """
    Attempts to create and return an instance of a particular type of object
    available to this factory.
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the instance
                  (used for error/debug messaging).
    @ In, kwargs, dict, a dicitonary specifying hte keywords and values needed to create the instance
    @ Out, returnInstance, instance, subclass object constructed with no arguments
  """
  try:
    return __interfaceDict[Type](caller.messageHandler,**kwargs)
  except KeyError as e:
    if Type not in __interfaceDict:
      caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)
    else:
      raise e

def returnClass(Type,caller):
  """
    Attempts to return a particular class type available to this factory.
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the class
                  (used for error/debug messaging).
    @ Out, returnClass, class, reference to the subclass
  """
  try:
    return __interfaceDict[Type]
  except KeyError:
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)
