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
Factory for generating the instances of the  Models Module
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------

from utils import utils

from .Model         import Model
from .Dummy         import Dummy
from .ROM           import ROM
from .ExternalModel import ExternalModel
from .Code          import Code
from .EnsembleModel import EnsembleModel
from .PostProcessor import PostProcessor
from .HybridModels import HybridModel

__base = 'Model'
__interFaceDict = {}

for classObj in utils.getAllSubclasses(eval(__base)):
  key = classObj.__name__
  __interFaceDict[key] = classObj

#here the class methods are called to fill the information about the usage of the classes
for classType in __interFaceDict.values():
  classType.generateValidateDict()
  classType.specializeValidateDict()

def knownTypes():
  """
    Return the known types
    @ In, None
    @ Out, knownTypes, list, list of known types
  """
  return __interFaceDict.keys()

needsRunInfo = True

def returnInstance(Type,runInfoDict,caller):
  """
    function used to generate a Model class
    @ In, Type, string, Model type
    @ Out, returnInstance, instance, Instance of the Specialized Model class
  """
  try:
    return __interFaceDict[Type](runInfoDict)
  except KeyError:
    availableClasses = ','.join(__interFaceDict.keys())
    caller.raiseAnError(NameError,
      'Requested {}, i.e. "{}", is not recognized (Available options: {})'.format(__base, Type, availableClasses))

def validate(className,role,what,caller):
  """
    This is the general interface for the validation of a model usage
    @ In, className, string, the name of the class
    @ In, role, string, the role assumed in the Step
    @ In, what, string, type of object
    @ In, caller, instance, the instance of the caller
    @ Out, None
  """
  if className in __interFaceDict:
    return __interFaceDict[className].localValidateMethod(role,what)
  else:
    caller.raiseAnError(IOError, 'The model "{}" is not registered for class "{}"'.format(className, __base))
