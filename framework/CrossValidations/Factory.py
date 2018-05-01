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
Created on Sept 5 2017

@author: wangc
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3-------------------------------------------

################################################################################
from utils import utils
from .CrossValidation import CrossValidation
from .SklCrossValidation import SciKitLearn
## [ Add new class here ]
"""
  Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'CrossValidation'
__interFaceDict = {}

for classObj in utils.getAllSubclasses(eval(__base)):
  __interFaceDict[classObj.__name__] = classObj


def knownTypes():
  """
    Returns a list of strings that define the types of instantiable objects for
    this base factory.
    @ In, None
    @ Out, knownTypes, list, list of known types
  """
  return __interFaceDict.keys()


def returnInstance(Type, caller, **kwargs):
  """
    This function return an instance of the request model type
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the class (used for error/debug messaging).
    @ In, kwargs, dict, a dictionary specifying the keywords and values needed to create the instance.
    @ Out, object, instance,  an instance of a cross validation
  """
  try:
    return __interFaceDict[Type](caller.messageHandler, **kwargs)
  except KeyError:
    caller.raiseAnError(NameError, 'unSupervisedLearning',
                        'Unknown ' + __base + ' type ' + str(Type))


def returnClass(Type, caller):
  """
    Attempts to return a particular class type available to this factory.
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the class (used for error/debug messaging).
    @ Out, returnClass, class, reference to the subclass
  """
  try:
    return __interFaceDict[Type]
  except KeyError:
    caller.raiseAnError(NameError, __name__ + ': unknown ' + __base + ' type ' + Type)
