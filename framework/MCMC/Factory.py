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
  Markov Chain Monte Carlo

  created on June 26, 2020
  @author: wangc
"""

from .Metropolis import Metropolis


"""
 Interface Dictionary (factory) (private)
"""
# This machinery will automatically populate the "knownTypes" given the
# imports defined above.

__base = 'MCMC'
__interFaceDict = {}

__interFaceDict['Metropolis'           ] = Metropolis

__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  """
    Returns a list of strings that define the types of instantiable objects for
    this base factory.
    @ In, None
    @ Out, knownTypes, list, the known types
  """
  return __interFaceDict.keys()


def returnInstance(instanceType, caller):
  """
    Attempts to create and return an instance of a particular type of object
    available to this factory.
    @ In, instanceType, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the instance (used for error/debug messaging).
    @ Out, returnInstance, instance, subclass object constructed with no arguments
  """
  try:
    return __interFaceDict[instanceType]()
  except KeyError:
    print(knownTypes())
    caller.raiseAnError(NameError, __name__, ': unknown', __base, 'type', instanceType)

def returnClass(instanceType, caller):
  """
    Attempts to return a particular class type available to this factory.
    @ In, instanceType, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the class (used for error/debug messaging).
    @ Out, returnClass, class, reference to the subclass
  """
  try:
    return __interFaceDict[instanceType]
  except KeyError:
    caller.raiseAnError(NameError, __name__, ': unknown', __base, 'type', instanceType)
