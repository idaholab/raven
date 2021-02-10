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
  This file contains the mathematical methods used in the framework.
  Specifically for identifying or manipulating variable types.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np
import six

#############
# conversions
def convertNumpyToLists(inputDict):
  """
    Method aimed to convert a dictionary containing numpy
    arrays or a single numpy array in list
    @ In, inputDict, dict or numpy array,  object whose content needs to be converted
    @ Out, response, dict or list, same object with its content converted
  """
  returnDict = inputDict
  if type(inputDict) == dict:
    for key, value in inputDict.items():
      if   type(value) == np.ndarray:
        returnDict[key] = value.tolist()
      elif type(value) == dict:
        returnDict[key] = (convertNumpyToLists(value))
      else:
        returnDict[key] = value
  elif type(inputDict) == np.ndarray:
    returnDict = inputDict.tolist()
  return returnDict

def toListFromNumpyOrC1array(array):
  """
    This method converts a numpy or c1darray into list
    @ In, array, numpy or c1array,  array to be converted
    @ Out, response, list, the casted value
  """
  response = array
  if type(array).__name__ == 'ndarray':
    response = array.tolist()
  elif type(array).__name__.split(".")[0] == 'c1darray':
    response = numpy.asarray(array).tolist()
  return response

def toListFromNumpyOrC1arrayIterative(array):
  """
    Method aimed to convert all the string-compatible content of
    an object (dict, list, or string) in type list from numpy and c1darray types (recursively call toBytes(s))
    @ In, array, object,  object whose content needs to be converted
    @ Out, response, object, a copy of the object in which the string-compatible has been converted
  """
  if type(array) == list:
    return [toListFromNumpyOrC1array(x) for x in array]
  elif type(array) == dict:
    if len(array.keys()) == 0:
      return None
    tempdict = {}
    for key,value in array.items():
      tempdict[toBytes(key)] = toListFromNumpyOrC1arrayIterative(value)
    return tempdict
  else:
    return toBytes(array)

#############
# determining types
def npZeroDToEntry(a):
  """
    Cracks the shell of the numpy array and gets the sweet sweet value inside
    @ In, a, object, thing to crack open (might be anything, hopefully a zero-d numpy array)
    @ Out, a, object, thing that was inside the thing in the first place
  """
  if isinstance(a, np.ndarray) and a.shape == ():
    # make the thing we're checking the thing inside to the numpy array
    a = a.item()
  return a

def isSingleValued(val, nanOk=True, zeroDOk=True):
  """
    Determine if a single-entry value (by traditional standards).
    Single entries include strings, numbers, NaN, inf, None
    NOTE that Python usually considers strings as arrays of characters.  Raven doesn't benefit from this definition.
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ In, zeroDOk, bool, optional, if True then a zero-d numpy array with a single-valued entry is A-OK
    @ Out, isSingleValued, bool, result
  """
  # TODO most efficient order for checking?
  if zeroDOk:
    # if a zero-d numpy array, then technically it's single-valued, but we need to get into the array
    val = npZeroDToEntry(val)
  return isAFloatOrInt(val,nanOk=nanOk) or isABoolean(val) or isAString(val) or (val is None)

def isAString(val):
  """
    Determine if a string value (by traditional standards).
    @ In, val, object, check
    @ Out, isAString, bool, result
  """
  return isinstance(val, six.string_types)

def isAFloatOrInt(val,nanOk=True):
  """
    Determine if a float or integer value
    Should be faster than checking (isAFloat || isAnInteger) due to checking against numpy.number
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ Out, isAFloatOrInt, bool, result
  """
  return isAnInteger(val,nanOk) or  isAFloat(val,nanOk)

def isAFloat(val,nanOk=True):
  """
    Determine if a float value (by traditional standards).
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ Out, isAFloat, bool, result
  """
  if isinstance(val,(float,np.number)):
    # exclude ints, which are numpy.number
    if isAnInteger(val):
      return False
    # numpy.float32 (or 16) is niether a float nor a numpy.float (it is a numpy.number)
    if nanOk:
      return True
    elif val not in [np.nan, np.inf]:
      return True
  return False

def isAnInteger(val,nanOk=False):
  """
    Determine if an integer value (by traditional standards).
    @ In, val, object, check
    @ In, nanOk, bool, optional, if True then NaN and inf are acceptable
    @ Out, isAnInteger, bool, result
  """
  if isinstance(val, six.integer_types) or isinstance(val, np.integer):
    # exclude booleans
    if isABoolean(val):
      return False
    return True
  # also include inf and nan, if requested
  if nanOk and isinstance(val,float) and val in [np.nan, np.inf]:
    return True
  return False

def isABoolean(val):
  """
    Determine if a boolean value (by traditional standards).
    @ In, val, object, check
    @ Out, isABoolean, bool, result
  """
  return isinstance(val, (bool, np.bool_))
