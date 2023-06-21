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
  This Module performs Unit Tests for the TSA.ZeroFilter class.
  It can not be considered part of the active code but of the regression test system
"""
import os
import sys
import numpy as np

# add RAVEN to path
ravenDir =  os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4)))
frameworkDir = os.path.join(ravenDir, 'framework')
if ravenDir not in sys.path:
  sys.path.append(ravenDir)

from ravenframework.utils import xmlUtils

from ravenframework.TSA import OutTruncation

print('Modules undergoing testing:')
print(OutTruncation)
print('')

results = {"pass":0, "fail":0}

def updateRes(comment, dtype, res, value, expected, printComment=True):
  """
    Updates results dictionary
    @ In, comment, string, a comment printed out if it fails
    @ In, dtype, type, the type of the value
    @ In, res, bool, the result of the check
    @ In, value, object, the value to compare
    @ In, expected, object, the expected value
    @ In, printComment, bool, optional, if False then don't print comment
  """
  if res:
    results['pass'] += 1
  else:
    results['fail'] += 1
    if printComment:
      print(f'checking {str(dtype)} {comment} | {value} != {expected}')

def checkFloat(comment, value, expected, tol=1e-10, update=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  if np.isnan(value) and np.isnan(expected):
    res = True
  elif np.isnan(value) or np.isnan(expected):
    res = False
  else:
    res = abs(value - expected) <= tol

  if update:
    updateRes(comment, float, res, value, expected)

  return res

def checkSame(comment, value, expected, update=True):
  """
    This method is aimed to compare two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    updateRes(comment, type(value), res, value, expected)
  return res

def checkBool(comment, value, expected, update=True):
  """
    Checks if value is True
    @ In, comment, string, a comment printed out if it fails
    @ In, value, bool, boolean value to check
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if all values are True
  """
  res = value == expected
  if update:
    updateRes(comment, type(value), res, value, expected)
  return res

def checkArray(comment, first, second, dtype, tol=1e-10, update=True):
  """
    This method is aimed to compare two arrays
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = True

  if first.shape != second.shape:
    res = False
    print(f'checking answer {comment} | array shapes do not match: {first.shape} {second.shape}')
    return res

  for indices, _ in np.ndenumerate(first):
    if dtype == float:
      pres = checkFloat('', first[indices], second[indices], tol, update=False)
    elif dtype == str:
      pres = checkSame('', first[indices], second[indices], update=False)
    elif dtype == bool:
      pres = checkBool('', first[indices], second[indices], update=False)
    else:
      pres = False

    if not pres:
      print(f'checking array {comment} | entry "{indices}" does not match: {first[indices]} != {second[indices]}')
      res = False

  if update:
    updateRes(comment, dtype, res, first, second, printComment=False)

  return res

######################################
#            CONSTRUCTION            #
######################################
def createOutTruncation(targets, domain):
  """
    Creates a transformer of the given type.
    @ In, targets, list(str), names of targets
    @ In, domain, str, domain of truncation
    @ Out, transformer, subclass of TimeSeriesTransformer, transformer instance
  """
  transformer = OutTruncation()
  xml = xmlUtils.newNode('outtruncation', attrib={'target':','.join(targets), 'domain': domain})
  inputSpec = transformer.getInputSpecification()()
  inputSpec.parseNode(xml)
  transformer.handleInput(inputSpec)
  return transformer

###################
#      Tests      #
###################
targets = ['A']
pivot = np.arange(11)
signals = np.linspace(-1, 1, 11).reshape(-1, 1)

# Test positive domain
outTruncationPositive = createOutTruncation(targets, 'positive')
params = outTruncationPositive.fit(signals, pivot, targets, {})
residual = outTruncationPositive.getResidual(signals, params, pivot, {})
# Forward transformation should be the identity function
checkArray('OutTruncation positive residual', residual, signals, float)
# Inverse transformation should not have any negative values
inverse = outTruncationPositive.getComposite(signals, params, pivot, {})
checkArray('OutTruncation positive inverse', inverse >= 0, np.full_like(inverse, True), float)
# Inverse transformation should reflect negative values back into positive domain with absolute value
checkArray('OutTruncation positive inverse', inverse, np.abs(signals), float)


# Test negative domain
outTruncationNegative = createOutTruncation(targets, 'negative')
params = outTruncationNegative.fit(signals, pivot, targets, {})
residual = outTruncationNegative.getResidual(signals, params, pivot, {})
# Forward transformation should be the identity function
checkArray('OutTruncation negative residual', residual, signals, float)
# Inverse transformation should not have any positive values
inverse = outTruncationNegative.getComposite(signals, params, pivot, {})
checkArray('OutTruncation negative inverse', inverse <= 0, np.full_like(inverse, True), float)
# Inverse transformation should reflect positive values back into negative domain with absolute value
checkArray('OutTruncation positive inverse', inverse, -np.abs(signals), float)

print(results)

sys.exit(results["fail"])
