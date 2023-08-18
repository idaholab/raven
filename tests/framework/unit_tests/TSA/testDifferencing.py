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
  This Module performs Unit Tests for the TSA.Differencing class.
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

from ravenframework.TSA import Differencing

print('Modules undergoing testing:')
print(Differencing)
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
def createDifferencing(targets, order):
  """
    Creates a TSA.Differencing instance
    @ In, targets, list(str), names of targets
    @ In, order, int, order of differencing
    @ Out, transformer, subclass of TimeSeriesTransformer, transformer instance
    @ Out, settings, dict, settings for transformer
  """
  transformer = Differencing()
  attribs = {'target': ','.join(targets)}
  xml = xmlUtils.newNode(transformer.name.lower(), attrib=attribs)
  xml.append(xmlUtils.newNode('order', text=str(order)))
  inputSpec = transformer.getInputSpecification()()
  inputSpec.parseNode(xml)
  settings = transformer.handleInput(inputSpec)
  return transformer, settings

###################
#      Tests      #
###################
# Test first-order differencing
targets = ['A']
pivot = np.arange(11)
signals = np.linspace(-1, 1, 11).reshape(-1, 1)

# Test fit
diff1, settings = createDifferencing(targets=targets, order=1)
params = diff1.fit(signals, pivot, targets, settings)
checkArray('first-order differencing fit', params['A']['initValues'], signals[:1], float)

# Test forward transformation
transformed1 = diff1.getResidual(signals, params, pivot, settings)
transformed1True = np.r_[np.diff(signals, axis=0).ravel(), [np.nan] * settings['order']].reshape(-1, 1)
checkArray('first-order differencing getResidual', transformed1, transformed1True, float)

# Test inverse transformation
inverse1 = diff1.getComposite(transformed1True, params, pivot, settings)
# The inverse of first-order differencing is the cumulative sum, which should recover the original signal
checkArray('first-order differencing getComposite', inverse1, signals, float)


# Test second-order differencing
targets = ['B']
pivot = np.arange(11)
signals = np.linspace(-1, 1, 11).reshape(-1, 1)

# Test fit
diff2, settings = createDifferencing(targets=targets, order=2)
params = diff2.fit(signals, pivot, targets, settings)
checkArray('second-order differencing fit', params['B']['initValues'], signals[:2], float)

# Test forward transformation
transformed2 = diff2.getResidual(signals, params, pivot, settings)
transformed2True = np.r_[np.diff(signals, n=2, axis=0).ravel(), [np.nan] * settings['order']].reshape(-1, 1)
checkArray('second-order differencing getResidual', transformed2, transformed2True, float)

# Test inverse transformation
inverse2 = diff2.getComposite(transformed2True, params, pivot, settings)
checkArray('second-order differencing getComposite', inverse2, signals, float)

print(results)

sys.exit(results["fail"])
