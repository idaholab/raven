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

from ravenframework.TSA import ZeroFilter

print('Module undergoing testing:')
print(ZeroFilter)
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

  if len(first) != len(second):
    res = False
    print(f'checking answer {comment} | lengths do not match: {len(first)} {len(second)}')
    return res

  for i, _ in enumerate(first):
    if dtype == float:
      pres = checkFloat('', first[i], second[i], tol, update=False)
    elif dtype == str:
      pres = checkSame('', first[i], second[i], update=False)
    else:
      pres = False

    if not pres:
      print(f'checking array {comment} | entry "{i}" does not match: {first[i]} != {second[i]}')
      res = False

  if update:
    updateRes(comment, dtype, res, first, second, printComment=False)

  return res

def checkDictSubset(comment, first, second, keys, update=True):
  checkSame(
    comment,
    {k: first[k] for k in first.keys() & set(keys)},
    {k: first[k] for k in second.keys() & set(keys)},
    update
  )

######################################
#            CONSTRUCTION            #
######################################
def createZeroFilter(targets, fill=None, tol=None):
  """
    Creates a ZeroFilter object
    @ In, targets, list(str), list of targets
    @ In, fill, str | float | None, optional, fill value for masked elements
    @ In, tol, float | None, optional, absolute tolerance
    @ Out, zeroFilter, ZeroFilter, zero filter object
    @ Out, settings, dict, filter settings
  """
  attribs = {
    'target': ','.join(targets),
  }
  if fill is not None:
    attribs['fill'] = str(fill)
  if tol is not None:
    attribs['tol'] = str(tol)
  xml = xmlUtils.newNode('zerofilter', attrib=attribs)
  zeroFilter = ZeroFilter()
  inputSpec = ZeroFilter.getInputSpecification()()
  inputSpec.parseNode(xml)
  settings = zeroFilter.handleInput(inputSpec)
  return zeroFilter, settings

###################
#  Tests          #
###################
targets = ['A', 'B']
pivot = np.arange(10)
signalA = np.ones(10)
signalA[3:6] = 0
signalB = np.ones(10)
signalB[0] = 0
signals = np.vstack([signalA, signalB]).T

# Test fit and getResidual
zf, settings = createZeroFilter(targets)
defaultSettings = {'fillValue': np.nan, 'tol': 1e-8}
checkDictSubset('Default settings', settings, defaultSettings, ['fill', 'tol'])
params = zf.fit(signals, pivot, targets, settings)
filteredSignals = zf.getResidual(signals, params, pivot, settings)
okayResidual = signals.copy()
okayResidual[okayResidual == 0] = np.nan

checkArray('Simple getResidual', filteredSignals.ravel(), okayResidual.ravel(), float)

# Test getComposite
composite = np.ones_like(signals)
composite = zf.getComposite(composite, params, pivot, settings)
# composite should have values replaced where they were in the original signal
checkArray('Simple getComposite', composite.ravel(), signals.ravel(), float)

# Test fill value setting
targets = ['C']
fillValue = 0.0
tol = 1e-6
signals = np.array([-1e-10, 0, 1e-7, 1e-5, 1]).reshape(-1, 1)
okayResidual = np.array([fillValue, fillValue, fillValue, 1e-5, 1]).reshape(-1, 1)
pivot = np.arange(signals.shape[0])

zf, settings = createZeroFilter(targets, fill=fillValue, tol=tol)
customSettings = {'fillValue': fillValue, 'tol': tol}
checkDictSubset('Custom fill and tol settings', settings, customSettings, ['fill', 'tol'])
params = zf.fit(signals, pivot, targets, settings)
filteredSignals = zf.getResidual(signals, params, pivot, settings)

checkArray('Custom fill and tol getResidual', filteredSignals.ravel(), okayResidual.ravel(), float)

# Test getComposite
# Input array can have any value in the place of the masked elements (indexes 0-2 in this case)
composite = np.array([np.inf, np.nan, -1000, 1e-5, 1]).reshape(-1, 1)
composite = zf.getComposite(composite, params, pivot, settings)
# composite should have values replaced where they were in the original signal
checkArray('Custom fill and tol getComposite', composite.ravel(), signals.ravel(), float)

print(results)

sys.exit(results["fail"])
