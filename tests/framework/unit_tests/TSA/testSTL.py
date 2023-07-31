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
  This Module performs Unit Tests for the TSA.PreserveCDF class.
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

from ravenframework.TSA import STL

print('Modules undergoing testing:')
print(STL)
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
def createSTL(targets, period=None, seasonal=None, trend=None):
  """
    Creates a TSA.STL instance
    @ In, targets, list(str), names of targets
    @ In, period, int, optional, period of seasonality
    @ In, seasonal, int, optional, length of seasonal smoother; must be odd
    @ In, trend, int, optional, length of trend smoother; must be odd
    @ Out, transformer, subclass of TimeSeriesTransformer, transformer instance
    @ Out, settings, dict, settings for transformer
  """
  transformer = STL()
  xml = xmlUtils.newNode(transformer.name.lower(), attrib={'target': ','.join(targets)})
  if period is not None:
    xml.append(xmlUtils.newNode('period', text=period))
  if seasonal is not None:
    xml.append(xmlUtils.newNode('seasonal', text=seasonal))
  if trend is not None:
    xml.append(xmlUtils.newNode('trend', text=trend))
  print(xml)
  inputSpec = transformer.getInputSpecification()()
  inputSpec.parseNode(xml)
  settings = transformer.handleInput(inputSpec)
  return transformer, settings

###################
#      Tests      #
###################
targets = ['A']
pivot = np.arange(1000)
period = 100
trend = pivot / 1000
seasonal = np.sin(2 * np.pi * pivot / period)
signals = (trend + seasonal).reshape(-1, 1)

stl, settings = createSTL(targets, period=period)
params = stl.fit(signals, pivot, targets, settings)

# Test forward transform
# There is no randomness in the signal, and the trends and seasonality are well-defined. The residual
# values should all be near zero.
transformed = stl.getResidual(signals, params, pivot, settings)
checkArray('residual', transformed, np.zeros_like(transformed), float)

# Test inverse transform
# The inverse transform (getComposite) on an array of zeros will return the sum of the trend and
# seasonal components of the STL decomposition.
inverseTransformed = stl.getComposite(np.zeros_like(signals), params, pivot, settings)
checkArray('inverse', inverseTransformed, signals, float)

# The results of the above inverse transformation should also be the same as calling the generate()
# method directly.
generated = stl.generate(params, pivot, settings)
checkArray('generate', generated, inverseTransformed, float)

print(results)

sys.exit(results["fail"])
