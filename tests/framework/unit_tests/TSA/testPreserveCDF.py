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
from scipy.stats import norm

# add RAVEN to path
ravenDir =  os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4)))
frameworkDir = os.path.join(ravenDir, 'framework')
if ravenDir not in sys.path:
  sys.path.append(ravenDir)

from ravenframework.utils import xmlUtils

from ravenframework.TSA import PreserveCDF

print('Modules undergoing testing:')
print(PreserveCDF)
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
def createPreserveCDF(targets):
  """
    Creates a TSA.PreserveCDF instance
    @ In, targets, list(str), names of targets
    @ Out, transformer, subclass of TimeSeriesTransformer, transformer instance
  """
  transformer = PreserveCDF()
  xml = xmlUtils.newNode(transformer.name.lower(), attrib={'target': ','.join(targets)})
  inputSpec = transformer.getInputSpecification()()
  inputSpec.parseNode(xml)
  transformer.handleInput(inputSpec)
  return transformer

###################
#      Tests      #
###################
np.random.seed(42)

targets = ['A']
pivot = np.arange(1000)
signals = np.random.normal(size=(len(pivot), len(targets)))

cdf = createPreserveCDF(targets=targets)
settings = {}
params = cdf.fit(signals, pivot, targets, settings)

# Test forward transform
# The forward transformation should just pass the original signal through (identity transformation)
transformed = cdf.getResidual(signals, params, pivot, settings)
checkArray('PreserveCDF.getResidual', transformed, signals, float)

# Test inverse transform
# The inverse transformation will take the input data and map it to the distribution of the original data
# We'll generate uniformly-distributed data, which will then be mapped to the distribution of the original data
# Note that the inverse transform is not exact (particularly in the tails), so this is somewhat difficult
# to test. We'll just check that the mean and standard deviation are close.
signals2 = np.random.uniform(size=(len(pivot), len(targets)))
inverse = cdf.getComposite(signals2, params, pivot, settings)
mean = np.mean(inverse)
std = np.std(inverse)
# The quantile function for the U(0, 1) distribution is simply the identity function. We can pass
# the uniform data directly into the quantile function of the normal distribution to get the true
# inverse transform.
inverseTrue = norm.ppf(signals2)
meanTrue = np.mean(inverseTrue)
stdTrue = np.std(inverseTrue)
# checkArray('PreserveCDF.getComposite', inverse, inverseTrue, float, tol=1e-1)
checkFloat('PreserveCDF.getComposite mean', mean, meanTrue, tol=1.96*stdTrue/np.sqrt(len(pivot))
checkFloat('PreserveCDF.getComposite std', std, stdTrue, tol=1e-2)

print(results)

sys.exit(results["fail"])
