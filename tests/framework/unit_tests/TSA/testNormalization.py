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
  This Module performs Unit Tests for classes in TSA.Transformers.Normalizers.
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

from ravenframework.TSA import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

print('Modules undergoing testing:')
testedClasses = [MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer]
print(*testedClasses)
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

def checkFails(comment, errstr, function, update=True, args=None, kwargs=None):
  """
    Checks if expected error occurs
    @ In, comment, string, a comment printed out if it fails
    @ In, errstr, str, expected fail message
    @ In, function, method, method to run to test for failure
    @ In, update, bool, optional, if False then don't update results counter
    @ In, args, list, arguments to pass to function
    @ In, kwargs, dict, keyword arguments to pass to function
    @ Out, res, bool, True if failed as expected
  """
  print('Error testing ...')
  if args is None:
    args = []
  if kwargs is None:
    kwargs = {}
  try:
    function(*args,**kwargs)
    res = False
    msg = 'Function call did not error!'
  except Exception as e:
    res = checkSame('',e.args[0],errstr,update=False)
    if not res:
      msg = 'Unexpected error message.  \n    Received: "{}"\n    Expected: "{}"'.format(e.args[0],errstr)
  if update:
    if res:
      results["pass"] += 1
      print(' ... end Error testing (PASSED)')
    else:
      print("checking error",comment,'|',msg)
      results["fail"] += 1
      print(' ... end Error testing (FAILED)')
  print('')
  return res

######################################
#            CONSTRUCTION            #
######################################
def createTransformer(targets, transformerType, **kwargs):
  """
    Creates a transformer of the given type.
    @ In, targets, list(str), names of targets
    @ In, transformerType, type, type of transformer to create
    @ In, kwargs, dict, optional, keyword arguments to add as attributes to transformer XML node
    @ Out, transformer, subclass of TimeSeriesTransformer, transformer instance
  """
  transformer = transformerType()
  xml = xmlUtils.newNode(transformer.name.lower(), attrib={'target':','.join(targets), **kwargs})
  inputSpec = transformer.getInputSpecification()()
  inputSpec.parseNode(xml)
  transformer.handleInput(inputSpec)
  return transformer

###################
#      Tests      #
###################
# Test MaxAbsScaler
targets = ['A']
pivot = np.arange(11)
signals = np.linspace(-2, 2, 11).reshape(-1, 1)
settings = {}
maxAbsScaler = createTransformer(targets, MaxAbsScaler)
# Check fitted parameters
params = maxAbsScaler.fit(signals, pivot, targets, settings)
scale = params['A']['scale']
scaleTrue = 2
checkFloat('MaxAbsScaler fit', scale, scaleTrue)
# Check forward transform
transformed = maxAbsScaler.getResidual(signals, params, pivot, settings)
transformedTrue = np.linspace(-1, 1, 11).reshape(-1, 1)
checkArray('MaxAbsScaler getResidual', transformed, transformedTrue, float)
# Check inverse transform
# The inverse transform should recover the original signals
inverse = maxAbsScaler.getComposite(transformed, params, pivot, settings)
checkArray('MaxAbsScaler getComposite', inverse, signals, float)

# Test MinMaxScaler
targets = ['B']
pivot = np.arange(11)
signals = np.linspace(-2, 2, 11).reshape(-1, 1)
settings = {}
minMaxScaler = createTransformer(targets, MinMaxScaler)
# Check fitted parameters
params = minMaxScaler.fit(signals, pivot, targets, settings)
minValue = params['B']['dataMin']
maxValue = params['B']['dataMax']
minValueTrue = -2
maxValueTrue = 2
scaleTrue = 0.25  # 1 / (2 - (-2))
checkFloat('MinMaxScaler fit minValue', minValue, minValueTrue)
checkFloat('MinMaxScaler fit maxValue', maxValue, maxValueTrue)
# Check forward transform
transformed = minMaxScaler.getResidual(signals, params, pivot, settings)
transformedTrue = np.linspace(0, 1, 11).reshape(-1, 1)
checkArray('MinMaxScaler getResidual', transformed, transformedTrue, float)
# Check inverse transform
# The inverse transform should recover the original signals
inverse = minMaxScaler.getComposite(transformed, params, pivot, settings)
checkArray('MinMaxScaler getComposite', inverse, signals, float)

# Test StandardScaler
targets = ['C']
pivot = np.arange(11)
signals = np.linspace(-2, 2, 11).reshape(-1, 1)
settings = {}
standardScaler = createTransformer(targets, StandardScaler)
# Check fitted parameters
params = standardScaler.fit(signals, pivot, targets, settings)
mean = params['C']['mean']
scale = params['C']['scale']
meanTrue = 0
scaleTrue = np.std(signals)
checkFloat('StandardScaler fit mean', mean, meanTrue)
checkFloat('StandardScaler fit scale', scale, scaleTrue)
# Check forward transform
transformed = standardScaler.getResidual(signals, params, pivot, settings)
transformedTrue = (signals - meanTrue) / scaleTrue
checkArray('StandardScaler getResidual', transformed, transformedTrue, float)
# Check inverse transform
# The inverse transform should recover the original signals
inverse = standardScaler.getComposite(transformed, params, pivot, settings)
checkArray('StandardScaler getComposite', inverse, signals, float)

# Test RobustScaler
targets = ['D']
pivot = np.arange(11)
signals = np.linspace(-2, 2, 11).reshape(-1, 1)
settings = {}
robustScaler = createTransformer(targets, RobustScaler)
# Check fitted parameters
params = robustScaler.fit(signals, pivot, targets, settings)
center = params['D']['center']
scale = params['D']['scale']
centerTrue = 0  # median at 0
scaleTrue = 2  # quartiles at -1 and 1, so interquarter range is 2
checkFloat('RobustScaler fit center', center, centerTrue)
checkFloat('RobustScaler fit scale', scale, scaleTrue)
# Check forward transform
transformed = robustScaler.getResidual(signals, params, pivot, settings)
transformedTrue = np.linspace(-1, 1, 11).reshape(-1, 1)
checkArray('RobustScaler getResidual', transformed, transformedTrue, float)
# Check inverse transform
# The inverse transform should recover the original signals
inverse = robustScaler.getComposite(transformed, params, pivot, settings)
checkArray('RobustScaler getComposite', inverse, signals, float)

# Testing the QuantileTransformer is a bit more difficult because the internal operations are more
# involved. However, if the distributions of both the original and target distributions are known,
# an analytical solution for the transformation can be derived through a change of random variables.
# The QuantileTransformer can use either a normal or uniform distribution as the target distribution.
# With the target distribution as the normal distribution, the QuantileTransformer struggles to
# estimate the correct quantiles of the extreme values of the distribution due to the asymptotic
# nature of the tails of the distribution. This is not a problem with the uniform distribution due
# to the bounded domain of the distribution. We can minimize the impact of this tail behavior by
# increasing the number of samples used to estimate the quantiles, since the quantile function
# estimation should converge to the true quantile function as the number of samples increases.
# However, it is still necessary for "reasonable" sample sizes to remove the extreme values from the
# transformed distribution before comparing the analytical solution and the quantile transformation.
# Even with these measures, the test tolerance must be kept rather high to accommodate the growing
# error in the tails.
#   - j-bryan

# Test QuantileTransformer (normal)
targets = ['E']
pivot = np.arange(3000)
np.random.seed(42)
signals = np.random.uniform(0, 1, (3000, 1))  # uniform distribution
settings = {}
quantileTransformer = createTransformer(targets, QuantileTransformer, outputDistribution='normal')
params = quantileTransformer.fit(signals, pivot, targets, settings)
# Check forward transform
transformed = quantileTransformer.getResidual(signals, params, pivot, settings)
transformedTrue = norm.ppf(signals)  # norm.ppf is the inverse CDF of the normal distribution
# Remove values greater than 3 standard deviations from the mean to avoid tail estimation issues
# when checking results
transformed = transformed[np.abs(transformedTrue) < 3].reshape(-1, 1)
transformedTrue = transformedTrue[np.abs(transformedTrue) < 3].reshape(-1, 1)
checkArray('QuantileTransformer.getResidual() (uniform -> normal)', transformed, transformedTrue, float, tol=3e-1)
# Check inverse transform
# The inverse transform should recover the original signals
inverse = quantileTransformer.getComposite(transformed, params, pivot, settings)
checkArray('QuantileTransformer.getComposite() (uniform -> normal)', inverse, signals, float)

# Test QuantileTransformer (uniform)
targets = ['F']
pivot = np.arange(500)
np.random.seed(42)
signals = np.random.normal(0, 1, (500, 1))  # far fewer samples are required
settings = {}
quantileTransformer = createTransformer(targets, QuantileTransformer, outputDistribution='uniform')
params = quantileTransformer.fit(signals, pivot, targets, settings)
# Check forward transform
transformed = quantileTransformer.getResidual(signals, params, pivot, settings)
transformedTrue = norm.cdf(signals)  # norm.cdf is the CDF of the normal distribution
checkArray('QuantileTransformer.getResidual() (normal -> uniform)', transformed, transformedTrue, float, tol=3e-2)
# Check inverse transform
# The inverse transform should recover the original signals
inverse = quantileTransformer.getComposite(transformed, params, pivot, settings)
checkArray('QuantileTransformer.getComposite() (normal -> uniform)', inverse, signals, float)


print(results)

sys.exit(results["fail"])
