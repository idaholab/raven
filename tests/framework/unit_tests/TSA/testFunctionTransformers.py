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
  This Module performs Unit Tests for the data transformers that inherit from the TSA.SklTransformer
  base class. These are classes which wrap sklearn.preprocessing.FunctionTransformer objects to perform
  simple nonlinear transformations on the data.
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

from ravenframework.TSA import LogTransformer, ArcsinhTransformer, SigmoidTransformer, TanhTransformer

print('Modules undergoing testing:')
testedClasses = [LogTransformer, ArcsinhTransformer, SigmoidTransformer, TanhTransformer]
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
def createTransformer(targets, transformerType):
  """
    Creates a transformer of the given type.
    @ In, targets, list(str), names of targets
    @ In, transformerType, type, type of transformer to create
    @ Out, transformer, subclass of TimeSeriesTransformer, transformer instance
  """
  transformer = transformerType()
  xml = xmlUtils.newNode(transformer.name.lower(), attrib={'target':','.join(targets)})
  inputSpec = transformer.getInputSpecification()()
  inputSpec.parseNode(xml)
  transformer.handleInput(inputSpec)
  return transformer

###################################
#            UTILITIES            #
###################################
def extractTransformerFunctions(transformer):
  """
    Extracts the functions from a transformer.
    @ In, transformer, subclass of TimeSeriesTransformer, transformer instance
    @ Out, functions, tuple, tuple of functions (func, inverseFunc)
  """
  params = transformer.templateTransformer.get_params()

  # Forward and inverse transformation functions may have been given as None, in which no transformation
  # is applied. In this case, we set the function to the identity function.
  func = params['func'] if params['func'] is not None else lambda x: x
  inverseFunc = params['inverse_func'] if params['inverse_func'] is not None else lambda x: x

  return func, inverseFunc

###################
#      Tests      #
###################
# Test positive values only
targets = ['A']
pivot = np.arange(11)
signals = np.linspace(1, 2, 11).reshape(-1, 1)

for transformerType in testedClasses:
  transformer = createTransformer(targets, transformerType)
  func, inverseFunc = extractTransformerFunctions(transformer)

  params = transformer.fit(signals, pivot, targets, {})

  # Test forward transformation
  transformed = transformer.getResidual(signals, params, pivot, {})
  transformedTrue = func(signals)
  checkArray(f'{transformer.name} forward transform (all positive values)', transformed, transformedTrue, float)

  # Test inverse transformation
  inverse = transformer.getComposite(transformed, params, pivot, {})
  inverseTrue = inverseFunc(transformedTrue)
  checkArray(f'{transformer.name} inverse transform (all positive values)', inverse, inverseTrue, float)

  # NOTE The forward and inverse transformation functions are not necessarily inverses of each other,
  # and inverseFunc(func(signal)) may not recover the original signal. However, for all tested
  # transformations, the inverse of the inverse should recover the original signal. We test this here.
  # This test may not apply to transformations added in the future.
  checkArray(f'{transformer.name} inverse of inverse (all positive values)', inverse, signals, float)

# Test negative values only
# This is important because some transforms (e.g. log transform) will fail on negative values
targets = ['B']
signals = np.linspace(-1, 1, 11).reshape(-1, 1)

for transformerType in testedClasses:
  transformer = createTransformer(targets, transformerType)
  func, inverseFunc = extractTransformerFunctions(transformer)

  params = transformer.fit(signals, pivot, targets, {})

  if transformer.name == 'LogTransformer':
    # Log transform should fail on negative values
    checkFails(f'{transformer.name} forward transform (all negative values)',
               'Log transformation requires strictly positive values, and negative values '
               'were found in target "B"! If negative values were expected, perhaps '
               'an ArcsinhTransformer would be more appropriate?',
               transformer.getResidual,
               args=(signals, params, pivot, {}))
    continue

  # Test forward transformation
  transformed = transformer.getResidual(signals, params, pivot, {})
  transformedTrue = func(signals)
  checkArray(f'{transformer.name} forward transform (mixed sign values)', transformed, transformedTrue, float)

  # Test inverser transformation
  inverse = transformer.getComposite(transformed, params, pivot, {})
  inverseTrue = inverseFunc(transformedTrue)
  checkArray(f'{transformer.name} inverse transform (mixed sign values)', inverse, inverseTrue, float)

  # NOTE The forward and inverse transformation functions are not necessarily inverses of each other,
  # and inverseFunc(func(signal)) may not recover the original signal. However, for all tested
  # transformations, the inverse of the inverse should recover the original signal. We test this here.
  # This test may not apply to transformations added in the future.
  checkArray(f'{transformer.name} inverse of inverse (mixed sign values)', inverse, signals, float)


print(results)

sys.exit(results["fail"])
