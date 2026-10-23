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
  Unit tests for GeneticAlgorithm fitness normalization: the <normalize> option parsing
  and the zscore normalization math.
"""
import os
import sys
import numpy as np

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()
from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm

results = {'pass': 0, 'fail': 0}

def checkSame(comment, value, expected, update=True):
  """
    Compares two values for equality.
    @ In, comment, str, description printed if it fails
    @ In, value, any, the value to compare
    @ In, expected, any, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking value', comment, '|', value, '!=', expected)
      results['fail'] += 1
  return res

def checkFloat(comment, value, expected, tol=1e-10, update=True):
  """
    Compares two floats within a tolerance.
    @ In, comment, str, description printed if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, absolute tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same within tolerance
  """
  res = abs(value - expected) < tol
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking float', comment, '|', value, '!=', expected)
      results['fail'] += 1
  return res

def checkRaises(comment, callable, exceptionType, update=True):
  """
    Confirms that calling `callable` raises an exception of the given type.
    @ In, comment, str, description printed if it fails
    @ In, callable, callable, zero-argument callable to invoke
    @ In, exceptionType, type, expected exception type
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if the expected exception was raised
  """
  res = False
  try:
    callable()
  except exceptionType:
    res = True
  except Exception as err:
    print('checking raise', comment, '| wrong exception type raised:', type(err), err)
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking raise', comment, '| no', exceptionType, 'was raised')
      results['fail'] += 1
  return res

ga = GeneticAlgorithm()

#
# <normalize> option parsing
#
checkSame("normalize option: absent -> disabled", ga._resolveNormalizeFitnessOption(None), False)
checkSame("normalize option: empty string -> disabled", ga._resolveNormalizeFitnessOption(''), False)
checkSame("normalize option: 'none' -> NoneType", ga._resolveNormalizeFitnessOption('none'), None)
checkSame("normalize option: 'None' (case-insensitive) -> NoneType", ga._resolveNormalizeFitnessOption('None'), None)
checkSame("normalize option: 'true' -> zscore", ga._resolveNormalizeFitnessOption('true'), 'zscore')
checkSame("normalize option: 'True' (case-insensitive) -> zscore", ga._resolveNormalizeFitnessOption('True'), 'zscore')
checkSame("normalize option: 'zscore' -> zscore", ga._resolveNormalizeFitnessOption('zscore'), 'zscore')
checkRaises("normalize option: 'maxmin' is rejected (not implemented downstream)",
            lambda: ga._resolveNormalizeFitnessOption('maxmin'), IOError)
checkRaises("normalize option: unrecognized string is rejected",
            lambda: ga._resolveNormalizeFitnessOption('garbage'), IOError)

#
# zscore normalization math
#
values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
expectedMean = 3.0
expectedStd = np.std(values)
mean, std, normalized = ga._computeZscoreNormalization(values)
checkFloat("zscore mean matches expected", mean, expectedMean)
checkFloat("zscore std matches expected", std, expectedStd)
for i, val in enumerate(values):
  checkFloat(f"zscore normalized value[{i}] matches manual computation",
             normalized[i], (val - expectedMean) / expectedStd)

# constant array: std is 0, normalization would divide by zero -> guarded to 0.0, not NaN
constantValues = np.array([7.0, 7.0, 7.0])
_, constStd, constNormalized = ga._computeZscoreNormalization(constantValues)
checkFloat("zscore std of a constant array is 0", constStd, 0.0)
for i, val in enumerate(constNormalized):
  checkSame(f"zscore normalization of constant array[{i}] is guarded to 0.0, not NaN", val, 0.0)

#
# end
#
print('Results:', results)
sys.exit(results['fail'])
