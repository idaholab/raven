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
  Testing for rankSelection mechanism for parent selection
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""
import os
import sys
import xarray as xr
import numpy as np

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()

from ravenframework.utils import randomUtils
randomUtils.randomSeed(5489) #initialize numpy RNG with set seed

from ravenframework.Optimizers.parentSelectors.parentSelectors import returnInstance


rankSelection = returnInstance('tester', 'rankSelection')

#
#
# checkers
#
def checkSameListOfInt(comment, value, expected, update=True):
  """
    This method compares two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, value, list, the value to compare
    @ In, expected, list, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = not abs(value - expected).any()
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking string", comment, '|', value, "!=", expected)
      results["fail"] += 1
  return res

def checkSameDataArrays(comment, resultedDA, expectedDA, update=True):
  """
    This method compares two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, resultedDA, xr.DataArray, the resulted DataArray to be tested
    @ In, expectedDA, xr.DataArray, the expected DataArray
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = resultedDA.identical(expectedDA)
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking string", comment, '|', resultedDA, "!=", expectedDA)
      results["fail"] += 1
  return res

results = {'pass': 0, 'fail': 0}

#
#
# formatters
#
def formatSample(vars):
  return dict ((x, np.atleast_1d(y)) for x, y in vars.items())

#
#
# initialization
#
optVars = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
population =[[1,2,3,4,5,6],
             [2,1,3,4,6,5],
             [6,5,4,3,2,1],
             [3,5,6,2,1,4]]

population = xr.DataArray(population,
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(population)[0]),
                                  'Gene':optVars})

popFitness = [7.2,1.3,9.5,2.0]
popFitness = xr.DataArray(popFitness,
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(np.shape(popFitness)[0])})
# any objective name must work (it used to require the literal 'test_RankSelection')
popFitness = popFitness.to_dataset(name = 'ans')

print('*'*39)
print('Rank based Parent Selection')
print('*'*39)
# Linear ranking: weights 1..n from worst to best, so P(pick individual k first) = rank_k / sum(ranks).
# Ranks for fitness [7.2, 1.3, 9.5, 2.0] are [3, 1, 4, 2]; sum = 10.
expectedProb = np.array([0.3, 0.1, 0.4, 0.2])
nDraws = 4000
counts = np.zeros(4)
randomUtils.randomSeed(2026)
for _ in range(nDraws):
  parents = rankSelection(population, variables=optVars, fitness=popFitness, nParents=1)
  counts[population.values.tolist().index(parents.values[0].tolist())] += 1
freq = counts / nDraws
print('selection frequencies: {} (expected {})'.format(freq, expectedProb))

## TESTING
# the best individual is the most likely parent, and frequencies match the linear-ranking probabilities
checkSameListOfInt('Best individual is the most likely parent', np.array([int(np.argmax(freq))]), np.array([2]))
if np.allclose(freq, expectedProb, atol=0.03):
  results["pass"] += 1
else:
  print('frequencies', freq, 'differ from', expectedProb)
  results["fail"] += 1
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
