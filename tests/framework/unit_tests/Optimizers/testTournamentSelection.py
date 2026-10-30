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
  Testing for tournamentSelection mechanism for parent selection
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

tournamentSelection = returnInstance('tester', 'tournamentSelection')

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
objVar = ['obj1']
population =[[1,2,3,4,5,6],[2,1,3,4,6,5],[6,5,4,3,2,1],[3,5,6,2,1,4]]
population = xr.DataArray(population,
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(population)[0]),
                                  'Gene':optVars})

popFitness = np.atleast_1d([7.2,1.3,9.5,2.0])
FitnessSet = xr.Dataset()
FitnessSet[objVar] = xr.DataArray(popFitness, dims=['chromosome'], coords={'chromosome': np.arange(np.shape(population)[0])})
nParents = 2
kSelection = 2
parents = tournamentSelection(population, variables=optVars, fitness=FitnessSet, nParents=nParents, objVar=objVar, kSelection=kSelection, isMultiObjective=False)
print('Parent Selection with TournamentSelection algorithm')
print('*'*19)
print('selected parents are: {}'.format(parents))
expectedParents = xr.DataArray([[1,2,3,4,5,6],
                                [6,5,4,3,2,1]],
                                dims=['chromosome','Gene'],
                                coords={'chromosome':np.arange(nParents),
                                        'Gene': optVars})

## TESTING
# Test survivor population
checkSameDataArrays('Check survived population data array',parents,expectedParents)
#
# end
#
originalRandomChoice = randomUtils.randomChoice
try:
  choices = iter([[1, 2], [1, 2], [0, 1]])
  def fakeChoice(values, size=None, replace=True, engine=None):
    values = list(values)
    assert values == [0, 1, 2]
    assert size == 2
    assert replace is False
    return np.array(next(choices), dtype=int)

  randomUtils.randomChoice = fakeChoice
  shiftedPopulation = xr.DataArray(np.array([[1, 10], [2, 20], [3, 30]], dtype=float),
                                   dims=['chromosome', 'Gene'],
                                   coords={'chromosome': np.array([10, 20, 30]),
                                           'Gene': ['x1', 'x2']})
  shiftedFitness = xr.Dataset()
  shiftedFitness['obj1'] = xr.DataArray(np.array([1.0, 2.0, 3.0]),
                                             dims=['chromosome'],
                                             coords={'chromosome': shiftedPopulation.coords['chromosome']})
  parents = tournamentSelection(shiftedPopulation, variables=['x1', 'x2'], fitness=shiftedFitness,
                                nParents=3, objVar=objVar, kSelection=2, isMultiObjective=False)
  expectedParents = xr.DataArray(np.array([[3, 30], [3, 30], [2, 20]], dtype=float),
                                 dims=['chromosome', 'Gene'],
                                 coords={'chromosome': np.arange(3),
                                         'Gene': ['x1', 'x2']})
  checkSameDataArrays('Check tournament uses row positions and permits reselection', parents, expectedParents)

  randomUtils.randomChoice = lambda values, size=None, replace=True, engine=None: np.array([0, 1, 2], dtype=int)
  rank = xr.DataArray(np.array([2, 1, 1]), dims=['rank'], coords={'rank': np.arange(3)})
  crowding = xr.DataArray(np.array([0.0, 0.2, 0.9]), dims=['CrowdingDistance'],
                          coords={'CrowdingDistance': np.arange(3)})
  moFitness = xr.Dataset()
  moFitness['obj1'] = xr.DataArray(np.zeros(3), dims=['chromosome'], coords={'chromosome': np.arange(3)})
  parents = tournamentSelection(shiftedPopulation, variables=['x1', 'x2'], fitness=moFitness,
                                nParents=1, objVar=objVar, kSelection=3, isMultiObjective=True,
                                rank=rank, crowdDistance=crowding)
  expectedParents = xr.DataArray(np.array([[3, 30]], dtype=float), dims=['chromosome', 'Gene'],
                                 coords={'chromosome': np.arange(1), 'Gene': ['x1', 'x2']})
  checkSameDataArrays('Check multi-objective tournament tie break returns selected row position', parents, expectedParents)

  def fakeMutatingChoice(values, size=None, replace=True, engine=None):
    assert values == [0, 1, 2, 3]
    return np.asarray(originalRandomChoice(values, size=size, replace=replace, engine=engine), dtype=int)

  randomUtils.randomSeed(5489)
  randomUtils.randomChoice = fakeMutatingChoice
  parents = tournamentSelection(population, variables=optVars, fitness=FitnessSet,
                                nParents=3, objVar=objVar, kSelection=2, isMultiObjective=False)
  checkSameListOfInt('Check tournament candidate pool is reset for each parent',
                     np.asarray(parents.shape), np.asarray([3, 6]))
finally:
  randomUtils.randomChoice = originalRandomChoice

print('Results:', results)
sys.exit(results['fail'])
