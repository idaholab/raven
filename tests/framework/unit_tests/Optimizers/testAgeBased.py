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
  Testing for ageBased survivor selection mechanism
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
from ravenframework.Optimizers.survivorSelectors.survivorSelectors import returnInstance

ageBased = returnInstance('tester', 'ageBased')

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
objVar = ['ans']
population =[[1,2,3,4,5,6],
             [2,1,3,4,6,5],
             [6,5,4,3,2,1],
             [3,5,6,2,1,4]]

population = xr.DataArray(population,
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(population)[0]),
                                  'Gene':optVars})

popFitness = [7.2,1.3,9.5,2.0]
popFitnessDS = xr.Dataset()
popFitnessDS[objVar] = xr.DataArray(popFitness,
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(np.shape(popFitness)[0])})

popAge = [3,1,7,1]

offSprings = [[2,3,4,5,6,1],
              [1,3,5,2,4,6],
              [1,2,4,3,6,5]]

offSpringsFitness = [1.1,2.0,3.2]
offSpringsFitnessDS = xr.Dataset()
offSpringsFitnessDS[objVar] = xr.DataArray(offSpringsFitness,
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(np.shape(offSpringsFitness)[0])})
rlz =[]
for i in range(np.shape(offSprings)[0]):
  d = {}
  for j in range(np.shape(offSprings)[1]):
    var = optVars[j]
    val = offSprings[i][j]
    d[var] = {'dims':() ,'data': val}
  rlz.append(xr.Dataset.from_dict(d))
rlz = xr.concat(rlz,dim='data')

newPop2,newFit2,newAge2,popObjVal2 = ageBased(rlz,
                                              age=popAge,
                                              popSize=np.shape(population)[0],
                                              variables=optVars,
                                              objVar=objVar,
                                              population=population,
                                              fitness=popFitnessDS,
                                              offSpringsFitness=offSpringsFitnessDS,
                                              popObjectiveVal=popFitnessDS)

print('Age Based Selection')
print('*'*19)
print('new population: {}, \n new Fitness {}, \n new age'.format(newPop2,newFit2,newAge2))
print('Note that the second and forth chromosome had the same age, but for the age based mechanism it omitted the one with the lowest fitness')
expectedPop = xr.DataArray([[3,5,6,2,1,4],
                            [1,2,4,3,6,5],
                            [1,3,5,2,4,6],
                            [2,3,4,5,6,1]],
                            dims=['chromosome','Gene'],
                            coords={'chromosome':np.arange(np.shape(population)[0]),
                                    'Gene': optVars})

expectedFit = xr.DataArray([2.0,3.2,2.0,1.1],
                           dims=['chromosome'],
                           coords={'chromosome':np.arange(np.shape(population)[0])})
expectedFitDS = xr.Dataset()
expectedFitDS[objVar] = xr.DataArray(expectedFit,
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(np.shape(expectedFit)[0])})
expectedAge = [2,0,0,0]

## TESTING
# Test survivor population
checkSameDataArrays('Check survived population data array',newPop2,expectedPop)
# Test survivor fitnesses
checkSameDataArrays('Check fitness for survived population data array',newFit2,expectedFitDS)
# Test survivor Ages
checkSameListOfInt('Check fitness for survived individuals',np.array(newAge2),np.array(expectedAge))
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
