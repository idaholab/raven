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
  Testing for rankNcrowdingBased survivor selection mechanism
  @authors: Junyung Kim, Mohammad Abdo
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
from ravenframework.utils import frontUtils

rankNcrowdingBased = returnInstance('tester', 'rankNcrowdingBased')

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

optVars = ['x1', 'x2']

population =[[0.913, 2.348],
             [0.599, 3.092],
             [0.139, 2.138],
             [0.867, 1.753],
             [0.885, 1.455],
             [0.658, 2.607],
             [0.788, 2.545],
             [0.342, 1.639]]

population = xr.DataArray(population,
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(population)[0]),
                                  'Gene':optVars})

offSprings =[[0.620, 3.050],
             [0.165, 1.379],
             [0.885, 2.295],
             [0.985, 2.380],
             [0.826, 1.226],
             [0.788, 2.545],
             [0.343, 1.639],
             [0.121, 1.946]]

rlz =[]
for i in range(np.shape(offSprings)[0]):
  d = {}
  for j in range(np.shape(offSprings)[1]):
    var = optVars[j]
    val = offSprings[i][j]
    d[var] = {'dims':() ,'data': val}
  rlz.append(xr.Dataset.from_dict(d))
rlz = xr.concat(rlz,dim='data')

newPop2,newRank2,newCD2 = rankNcrowdingBased(rlz,
                                             variables=optVars,
                                             population=population)


print('Rank and Crowding Based Selection')
print('*'*19)
print('new population: {}, \n new Rank {}, \n new Crowding Distance'.format(newPop2,newRank2,newCD2))

expectedPop = xr.DataArray([[ 0.826,  1.226],
                            [ 0.121,  1.946],
                            [ 0.165,  1.379],
                            [ 0.139,  2.138],
                            [ 0.885,  1.455],
                            [ 0.342,  1.639],
                            [ 0.343,  1.639],
                            [ 0.599,  3.092]],
                            dims=['chromosome','Gene'],
                            coords={'chromosome':np.arange(np.shape(population)[0]),
                                    'Gene': optVars})
expectedRank = xr.DataArray([1, 1, 1, 2, 2, 2, 3, 4],
                           dims=['rank'],
                           coords={'rank':np.arange(np.shape(population)[0])})
expectedCD = [np.inf, np.inf, 2., np.inf, np.inf, 2., np.inf, np.inf,]

## TESTING
# Test survivor population
checkSameDataArrays('Check survived population data array',newPop2,expectedPop)
# Test survivor rank
checkSameDataArrays('Check rank for survived population data array',newRank2,expectedRank)
# Test survivor Crowding Distance
checkSameListOfInt('Check crowding distance for survived individuals',newCD2.data, np.array(expectedCD))  ## Question: Why it says they are different????
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
