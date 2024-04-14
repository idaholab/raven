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
  Testing for the twoPointsCrossover method
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
from ravenframework.Optimizers.crossOverOperators.crossovers import returnInstance

twoPointsCrossover = returnInstance('tester', 'twoPointsCrossover')

#
#
# checkers
#
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
# initialization
#
optVars = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
population = [[11,12,13,14,15,16,17,18],
              [21,22,23,24,25,26,27,28],
              [31,32,33,34,35,36,37,38],
              [41,42,43,44,45,46,47,48]]
population = xr.DataArray(population,
                          dims   = ['chromosome','Gene'],
                          coords = {'chromosome': np.arange(np.shape(population)[0]),
                                    'Gene':optVars})
nParents = 2
crossoverProb = 0.5
children = twoPointsCrossover(population, crossoverProb=crossoverProb)

print('twoPointsCrossover')
print('*'*19)
print('generated children are: {}'.format(children))
expectedChildren = xr.DataArray([[ 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.],
                                 [ 21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.],
                                 [ 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.],
                                 [ 31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.],
                                 [ 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.],
                                 [ 41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.],
                                 [ 21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.],
                                 [ 31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.],
                                 [ 21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.],
                                 [ 41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.],
                                 [ 31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.],
                                 [ 41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.]],
                                 dims   = ['chromosome','Gene'],
                                 coords = {'chromosome': np.arange(12),
                                           'Gene'      : optVars})

## TESTING
# Test survivor population
checkSameDataArrays('Check survived population data array',children,expectedChildren)
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
