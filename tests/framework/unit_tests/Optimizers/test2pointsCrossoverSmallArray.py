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
optVars = ['x1', 'x2', 'x3']
population = [[11,12,13],
              [21,22,23],
              [31,32,33]]
population = xr.DataArray(population,
                          dims   = ['chromosome','Gene'],
                          coords = {'chromosome': np.arange(np.shape(population)[0]),
                                    'Gene':optVars})
nParents = 2
children = twoPointsCrossover(population)

print('twoPointsCrossover')
print('*'*19)
print('generated children are: {}'.format(children))
expectedChildren = xr.DataArray([[ 11.,  22.,  13.],
                                 [ 21.,  12.,  23.],
                                 [ 11.,  32.,  13.],
                                 [ 31.,  12.,  33.],
                                 [ 21.,  32.,  23.],
                                 [ 31.,  22.,  33.]],
                                 dims   = ['chromosome','Gene'],
                                 coords = {'chromosome': np.arange(6),
                                           'Gene'      : optVars})

## TESTING
# Test survivor population
checkSameDataArrays('Check survived population data array',children,expectedChildren)
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
