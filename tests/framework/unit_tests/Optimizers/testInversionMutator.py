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
  Testing for the inversionMutator method
  @authors: Junyung Kim and Mohammad Abdo
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

from ravenframework.Optimizers.mutators.mutators import returnInstance
import xml.etree.ElementTree as ET
from ravenframework import MessageHandler

from ravenframework import Distributions
from ravenframework.Distributions import UniformDiscrete
from ravenframework.Distributions import Categorical

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug'})

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

def createElement(tag,attrib={},text={}):
  """
    Method to create a dummy xml element readable by the distribution classes
    @ In, tag, string, the node tag
    @ In, attrib, dict, optional, the attribute of the xml node
    @ In, text, dict, optional, the dict containing what should be in the xml text
  """
  element = ET.Element(tag,attrib)
  element.text = text
  return element

def getDistribution(xmlElement):
  """
    Parses the xmlElement and returns the distribution
  """
  distributionInstance = Distributions.factory.returnInstance(xmlElement.tag)
  distributionInstance.setMessageHandler(mh)
  paramInput = distributionInstance.getInputSpecification()()
  paramInput.parseNode(xmlElement)
  distributionInstance._handleInput(paramInput)
  distributionInstance.initializeDistribution()
  return distributionInstance

uniformDiscreteElement = ET.Element("UniformDiscrete",{"name":"test"})
uniformDiscreteElement.append(createElement("lowerBound",text="1.0"))
uniformDiscreteElement.append(createElement("upperBound",text="6.0"))
uniformDiscreteElement.append(createElement("strategy",text="withReplacement"))

uniform = getDistribution(uniformDiscreteElement)

distDict = {}
distDict['x1'] = uniform
distDict['x2'] = uniform
distDict['x3'] = uniform
distDict['x4'] = uniform
distDict['x5'] = uniform

optVars = ['x1', 'x2', 'x3', 'x4', 'x5']

population = [[1,4,5,2,6],
              [2,3,1,1,5],
              [6,4,2,3,2]]

population = xr.DataArray(population,
                          dims   = ['chromosome','Gene'],
                          coords = {'chromosome': np.arange(np.shape(population)[0]),
                                    'Gene':optVars})

kwargs = {'locs': [1,4], 'mutationProb': 1.0, 'variables': ['x1', 'x2', 'x3', 'x4', 'x5']}

inversionMutator = returnInstance('tester', 'inversionMutator')

children = inversionMutator(population, distDict, **kwargs)

print('\n'*2)
print('*'*79)
print('Inversion Mutator unit test')
print('*'*79)
print('generated children are: {}'.format(children))
expectedChildren = xr.DataArray([[ 1,  6,  2,  5,  4],
                                 [ 2,  5,  1,  1,  3],
                                 [ 6,  2,  3,  2,  4]],
                                 dims   = ['chromosome','Gene'],
                                 coords = {'chromosome': np.arange(np.shape(population)[0]),
                                           'Gene'      : optVars})

## TESTING
# Test survivor population
checkSameDataArrays('Check survived population data array',children,expectedChildren)
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
