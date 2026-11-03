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
  Testing for the polynomial mutation (polynomialMutator) method.
  Validates bound-respect, the probability gate (0 -> unchanged, 1 -> changed),
  and reproducibility under a fixed RNG seed.
  @authors: Mohammad Abdo
"""
import os
import sys
import xml.etree.ElementTree as ET
import xarray as xr
import numpy as np

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()

from ravenframework.utils import randomUtils
from ravenframework import MessageHandler
from ravenframework import Distributions
from ravenframework.Optimizers.mutators.mutators import returnInstance

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity': 'debug'})

results = {'pass': 0, 'fail': 0}

def checkTrue(comment, condition):
  """
    Assert a boolean condition.
    @ In, comment, str, message printed on failure
    @ In, condition, bool, condition that must hold
    @ Out, None
  """
  if bool(condition):
    results['pass'] += 1
  else:
    print('FAILED:', comment)
    results['fail'] += 1

def createElement(tag, attrib={}, text={}):
  """
    Build a dummy xml element readable by the distribution classes.
    @ In, tag, str, the node tag
    @ In, attrib, dict, optional, the node attributes
    @ In, text, str, optional, the node text
    @ Out, element, ET.Element, the constructed element
  """
  element = ET.Element(tag, attrib)
  element.text = text
  return element

def getDistribution(xmlElement):
  """
    Parse the xmlElement and return an initialized distribution instance.
    @ In, xmlElement, ET.Element, the distribution definition
    @ Out, distributionInstance, Distribution, the initialized distribution
  """
  distributionInstance = Distributions.factory.returnInstance(xmlElement.tag)
  distributionInstance.setMessageHandler(mh)
  paramInput = distributionInstance.getInputSpecification()()
  paramInput.parseNode(xmlElement)
  distributionInstance._handleInput(paramInput)
  distributionInstance.initializeDistribution()
  return distributionInstance

uniformX = ET.Element("Uniform", {"name": "ux"})
uniformX.append(createElement("lowerBound", text="0.0"))
uniformX.append(createElement("upperBound", text="10.0"))
distX = getDistribution(uniformX)
uniformY = ET.Element("Uniform", {"name": "uy"})
uniformY.append(createElement("lowerBound", text="-5.0"))
uniformY.append(createElement("upperBound", text="5.0"))
distY = getDistribution(uniformY)

optVars = ['x', 'y']
distDict = {'x': distX, 'y': distY}

def makeOffspring(rows):
  """
    Build an offspring DataArray from a list of gene rows.
    @ In, rows, list(list(float)), offspring gene values
    @ Out, da, xr.DataArray, offspring in (chromosome, Gene) layout
  """
  rows = np.array(rows, dtype=float)
  return xr.DataArray(rows, dims=['chromosome', 'Gene'],
                      coords={'chromosome': np.arange(rows.shape[0]), 'Gene': optVars})

polynomialMutator = returnInstance('tester', 'polynomialMutator')

offspring = makeOffspring([[2.0, 1.0], [8.0, -3.0]])

# ---- probability gate: 0 -> unchanged ----
mutatedNone = polynomialMutator(offspring=offspring, distDict=distDict, mutationProb=0.0, variables=optVars).values
checkTrue('polynomial mutation with prob 0 leaves genes unchanged', np.allclose(mutatedNone, offspring.values))

# ---- probability gate: 1 -> changed, and within bounds ----
randomUtils.randomSeed(13)
mutatedAll = polynomialMutator(offspring=offspring, distDict=distDict, mutationProb=1.0, variables=optVars).values
checkTrue('polynomial mutation with prob 1 changes genes', not np.allclose(mutatedAll, offspring.values))
checkTrue('polynomial mutation respects x bounds [0,10]', np.all((mutatedAll[:, 0] >= 0.0) & (mutatedAll[:, 0] <= 10.0)))
checkTrue('polynomial mutation respects y bounds [-5,5]', np.all((mutatedAll[:, 1] >= -5.0) & (mutatedAll[:, 1] <= 5.0)))
checkTrue('polynomial mutation produces no NaNs', not np.any(np.isnan(mutatedAll)))
checkTrue('polynomial mutation does not modify the input array', np.allclose(offspring.values, [[2.0, 1.0], [8.0, -3.0]]))

# ---- reproducibility under fixed seed ----
randomUtils.randomSeed(21)
runA = polynomialMutator(offspring=offspring, distDict=distDict, mutationProb=0.5, variables=optVars).values
randomUtils.randomSeed(21)
runB = polynomialMutator(offspring=offspring, distDict=distDict, mutationProb=0.5, variables=optVars).values
checkTrue('polynomial mutation is reproducible under a fixed seed', np.allclose(runA, runB))

print(results)
sys.exit(results['fail'])
