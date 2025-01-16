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

import os
import sys

ravenPath = os.path.join(os.path.dirname(__file__), *(['..']*4))
sys.path.append(ravenPath)

from ravenframework.utils import InputData, TreeStructure
import xml.etree.ElementTree as ET
print('Testing:', InputData)

results = {'pass':0, 'fail':0}



####################################
# Test arbitrary XML spec
#
# Write the spec
ASpec = InputData.parameterInputFactory('A', descr='first')
BSpec = InputData.parameterInputFactory('B', descr='second')
BSpec.setStrictMode(False)
ASpec.addSub(BSpec)
# Write the tree
# <A>
#   <B>
#     <TheQuestion really='True'>unknown</TheQuestion>
#     <TheAnswer>42</TheAnswer>
#   </B>
# </A>
ANode = TreeStructure.InputNode('A')
BNode = TreeStructure.InputNode('B')
TQNode = TreeStructure.InputNode('TheQuestion', attrib={'really': 'True'}, text='unknown')
BNode.append(TQNode)
TANode = TreeStructure.InputNode('TheAnswer', text='42')
BNode.append(TANode)
ANode.append(BNode)
# parse and check
A = ASpec()
A.parseNode(ANode)
B = A.findFirst('B')

if B.additionalInput[0].tag == 'TheQuestion' and \
      B.additionalInput[0].attrib['really'] == 'True' and\
      B.additionalInput[0].text == 'unknown':
  results['pass'] += 1
else:
  print('InputData Arbitrary Custom XML 0 did not match!')
  results['fail'] += 1

if B.additionalInput[1].tag == 'TheAnswer' and \
      B.additionalInput[1].text == '42':
  results['pass'] += 1
else:
  print('InputData Arbitrary Custom XML 1 did not match!')
  results['fail'] += 1


####################################
# load libraries for all of RAVEN
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()
DriverUtils.setupBuiltins()

####################################
# Test InputData creating LaTeX
#
# test MultiRun Step
from ravenframework import Steps

# write tex
stepSpec = Steps.factory.returnClass('MultiRun').getInputSpecification()()
tex = stepSpec.generateLatex()
fName = 'example_multirun_spec.tex'
with open(fName, 'w') as f:
  f.writelines(tex)

# compare
with open('gold/tex/multirun_spec.tex', 'r') as test:
  with open (fName, 'r') as gold:
    t = test.read()
    g = gold.read()
    same = t == g

if same:
  results['pass'] += 1
else:
  print('InputData MultiRun LatexSpecs did not match gold file!')
  results['fail'] += 1


# cleanup
try:
  os.remove(fName)
except:
  print('Test multirun spec file was not able to be removed; continuing ...')


print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.inputData</name>
    <author>talbpaul</author>
    <created>2020-01-08</created>
    <classesTested>utils.InputData</classesTested>
    <description>
       This test performs Unit Tests for the InputData methods
    </description>
  </TestInfo>
"""
