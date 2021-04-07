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

frameworkPath = os.path.join(os.path.dirname(__file__), *(['..']*4), 'framework')
sys.path.append(frameworkPath)

from utils import InputData
print('Testing:', InputData)

results = {'pass':0, 'fail':0}

####################################
# Test InputData creating LaTeX
#
# load libraries for all of RAVEN
import Driver
# test MultiRun Step
import Steps

# write tex
stepSpec = Steps.MultiRun.getInputSpecification()()
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
