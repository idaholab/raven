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
Created on 2017-Aug-24

@author: cogljj

This is used to check strict mode with InputData
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import xml.etree.ElementTree as ET
import sys, os

ravenDir = os.path.dirname(os.path.dirname(os.getcwd()))
frameworkDir = os.path.join(ravenDir,"framework")
sys.path.append(frameworkDir)

from utils import InputData
import test_classes

passFails = [0,0]
def checkAnswer(expected, actual):
  """
    checks to see if the actual value matches the expected value
    @ In, expected, Any, the expected value
    @ In, actual, Any, the actual value
    @ Out, None
  """
  if expected == actual:
    passFails[0] += 1
  else:
    print("failed expected:",expected," got:",actual)
    passFails[1] += 1

errors = []

testStrictFilename = os.path.join("TestXSD","test_strict.xml")

parser = ET.parse(testStrictFilename)

outerInput = test_classes.OuterInput()

outerInput.parseNode(parser.getroot(), errors)


print(errors)

checkAnswer('Required parameter required_string not in inner', errors[0])
checkAnswer('no_such_element not in attributes and strict mode on in inner', errors[1])
checkAnswer('Child \"no_such_sub\" not allowed as sub-element of \"ordered\"', errors[2])

print("passes",passFails[0],"fails",passFails[1])
sys.exit(passFails[1])
