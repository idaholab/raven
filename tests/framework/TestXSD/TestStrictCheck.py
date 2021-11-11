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
frameworkDir = os.path.join(ravenDir,"../framework")
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

testStrictFilename = "test_strict.xml"

parser = ET.parse(testStrictFilename)

outerInput = test_classes.OuterInput()

outerInput.parseNode(parser.getroot(), errors)


print(errors)

checkAnswer('outer.inner: Required attribute "required_string" not in "inner"', errors[0])
checkAnswer('outer.inner: "no_such_element" not in node attributes and strict mode on in "inner"', errors[1])
checkAnswer('outer.ordered: Unrecognized input node "no_such_sub"! Allowed: [sub_1, sub_2, sub_3, sub_bool], tried []', errors[2])

print("passes",passFails[0],"fails",passFails[1])
sys.exit(passFails[1])

"""
 <TestInfo>
    <name>crow.test_normal</name>
    <author>cogljj</author>
    <created>2017-08-30</created>
    <classesTested> </classesTested>
    <description>
      This test is a Unit Test for the RAVEN input checker. This test is aimed to check that
      the input checker (Strict mode) is able to detect input errors.
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""
