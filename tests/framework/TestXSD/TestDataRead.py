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
Created on 2016-Jan-26

@author: cogljj

This was used to test the xsd program.
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import xml.etree.ElementTree as ET
import sys, os

ravenDir = os.path.dirname(os.path.dirname(os.getcwd()))
frameworkDir = os.path.join(ravenDir,"framework")
sys.path.append(frameworkDir)

from utils import InputData
import test_classes


outerInput = test_classes.OuterInput()

testMoreFilename = os.path.join("TestXSD","test_more.xml")
testMoreXSDFilename = os.path.join("TestXSD","test_more.xsd")

parser = ET.parse(testMoreFilename)

outerInput.parseNode(parser.getroot())

#first inner
firstInner = outerInput.subparts[0]

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

checkAnswer('value_1', firstInner.parameterValues['data_1'])
checkAnswer(42, firstInner.parameterValues['int_value'])
checkAnswer('hi', firstInner.parameterValues['required_string'])

sub3Index = None
for i in range(len(firstInner.subparts)):
  if firstInner.subparts[i].name == 'sub_3':
    sub3Index = i
checkAnswer(23, firstInner.subparts[sub3Index].value)

checkAnswer(23, firstInner.findFirst('sub_3').value)
checkAnswer(None, firstInner.findFirst('no_such_sub'))

print('sub_value_2', repr(outerInput.subparts[2].subparts[0].value),
      repr(outerInput.subparts[2].subparts[0].name))
checkAnswer('sub_value_2', outerInput.subparts[2].subparts[0].value)

checkAnswer(["1", "abc", "3"], outerInput.subparts[1].parameterValues['list_of_strings'])

outside = InputData.createXSD(test_classes.OuterInput)
outsideTree = ET.ElementTree(outside)
outsideTree.write(testMoreXSDFilename)

try:
  from lxml import etree

  schemaDoc = etree.parse(open(testMoreXSDFilename,"r"))

  schema = etree.XMLSchema(schemaDoc)

  testDoc = etree.parse(open(testMoreFilename,"r"))

  valid = schema.validate(testDoc)
  checkAnswer(True, valid)
  print("valid",valid)
except ImportError:
  print("Unable to import lxml")

print("passes",passFails[0],"fails",passFails[1])
sys.exit(passFails[1])
"""
  <TestInfo>
    <name>framework.test_xsd_input_data</name>
    <author>cogljj</author>
    <created>2016-04-11</created>
    <classesTested>None</classesTested>
    <description>
       This test is aimed to check the functionality of the XSD python validator
    </description>
    <revisions>
      <revision author="cogljj" date="2016-04-12">Adding a findFirst function to the xml reader.</revision>
      <revision author="cogljj" date="2016-04-12">Renaming text to value in ParameterInput</revision>
      <revision author="cogljj" date="2016-07-05">Add ability to run without lxml. The first checkes for lxml and skips the test if missing. The second checkes for lxml before running part of the test.</revision>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
    </revisions>
  </TestInfo>
"""
