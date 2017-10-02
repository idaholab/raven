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
Created on 2016-Apr-7

@author: cogljj

This was used to test the xsd program.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import xml.etree.ElementTree as ET
import sys, os
from lxml import etree

testDir = os.getcwd()
print("testDir", testDir)
ravenDir = os.path.dirname(os.path.dirname(testDir))
frameworkDir = os.path.join(ravenDir,"framework")
sys.path.append(frameworkDir)


from utils import InputData
import test_classes



schemaDoc = etree.parse(open(os.path.join(testDir,"TestXSD","test_more.xsd"),"r"))

schema = etree.XMLSchema(schemaDoc)

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


for xmlFile in ["test_fail_no_required_attr.xml","test_fail_wrong_content.xml",
                "test_fail_wrong_order.xml","test_fail_wrong_attribute_type.xml"]:
  testDoc = etree.parse(open(os.path.join("TestXSD",xmlFile),"r"))

  valid = schema.validate(testDoc)
  print("valid", valid, xmlFile)
  checkAnswer(False, valid)



print("passes",passFails[0],"fails",passFails[1])
sys.exit(passFails[1])
"""
  <TestInfo>
    <name>framework.test_xsd_input_fails</name>
    <author>cogljj</author>
    <created>2016-04-11</created>
    <classesTested> </classesTested>
    <description>
       This test is aimed to check the functionality of the XSD python validator (failure)
    </description>
    <revisions>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
    </revisions>
  </TestInfo>
"""
