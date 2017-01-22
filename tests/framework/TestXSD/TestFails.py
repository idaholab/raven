"""
Created on 2016-Apr-7

@author: cogljj

This was used to test the xsd program.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import xml.etree.ElementTree as ET
import sys, os
from lxml import etree

ravenDir = os.path.dirname(os.path.dirname(os.getcwd()))
frameworkDir = os.path.join(ravenDir,"framework")
sys.path.append(os.path.join(frameworkDir,'utils'))


import InputData
import test_classes



schemaDoc = etree.parse(open(os.path.join("TestXSD","test_more.xsd"),"r"))

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
