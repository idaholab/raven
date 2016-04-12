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
sys.path.append(os.path.join(frameworkDir,'utils'))

import InputData
import test_classes


#print("OuterInput params",OuterInput.parameters)

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


outside = InputData.createXSD(test_classes.OuterInput)
outsideTree = ET.ElementTree(outside)
outsideTree.write(testMoreXSDFilename)

from lxml import etree

schemaDoc = etree.parse(open(testMoreXSDFilename,"r"))

schema = etree.XMLSchema(schemaDoc)

testDoc = etree.parse(open(testMoreFilename,"r"))

valid = schema.validate(testDoc)
checkAnswer(True, valid)
print("valid",valid)
print("passes",passFails[0],"fails",passFails[1])
sys.exit(passFails[1])
