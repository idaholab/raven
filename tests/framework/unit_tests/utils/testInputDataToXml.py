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
import copy

ravenPath = os.path.join(os.path.dirname(__file__), *(['..']*4))
sys.path.append(ravenPath)
sys.path.append(os.path.join(ravenPath,"rook"))

import xml.etree.ElementTree as ET
from ravenframework.utils import InputData,InputTypes,xmlUtils
from rook import XMLDiff
print('Testing:', InputData)

results = {'pass':0, 'fail':0}

####################################
class ObjectsInputData(object):
  """
  Helper class with matching input specs for XML test
  """
  def __init__(self):
    """
      initialization
      @ In, None
      @ Out, None
    """

  @classmethod
  def get_input_specs(cls):
    """
      Method to get a reference to a class that specifies the input data for ObjectsInputData class.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    # parent node
    spec = InputData.parameterInputFactory('Objects', ordered=False, baseNode=None)
    # -- subnodes
    Object = InputData.parameterInputFactory('Object', ordered=False, baseNode=None)
    # -- subnode attributes
    Object.addParam('name', required=True, param_type=InputTypes.StringType,descr=r"""Object attribute: name""")
    Object.addParam('type', required=True, param_type=InputTypes.StringType,descr=r"""Object attribute: type""")
    # -- -- subsubnode
    paramsNode = InputData.parameterInputFactory('params', ordered=False, baseNode=None)
    alphaNode = InputData.parameterInputFactory('alpha', ordered=False, baseNode=None)
    # -- add subsubnodes to subnode
    Object.addSub(paramsNode)
    Object.addSub(alphaNode)
    # add subnode to parent node
    spec.addSub(Object)
    return spec
####################################

# get original XML filepath
XMLTestDir = os.path.join(os.path.dirname(__file__), 'parse')
testXMLPath = os.path.join(XMLTestDir,'example_xml.xml')

# load test XML and node to test
with open(testXMLPath,'r',encoding="utf8") as f:
  testXML, _ = xmlUtils.loadToTree(f)
testXMLNode = testXML.find('Objects')

# create an instance of testing object
testObj = ObjectsInputData()
# create input specs and load data from XML
testInputData = testObj.get_input_specs()()
testInputData.parseNode(testXMLNode)

# convert back from InputDate to XML
convertedXMLNode = testInputData.convertToXML()

# write two XML scripts for test and converted XML
gold_file = os.path.join(XMLTestDir,'inputDataToXML__gold.xml')
out_file  = os.path.join(XMLTestDir,'inputDataToXML__out.xml')
xmlUtils.toFile(gold_file, testXMLNode)
xmlUtils.toFile(out_file,  convertedXMLNode)

# COMPARISON TEST with XMLDiff from Rook
differ = XMLDiff.XMLDiff([out_file],[gold_file], ignored_nodes=None, alt_root=None)
same, msg = differ.diff()

if same:
  results['pass'] += 1
else:
  print(msg)
  results['fail'] += 1

# cleanup
try:
  os.remove(gold_file)
  os.remove(out_file)
except Exception:
  print(f'Test files {gold_file} and {out_file} were not able to be removed; continuing ...')

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.inputData</name>
    <author>sotogj</author>
    <created>2024-03-19</created>
    <classesTested>utils.InputData</classesTested>
    <description>
       This test performs Unit Tests for the InputData convertToXML method
    </description>
  </TestInfo>
"""
