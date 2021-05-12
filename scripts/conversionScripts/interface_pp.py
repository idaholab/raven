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
import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os

def convert(tree,fileName=None):
  """
    Converts input files to be compatible with merge request #1533
    The InterfacedPostProcessor has been removed, and the subType of given
    PostProcessor has been replaced with text from method node
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  updateTestInfo = False
  if models is not None:
    postProcessors = models.findall('PostProcessor')
    for pp in postProcessors:
      subType = pp.get('subType')
      if subType == 'InterfacedPostProcessor':
        method = pp.find('method')
        pp.set('subType', method.text.strip())
        pp.remove(method)
        updateTestInfo = True
        dataType = pp.find('dataType')
        if dataType is not None:
          pp.remove(dataType)

  if updateTestInfo:
    TestInfo = simulation.find('TestInfo')
    if TestInfo is not None:
      revisions = TestInfo.find('revisions')
      hasRev = True
      if revisions is None:
        revisions = ET.Element('revisions')
        hasRev = False
      rev = ET.Element('revision')
      rev.attrib['author'] = 'wangc'
      rev.attrib['date'] = '2020-05-07'
      rev.text = 'Convert InterfacedPostProcessor: subType will be replaced with the text from method node, and method node will be removed'
      revisions.append(rev)
      if not hasRev:
        TestInfo.append(revisions)

  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
