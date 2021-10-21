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
    Converts input files to be compatible with merge request #1583
    Restructure the Validation PostProcessor, use the subType to indicate the algorithm
    used by the Validation. Remove the specific node 'Probailistic'.
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
      if subType == 'Validation':
        prob = pp.find('Probabilistic')
        if prob is not None:
          pp.set('subType', prob.tag.strip())
          pp.remove(prob)
          updateTestInfo = True

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
      rev.attrib['date'] = '2021-09-28'
      rev.text = 'Convert Validation PostProcessor: subType will be replaced with the Probabilistic node tag, and Probabilistic node is removed'
      revisions.append(rev)
      if not hasRev:
        TestInfo.append(revisions)

  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
