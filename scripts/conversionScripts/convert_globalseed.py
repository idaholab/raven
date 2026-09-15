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

def convert(tree,fileName=None):
  """
    Converts input files to include the <globalSeed> argument under <RunInfo>. This enables
    backwards compatibility following updates made to randomUtils.py.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  newTag = 'globalSeed'
  newValue = 5489
  revisionStr = "'globalSeed' tag added to node 'RunInfo' to enable backwards compatability following updates made to randomUtils.py."

  simulation = tree.getroot()
  # Add 'globalSeed' tag to 'RunInfo' node
  runInfo = simulation.find('RunInfo')
  if runInfo is not None:
    elemExists = False
    for child in runInfo:
      if child.tag == newTag:
        elemExists = True
    if not elemExists:
      newElem = ET.Element(newTag)
      newElem.text = str(newValue)
      runInfo.append(newElem)
  # Update 'TestInfo' tag
    testInfo = simulation.find('TestInfo')
    if testInfo is not None:
      elemExists = False
      for child in testInfo:
        if child.tag == 'revisions':
          elemExists = True
      if elemExists:
        revisions = testInfo.find('revisions')
        newRevision = ET.Element('revision')
        newRevision.attrib['author'] = "rollnk"
        newRevision.attrib['date'] = "2025-09-10"
        newRevision.text = revisionStr
        revisions.append(newRevision)
      else:
        revisions = ET.Element('revisions')
        newRevision = ET.Element('revision')
        newRevision.attrib['author'] = "rollnk"
        newRevision.attrib['date'] = "2025-09-10"
        newRevision.text = revisionStr
        revisions.append(newRevision)
        testInfo.append(revisions)
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
  ## the following is provided as an example format for the expected content of the system arguments (sys.argv):
  #sys.argv = ['/path/to/raven/scripts/conversionScripts/convert_globalseed.py','--tests','--no-rewrite','--remove-comments','/path/to/raven/tests/']
