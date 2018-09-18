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
    Converts input files to be compatible with merge request #785:
      Where ARMA exists, removes <Pmax>, <Pmin>, <Qmax>, and <Qmin>, and adds <P> and <Q>
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  if models is not None:
    for child in models:
      if child.tag == 'ROM' and child.attrib['subType'] == 'ARMA':
        # store outdated nodes if existing
        nodeP = child.find('Pmax')
        nodeQ = child.find('Qmax')
        # replace outdated nodes with new node
        if nodeP is not None:
          newP = ET.Element('P')
          newP.text = nodeP.text
          child.remove(nodeP)
          child.remove(child.find('Pmin'))
          child.append(newP)
        if nodeQ is not None:
          newQ = ET.Element('Q')
          newQ.text = nodeQ.text
          child.remove(nodeQ)
          child.remove(child.find('Qmin'))
          child.append(newQ)
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
