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
    Converts input files to be compatible with merge request #789.
    Remove nodes random_state, train_size, tset_size if they are None
    Change text of labels node to list
    Change node n_iter to n_splits
    Change node p to n_groups for LeavePLabelOut
    Change node y to labels
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a
      RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN
      input file
  """
  rootNode = tree.getroot()
  if rootNode.tag not in ['Simulation', 'PostProcessor', 'SciKitLearn']:
    ## This is not a valid input file, or at least not one we care about for
    ## this conversion
    return tree

  for cvNode in rootNode.iter('PostProcessor'):
    if cvNode.attrib['subType'] == 'CrossValidation':
      sklNode = cvNode.find('SciKitLearn')
      trainSize = sklNode.find('train_size')
      if trainSize is not None and trainSize.text.strip() == 'None':
        sklNode.remove(trainSize)
      testSize = sklNode.find('test_size')
      if testSize is not None and testSize.text.strip() == 'None':
        sklNode.remove(testSize)
      randomState = sklNode.find('random_state')
      if randomState is not None and randomState.text.strip() == 'None':
        sklNode.remove(randomState)
      nIter = sklNode.find('n_iter')
      if nIter is not None:
        param = nIter.text.strip()
        nSplits = ET.Element('n_splits')
        nSplits.text = param
        sklNode.remove(nIter)
        sklNode.append(nSplits)
      labelNode = sklNode.find('labels')
      if labelNode is not None:
        params = labelNode.text.strip('[').strip(']')
        labelNode.text = params
      yNode = sklNode.find('y')
      if yNode is not None:
        params = yNode.text.strip('[').strip(']')
        labelNode = ET.Element('labels')
        labelNode.text = params
        sklNode.remove(yNode)
        sklNode.append(labelNode)
      pNode = sklNode.find('p')
      if pNode is not None and sklNode.find('SKLtype').text.strip() == 'LeavePLabelOut':
        params = pNode.text.strip()
        nGroups = ET.Element('n_groups')
        nGroups.text = params
        sklNode.remove(pNode)
        sklNode.append(nGroups)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
