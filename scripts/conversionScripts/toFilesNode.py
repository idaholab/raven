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
    Converts input files to be compatible with merge request 255 (Talbpaul/redundant input).  Removes the <Files> node
    from the RunInfo block and makes it into its own node.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  runinfo = simulation.find('RunInfo')
  oldFilesNode = runinfo.find('Files')
  if oldFilesNode is not None:
    fileNameList = oldFilesNode.text.split(',')
    runinfo.remove(oldFilesNode)
    newFiles = ET.Element('Files')
    for name in fileNameList:
      name = name.strip()
      newfile = ET.Element('Input')
      newfile.set('name',name)
      newfile.set('type','')
      newfile.text = name
      newFiles.append(newfile)
    simulation.insert(1,newFiles)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
