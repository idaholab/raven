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
    Converts input files to be compatible with merge request 542.
    Changes list of <variable> nodes to <variables> nodes.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @ Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation' and simulation.tag!='ExternalModel': return tree #this isn't an input file
  dbNodes = None
  if simulation.tag=='Simulation':
    dbNodes = simulation.find('Databases')
  elif simulation.tag=='Databases': #externalNode case
    dbNodes = simulation
  if dbNodes is not None:
    for dbNode in dbNodes:
      if 'filename' in dbNode.attrib.keys() and 'readMode' not in dbNode.attrib.keys():
        dbNode.attrib['readMode'] = 'read'
      else:
        dbNode.attrib['readMode'] = 'overwrite'
      if 'type' in dbNode.attrib.keys():
        del dbNode.attrib['type']
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
