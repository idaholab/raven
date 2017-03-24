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
    Converts input files to be compatible with merge request ???.
    Changes the <OutStreamManager> nodes to <OutStreams> nodes.
    Changes class="OutStreamManager" to class="OutStreams".
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a
      RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN
      input file
  """
  rootNode = tree.getroot()
  if rootNode.tag not in ['Simulation', 'OutStreamManager', 'Steps']:
    ## This is not a valid input file, or at least not one we care about for
    ## this conversion
    return tree
  osmNode = None
  stepsNode = None
  if rootNode.tag == 'Simulation':
    osmNode = rootNode.find('OutStreamManager')
    stepsNode = rootNode.find('Steps')
  elif rootNode.tag == 'outstreamManager':
    ## Case for when the OutStreamManager node is specified in an external file.
    ## (Steps should not be in this file?)
    osmNode = rootNode
  elif rootNode.tag == 'Steps':
    ## Case for when the Steps node is specified in an external file.
    ## (OutStreamManager should not be in this file?)
    stepsNode = rootNode

  if osmNode is not None:
    osmNode.tag = 'OutStreams'

  if stepsNode is not None:
    for outputNode in stepsNode.iter('Output'):
      if 'class' in outputNode.attrib and outputNode.attrib['class'] == 'OutStreamManager':
        outputNode.attrib['class'] = 'OutStreams'

  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
