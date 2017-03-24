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

def addToNode(node):
  """
    Adds required node to postprocessor
    @ In node, xml.etree.ElementTree.Element, postprocessor node
    @ Out, none
  """
  text = node.find('type').text
  if text.strip() == 'avg':
    text.replace('avg','average')
    node.find('type').text = text

def convert(tree,fileName=None):
  """
    Converts input files to be compatible with merge request 579.
    Changes Models.PostProcessors(Interfaced).type "avg" -> "average"
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a
      RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN
      input file
  """
  rootNode = tree.getroot()
  if rootNode.tag not in ['Simulation', 'Models', 'PostProcessor']:
    ## This is not a valid input file, or at least not one we care about for
    ## this conversion
    return tree
  ppNode = None
  stepsNode = None
  if rootNode.tag == 'Simulation':
    mods = rootNode.find('Models')
    if mods is None: return tree
    for child in mods:
      if child.tag == 'PostProcessor' and child.attrib['subType'] == 'InterfacedPostProcessor':
        if child.find('method').text == 'HistorySetSnapShot' and child.find('type') is not None:
  elif rootNode.tag == 'Models':
    ## Case for when the Models node is specified in an external file.
    for child in rootNode:
      if child.tag == 'PostProcessor' and child.attrib['subType'] == 'InterfacedPostProcessor':
        if child.find('method').text == 'HistorySetSnapShot' and child.find('type') is not None:
          addToNode(child)
  elif rootNode.tag == 'PostProcessor' and rootNode.attrib['subType'] == 'InterfacedPostProcessor':
    child = rootNode
    ## Case for when the PostProcessor is specified in an external file.
    if child.find('method').text == 'HistorySetSnapShot' and child.find('type') is not None:
      addToNode(child)

  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
