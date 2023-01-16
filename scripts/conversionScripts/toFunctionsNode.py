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
    Converts input files to be compatible with merge request #935.
    Changes the <Functions> <External> from using multiple
    <variable> nodes to using a single comma separated list of variables
    in node <variables>.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a
      RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN
      input file
  """
  rootNode = tree.getroot()
  if rootNode.tag not in ['Simulation', 'Functions']:
    ## This is not a valid input file, or at least not one we care about for
    ## this conversion
    return tree
  osmNode = None
  stepsNode = None
  if rootNode.tag == 'Simulation':
    functionsNode = rootNode.find('Functions')

  elif rootNode.tag == 'Functions':
    ## Case for when the Functions node is specified in an external file.
    functionsNode = rootNode

  if functionsNode is not None:
    for functionNode in functionsNode:
      variables = []
      for var in functionNode.findall('variable'):
        varName = var.text
        if varName is not None:
          variables.append(var.text.strip())
        functionNode.remove(var)
      variablesNode = ET.Element('variables')
      variablesNode.text = ",".join(variables)
      functionNode.append(variablesNode)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
