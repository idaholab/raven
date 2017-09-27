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

def ExternalXMLread(externalXMLFile,externalXMLNode,xmlFileName=None):
  """
    parses the external xml input file
    @ In, externalXMLFile, the filename for the external xml file that will be loaded
    @ In, externalXMLNode, decribes which node will be loaded to raven input file
    @ In, xmlFileName, the raven input file name
    @ Out, externalElemment, xml.etree.ElementTree object that will be added to the current tree of raven input
  """
  if '~' in externalXMLFile: externalXMLFile = os.path.expanduser(externalXMLFile)
  if not os.path.isabs(externalXMLFile):
    if xmlFileName == None:
      raise IOError('Relative working directory requested but input xmlFileName is None.')
    xmlDirectory = os.path.dirname(os.path.abspath(xmlFileName))
    externalXMLFile = os.path.join(xmlDirectory,externalXMLFile)
  if os.path.exists(externalXMLFile):
    externalTree = ET.parse(externalXMLFile)
    externalElement = externalTree.getroot()
    if externalElement.tag != externalXMLNode:
      raise IOError('The required type is: ' + externalXMLNode + 'is different from the provided external xml type: ' + externalElement.tag)
  else:
    raise IOError('The external xml input file ' + externalXMLFile + ' does not exist!')
  return externalElement

def XMLpreprocess(xmlNode,xmlFileName=None):
  """
    Preprocess the xml input file, load external xml files into the main ET
    @ In, xmlNode, xml.etree.ElementTree object, root element of RAVEN input file
    @ In, xmlFileName, the raven input file name
  """
  for element in xmlNode.iter():
    for s,subElement in enumerate(element):
      if subElement.tag == 'ExternalXML':
        print('-'*2+' Loading external xml within block '+ element.tag+ ' for: {0:15}'.format(str(subElement.attrib['node']))+2*'-')
        nodeName = subElement.attrib['node']
        xmlToLoad = subElement.attrib['xmlToLoad'].strip()
        newElement = ExternalXMLread(xmlToLoad,nodeName,xmlFileName)
        # append/remove destroys the order of the entries.  Instead, replace in place
        #element.append(newElement)
        #element.remove(subElement)
        element[s] = newElement
        XMLpreprocess(xmlNode,xmlFileName)

def convert(tree,fileName=None):
  """
    Converts input files to be compatible with merge request  (wangc/external_xml input).  Replace the <ExternalXML> node
    with content defined in the external xml files.
    @ In, tree, xml.etree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  XMLpreprocess(simulation,fileName)
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
