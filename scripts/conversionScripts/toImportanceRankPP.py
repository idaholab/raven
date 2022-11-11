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
    Converts input files to be compatible with merge request !624
    Change the structure of the ImportanceRank Postprocessor
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation': return tree #this isn't an input file
  changeNodeList = []
  for ppNode in simulation.iter('PostProcessor'):
    ppSubType = ppNode.get('subType')
    if ppSubType == 'ImportanceRank':
      changeNodeList.append(ppNode)
  if changeNodeList == []: return tree
  for ppNode in changeNodeList:
    hasDim = False
    dimNode = ppNode.find('dimensions')
    if dimNode is not None:
      hasDim = True
      dim = dimNode.text
    featuresNode = ppNode.find('features')
    featuresType = featuresNode.get('type')
    if featuresType == None: return tree
    features = featuresNode.text
    ppNode.remove(featuresNode)
    newNode = ET.Element('features')
    if featuresType == "":
      manifestNode = ET.SubElement(newNode,'manifest')
      varNode = ET.SubElement(manifestNode,'variables')
      varNode.text = features
      if hasDim:
        manifestDim = ET.SubElement(manifestNode,'dimensions')
        manifestDim.text = dim
    elif featuresType == "latent":
      latentNode = ET.SubElement(newNode,'latent')
      varNode = ET.SubElement(latentNode,'variables')
      varNode.text = features
      if hasDim:
        latentDim = ET.SubElement(latentNode,'dimensions')
        latentDim.text = dim
    ppNode.insert(1,newNode)
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
