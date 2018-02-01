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
    Converts input files to be compatible with merge request #460
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  steps = simulation.find('Steps')
  postProcess = steps.findall('PostProcess')
  outStreams  = simulation.find('OutStreams')
  PointSets  = simulation.find('DataObjects').findall('PointSet')
  files = simulation.find('Files')
  filesInput = files.findall('Input')
  filesInputOutstreams = {}
  for fileObj in filesInput:
    PrintElement = ET.Element('Print')
    PrintElement.attrib['name'] = fileObj.text.split(".")[0]
    typeElem = ET.Element('type')
    typeElem.text = 'csv'
    PrintElement.append(typeElem)
    filesInputOutstreams[fileObj.attrib['name']] = PrintElement
  limitSurfaceIntegralNames = []
  limitSurfaceNames = []
  outputNamesVariables = []
  if models is None:
    return tree # no models, no LimitSurfaceIntegral
  for model in models:
    if model.tag == 'PostProcessor' and model.attrib['subType'] == 'LimitSurfaceIntegral':
      limitSurfaceIntegralNames.append(model.attrib['name'])
      #note that this converts exactly, it asks for everything with respect to everything
      if model.find('outputName') is None:
        # add the node with the original default name
        outputName = ET.Element('outputName')
        outputName.text = 'EventProbability'
        model.append(outputName)
      outputNamesVariables.append(model.find('outputName').text)
    if model.tag == 'PostProcessor' and model.attrib['subType'] == 'LimitSurface':
      limitSurfaceNames.append(model.attrib['name'])
  # check the steps if there are files and replace them with a dataObject outstream
  for pp in postProcess:
    if pp.find("Model").text in limitSurfaceIntegralNames + limitSurfaceNames:
      Outputs = pp.findall('Output')
      outstreamFound = False
      dataObjectName = ''
      for out in Outputs:
        if out.attrib['class'] == 'OutStreams':
          outstreamFound = True
        if out.attrib['class'] == 'DataObjects':
          dataObjectName = out.text
      # if LimitSurfaceIntegral, check the DataObject output to be sure it has the outputName
      for pointSetNode in PointSets:
        if pointSetNode.attrib['name'] == dataObjectName:
          outputPS = pointSetNode.find('Output')
          foundVar = False
          for ov in outputPS.text.split(","):
            if ov in outputNamesVariables:
              foundVar = True
              break
            if not foundVar:
              outputPS.text+=","+"EventProbability"
      for out in Outputs:
        if out.text in filesInputOutstreams.keys():
          if not outstreamFound:
            # replace it with an outstrem
            out.attrib['class'] = 'OutStreams'
            out.attrib['type'] = 'Print'
            # add the OutStream in the right node
            sourceET = ET.Element('source')
            sourceET.text = dataObjectName
            filesInputOutstreams[out.text].append(sourceET)
            outStreams.append(filesInputOutstreams[out.text])
            out.text = filesInputOutstreams[out.text].attrib['name']
          else:
            #remove it
            pp.remove(out)
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
