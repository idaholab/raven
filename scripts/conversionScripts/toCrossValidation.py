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
    Converts the metric input files to use the new data objects
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  hasDataObjects = True
  hasOutStreams = True

  dataObjects = simulation.find('DataObjects')
  if dataObjects is None:
    dataObjects = ET.Element('DataObjects')
    hasDataObjects = False
  outStreams = simulation.find('OutStreams')
  if outStreams is None:
    outStreams = ET.Element('OutStreams')
    hasOutStreams = False

  steps = simulation.find('Steps')
  postProcess = steps.findall('PostProcess')

  if models is None: return tree # no models, no BasicStats
  modelNames = []
  for model in models:
    if model.tag == 'PostProcessor' and model.attrib['subType'] == 'CrossValidation':
      metrics = model.findall('Metric')
      metricNames = []
      modelNames.append(model.attrib['name'])
      for metric in metrics:
        metricName = metric.text.strip()
        metricNames.append(metricName)

  for modelName in modelNames:

    dataSetName = modelName + '_cv'
    printNode = ET.Element('Print')
    printNode.attrib['name'] = dataSetName + '_dump'
    typeNode = ET.SubElement(printNode,'type')
    typeNode.text = 'csv'
    sourceNode = ET.SubElement(printNode,'source')
    sourceNode.text = dataSetName
    outStreams.append(printNode)
    for pp in postProcess:
      if modelName == pp.find('Model').text.strip():
        inputs = pp.findall('Input')
        for inputObj in inputs:
          if inputObj.attrib['class'] == 'Models':
            romName = inputObj.text.strip()
            varNames = []
            for metricName in metricNames:
              varName = 'cv' + '_' + metricName + '_' + romName
              varNames.append(varName)
            dataSet = ET.Element('PointSet')
            dataSet.attrib['name'] = dataSetName
            outNode = ET.SubElement(dataSet,'Output')
            outNode.text = ','.join(varNames)
            dataObjects.append(dataSet)
        outputs = pp.findall('Output')
        remove = False
        hasPrint = False
        for output in outputs:
          if output.attrib['class'] == 'Files':
            output.attrib['class'] = 'DataObjects'
            output.attrib['type'] = 'PointSet'
            output.text = dataSetName
            if remove:
              pp.remove(output)
            else:
              remove = True
          elif output.attrib['class'] == 'OutStreams' and output.attrib['type'] == 'Print':
            output.text = dataSetName + '_dump'
            hasPrint = True
          elif output.attrib['class'] == 'DataObjects':
            pp.remove(output)
        if not hasPrint:
          printNode = ET.SubElement(pp, 'Output')
          printNode.attrib['class'] = 'OutStreams'
          printNode.attrib['type'] = 'Print'
          printNode.text = dataSetName + '_dump'
    if not hasDataObjects:
      simulation.append(dataObjects)
    if not hasOutStreams:
      simulation.append(outStreams)

  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
