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
    Converts input files of importance rank post-processors
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')

  hasVariableGroups = True
  hasDataObjects = True
  hasOutStreams = True

  variableGroups = simulation.find('VariableGroups')
  if variableGroups is None:
    variableGroups = ET.Element('VariableGroups')
    hasVariableGroups = False
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


  if models is None: return tree # no models
  timeDep = {}
  ppDict = {}
  paramDict = {}
  for model in models:
    if model.tag == 'PostProcessor' and model.attrib['subType'] == 'ImportanceRank':
      #note that this converts exactly, it asks for everything with respect to everything
      name = model.attrib['name']
      ppDict[name] = {}
      paramDict[name] = []
      timeDep[model.attrib['name']] = model.find('pivotParameter')
      what = [metric.lower().strip() for metric in model.find('what').text.split(',')]
      if 'all' in what:
        what = list(set(what + ['sensitivityindex', 'importanceindex', 'pcaindex']))
      ppDict[name]['what'] = what
      targs = [targ.strip() for targ in model.find('targets').text.split(',')]
      ppDict[name]['targets'] = targs
      featNode = model.find('features')
      manifestNode = featNode.find('manifest')
      latentNode = featNode.find('latent')
      mVar = [var.strip() for var in manifestNode.find('variables').text.split(',')]
      if latentNode is not None:
        lVar = [var.strip() for var in latentNode.find('variables').text.split(',')]
        for metric in what:
          if metric in ['sensitivityindex']:
            metric = 'sensitivityIndex'
            for targ in targs:
              for feat in lVar:
                varName = metric + '_' + targ + '_' + feat
                paramDict[name].append(varName)
          elif metric in ['importanceindex']:
            metric = 'importanceIndex'
            for targ in targs:
              for feat in lVar:
                varName = metric + '_' + targ + '_' + feat
                paramDict[name].append(varName)
          elif metric in ['pcaindex']:
            metric = 'pcaIndex'
            for feat in lVar:
              varName = metric + '_' + feat
              paramDict[name].append(varName)
          elif metric in ['manifestsensitivity']:
            metric = 'manifestSensitivity'
            for targ in targs:
              for feat in mVar:
                varName = metric + '_' + targ + '_' + feat
                paramDict[name].append(varName)
          elif metric in ['transformation']:
            metric = 'transformation'
            for mFeat in mVar:
              for lFeat in lVar:
                varName = metric + '_' + mFeat + '_' + lFeat
                paramDict[name].append(varName)
          elif metric in ['inversetransformation']:
            metric = 'inverseTransformation'
            for lFeat in lVar:
              for mFeat in mVar:
                varName = metric + '_' + lFeat + '_' + mFeat
                paramDict[name].append(varName)
      else:
        for metric in what:
          if metric in ['sensitivityindex']:
            metric = 'sensitivityIndex'
            for targ in targs:
              for feat in mVar:
                varName = metric + '_' + targ + '_' + feat
                paramDict[name].append(varName)
          elif metric in ['importanceindex']:
            metric = 'importanceIndex'
            for targ in targs:
              for feat in mVar:
                varName = metric + '_' + targ + '_' + feat
                paramDict[name].append(varName)

        # add variable groups
      group = ET.Element('Group')
      group.attrib['name'] = name + '_vars'
      group.text = ',\n                 '.join(paramDict[name])
      variableGroups.append(group)

  if variableGroups.find('Group') is not None:
    if not hasVariableGroups:
      simulation.append(variableGroups)

  for modelName, pivotParam in timeDep.items():
    dataSetName = modelName + '_Dataset'
    if pivotParam is None:
      dataSet = ET.Element('PointSet')
    else:
      dataSet = ET.Element('HistorySet')
      option = ET.SubElement(dataSet, 'options')
      pivotNode = ET.SubElement(option,'pivotParameter')
      pivotNode.text = pivotParam.text

    dataSet.attrib['name'] = dataSetName
    outNode = ET.SubElement(dataSet,'Output')
    outNode.text = modelName + '_vars'
    dataObjects.append(dataSet)
    if not hasDataObjects:
      simulation.append(dataObjects)

    printNode = ET.Element('Print')
    printNode.attrib['name'] = dataSetName + '_dump'
    typeNode = ET.SubElement(printNode,'type')
    typeNode.text = 'csv'
    sourceNode = ET.SubElement(printNode,'source')
    sourceNode.text = dataSetName
    outStreams.append(printNode)
    if not hasOutStreams:
      simulation.append(outStreams)

    for pp in postProcess:
      if modelName == pp.find('Model').text.strip():
        outputs = pp.findall('Output')
        remove = False
        hasPrint = False
        for output in outputs:
          if output.attrib['class'] == 'Files':
            output.attrib['class'] = 'DataObjects'
            output.attrib['type'] = 'PointSet' if pivotParam is None else 'HistorySet'
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

  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
