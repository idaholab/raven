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
    Converts input files to be compatible with merge request #460:
     - Removes "all" node
     - Sets default variable names
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
  TestInfo = simulation.find('TestInfo')
  if TestInfo is not None:
    revisions = TestInfo.find('revisions')
    hasRev = True
    if revisions is None:
      revisions = ET.Element('revisions')
      hasRev = False
    rev = ET.Element('revision')
    rev.attrib['author'] = 'wangc'
    rev.attrib['date'] = '2017-12-20'
    rev.text = 'convert test to use the new DataObjects with the new structure of basic statistic'
    revisions.append(rev)
    if not hasRev:
      TestInfo.append(revisions)

  toRemove = []

  if models is None: return tree # no models, no BasicStats
  timeDep = {}
  for model in models:
    if model.tag == 'PostProcessor' and model.attrib['subType'] == 'BasicStatistics':
      #note that this converts exactly, it asks for everything with respect to everything
      params = []

      timeDep[model.attrib['name']] = model.find('pivotParameter')
      if model.find('all') is not None:
        anode = model.find('all')
        targNode = anode.find('targets')
        featNode = anode.find('features')
        targs = targNode.text
        feats = featNode.text

        model.remove(model.find('all'))

        metricDict = {'expectedValue':'mean',
              'minimum':'min',
              'maximum':'max',
              'median':'median',
              'variance':'var',
              'sigma':'sigma',
              'percentile':'percentile',
              'variationCoefficient':'vc',
              'skewness':'skew',
              'kurtosis':'kurt',
              'samples':'samp'
              }
        for metric, prefix in metricDict.items():
          node = ET.Element(metric)
          node.text = targs
          node.attrib['prefix'] = prefix
          model.append(node)
          for targ in targs.split(','):
            if metric != 'percentile':
              params.append(prefix+'_'+targ.strip())
            else:
              params.append(prefix+'_5_'+targ.strip())
              params.append(prefix+'_95_'+targ.strip())

        metricDict = {'sensitivity': 'sen',
             'covariance':'cov',
             'pearson':'pear',
             'NormalizedSensitivity':'nsen',
             'VarianceDependentSensitivity':'vsen'
             }

        for metric, prefix in metricDict.items():
          node = ET.Element(metric)
          node.attrib['prefix'] = prefix
          node.append(targNode)
          node.append(featNode)
          model.append(node)
          for targ in targs.split(','):
            for feat in feats.split(','):
              params.append(prefix+'_'+targ.strip()+'_'+feat.strip())

      else:
        metricDict = {'expectedValue':'mean',
              'minimum':'min',
              'maximum':'max',
              'median':'median',
              'variance':'var',
              'sigma':'sigma',
              'percentile':'percentile',
              'variationCoefficient':'vc',
              'skewness':'skew',
              'kurtosis':'kurt',
              'samples':'samp',
              'sensitivity': 'sen',
              'covariance':'cov',
              'pearson':'pear',
              'NormalizedSensitivity':'nsen',
              'VarianceDependentSensitivity':'vsen'
             }

        metricDict1 = {'expectedValue':'mean',
              'minimum':'min',
              'maximum':'max',
              'median':'median',
              'variance':'var',
              'sigma':'sigma',
              'percentile':'percentile',
              'variationCoefficient':'vc',
              'skewness':'skew',
              'kurtosis':'kurt',
              'samples':'samp'
              }

        for child in model:
          if child.tag in metricDict.keys():
            child.attrib['prefix'] = metricDict[child.tag]
            if child.tag in metricDict1.keys():
              for var in child.text.split(','):
                if child.tag != 'percentile':
                  params.append(metricDict[child.tag] + '_' + var.strip())
                else:
                  if 'percent' in child.attrib.keys():
                    params.append(metricDict[child.tag]+'_'+child.attrib['percent']+'_'+var.strip())
                  else:
                    params.append(metricDict[child.tag]+'_5_'+var.strip())
                    params.append(metricDict[child.tag]+'_95_'+var.strip())

            else:
              targNode = child.find('targets')
              featNode = child.find('features')
              for targ in targNode.text.split(','):
                for feat in featNode.text.split(','):
                  params.append(metricDict[child.tag]+'_'+targ.strip()+'_'+feat.strip())

        # add variable groups
      group = ET.Element('Group')
      group.attrib['name'] = model.attrib['name'] + '_vars'
      group.text = ',\n                 '.join(params)
      variableGroups.append(group)

  if variableGroups.find('Group') is not None:
    if not hasVariableGroups:
      simulation.append(variableGroups)
    for modelName, pivotParam in timeDep.items():

      dataSetName = modelName + '_basicStatPP'
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
              toRemove.append(output.text)
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
  # move unused files
  if len(toRemove) > 0:
    files = simulation.find('Files')
    for inputFile in files:
      if inputFile.attrib['name'] in toRemove:
        files.remove(inputFile)

  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
