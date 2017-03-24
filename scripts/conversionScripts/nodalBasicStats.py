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
    Converts input files to be compatible with merge request ###, where BasicStatistics is given the power
    to be more nodalized than before.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  if models is None: return tree # no models, no BasicStats
  for model in models:
    if model.tag == 'PostProcessor' and model.attrib['subType'] == 'BasicStatistics':
      #note that this converts exactly, it asks for everything with respect to everything
      if model.find('what') is None:
        #fix one botched attempt
        if model.find('all') is not None:
          anode = model.find('all')
          if anode.find('targets') is None:
            params = anode.text
            anode.text = ''
            targetNode = ET.Element('targets')
            targetNode.text = params
            featureNode = ET.Element('features')
            featureNode.text = params
            anode.append(targetNode)
            anode.append(featureNode)
        #already converted
        return tree
      #get the metrics
      what = model.find('what').text.strip()
      model.remove(model.find('what'))
      #get the parameters
      params = model.find('parameters').text.strip()
      model.remove(model.find('parameters'))
      #targets and features
      targetNode = ET.Element('targets')
      targetNode.text = params
      featureNode = ET.Element('features')
      featureNode.text = params
      #parameters
      if 'all' in what:
        allNode = ET.Element('all')
        allNode.append(targetNode)
        allNode.append(featureNode)
        model.append(allNode)
      else:
        needsFeatures = ['sensitivity','covariance','pearson','NormalizedSensitivity','VarianceDependentSensitivity']
        for w in (i.strip() for i in what.split(',')):
          node = ET.Element(w)
          if w in needsFeatures:
            node.append(targetNode)
            node.append(featureNode)
          else:
            node.text = params
          model.append(node)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
