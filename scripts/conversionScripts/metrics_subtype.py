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

def convert(tree,fileName=None):
  """
    Converts input files to be compatible with merge request #1573:
    Change Metrics to use subType formula
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  # metrics
  metrics = simulation.find('Metrics')
  if metrics is not None:
    for child in metrics:
      if child.tag != 'Metric':
        subType = child.tag
        child.tag = 'Metric'
        child.attrib['subType'] = subType
  # postprocessors
  models = simulation.find('Models')
  if models is not None:
    for pp in models.findall('PostProcessor'):
      for node in pp.findall('Metric'):
        if node.attrib.get('class', None) == 'Metrics':
          node.attrib['type'] = 'Metric'
  return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
