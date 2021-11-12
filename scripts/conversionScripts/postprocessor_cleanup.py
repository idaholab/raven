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
    Converts input files to be compatible with merge request #971
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  models = simulation.find('Models')
  distributions = simulation.find('Distributions')
  distDict = {}
  if distributions is not None:
    for distribution in distributions:
      distDict[distribution.attrib['name']] = distribution.tag
  else:
    return tree
  if models is not None:
    postProcessors = models.findall('PostProcessor')
    for pp in postProcessors:
      for dist in pp.iter('distribution'):
        if dist.get('class') is None:
          dist.set('class', 'Distributions')
          dist.set('type',distDict[dist.text.strip()])
      dists = pp.findall('mvnDistribution')
      for dist in dists:
        if dist.get('class') is None:
          dist.set('class', 'Distributions')
          dist.set('type',distDict[dist.text.strip()])
  else:
    return tree

if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
