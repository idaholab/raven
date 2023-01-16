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
    Converts input files to be compatible with merge request ???.
    Changes list of <variable> nodes to <variables> nodes.
    @ In, tree, xml.etree.ElementTree.ElementTree object, the contents of a RAVEN input file
    @ In, fileName, the name for the raven input file
    @Out, tree, xml.etree.ElementTree.ElementTree object, the modified RAVEN input file
  """
  simulation = tree.getroot()
  if simulation.tag!='Simulation' and simulation.tag!='ExternalModel': return tree #this isn't an input file
  extmod = None
  if simulation.tag=='Simulation':
    models = simulation.find('Models')
    if models is not None:
      extmod = models.find('ExternalModel')
  elif simulation.tag=='ExternalModel': #externalNode case
    extmod = simulation
  if extmod is not None:
    vars = []
    toRemove = []
    for child in extmod:
      if child.tag=='variable':
        vars.append(child.text)
        toRemove.append(child)
    for child in toRemove:
      extmod.remove(child)
    if len(vars)>0:
      variables = ET.Element('variables')
      extmod.append(variables)
      variables.text = ','.join(vars)
  return tree


if __name__=='__main__':
  import convert_utils
  import sys
  convert_utils.standardMain(sys.argv,convert)
