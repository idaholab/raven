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

def modifyInput(root,modDict):
  """
    Manipulate the input in a testable fashion.
    @ In, root, xml.etree.ElementTree.Element, root Simulation node of RAVEN input file
    @ In, modDict, dict, dictionary of modifications that were made to the input
    @ Out, root, xml.etree.ElementTree.Element, modified Simulation root node.
  """
  # create the input for the external model
  new = ET.Element('ExternalModel')
  new.attrib['name'] = 'PythonModule'
  new.attrib['ModuleToLoad'] = 'TMI_fake'
  new.attrib['subType'] = ''
  var = ET.Element('variables')
  var.text = 'GRO_targets,time,CladTempThreshold'
  new.append(var)
  # add new node to the outstreams
  models = root.find('Models')
  models.append(new)
  return root




