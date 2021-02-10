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
"""
Other common manipulations for RAVEN entities.
Unlike utils, expects that BaseClass is loaded before use.
"""
from .graphStructure import graphObject

def readVariableGroups(xmlNode, messageHandler, caller):
  """
    Reads the XML for the variable groups and initializes them
    Placed in mathUtils because it uses VariableGroups, which inherit from BaseClasses
    -> and hence all the rest of the required libraries.
    NOTE: maybe we should have a thirdPartyUtils that is different from utils and mathUtils?
    @ In, xmlNode, ElementTree.Element, xml node to read in
    @ In, messageHandler, MessageHandler.MessageHandler instance, message handler to assign to the variable group objects
    @ In, caller, MessageHandler.MessageUser instance, entity calling this method (needs to inherit from MessageHandler.MessageUser)
    @ Out, varGroups, dict, dictionary of variable groups (names to the variable lists to replace the names)
  """
  import VariableGroups
  # first find all the names
  names = [node.attrib['name'] for node in xmlNode]

  # find dependencies
  deps = {}
  nodes = {}
  initials = []
  for child in xmlNode:
    name = child.attrib['name']
    nodes[name] = child
    if child.text is None:
      needs = []
    else:
      needs = [s.strip().strip('-+^%') for s in child.text.split(',')]
    for n in needs:
      if n not in deps and n not in names:
        deps[n] = []
    deps[name] = needs
    if len(deps[name]) == 0:
      initials.append(name)
  graph = graphObject(deps)
  # sanity checking
  if graph.isALoop():
    caller.raiseAnError(IOError, 'VariableGroups have circular dependency!')
  # ordered list (least dependencies first)
  hierarchy = list(reversed(graph.createSingleListOfVertices(graph.findAllUniquePaths(initials))))

  # build entities
  varGroups = {}
  for name in hierarchy:
    if len(deps[name]):
      varGroup = VariableGroups.VariableGroup()
      varGroup.readXML(nodes[name], messageHandler, varGroups)
      varGroups[name] = varGroup

  return varGroups
