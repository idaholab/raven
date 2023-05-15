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
Created on Mar 25, 2013

@author: crisr
"""

import xml.etree.ElementTree as ET
import os
import copy
import numpy as np
from ravenframework.utils.utils import toBytes, toStrish, compare, toString
from . import MooseInputParser

class MOOSEparser():
  """
    Import the MOOSE input as xml tree, provide methods to add/change entries and print it back
  """
  #--------------------
  # CONSTRUCTION
  #--------------------
  def __init__(self, inputFile):
    """
      Constructor
      @ In, inputFile, string, input file name
      @ Out, None
    """
    self.printTag = 'MOOSE_PARSER'         # tag for identifying in message printing
    self.inputFile = inputFile             # location of original input file
    self.roots = self.loadInput(inputFile) # structure of native input file

  #--------------------
  # API
  #--------------------
  def loadInput(self, inputFile):
    """
      Reads the input to class members.
      @ In, inputFile, string, name of input file
      @ Out, roots, list, root nodes of input file as utils.TreeStructure.InputTree instances
    """
    if not os.path.exists(inputFile):
      raise IOError('MOOSE input file not found: "{}"'.format(inputFile))
    with open(inputFile, 'r') as f:
      roots = MooseInputParser.getpotToInputTree(f)
    return roots

  def modifyOrAdd(self, modiDictionaryList):
    """
      modiDictionaryList is a list of dictionaries of the required addition or modification
      -name- key should return a ordered list of the name e.g. ['Components','Pipe']
      the other keywords possible are used as attribute names
      @ In, modiDictionaryList, list, list of dictionaries containing the info to modify the XML tree
      @ Out, modified, list(TreeStructure.InputTrees), the tree(s) that got modified
    """
    modified = copy.deepcopy(self.roots)
    for mod in modiDictionaryList:
      name = mod.pop('name')
      modified = self._modifySingleEntry(modified, name, mod)
    return modified

  def printInput(self, outFile, trees):
    """
      Method to print out the new input
      @ In, outFile, string, output file name to write to
      @ In, trees, list(TreeStructure.InputTrees), the tree(s) to write
      @ Out, None
    """
    gp = MooseInputParser.writeGetpot(trees)
    with open(outFile, 'w') as f:
      f.writelines(gp)

  #--------------------
  # UTILITIES
  #--------------------
  def _modifySingleEntry(self, trees, target, modification):
    """
      Searches trees for appropriate target to change to modification.
      @ In, trees, list(TreeStructure.InputTrees), the tree(s) that get modified
      @ In, target, list(string), target modification location as e.g. [Block1, Block2, Entry]
      @ In, modification, dict, change to be made including special commands
      @ Out, None
    """
    specials = modification.pop('special', set())
    # get the target object to modify
    found = MooseInputParser.findInGetpot(trees, target)

    # if asking to erase a block ...
    # TODO is this untested code?
    if 'erase_block' in specials:
      if found:
        found[-2].remove(found[-1])

    # if asking to check on a block ...
    # TODO is this untested code?
    elif 'assert_match' in specials:
      if found is None:
        raise IOError('MOOSEParser: Target path not found in provided input file: "{}"'.format(target))

    # if asking to modify on a block ...
    else:
      shortName, mod = next(iter(modification.items()))
      # modification or addition
      if found:
        # modification
        if isinstance(mod, tuple):
          ## location index start from 1
          loc, modVal = mod[0], mod[1]
          val = found[-1].text.split()
          val[loc-1] = str(modVal)
          found[-1].text = ' '.join(val)
        else:
          found[-1].text = str(mod)
      else:
        # addition
        trees = MooseInputParser.addNewPath(trees, target, mod)
    return trees

  def vectorPostProcessor(self):
    """
      This method finds and process the vector post processor
      @ In, None
      @ Out, (found, vectorPPDict), tuple,
      found: boolean for the presence of the vector PP
      vectorPPDict: Dictionary for the properties related to the vector PP
    """
    vectorPPDict = {}
    foundIntegral = MooseInputParser.findInGetpot(self.roots, ['DomainIntegral'])
    foundNumSteps = MooseInputParser.findInGetpot(self.roots, ['Executioner', 'num_steps'])
    if foundNumSteps:
      vectorPPDict['timeStep'] = foundNumSteps[-1].text.strip("\' \n").split(' ') #TODO: define default num_steps in case it is not in moose input
    return foundIntegral is not None, vectorPPDict
