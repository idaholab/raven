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
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import os
import copy
import numpy as np
from utils.utils import toBytes, toStrish, compare, toString

# numpy with version 1.14.0 and upper will change the floating point type and print
# https://docs.scipy.org/doc/numpy-1.14.0/release.html
if int(np.__version__.split('.')[1]) > 13:
  np.set_printoptions(**{'legacy':'1.13'})

class MOOSEparser():
  """
    Import the MOOSE input as xml tree, provide methods to add/change entries and print it back
  """
  def __init__(self, inputFile):
    """
      Constructor
      @ In, inputFile, string, input file name
      @ Out, None
    """
    self.printTag = 'MOOSE_PARSER'
    if not os.path.exists(inputFile):
      raise IOError('not found MOOSE input file')
    IOfile = open(inputFile,'r')
    self.inputfile = inputFile
    lines = IOfile.readlines()
    self.root = ET.Element('root')
    current = self.root
    current.tail = []
    parents = []
    parents.append(self.root)
    for line in lines:
      line = line.lstrip().strip('\n')
      if line.startswith('['):
        line = line.strip()
        if line.startswith('[]') or line.startswith('[../]'):
          current = parents.pop(len(parents)-1)
        else:
          #name = line.strip('[').strip(']').strip('../')
          name = line[line.index('[')+1:line.index(']')].strip('../').strip('./')
          parents.append(current)
          current      = ET.SubElement(current,name)
          current.tail = []
          if '#' in line[line.index(']'):]:
            current.tail.append(line[line.index(']')+1:].strip('\n').lstrip())
      elif len(line)!=0:
        if not line.startswith('#'):
          ind = line.find('=')
          if ind != -1:
            listLine = line.split('=', 1)
            attribName = listLine[0].strip()
            if '#' not in listLine[1]:
              attribValue = listLine[1].strip()
            else:
              attribValue = listLine[1][:listLine[1].index('#')]
            current.attrib[attribName] = attribValue
          else:
            if '#' not in line:
              attribValue = attribValue + '\n' + line
            else:
              attribValue = attribValue + '\n' + line[:line.index('#')]
            current.attrib[attribName] = attribValue
        else:
          current.tail.append(line)

  def printInput(self,outfile=None):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    def write(IOfile, indent, xmlnode):
      """
        Method to print out the key and value pairs
       @ In, IOfile, file, the file to write to
       @ In, indent, string, the string to print before the key
       @ In, xmlnode, ElementNode, the node with the attributes
       @ Out, None
      """
      for key in sorted(xmlnode.attrib.keys()):
        value = xmlnode.attrib[key]
        if type(value) == float:
          valueStr = repr(value)
        else:
          valueStr = toStrish(value)
        IOfile.write(indent+toString(key)+' = '+valueStr+'\n')
    # 4 sub levels maximum
    def printSubLevels(xmlnode,IOfile,indentMultiplier):
      IOfile.write('  '*indentMultiplier+'[./'+xmlnode.tag+']\n')
      for string in xmlnode.tail if xmlnode.tail else []:
        IOfile.write('    '*indentMultiplier+string+'\n')
      write(IOfile, '    '*indentMultiplier, xmlnode)
    if outfile==None:
      outfile =self.inputfile
    IOfile = open(outfile,'w')
    for child in self.root:
      IOfile.write('['+child.tag+']\n')
      if child.tail:
        for string in child.tail:
          IOfile.write('  '+string+'\n')
      write(IOfile, '  ', child)
      for childChild in child:
        printSubLevels(childChild,IOfile,1)
        for childChildChild in childChild:
          printSubLevels(childChildChild,IOfile,2)
          for childChildChildChild in childChildChild:
            printSubLevels(childChildChildChild,IOfile,3)
            IOfile.write('      [../]\n')
          IOfile.write('    [../]\n')
        IOfile.write('  [../]\n')
      IOfile.write('[]\n')

  def findNodeInXML(self,name):
    """
      Find node in xml and return it, if not found... None is returned
      @ In, name, string, name of the node that needs to be found in the XML tree
      @ Out, returnNode, xml.etree.ElementTree.Element, found node (if not found, return None)
    """
    returnNode = None
    self.root.find(name)
    for child in self.root:
      if name.strip() == child.tag:
        returnNode = child
    return returnNode


  def __findInXML(self,element,name):
    """
      Checks if there is a tag with name or binary name in
      element, and returns the (found,actual_name)
      @ In, element, xml.etree.ElementTree.Element, element where the 'name' needs to be found
      @ In, name, string, name of the node to be found
      @ Out, returnTuple, tuple, response. returnTuple[0]  bool (True if found) and returnTuple[1] string (binary name in the element)
    """
    returnTuple = None
    if element.find(name) is not None:
      returnTuple = (True,name)
    else:
      binaryName = toBytes(name)
      if element.find(binaryName) is not None:
        returnTuple = (True,binaryName)
      else:
        returnTuple = (False,None)
    return returnTuple

  def __updateDict(self,dictionary,other):
    """
      Add all the keys and values in other into dictionary
      @ In, dictionary, dict, dictionary that needs to be updated
      @ In, other, dict, dictionary from which the valued need to be taken
      @ Out, None
    """
    for key in other:
      if key in dictionary:
        dictionary[key] = other[key]
      else:
        binKey = toBytes(key)
        if binKey in dictionary:
          dictionary[binKey] = other[key]
        else:
          dictionary[key] = other[key]

  def __matchDict(self,dictionary,other):
    """
      Method to check the consistency of two dictionaries
      Returns true if all the keys and values in other
      match all the keys and values in dictionary.
      Note that it does not check that all the keys in dictionary
      match all the keys and values in other.
      @ In, dictionary, dict, first dictionary to check
      @ In, other, dict, second dictionary to check
      @ Out, returnBool, bool, True if all the keys and values in other match all the keys and values in dictionary.
    """
    returnBool = True
    for key in other:
      if key in dictionary:
        #if dictionary[key] != other[key]:
        if not compare(dictionary[key],other[key]):
          print("Missmatch ",key,repr(dictionary[key]),repr(other[key]))
          returnBool = False
      else:
        binKey = toBytes(key)
        if binKey in dictionary:
          if not compare(dictionary[binKey],other[key]):
            print("Missmatch_b ",key,dictionary[binKey],other[key])
            returnBool = False
        else:
          print("No_key ",key,other[key])
          returnBool = False
    return returnBool

  def __modifyOrAdd(self,returnElement,name,modiDictionary):
    """
      Modify the XML tree with the information in name and modiDictionary
      If erase_block in modiDictionary, then remove name from returnElement
      else modify name in returnElement
      @ In, returnElement, xml.etree.ElementTree.Element, the tree that needs to be updated
      @ In, name, list, list of instruction to reach the node to be modified
      @ In, modiDictionary, dict, dictionary contained the info to modify the tree
      @ Out, None
    """
    assert(len(name) > 0)
    specials  = modiDictionary['special'] if 'special' in modiDictionary.keys() else set()
    #If erase_block is true, then erase the entire block
    has_erase_block = 'erase_block' in specials
    #If assert_match is true, then fail if any of the elements do not exist
    has_assert_match = 'assert_match' in specials
    #If name[0] is not found and in erase_block, then done
    found,true_name = self.__findInXML(returnElement,name[0])
    if not found and has_erase_block:
      #Not found, and just wanted to erase it, so quit.
      return
    if not found and has_assert_match:
      #Not found, and just checking to see if there was a match
      return
    #If len(name) == 1, then don't recurse anymore.  Either
    # erase block or modify the element.
    if len(name) == 1:
      modiDictionary.pop('special',None)
      if has_erase_block:
        returnElement.remove(returnElement.find(true_name))
      elif has_assert_match:
        self.__matchDict(returnElement.find(true_name).attrib,modiDictionary)
        assert(self.__matchDict(returnElement.find(true_name).attrib,modiDictionary))
      elif found:
        self.__updateDict(returnElement.find(true_name).attrib,modiDictionary)
      else:
        ET.SubElement(returnElement,name[0],modiDictionary)
    else:
      if not found:
        subElement = ET.SubElement(returnElement,name[0])
        #if len(name) > 1, then if not found (and since we already checked for erasing) then add it and recurse.
      else:
        # if len(name) > 1 and found, then recurse on child
        subElement = returnElement.find(true_name)
      self.__modifyOrAdd(subElement,name[1:],modiDictionary)

  def modifyOrAdd(self,modiDictionaryList,save=True):
    """
      modiDictionaryList is a list of dictionaries of the required addition or modification
      -name- key should return a ordered list of the name e.g. ['Components','Pipe']
      the other keywords possible are used as attribute names
      @ In, modiDictionaryList, list, list of dictionaries containing the info to modify the XML tree
      @ In, save, bool, optional, True if the original tree needs to be saved
      @ Out, returnElement, xml.etree.ElementTree.Element, the tree that got modified
    """
    if save:
      returnElement = copy.deepcopy(self.root)         #make a copy if save is requested
    else:
      returnElement = self.root                           #otherwise return the original modified
    for i in range(len(modiDictionaryList)):
      name = modiDictionaryList[i]['name']
      del modiDictionaryList[i]['name']
      self.__modifyOrAdd(returnElement,name,modiDictionaryList[i])
    if save:
      return returnElement

  def vectorPostProcessor(self):
    """
      This method finds and process the vector post processor
      @ In, None
      @ Out, (found, vectorPPDict), tuple,
      found: boolean for the presence of the vector PP
      vectorPPDict: Dictionary for the properties related to the vector PP
    """
    vectorPPDict = {}
    found = False
    for child in self.root:
      if 'DomainIntegral' in child.tag:
        found = True
        #Below are not necessarily at his stage but may be for future use
        #if 'radius_inner' in child.keys():
        #  vectorPPDict['rings'] = child.attrib['radius_inner'].strip("'").strip().split(' ')
        #if 'integrals' in child.keys():
        #  vectorPPDict['integrals'] = child.attrib['integrals'].strip("'").strip().split(' ')
      if 'Executioner' in child.tag:
        if 'num_steps' in child.keys():
          vectorPPDict['timeStep'] = child.attrib['num_steps'].strip("'").strip().split(' ') #TODO: define default num_steps in case it is not in moose input
    return found, vectorPPDict
