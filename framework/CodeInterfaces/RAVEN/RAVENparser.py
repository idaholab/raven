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
Created on Sept 10, 2017

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import copy
#from utils.utils import toBytes, toStrish, compare

class RAVENparser():
  """
    Import the RAVEN input as xml tree, provide methods to add/change entries and print it back
  """
  def __init__(self, inputFile):
    """
      Constructor
      @ In, inputFile, string, input file name
      @ Out, None
    """
    self.printTag  = 'RAVEN_PARSER'
    self.inputFile = inputFile
    if not os.path.exists(inputFile):
      raise IOError('not found RAVEN input file')
    try:
      tree = ET.parse(file(inputFile,'r'))
    except ET.InputParsingError as e:
      raise IOError(self.printTag+' ERROR: Input Parsing error!\n' +str(e)+'\n')
    self.tree = tree.getroot()
    # do some sanity checks
    sequence = [step.strip() for step in self.tree.find('.//RunInfo/Sequence').text.split(",")]
    # firstly no multiple sublevels of RAVEN can be handled now
    for code in self.tree.findall('.//Models/Code'):
      if 'subType' not in code.attrib:
        raise IOError(self.printTag+' ERROR: Not found subType attribute in <Code> XML blocks!')
      if code.attrib['subType'].strip() == 'RAVEN':
        raise IOError(self.printTag+' ERROR: Only one level of RAVEN runs are allowed (Not a chain of RAVEN runs). Found a <Code> of subType RAVEN!')
    # find steps and check if there are active outstreams (Print)
    foundOutStreams = False
    lupo = self.tree.find('.//Steps')
    for step in self.tree.find('.//Steps'):
      if step.attrib['name'] in sequence:
        for role in step:
          if role.tag.strip() == 'Output':
            mainClass, subType = role.attrib['class'].strip(), role.attrib['type'].strip()
            if mainClass == 'OutStreams' and subType == 'Print':
              foundOutStreams = True
              break
    if not foundOutStreams:
      raise IOError(self.printTag+' ERROR: at least one <OutStreams> of type "Print" needs to be inputted in the active Steps!!')

  def printInput(self,rootToPrint,outfile=None):
    """
      Method to print out the new input
      @ In, rootToPrint, xml.etree.ElementTree.Element, the Element containing the input that needs to be printed out
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    xmlObj = xml.dom.minidom.parseString(ET.tostring(rootToPrint))
    inputAsString = xmlObj.toprettyxml()
    inputAsString = "".join([s for s in inputAsString.strip().splitlines(True) if s.strip()])
    if outfile==None:
      outfile =self.inputfile
    IOfile = open(outfile,'wb')
    IOfile.write(inputAsString)
    IOfile.close()

  def modifyOrAdd(self,modiDictionary={},save=True):
    """
      modiDictionary a dict of dictionaries of the required addition or modification
      {"variableToChange":value }
      @ In, modiDictionary, dict, dictionary of variables to modify
            syntax:
            {'Node|SubNode|SubSubNode:value1','Node|SubNode|SubSubNode@attribute:attributeValue|SubSubSubNode':value2
                      'Node|SubNode|SubSubNode@attribute':value3}
             TODO: handle added XML nodes
      @ In, save, bool, optional, True if the original tree needs to be saved
      @ Out, returnElement, xml.etree.ElementTree.Element, the tree that got modified
    """
    if save:
      returnElement = copy.deepcopy(self.tree)            #make a copy if save is requested
    else:
      returnElement = self.tree                           #otherwise return the original modified
    for node, value in modiDictionary.items():
      if "|" not in node:
        raise IOError(self.printTag+' ERROR: the variable '+node.strip()+' does not contain "|" separator and can not be handled!!')
      changeTheNode = True
      if "@" in node:
        # there are attributes that are needed to locate the node
        splittedComponents = node.split("|")
        # check the first
        pathNode = './'
        attribName = ''
        for cnt, subNode in enumerate(splittedComponents):
          splittedComp = subNode.split("@")
          component = splittedComp[0]
          attribPath = ""
          if "@" in subNode:
            # more than an attribute locator
            for attribComp in splittedComp[1:]:
              if ":" in attribComp.strip():
                # it is a locator
                attribPath +='[@'+attribComp.split(":")[0].strip()+('="'+attribComp.split(":")[1].strip()+'"]')
              else:
                # it is actually the attribute that needs to be changed
                # check if it is the last component
                if cnt+1 != len(splittedComponents):
                  raise IOError(self.printTag+' ERROR: the variable '+node.strip()+' follows the syntax "Node|SubNode|SubSubNode@attribute"'+
                                              ' but the attribute is not the last component. Please check your input!')
                attribPath +='[@'+attribComp.strip()+']'
                attribName = attribComp
          pathNode += "/" + component.strip()+attribPath
        if pathNode.endswith("]"):
          changeTheNode = False
        else:
          changeTheNode = True     
      else:
        # there are no attributes that are needed to track down the node to change
        pathNode = './/' + node.replace("|","/").strip()
      # look for the node with XPath directives 
      foundNodes = returnElement.findall(pathNode)
      if len(foundNodes) > 1:
        raise IOError(self.printTag+' ERROR: multiple nodes have been found corresponding to path -> '+node.strip()+'. Please use the attribute identifier "@" to nail down to a specific node !!')
      if len(foundNodes) == 0:
        raise IOError(self.printTag+' ERROR: no node has been found corresponding to path -> '+node.strip()+'. Please check the input!!')
      nodeToChange = foundNodes[0]
      pathNode     = './/'      
      if changeTheNode:
        nodeToChange.text = str(value).strip()
      else:
        nodeToChange.attrib[attribName] = str(value).strip()

    self.printInput(returnElement,outfile="lupo.xml")
    if save:
      return returnElement

  #def findNodeInXML(self,name):
    #"""
      #Find node in xml and return it, if not found... None is returned
      #@ In, name, string, name of the node that needs to be found in the XML tree
      #@ Out, returnNode, xml.etree.ElementTree.Element, found node (if not found, return None)
    #"""
    #returnNode = None
    #self.root.find(name)
    #for child in self.root:
      #if name.strip() == child.tag:
        #returnNode = child
    #return returnNode


  #def __findInXML(self,element,name):
    #"""
      #Checks if there is a tag with name or binary name in
      #element, and returns the (found,actual_name)
      #@ In, element, xml.etree.ElementTree.Element, element where the 'name' needs to be found
      #@ In, name, string, name of the node to be found
      #@ Out, returnTuple, tuple, response. returnTuple[0]  bool (True if found) and returnTuple[1] string (binary name in the element)
    #"""
    #returnTuple = None
    #if element.find(name) is not None:
      #returnTuple = (True,name)
    #else:
      #binaryName = toBytes(name)
      #if element.find(binaryName) is not None:
        #returnTuple = (True,binaryName)
      #else:
        #returnTuple = (False,None)
    #return returnTuple

  #def __updateDict(self,dictionary,other):
    #"""
      #Add all the keys and values in other into dictionary
      #@ In, dictionary, dict, dictionary that needs to be updated
      #@ In, other, dict, dictionary from which the valued need to be taken
      #@ Out, None
    #"""
    #for key in other:
      #if key in dictionary:
        #dictionary[key] = other[key]
      #else:
        #binKey = toBytes(key)
        #if binKey in dictionary:
          #dictionary[binKey] = other[key]
        #else:
          #dictionary[key] = other[key]

  #def __matchDict(self,dictionary,other):
    #"""
      #Method to check the consistency of two dictionaries
      #Returns true if all the keys and values in other
      #match all the keys and values in dictionary.
      #Note that it does not check that all the keys in dictionary
      #match all the keys and values in other.
      #@ In, dictionary, dict, first dictionary to check
      #@ In, other, dict, second dictionary to check
      #@ Out, returnBool, bool, True if all the keys and values in other match all the keys and values in dictionary.
    #"""
    #returnBool = True
    #for key in other:
      #if key in dictionary:
        ##if dictionary[key] != other[key]:
        #if not compare(dictionary[key],other[key]):
          #print("Missmatch ",key,repr(dictionary[key]),repr(other[key]))
          #returnBool = False
      #else:
        #binKey = toBytes(key)
        #if binKey in dictionary:
          #if not compare(dictionary[binKey],other[key]):
            #print("Missmatch_b ",key,dictionary[binKey],other[key])
            #returnBool = False
        #else:
          #print("No_key ",key,other[key])
          #returnBool = False
    #return returnBool

  #def __modifyOrAdd(self,returnElement,name,modiDictionary):
    #"""
      #Modify the XML tree with the information in name and modiDictionary
      #If erase_block in modiDictionary, then remove name from returnElement
      #else modify name in returnElement
      #@ In, returnElement, xml.etree.ElementTree.Element, the tree that needs to be updated
      #@ In, name, list, list of instruction to reach the node to be modified
      #@ In, modiDictionary, dict, dictionary contained the info to modify the tree
      #@ Out, None
    #"""
    #assert(len(name) > 0)
    #specials  = modiDictionary['special'] if 'special' in modiDictionary.keys() else set()
    ##If erase_block is true, then erase the entire block
    #has_erase_block = 'erase_block' in specials
    ##If assert_match is true, then fail if any of the elements do not exist
    #has_assert_match = 'assert_match' in specials
    ##If name[0] is not found and in erase_block, then done
    #found,true_name = self.__findInXML(returnElement,name[0])
    #if not found and has_erase_block:
      ##Not found, and just wanted to erase it, so quit.
      #return
    #if not found and has_assert_match:
      ##Not found, and just checking to see if there was a match
      #return
    ##If len(name) == 1, then don't recurse anymore.  Either
    ## erase block or modify the element.
    #if len(name) == 1:
      #modiDictionary.pop('special',None)
      #if has_erase_block:
        #returnElement.remove(returnElement.find(true_name))
      #elif has_assert_match:
        #self.__matchDict(returnElement.find(true_name).attrib,modiDictionary)
        #assert(self.__matchDict(returnElement.find(true_name).attrib,modiDictionary))
      #elif found:
        #self.__updateDict(returnElement.find(true_name).attrib,modiDictionary)
      #else:
        #ET.SubElement(returnElement,name[0],modiDictionary)
    #else:
      #if not found:
        #subElement = ET.SubElement(returnElement,name[0])
        ##if len(name) > 1, then if not found (and since we already checked for erasing) then add it and recurse.
      #else:
        ## if len(name) > 1 and found, then recurse on child
        #subElement = returnElement.find(true_name)
      #self.__modifyOrAdd(subElement,name[1:],modiDictionary)

if __name__ == '__main__':
  parser = RAVENparser("/Users/alfoa/projects/raven_github/raven/tests/framework/test_Grid_Sampler.xml")
  modifyDict = {'Distributions|Normal@name:Gauss1|mean':1.2345,'RunInfo|batchSize':2,'Models|Dummy@name:MyDummy@subType':"lupo"}
  parser.modifyOrAdd(modifyDict)
