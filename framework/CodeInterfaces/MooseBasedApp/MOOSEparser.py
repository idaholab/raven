'''
Created on Mar 25, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
from macpath import split
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import os
import copy
from utils import toBytes, toStrish, compare

class MOOSEparser():
  '''import the MOOSE input as xml tree, provide methods to add/change entries and print it back'''
  def __init__(self,inputFile):
    self.printTag = 'MOOSE_PARSER'
    if not os.path.exists(inputFile): raise IOError('not found MOOSE input file')
    IOfile = open(inputFile,'rb')
    self.inputfile = inputFile
    lines = IOfile.readlines()
    self.root = ET.Element('root')
    current = self.root
    current.tail = []
    parents = []
    parents.append(self.root)
    for line in lines:
      line = line.lstrip().strip(b'\n')
      if line.startswith(b'['):
        line = line.strip()
        if line.startswith(b'[]') or line.startswith(b'[../]'):
          current = parents.pop(len(parents)-1)
        else:
          #name = line.strip(b'[').strip(b']').strip(b'../')
          name = line[line.index(b'[')+1:line.index(b']')].strip(b'../').strip(b'./')
          parents.append(current)
          current      = ET.SubElement(current,name)
          current.tail = []
          if b'#' in line[line.index(b']'):]: current.tail.append(line[line.index(b']')+1:].strip(b'\n').lstrip())
      elif len(line)!=0:
        if not line.startswith(b'#'):
          listline = line.split(b'=')
          if b'#' not in listline[0]: current.attrib[listline[0].strip()]=listline[1].strip()
          else: current.attrib[listline[0].strip()]=listline[1][:listline[1].index('#')].strip()
        else:
          current.tail.append(line)

  def printInput(self,outfile=None):
    # 4 sub levels maximum
    def printSubLevels(xmlnode,IOfile,indentMultiplier):
      IOfile.write(b'  '*indentMultiplier+b'[./'+toBytes(xmlnode.tag)+b']\n')
      for string in xmlnode.tail if xmlnode.tail else []:
        IOfile.write(b'    '*indentMultiplier+string+b'\n')
      for key in xmlnode.attrib.keys():
        IOfile.write(b'    '*indentMultiplier+toBytes(key)+b' = '+toBytes(toStrish(xmlnode.attrib[key]))+b'\n')

    if outfile==None: outfile =self.inputfile
    IOfile = open(outfile,'wb')
    for child in self.root:
      IOfile.write(b'['+toBytes(child.tag)+b']\n')
      if child.tail:
        for string in child.tail:IOfile.write(b'  '+string+b'\n')
      for key in child.attrib.keys():
        IOfile.write(b'  '+toBytes(key)+b' = '+toBytes(toStrish(child.attrib[key]))+b'\n')
      for childChild in child:
        printSubLevels(childChild,IOfile,1)
        for childChildChild in childChild:
          printSubLevels(childChildChild,IOfile,2)
          for childChildChildChild in childChildChild:
            printSubLevels(childChildChildChild,IOfile,3)
            IOfile.write(b'      [../]\n')
          IOfile.write(b'    [../]\n')
        IOfile.write(b'  [../]\n')
      IOfile.write(b'[]\n')

  def findNodeInXML(self,name):
    """find node in xml and return it, if not found... None is returned"""
    returnNode = None
    self.root.find(name)
    for child in self.root:
      if name.strip() == child.tag: returnNode = child
    return returnNode


  def __findInXML(self,element,name):
    """Checks if there is a tag with name or binary name in
    element, and returns the (found,actual_name)"""
    if element.find(name) is not None:
      return (True,name)
    else:
      binary_name = toBytes(name)
      if element.find(binary_name) is not None:
        return (True,binary_name)
      else:
        return (False,None)

  def __updateDict(self,dictionary,other):
    """Add all the keys and values in other into dictionary"""
    for key in other:
      if key in dictionary:
        dictionary[key] = other[key]
      else:
        bin_key = toBytes(key)
        if bin_key in dictionary:
          dictionary[bin_key] = other[key]
        else:
          dictionary[key] = other[key]

  def __matchDict(self,dictionary,other):
    """ Returns true if all the keys and values in other
    match all the keys and values in dictionary.
    Note that it does not check that all the keys in dictionary
    match all the keys and values in other.
    """
    for key in other:
      if key in dictionary:
        #if dictionary[key] != other[key]:
        if not compare(dictionary[key],other[key]):
          print("Missmatch ",key,repr(dictionary[key]),repr(other[key]))
          return False
      else:
        bin_key = toBytes(key)
        if bin_key in dictionary:
          if not compare(dictionary[bin_key],other[key]):
          #if dictionary[bin_key] != other[key]:
            print("Missmatch_b ",key,dictionary[bin_key],other[key])
            return False
        else:
          print("No_key ",key,other[key])
          return False
    return True

  def __modifyOrAdd(self,returnElement,name,modiDictionary):
    """ If erase_block in modiDictionary, then remove name from returnElement
    else modify name in returnElement
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
    '''ModiDictionaryList is a list of dictionaries of the required addition or modification
    -name- key should return a ordered list of the name e.g. ['Components','Pipe']
    the other keywords possible are used as attribute names'''
    if save: returnElement = copy.deepcopy(self.root)         #make a copy if save is requested
    else: returnElement = self.root                           #otherwise return the original modified
    for i in xrange(len(modiDictionaryList)):
      name = modiDictionaryList[i]['name']
      del modiDictionaryList[i]['name']
      self.__modifyOrAdd(returnElement,name,modiDictionaryList[i])
    if save: return returnElement

  def vectorPostProcessor(self):
    """this method finds and process the vector post processor
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
        ''' Below are not necessarily at his stage but may be for future use 
        if 'radius_inner' in child.keys():
          vectorPPDict['rings'] = child.attrib['radius_inner'].strip("'").strip().split(' ')
        if 'integrals' in child.keys():
          vectorPPDict['integrals'] = child.attrib['integrals'].strip("'").strip().split(' ')
        '''  
      if 'Executioner' in child.tag:
        if 'num_steps' in child.keys():
          vectorPPDict['timeStep'] = child.attrib['num_steps'].strip("'").strip().split(' ')
    return found, vectorPPDict

