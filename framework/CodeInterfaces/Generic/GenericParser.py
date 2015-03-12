'''
Created on Mar 10, 2015

@author: talbpaul
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

#import xml.etree.ElementTree as ET
import os
import copy
#from utils import toBytes, toStrish, compare

def toBytes(s):
  if type(s) == type(""):
    return s.encode()
  elif type(s).__name__ in ['unicode','str','bytes']: return bytes(s)
  else:
    return s

class GenericParser:
  '''import the user-edited input file, build list of strings with replacable parts'''

  def __init__(self,inputFile,prefix='$RAVEN-',postfix='$'):
    #TODO use user-defined string catch as an option?
    if not os.path.exists(inputFile): raise IOError('Input file not found!')
    IOfile = open(inputFile,'rb')
    self.inputfile = inputFile
    lines = IOfile.readlines()
    self.varPlaces={}
    self.segments = []
    self.prefixKey=prefix
    self.postfixKey=postfix
    seg = ''
    for line in lines:
      if self.prefixKey in line and self.postfixKey in line:
        self.segments.append(toBytes(seg))
        start = line.find(self.prefixKey)
        end = line.find(self.postfixKey,start+1)
        var = line[start+len(self.prefixKey):end]
        self.segments.append(toBytes(line[:start]))
        self.segments.append(toBytes(var))
        self.varPlaces[var] = len(self.segments)
        self.segments.append(toBytes(line[end+1:]))
        seg = ''
      else:
        seg+=line
    self.segments.append(toBytes(seg))

  def printInput(self,outfile=None):
    # 4 sub levels maximum
    def printSubLevels(xmlnode,IOfile,indentMultiplier):
      IOfile.write(b'  '*indentMultiplier+b'[./'+toBytes(xmlnode.tag)+b']\n')
      for string in xmlnode.tail if xmlnode.tail else []:
        IOfile.write(b'    '*indentMultiplier+string+b'\n')
      for key in xmlnode.attrib.keys():
        IOfile.write(b'    '*indentMultiplier+toBytes(key)+b' = '+toBytes(toStrish(xmlnode.attrib[key]))+b'\n')

    if outfile==None: outfile = self.inputfile
    IOfile = open(outfile,'wb')
    IOfile.write(b''.join(self.segments))
    return

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
    if save: returnElement = copy.deepcopy(self.segments)         #make a copy if save is requested
    else: returnElement = self.segments
    #print(modiDictionaryList)
    for i in xrange(len(modiDictionaryList)):
      name = modiDictionaryList[i]['name']
      del modiDictionaryList[i]['name']
      self.__modifyOrAdd(returnElement,name,modiDictionaryList[i])
    if save: return returnElement
