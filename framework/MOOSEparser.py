'''
Created on Mar 25, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import os
import copy
from utils import *

class MOOSEparser:
  '''import the MOOSE input as xml tree, provide methods to add/change entries and print it back'''

  def __init__(self,inputFile):
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
        if line =='[]' or line =='[../]':
          current = parents.pop(len(parents)-1)
        else:
#          name = line.strip(b'[').strip(b']').strip(b'../')
          name = line[line.index(b'[')+1:line.index(b']')].strip(b'../').strip(b'./')
          print(name)
          parents.append(current)
          current      = ET.SubElement(current,name)
          current.tail = []
          if b'#' in line[line.index(b']'):]: current.tail.append(line[line.index(b']')+1:].strip(b'\n').lstrip())
      elif len(line)!=0:
        if not line.startswith(b'#'):
          listline = line.split(b'=')
          if b'#' not in listline[0]: current.attrib[listline[0].strip()]=listline[1]
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
        IOfile.write(b'    '*indentMultiplier+toBytes(key)+b' = '+toBytes(xmlnode.attrib[key])+b'\n')

    if outfile==None: outfile =self.inputfile
    IOfile = open(outfile,'wb')
    for child in self.root:
      IOfile.write(b'['+toBytes(child.tag)+b']\n')
      if child.tail:
        for string in child.tail:IOfile.write(b'  '+string+b'\n')
      for key in child.attrib.keys(): 
        IOfile.write(b'  '+key+b' = '+toBytes(child.attrib[key])+b'\n')
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

  def modifyOrAdd(self,modiDictionaryList,save=True):
    '''ModiDictionaryList is a list of dictionaries of the required addition or modification
    -name- key should return a ordered list of the name e.g. ['Components','Pipe']
    the other keywords possible are used as attribute names'''
    if save: returnElement = copy.deepcopy(self.root)         #make a copy if save is requested
    else: returnElement = self.root                           #otherwise return the original modified
    for i in xrange(len(modiDictionaryList)):
      name = modiDictionaryList[i]['name']
      del modiDictionaryList[i]['name']
      if returnElement.find(name[0])!=None:   #if the first level name is present
        if 'erase_block' in modiDictionaryList[i].keys():
          if modiDictionaryList[i]['erase_block']:
            returnElement.remove(returnElement.find(name[0]))
        else:
          child = returnElement.find(name[0])
          if len(name)>1:
            if child.find(name[1])!=None: child.find(name[1]).attrib.update(modiDictionaryList[i])
            else: ET.SubElement(child,name[1],modiDictionaryList[i])
          else:
            child.attrib.update(modiDictionaryList[i])
      else:
        if not 'erase_block' in modiDictionaryList[i].keys():
          ET.SubElement(returnElement,name[0])
          if len(name) > 1:
            child = returnElement.find(name[0])
            ET.SubElement(child,name[1],modiDictionaryList[i])
    if save: return returnElement

