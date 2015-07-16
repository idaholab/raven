'''
Created on Jul 09, 2015

@author: tompjame
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import os
import sys
import copy
import re
import collections
from utils import toBytes, toStrish, compare
import MessageHandler

class ParserBisonMeshScript(MessageHandler.MessageUser):
  '''Import Bison Mesh Script input, provide methods to add/change entries and print input back'''

  def __init__(self,messageHandler,inputFile):
    '''Open and read file content into a dictionary'''
    self.printTag = 'BISONMESHSCRIPT_PARSER'
    self.messageHandler = messageHandler
    if not os.path.exists(inputFile): self.raiseAnError(IOError,'Input file not found: '+inputFile)
    # Initialize file dictionary, storage order, and internal variables
    self.AllVarDict = collections.OrderedDict()
    self.fileOrderStorage = []

    quote_comment = False
    quote_comment_line = False
    apostrophe_comment = False
    apostrophe_comment_line = False

    between_str = ''

    # Open file
    IOfile = open(inputFile)
    self.inputfile = inputFile

    # Store lines into a list
    lines = IOfile.readlines()

    # self.keywordDictionary dictionary
    for line in lines:
      if '"""' in line or "'''" in line:
        if '"""' in line and quote_comment == True:
          quote_comment_line = True
          splitline = line.split('"""')
          between_str += splitline[0] + '"""'
          line = splitline[1]
          quote_comment = not quote_comment
        elif '"""' in line and quote_comment == False:
          quote_comment_line = True
          splitline = line.split('"""')
          line = splitline[0]
          quote_comment = not quote_comment
        elif "'''" in line and apostrophe_comment == True:
          apostrophe_comment_line = True
          splitline = line.split("'''")
          between_str += splitline[0] + "'''"
          line = splitline[1]
          apostrophe_comment = not apostrophe_comment
        elif "'''" in line and apostrophe_comment == False:
          apostrophe_comment_line = True
          splitline = line.split("'''")
          line = splitline[0]
          apostrophe_comment = not apostrophe_comment
        # parse stuff that is left over on the line
        if len(line) == 0:
          between_str += line
        elif line.isspace():
          between_str += line
        elif line.startswith('#'):
          between_str += line
        elif '{}' in line:
          between_str += line
        elif line.startswith('pellets'):
          between_str += line
        else:
          # Append string of non-varying parts of input file to file storage and reset the collection string
          if len(between_str) > 0: self.fileOrderStorage.append(between_str); between_str = ''
          dictname, varname, varvalue = re.split("\['|'] = |'] =|']= ", line)
          if dictname in self.AllVarDict.keys():
            self.AllVarDict[dictname][varname] = varvalue.strip()
          else:
            self.fileOrderStorage.append([dictname])
            self.AllVarDict[dictname] = collections.OrderedDict()
            self.AllVarDict[dictname][varname] = varvalue.strip()
        # Add comment contents to storage for ''' or """ that starts comment block after code on same line
        if quote_comment_line == True and quote_comment == True:
          between_str += '"""' + splitline[1]
        elif apostrophe_comment_line == True and apostrophe_comment == True:
          between_str += "'''" + splitline[1]
        quote_comment_line = False
        apostrophe_comment_line = False
      else: # Didn't find a comment block flag
        if quote_comment == True or apostrophe_comment == True:
          between_str += line
          continue
        else: # Outside of comment block (in code)
          if len(line) == 0:
            between_str += line
          elif line.isspace():
            between_str += line
          elif line.startswith('#'):
            between_str += line
          elif '{}' in line:
            between_str += line
          elif line.startswith('pellets'):
            between_str += line
          else:
            # Append string of non-varying parts of input file to file storage and reset the collection string
            if len(between_str) > 0: self.fileOrderStorage.append(between_str); between_str = ''
            dictname, varname, varvalue = re.split("\['|'] = |'] =|']= ", line)
            if dictname in self.AllVarDict.keys():
              self.AllVarDict[dictname][varname] = varvalue.strip()
            else:
              self.fileOrderStorage.append([dictname])
              self.AllVarDict[dictname] = collections.OrderedDict()
              self.AllVarDict[dictname][varname] = varvalue.strip()
    if len(between_str) > 0: self.fileOrderStorage.append(between_str)
    IOfile.close()

  def modifyInternalDictionary(self,inDictionary):
    # Parse the input dictionary and replace matching keywords in internal dictionary
    for keyword, newvlaue in inDictionary.items():
      garb, keyword1, keyword2 = keyword.split('|')
      self.AllVarDict[keyword1][keyword2] = newvalue


  def printInput(self,outfile=None):
    if outfile==None: outfile = self.inputfile
    IOfile = open(outfile,'w')
    for e, entry in enumerate(self.fileOrderStorage):
      if type(entry) == str:
        IOfile.writelines(entry)
      elif type(entry) == list:
        DictBlockName = self.fileOrderStorage[e][0]
        DictBlock = self.AllVarDict[DictBlockName]
        for key, value in DictBlock.items():
          IOfile.writelines(DictBlockName + "['" + key + "'] = " + value + '\n')
    IOfile.close()
