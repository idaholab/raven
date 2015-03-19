'''
Created on Mar 10, 2015

@author: talbpaul
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import os
import copy
from utils import toBytes, toStrish, compare

class GenericParser:
  '''import the user-edited input file, build list of strings with replacable parts'''
  def __init__(self,inputFiles,prefix='$RAVEN-',postfix='$',defaultDelim=':'):
    '''Accept the input file and parse it by the prefix-postfix breaks.'''
    self.inputfiles = inputFiles
    self.prefixKey=prefix
    self.postfixKey=postfix
    self.varPlaces = {} #varPlaces[var][inputFile]
    self.defaults = {}  # defaults[var][inputFile]
    self.segments = {}  # segments[inputFile]
    for inputFile in self.inputFiles:
      self.segments[inputFile] = []
      if not os.path.exists(inputFile): raise IOError('Input file not found: '+inputFile)
      IOfile = open(inputFile,'rb')
      foundSome = False
      seg = ''
      lines = IOfile.readlines()
      for line in lines:
        while self.prefixKey in line and self.postfixKey in line:
          self.segments[inputFile].append(toBytes(seg))
          start = line.find(self.prefixKey)
          end = line.find(self.postfixKey,start+1)
          var = line[start+len(self.prefixKey):end]
          if defaultDelim in var:
            var,defval = var.split(defaultDelim)
            if var in self.defaults.keys(): print('Parser WARNING: multiple default values given for variable',var)
            #TODO allow the user to specify take-last or take-first?
            self.defaults[var][inputFile]=defval
          self.segments[inputFile].append(toBytes(line[:start]))
          self.segments[inputFile].append(toBytes(var))
          if var not in self.varPlaces.keys(): self.varPlaces[var] = {inputFile:[len(self.segments[inputFile])-1]}
          elif inputFile not in self.varPlaces[var].keys(): self.varPlaces[var][inputFile]=[len(self.segments[inputFile])-1]
          else: self.varPlaces[var][inputFile].append(len(self.segments[inputFile])-1)
          #self.segments.append(toBytes(line[end+1:]))
          line=line[end+1:]
          seg = ''
        else:
          seg+=line
      self.segments[inputFile].append(toBytes(seg))

  def modifyInternalDictionary(self,moddict):
    newFileStrings={}
    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile]:
          if var in moddict.keys(): self.segments[inputFiles][p] = moddict[var]
          elif var in self.defaults.keys(): self.segments[inputFiles][p] = self.defaults[var]
          else: raise IOError('For variable '+var+' no distribution was sampled and no default given!')

  def writeNewInput(self,infileNames):
    for fileName in infileNames:
      outfile = file(fileName,'w')
      outfile.writelines(toBytes(''.join(self.segments[fileName])))
      outfile.close()
