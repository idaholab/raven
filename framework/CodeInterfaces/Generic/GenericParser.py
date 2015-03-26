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

def justFile(fullfile):
  return fullfile.split('/')[-1]

class GenericParser:
  '''import the user-edited input file, build list of strings with replacable parts'''
  def __init__(self,inputFiles,prefix='$RAVEN-',postfix='$',defaultDelim=':'):
    '''
    Accept the input file and parse it by the prefix-postfix breaks. Someday might be able to change prefix,postfix,defaultDelim from input file, but not yet.
    @ In, inputFiles, string list of input filenames that might need parsing.
    @ In, prefix, the string prefix to find input variables within the input files
    @ In, postfix, the string postfix signifying hte end of an input variable within an input file
    @ In, defaultDelim, the string used between prefix and postfix to set default values
    @Out, None.
    '''
    self.inputFiles = inputFiles
    self.prefixKey=prefix
    self.postfixKey=postfix
    self.varPlaces = {} #varPlaces[var][inputFile]
    self.defaults = {}  # defaults[var][inputFile]
    self.segments = {}  # segments[inputFile]
    for inputFile in self.inputFiles:
      infileName = justFile(inputFile)
      self.segments[infileName] = []
      if not os.path.exists(inputFile): raise IOError('Input file not found: '+inputFile)
      IOfile = open(inputFile,'rb')
      foundSome = False
      seg = ''
      lines = IOfile.readlines()
      for line in lines:
        while self.prefixKey in line and self.postfixKey in line:
          self.segments[infileName].append(toBytes(seg))
          start = line.find(self.prefixKey)
          end = line.find(self.postfixKey,start+1)
          var = line[start+len(self.prefixKey):end]
          if defaultDelim in var:
            var,defval = var.split(defaultDelim)
            if var in self.defaults.keys(): print('Parser WARNING: multiple default values given for variable',var)
            #TODO allow the user to specify take-last or take-first?
            if var not in self.defaults.keys(): self.defaults[var]={}
            self.defaults[var][infileName]=defval
          self.segments[infileName].append(toBytes(line[:start]))
          self.segments[infileName].append(toBytes(var))
          if var not in self.varPlaces.keys(): self.varPlaces[var] = {infileName:[len(self.segments[infileName])-1]}
          elif inputFile not in self.varPlaces[var].keys(): self.varPlaces[var][infileName]=[len(self.segments[infileName])-1]
          else: self.varPlaces[var][infileName].append(len(self.segments[infileName])-1)
          #self.segments.append(toBytes(line[end+1:]))
          line=line[end+1:]
          seg = ''
        else:
          seg+=line
      self.segments[infileName].append(toBytes(seg))

  def modifyInternalDictionary(self,**moddict):
    '''
    Edits the parsed file stored in self.segments to enter new variable values preperatory to a new run.
    @ In, moddict, the dictionary of variable:value to replace.
    @Out, None.
    '''
    newFileStrings={}
    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile] if inputFile in self.varPlaces[var].keys() else []:
          if var in moddict.keys(): self.segments[inputFile][place] = str(moddict[var])
          elif var in self.defaults.keys(): self.segments[inputFile][place] = self.defaults[var][inputFile]
          else: raise IOError('For variable '+var+' no distribution was sampled and no default given!')

  def writeNewInput(self,infileNames,origNames):
    '''
    Generates a new input file with the existing parsed dictionary.
    @ In, infileNames, string list of names for new input files to return
    @ In, origNames, the original list of filenames, used for key names
    @Out, None.
    '''
    for f,fileName in enumerate(infileNames):
      outfile = file(fileName,'w')
      outfile.writelines(toBytes(''.join(self.segments[justFile(origNames[f])])))
      outfile.close()
