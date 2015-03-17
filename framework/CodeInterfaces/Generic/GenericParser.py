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
  def __init__(self,inputFile,prefix='$RAVEN-',postfix='$',defaultDelim=':'):
    '''Accept the input file and parse it by the prefix-postfix breaks.'''
    if not os.path.exists(inputFile): raise IOError('Input file not found!')
    IOfile = open(inputFile,'rb')
    self.inputfile = inputFile
    lines = IOfile.readlines()
    self.varPlaces={}
    self.defaults={}
    self.segments = []
    self.prefixKey=prefix
    self.postfixKey=postfix
    seg = ''
    for line in lines:
      while self.prefixKey in line and self.postfixKey in line:
        #TODO what if it's in there multiple times?? (there being file OR line)
        self.segments.append(toBytes(seg))
        start = line.find(self.prefixKey)
        end = line.find(self.postfixKey,start+1)
        var = line[start+len(self.prefixKey):end]
        if defaultDelim in var:
          var,defval = var.split(defaultDelim)
          if var in self.defaults.keys(): print('Parser WARNING: multiple default values given for variable',var)
          #TODO allow the user to specify take-last or take-first?
          self.defaults[var]=defval
        self.segments.append(toBytes(line[:start]))
        self.segments.append(toBytes(var))
        if var not in self.varPlaces.keys(): self.varPlaces[var] = [len(self.segments)-1]
        else: self.varPlaces[var].append(len(self.segments)-1)
        #self.segments.append(toBytes(line[end+1:]))
        line=line[end+1:]
        seg = ''
      else:
        seg+=line
    self.segments.append(toBytes(seg))

  def printInput(self,moddict={},outfile=None):
    toPrint = self.segments[:]
    for var in self.varPlaces.keys():
      for p in self.varPlaces[var]:
        if var in moddict.keys(): toPrint[p] = moddict[var]
        elif var in self.defaults.keys(): toPrint[p] = self.defaults[var]
        else: raise IOError('For variable '+var+' no distribution was sampled and no default given!')
    toPrint = toBytes(''.join(toPrint))
    if outfile==None: outfile = self.inputfile
    IOfile = open(outfile,'wb')
    IOfile.write(toPrint)
    return

  def __updateDict(self,dictionary,other):
    """Add all the keys and values in other into dictionary"""
    raise IOError('Generic Parser has no __updateDict method.')

  def __matchDict(self,dictionary,other):
    """ Returns true if all the keys and values in other
    match all the keys and values in dictionary.
    Note that it does not check that all the keys in dictionary
    match all the keys and values in other.
    """
    raise IOError('Generic Parser has no __updateDict method.')
