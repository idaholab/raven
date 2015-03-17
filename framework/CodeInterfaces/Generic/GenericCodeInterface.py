'''
Created March 17th, 2015

@author: talbpaul
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
from CodeInterfaceBaseClass import CodeInterfaceBase

class GenericCodeInterface(CodeInterfaceBase):
  def generateComand(self,inputFiles,executable,flags=None):
    found = False
    for index,inputFile in enumerate(inputFiles):
      if inputFiles.endswith(('.i','.inp','.in')): #TODO makes this a user option
        found = True
        break
    if not found: raise IOError('GENERIC INTERFACE ERROR -> No input file with (.i, .inp, .in) extension found!')
    outfile = 'out~TODO'

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    import GenericParser
    parser = GenericParser.GenericParser(
