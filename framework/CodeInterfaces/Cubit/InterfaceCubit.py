'''
created on July 16, 2015

@author: tompjame
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

import os
import copy
import sys
import re
import collections
from utils import toBytes, toStrish, compare
import MessageHandler
from CodeInterfaceBaseClass import CodeInterfaceBase

class CubitInterface(CodeInterfaceBase):
  '''This class is used to couple raven to Cubit journal files for input to generate meshes (usually to run in another simulation)'''

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None):
    '''seek which is which of the inputs files and generate according to the running command'''
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.endswith(self.getInputExtension()):
        found = True
        break
    if not found: self.raiseAnError(IOError,'None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    executeCommand = (executable+ ' -batch ' + os.path.split(inputFiles[index])[1])
    self.raiseADebug(executeCommand)
    return executeCommand, self.outputfile
    # CHECK THIS DEFINITION

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    import ParserCubit
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.endswith(self.getInputExtension()):
        break
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.copy(currentInputFiles)
    newInputFiles[index] = os.path.join(os.path.split(temp)[0], os.path.split(temp)[1].split('.')[0] \
        +'_'+Kwargs['prefix']+'.'+os.path.split(temp)[1].split('.')[1])
    self.outputfile = ('mesh~'+os.path.split(temp)[1].split('.')[0])+'_'+Kwargs['prefix']
    Kwargs['SampledVars']['Cubit|out_name'] = "\"'"+self.outputfile+".e'\""
    parser = ParserCubit.ParserCubit(self.messageHandler, currentInputFiles[index])
    parser.modifyInternalDictionary(**Kwargs['SampledVars'])
    parser.writeNewInput(newInputFiles[index])
    return newInputFiles
    # CHECK THIS DEFINITION

  def getInputExtension(self):
    return(".jou")
