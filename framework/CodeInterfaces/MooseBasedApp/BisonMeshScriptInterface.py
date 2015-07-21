'''
created on July 13, 2015

@author: tompjame
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import sys
import re
import collections
from utils import toBytes, toStrish, compare
from CodeInterfaceBaseClass import CodeInterfaceBase

class BisonMeshScriptInterface(CodeInterfaceBase):
  '''This class is used to couple raven to the Bison Mesh Generation Script using cubit (python syntax, NOT Cubit journal file)'''

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None):
    '''seek which is which of the input files and generate according to the running command'''
    found = False
    for index, inputFile in enumerate(inputFiles):
      if '.'+inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    outputfile = 'mesh~'+inputFiles[index].getBase()
    executeCommand = ('python '+executable+ ' -i ' +inputFiles[index].getFilename()+' -o '+outputfile+'.e')
    return executeCommand,outputfile

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    import BisonMeshScriptParser
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getExt() == self.getInputExtension():
        break
    moddict = self.expandVarNames(**Kwargs)
    parser = BisonMeshScriptParser.BisonMeshScriptParser(currentInputFiles[index])
    parser.modifyInternalDictionary(**Kwargs['SampledVars'])
    temp = str(oriInputFiles[index])[:]
    newInputFiles = copy.copy(currentInputFiles)
    newInputFiles[index] = os.path.join(os.path.split(temp)[0], os.path.split(temp)[1].split('.')[0] \
	+'_'+Kwargs['prefix']+'.'+os.path.split(temp)[1].split('.')[1])
    parser.writeNewInput(newInputFiles[index])
    return newInputFiles

  def addDefaultExtension(self):
    self.addInputExtension(['.py'])
