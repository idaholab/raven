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
  def __init__(self):
    CodeInterfaceBase.__init__(self) #I think this isn't implemented
    self.inputExtensions  = [] #list of extensions for RAVEN to edit as inputs
    self.outputExtensions = [] #list of extensions for RAVEN to gather data from?
    self.execPrefix       = '' #executioner command prefix (e.g., 'python ')
    self.execPostfix      = '' #executioner command postfix (e.g. -zcvf)
    self.caseName         = None #base label for outgoing files, should default to inputFileName

  def _readMoreXML(self,xmlNode):
    pass

  def generateComand(self,inputFiles,executable,clargs=None):
    #check for duplicate extension use
    usedExt=[]
    for ext in list(clargs['input'][flag] for flag in clargs['input'].keys()):
      if ext not in usedExt: usedExt.append(ext)

    #check all required input files are there
    inFiles=inputFiles[:]
    for ext in list(clargs['input'][flag] for flag in clargs['input'].keys()):
      for inf in inputFiles:
        if inf.endswith(ext):
          inFiles.remove(inf)
          break
      #if not found
      raise IOError(self.printTag+': ERROR -> input extension "'+ext+'" listed in input but not in inputFiles!')
    #TODO if any remaining, check them against valid inputs

    #PROBLEM this doesn't work, since we can't figure out which .xml goes to -i and which to -d, for example.

#TODO missing some stuff like "executable" here

    outfile = 'out~'+self.caseName
    todo = ''
    todo += clargs['prepend']+' '
    todo += executable
    for flag,exts in clargs['input'].items():
      if flag == 'noarg':
        for ext in exts:
          todo+=' '+ext
        continue
      todo += ' '+flag
      for ext in exts:
        todo+' '+ext
    executeCommand = (self.execPrefix+executable+self.execPostfix)
    #TODO how to specify where the output is set?  -> for now, use special keyword $RAVEN-outFileName$

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    import GenericParser
    indexes=[]
    infiles=[]
    for index,inputFile in enumerate(inputFiles):
      if inputFile.endswith(self.getInputExtension()):
        indexes.append(index)
        infiles.append(inputFile)
    parser = GenericParser.GenericParser(infiles) #TODO this is a list, so be careful
    parser.modifyInternalDictionary(**Kwargs['SampledVars'])
    temps = list(str(oriInputFiles[i][:]) for i in indexes)
    newInFiles = copy.deepcopy(currentInputFiles)
    for i in indexes:
      newInFiles[i] = os.path.join(os.path.split(temp)[0],Kwargs['prefix']+'~'+os.path.split(temp)[1])
    parser.writeNewInput(list(newInFiles[i] for i in indexes))
    return newInFiles
