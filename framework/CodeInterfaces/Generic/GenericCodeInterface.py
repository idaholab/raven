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
  '''This class is used as a generic code interface for Model.Code in Raven.  It expects
     input paremeters to be specified by input file, input files be specified by either
     command line or in a main input file, and produce a csv output.  It makes significant
     use of the 'clargs', 'fileargs', 'prepend', 'text', and 'postpend' nodes in the input
     XML file.  See base class for more details.
  '''
  def __init__(self):
    '''Initializes the GenericCode Interface.
       @ In, None
       @Out, None
    '''
    CodeInterfaceBase.__init__(self) #The base class doesn't actually implement this, but futureproofing.
    self.inputExtensions  = [] #list of extensions for RAVEN to edit as inputs
    self.outputExtensions = [] #list of extensions for RAVEN to gather data from?
    self.execPrefix       = '' #executioner command prefix (e.g., 'python ')
    self.execPostfix      = '' #executioner command postfix (e.g. -zcvf)
    self.caseName         = None #base label for outgoing files, should default to inputFileName

  def addDefaultExtension(self):
    '''The Generic code interface does not accept any default input types.
    @ In, None
    @Out, None
    '''
    pass

  def generateCommand(self,inputFiles,executable,clargs=None, fargs=None):
    '''
    See base class.  Collects all the clargs and the executable to produce the command-line call.
    '''
    if clargs==None:
      raise IOError('No input file was specified in clargs!')
    #check for output either in clargs or fargs
    if len(fargs['output'])<1 and 'output' not in clargs.keys():
      raise IOError('No output file was specified, either in clargs or fileargs!')
    #check for duplicate extension use
    usedExt=[]
    for ext in list(clargs['input'][flag] for flag in clargs['input'].keys()) + list(fargs['input'][var] for var in fargs['input'].keys()):
      if ext not in usedExt: usedExt.append(ext)
      else: raise IOError('GenericCodeInterface cannot handle multiple input files with the same extension.  You may need to write your own interface.')

    #check all required input files are there
    inFiles=inputFiles[:]
    for exts in list(clargs['input'][flag] for flag in clargs['input'].keys()) + list(fargs['input'][var] for var in fargs['input'].keys()):
      for ext in exts:
        found=False
        for inf in inputFiles:
          if inf.endswith(ext):
            found=True
            inFiles.remove(inf)
            break
        if not found: raise IOError('input extension "'+ext+'" listed in input but not in inputFiles!')
    #TODO if any remaining, check them against valid inputs

    #PROBLEM this is limited, since we can't figure out which .xml goes to -i and which to -d, for example.
    def getFileWithExtension(fileList,ext):
      '''
      Just a script to get the file with extension ext from the fileList.
      @ In, fileList, the string list of filenames to pick from.
      @Out, ext, the string extension that the desired filename ends with.
      '''
      for index,inputFile in enumerate(fileList):
        # already handled inputFile = inputFile.getAbsFile()
        if inputFile.endswith(ext):
          found=True
          break
      if not found: raise IOError('No InputFile with extension '+ext+'found!')
      return index,inputFile

    #prepend
    todo = ''
    todo += clargs['pre']+' '
    todo += executable
    index=None
    #inputs
    for flag,exts in clargs['input'].items():
      if flag == 'noarg':
        for ext in exts:
          idx,fname = getFileWithExtension(inputFiles,ext)
          todo+=' '+fname
          if index == None: index = idx
        continue
      todo += ' '+flag
      for ext in exts:
        idx,fname = getFileWithExtension(inputFiles,ext)
        todo+=' '+fname
        if index == None: index = idx
    #outputs
    #FIXME I think if you give multiple output flags this could result in overwriting
    self.caseName = os.path.split(inputFiles[index])[1].split('.')[0]
    outfile = 'out~'+self.caseName
    if 'output' in clargs.keys():
      todo+=' '+clargs['output']+' '+outfile
    #text flags
    todo+=' '+clargs['text']
    #postpend
    todo+=' '+clargs['post']
    executeCommand = (todo)
    self.raiseADebug('Execution Command: '+str(executeCommand))
    return executeCommand,outfile

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    '''
    See base class.  Loops over all input files to edit as many as needed to cover the input variables.
    '''
    import GenericParser
    indexes=[]
    infiles=[]
    #FIXME possible danger here from reading binary files
    for index,inputFile in enumerate(currentInputFiles):
      inputFile = inputFile.getAbsFile()
      if inputFile.endswith(self.getInputExtension()):
        indexes.append(index)
        infiles.append(inputFile)
    parser = GenericParser.GenericParser(infiles)
    parser.modifyInternalDictionary(**Kwargs)
    temps = list(str(origInputFiles[i][:]) for i in indexes)
    newInFiles = copy.deepcopy(currentInputFiles)
    for i in indexes:
      newInFiles[i] = os.path.join(os.path.split(temps[i])[0],Kwargs['prefix']+'~'+os.path.split(temps[i])[1])
    parser.writeNewInput(list(newInFiles[i] for i in indexes),list(origInputFiles[i] for i in indexes))
    #except TypeError: parser.writeNewInput(list(newInFiles[i] for i in indexes))
    return newInFiles
