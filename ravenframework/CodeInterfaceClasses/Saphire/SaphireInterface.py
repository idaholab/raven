# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on July 12, 2018

@author: wangc

comments: Interface for SAPHIRE Simulation
"""
import os
import copy

from ..Generic.GenericCodeInterface import GenericCode
from .SaphireData import SaphireData

class Saphire(GenericCode):
  """
    SAPHIRE Interface.
  """
  def __init__(self):
    """
      Initializes the SAPHIRE code interface
      @ In, None
      @ Out, None
    """
    GenericCode.__init__(self)
    self.codeOutputs = {} # dict of {codeOutputFileName:codeOutputFileType}, SAPHIRE can generate multiple output files
                          # such as uncertainty files for event trees or fault trees, and importance files for event
                          # trees or fault trees. This dictionary will store the users defined outputflies that
                          # will be collected.
    self.outputDest = 'Publish' # Saphire will dump its outputs to this folder
    self.ignoreInputExtensions = ['zip'] # the list of input extensions that will be ignored by the code interface.
                                         # i.e. files with extension 'zip' will not be perturbed.
    self.setRunOnShell(shell=False) # Saphire can not run through shell

  def addInputExtension(self,exts):
    """
      This method adds a list of extension the code interface accepts for the input files
      @ In, exts, list, list or other array containing accepted input extension (e.g.[".i",".inp"])
      @ Out, None
    """
    for e in exts:
      if e not in self.ignoreInputExtensions:
        self.inputExtensions.append(e)

  def addDefaultExtension(self):
    """
      The Generic code interface does not accept any default input types.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['mac'])

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this class and initialize some members
      based on inputs
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node
      @ Out, None
    """
    GenericCode._readMoreXML(self, xmlNode)
    for outNode in xmlNode.findall('codeOutput'):
      outType = outNode.get('type').strip()
      outName = outNode.text.strip()
      self.codeOutputs[outName] = outType

  def generateCommand(self,inputFiles,executable,clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< fileargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    if clargs==None:
      raise IOError('No input file was specified in clargs!')
    #check all required input files are there
    inFiles=inputFiles[:]
    #check for duplicate extension use
    usedExt=[]
    for elems in list(clargs['input'][flag] for flag in clargs['input'].keys()) + list(fargs['input'][var] for var in fargs['input'].keys()):
      for elem in elems:
        ext = elem[0]
        if ext not in usedExt:
          usedExt.append(ext)
        else:
          raise IOError('Saphire cannot handle multiple input files with the same extension')
        found=False
        for inf in inputFiles:
          if '.'+inf.getExt() == ext:
            found=True
            inFiles.remove(inf)
            break
        if not found:
          raise IOError('input extension "'+ext+'" listed in input but not in inputFiles!')

    #TODO if any remaining, check them against valid inputs

    #PROBLEM this is limited, since we can't figure out which .xml goes to -i and which to -d, for example.
    def getFileWithExtension(fileList,ext):
      """
      Just a script to get the file with extension ext from the fileList.
      @ In, fileList, the string list of filenames to pick from.
      @ Out, ext, the string extension that the desired filename ends with.
      """
      found = False
      for index,inputFile in enumerate(fileList):
        if inputFile.getExt() == ext:
          found=True
          break
      if not found:
        raise IOError('No InputFile with extension '+ext+' found!')
      return index,inputFile

    todo = ''
    todo += executable
    index=None
    #inputs
    for flag,elems in clargs['input'].items():
      if flag == 'noarg':
        continue
      todo += ' '+flag
      for elem in elems:
        ext, delimiter = elem[0], elem[1]
        idx,fname = getFileWithExtension(inputFiles,ext.strip('.'))
        todo += delimiter + fname.getAbsFile()
        if index == None:
          index = idx
    #outputs
    self.caseName = inputFiles[index].getBase()
    outFile = 'out~'+self.caseName
    if self.fixedOutFileName is not None:
      outFile = self.fixedOutFileName
    returnCommand = [('parallel',todo)],outFile
    print('Execution Command: '+str(returnCommand[0]))
    return returnCommand

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run. It will convert the SAPHIRE outputs into RAVEN
      compatible csv file.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
    """
    filesIn = []
    for outName, outType in self.codeOutputs.items():
      filesIn.append((os.path.join(workingDir, self.outputDest, outName), outType))
    outputParser = SaphireData(filesIn)
    outputParser.writeCSV(os.path.join(workingDir, output))

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by RAVEN at the end of each run if the return code is == 0.
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    return failure
