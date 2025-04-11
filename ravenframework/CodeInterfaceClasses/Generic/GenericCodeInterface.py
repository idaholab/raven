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
Created March 17th, 2015

@author: talbpaul
"""
import os
import pandas as pd
import numpy as np
from . import GenericParser
from ...Functions import factory as functionFactory
from ...Functions import returnInputParameter
from ...utils.TreeStructure import InputNode
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase

class GenericCode(CodeInterfaceBase):
  """
    This class is used as a generic code interface for Model.Code in Raven.  It expects
    input parameters to be specified by input file, input files be specified by either
    command line or in a main input file, and produce a csv output.  It makes significant
    use of the 'clargs', 'fileargs', 'prepend', 'text', and 'postpend' nodes in the input
    XML file.  See base class for more details.
  """
  def __init__(self):
    """
      Initializes the GenericCode Interface.
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self) # The base class doesn't actually implement this, but futureproofing.
    self.inputExtensions  = []       # list of extensions for RAVEN to edit as inputs
    self.outputExtensions = []       # list of extensions for RAVEN to gather data from?
    self.execPrefix       = ''       # executioner command prefix (e.g., 'python ')
    self.execPostfix      = ''       # executioner command postfix (e.g. -zcvf)
    self.caseName         = None     # base label for outgoing files, should default to inputFileName
    self.fixedOutFileName = None     # CSV output filename of the run code (in case it is hardcoded in the driven code)
    # function to be evaluated for the stopping condition if any
    self.stoppingCriteriaFunction = None
    self._stoppingCriteriaFunctionNode = None

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this class and
      initialize some members based on inputs.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    outFileName = xmlNode.find("outputFile")
    self.fixedOutFileName = outFileName.text if outFileName is not None else None
    if self.fixedOutFileName is not None:
      if '.' in self.fixedOutFileName and self.fixedOutFileName.split(".")[-1] != 'csv':
        raise IOError('user defined output extension "'+self.fixedOutFileName.split(".")[-1]+'" is not a "csv"!')
      else:
        self.fixedOutFileName = '.'.join(self.fixedOutFileName.split(".")[:-1])
    # we just store it for now because we need the runInfo to be applied before parsing it
    self._stoppingCriteriaFunctionNode = xmlNode.find("stoppingCriteriaFunction")

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    super().initialize(runInfo, oriInputFiles)
    if self._stoppingCriteriaFunctionNode is not None:
      # get instance of function and apply run info
      self.stoppingCriteriaFunction = functionFactory.returnInstance('External')
      self.stoppingCriteriaFunction.applyRunInfo(runInfo)
      # create a functions input node to use the Functions reader
      functs = InputNode('Functions')
      # change tag name to External (to use the parser)
      # this is very ugly but works fine
      self._stoppingCriteriaFunctionNode.tag = 'External'
      functs.append(self._stoppingCriteriaFunctionNode)
      inputParams = returnInputParameter()
      inputParams.parseNode(functs)
      self.stoppingCriteriaFunction.handleInput(inputParams.subparts[0])
      if self.stoppingCriteriaFunction.name not in self.stoppingCriteriaFunction.availableMethods():
        self.raiseAnError(ValueError, f"<stoppingCriteriaFunction> named '{self.stoppingCriteriaFunction.name}' "
                          f"not found in file '{self.stoppingCriteriaFunction.functionFile}'!")

  def addDefaultExtension(self):
    """
      The Generic code interface does not accept any default input types.
      @ In, None
      @ Out, None
    """
    pass

  def generateCommand(self,inputFiles,executable,clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (e.g. under the node < Code >< fileargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    if clargs==None:
      raise IOError('No input file was specified in clargs!')
    #check for output either in clargs or fargs
    #if len(fargs['output'])<1 and 'output' not in clargs.keys():
    #  raise IOError('No output file was specified, either in clargs or fileargs!')
    #check all required input files are there
    #check for duplicate extension use
    extsClargs = list(ext[0][0] for ext in clargs['input'].values() if len(ext) != 0)
    extsFargs  = list(ext[0] for ext in fargs['input'].values())
    usedExts = extsClargs + extsFargs
    if len(usedExts) != len(set(usedExts)):
      raise IOError('GenericCodeInterface cannot handle multiple input files with the same extension.  You may need to write your own interface.')
    for inf in inputFiles:
      ext = '.' + inf.getExt() if inf.getExt() is not None else ''
      try:
        usedExts.remove(ext)
      except ValueError:
        pass
    if len(usedExts) != 0:
      raise IOError('Input extension',','.join(usedExts),'listed in XML node Code, but not found in the list of Input of <Files>')

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

    #prepend
    todo = ''
    todo += clargs['pre']+' '
    todo += executable
    index=None
    #inputs
    for flag,elems in clargs['input'].items():
      if flag == 'noarg':
        for elem in elems:
          ext, delimiter = elem[0], elem[1]
          idx,fname = getFileWithExtension(inputFiles,ext.strip('.'))
          todo += delimiter + fname.getFilename()
          if index == None:
            index = idx
        continue
      todo += ' '+flag
      for elem in elems:
        ext, delimiter = elem[0], elem[1]
        idx,fname = getFileWithExtension(inputFiles,ext.strip('.'))
        todo += delimiter + fname.getFilename()
        if index == None:
          index = idx
    #outputs
    #FIXME I think if you give multiple output flags this could result in overwriting
    self.caseName = inputFiles[index].getBase()
    outFile = 'out~'+self.caseName
    if 'output' in clargs:
      todo+=' '+clargs['output']+' '+outFile
    if self.fixedOutFileName is not None:
      outFile = self.fixedOutFileName
    todo+=' '+clargs['text']
    #postpend
    todo+=' '+clargs['post']
    returnCommand = [('parallel',todo)],outFile
    print('Execution Command: '+str(returnCommand[0]))
    return returnCommand

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """
      This method is used to generate an input based on the information passed in.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    infiles=[]
    origfiles=[]
    #FIXME possible danger here from reading binary files
    for inputFile in currentInputFiles:
      if inputFile.getExt() in self.getInputExtension():
        infiles.append(inputFile)
    for inputFile in origInputFiles:
      if inputFile.getExt() in self.getInputExtension():
        origfiles.append(inputFile)
    parser = GenericParser.GenericParser(infiles)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(infiles,origfiles)
    return currentInputFiles

  def onlineStopCriteria(self, command, output, workDir):
    """
      This method is called by RAVEN during the simulation.
      It is intended to provide means for the code interface to monitor the execution of a run
      and stop it if certain criteria are met (defined at the code interface level)
      For example, the underlying code interface can check for a figure of merit in the output file
      produced by the driven code and stop the simulation if that figure of merit is outside a certain interval
      (e.g. Pressure > 2 bar, stop otherwise, continue).
      If the simulation is stopped because of this check, the return code is set artificially to 0 (normal termination) and
      the 'checkForOutputFailure' method is not called. So the simulation is considered to be successful.

      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, continueSim, bool, True if the job needs to continue being executed, False if it needs to be stopped
    """
    continueSim = True
    if self.stoppingCriteriaFunction is not None:
      outCsv = os.path.join(workDir,output+".csv")
      if os.path.exists(outCsv):
        df = pd.read_csv(outCsv)
        results =  dict((header, np.array(df[header])) for header in df.columns)
        continueSim = self.stoppingCriteriaFunction.evaluate(self.stoppingCriteriaFunction.name,results)
    return continueSim
