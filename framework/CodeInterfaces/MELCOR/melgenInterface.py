# Copyright 2017 University of Rome La Sapienza and Battelle Energy Alliance, LLC
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
  Created on April 18, 2017
  @author: Matteo Donorio (University of Rome La Sapienza)
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
from  __builtin__ import any as bAny
import GenericParser
from utils import utils
from CodeInterfaceBaseClass import CodeInterfaceBase

class MelgenApp(CodeInterfaceBase):
  """
    This class is the CodeInterface for MELGEN (a sub-module of Melcor)
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.inputExtensions = ['i','inp']

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
      Generate a command to run MELGEN (a sub-module of Melcor)
      Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    outputfile = 'OUTPUT_MELCOR'
    found = False

    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError("Unknown input extension. Expected input file extensions are "+ ",".join(self.getInputExtension()))
    if clargs:
      precommand = executable + clargs['text']
    else     :
      precommand = executable
    executeCommand = [('serial',precommand + ' '+inputFiles[index].getFilename())]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """
      This generates a new input file depending on which sampler has been chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    if "dynamicevent" in samplerType.lower():
      raise IOError("Dynamic Event Tree-based samplers not implemented for MELCOR yet!")
    indexes  = []
    inFiles  = []
    origFiles= []
    #FIXME possible danger here from reading binary files
    for index,inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        indexes.append(index)
        inFiles.append(inputFile)
    for index,inputFile in enumerate(origInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        origFiles.append(inputFile)
    parser = GenericParser.GenericParser(inFiles)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputFiles,origFiles)
    return currentInputFiles

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      (e.g. in MELCOR would be the expression "Normal termination")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = True
    errorWord = ["Normal termination"]
    try:
      outputToRead = open(os.path.join(workingDir,output+'.out'),"r")
    except IOError:
      # the output does not exist => MELCOR failed
      return failure
    readLines = outputToRead.readlines()
    for goodMsg in errorWord:
      if bAny(goodMsg in x for x in readLines):
        failure = False
        break
    return failure

