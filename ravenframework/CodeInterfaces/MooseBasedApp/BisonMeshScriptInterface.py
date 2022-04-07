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
created on July 13, 2015

@author: tompjame
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import os
import copy
from subprocess import Popen
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase

class BisonMeshScript(CodeInterfaceBase):
  """
    This class is used to couple raven to the Bison Mesh Generation Script using cubit (python syntax, NOT Cubit journal file)
  """
  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      Generate a command to run cubit using an input with sampled variables to output
      the perturbed mesh as an exodus file.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (lenght of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    outputfile = 'mesh~'+inputFiles[index].getBase()
    returnCommand = [('serial','python '+executable+ ' -i ' +inputFiles[index].getFilename()+' -o '+outputfile+'.e')], outputfile
    return returnCommand

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generates new perturbed input files.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    import BISONMESHSCRIPTparser
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getExt() == self.getInputExtension():
        break
    parser = BISONMESHSCRIPTparser.BISONMESHSCRIPTparser(currentInputFiles[index])
    parser.modifyInternalDictionary(**copy.deepcopy(Kwargs['SampledVars']))
    parser.writeNewInput(currentInputFiles[index].getAbsFile())
    return currentInputFiles

  def addDefaultExtension(self):
    """
      This method sets a list of default extensions a specific code interface accepts for the input files.
      This method should be overwritten if these are not acceptable defaults.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['py'])

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      this method is called by the RAVEN code at the end of each run (if the method is present).
      Cleans up files in the working directory that are not needed after the run
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method (in this case None)
    """
    # Append wildcard strings to workingDir for files wanted to be removed
    cubitjour_files = os.path.join(workingDir,'cubit*')
    pyc_files = os.path.join(workingDir,'*.pyc')
    # Inform user which files will be removed
    print('Interface attempting to remove files the following:\n    '+cubitjour_files+'\n    '+pyc_files)
    # Remove Cubit generated journal files
    self.rmUnwantedFiles(cubitjour_files)
    # Remove .pyc files created when running BMS python inputs
    self.rmUnwantedFiles(pyc_files)

  def rmUnwantedFiles(self, pathToFiles):
    """
      Method to remove unwanted files after completing the run
      @ In, pathToFiles, string, path to the files to be removed
      @ Out, None
    """
    try:
      p = Popen('rm '+pathToFiles)
    except OSError as e:
      print('    ...',"There was an error removing ",pathToFiles,'<',e,'> but continuing onward...')
