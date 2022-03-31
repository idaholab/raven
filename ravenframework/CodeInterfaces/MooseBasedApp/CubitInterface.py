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
created on July 16, 2015

@author: tompjame
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import copy
from subprocess import Popen
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase

class Cubit(CodeInterfaceBase):
  """
    This class is used to couple raven to Cubit journal files for input to generate
    meshes (usually to run in another simulation)
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
    returnCommand = [('serial',executable+ ' -batch ' + inputFiles[index].getFilename())], self.outputFile
    return returnCommand

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      This method is used to generate an input based on the information passed in.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    import CUBITparser
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getExt() == self.getInputExtension():
        break
    parser = CUBITparser.CUBITparser(oriInputFiles[index])
    self.outputFile = 'mesh~'+currentInputFiles[index].getBase()
    Kwargs['SampledVars']['Cubit@out_name'] = "\"'"+self.outputFile+".e'\""
    parser.modifyInternalDictionary(**copy.deepcopy(Kwargs['SampledVars']))
    # Write new input files
    parser.writeNewInput(currentInputFiles[index].getAbsFile())
    return currentInputFiles

  def getInputExtension(self):
    """
      This method returns a list of extension the code interface accepts for the input file (the main one)
      @ In, None
      @ Out, tuple, tuple of strings containing accepted input extension (e.g.(".i",".inp"]) )
    """
    return("jou")

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Cleans up files in the working directory that are not needed after the run
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method. (not in case)
    """
    # Append wildcard strings to workingDir for files wanted to be removed
    cubitjour_files = os.path.join(workingDir,'cubit*')
    # Inform user which files will be removed
    print('Interface attempting to remove files: \n'+cubitjour_files)
    # Remove Cubit generated journal files
    self.rmUnwantedFiles(cubitjour_files)

  def rmUnwantedFiles(self, pathToFiles):
    """
      Method to remove unwanted files after completing the run
      @ In, pathToFiles, string, path to the files to be removed
      @ Out, None
    """
    try:
      p = Popen('rm '+pathToFiles)
    except OSError as e:
      print('  ...',"There was an error removing ",pathToFiles,'<',e,'>','but continuing onward...')
