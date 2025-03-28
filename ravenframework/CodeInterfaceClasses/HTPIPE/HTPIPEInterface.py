# Copyright 2025 NuCube Energy and Battelle Energy Alliance, LLC
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
  Created on Mar 27, 2025
  @author: Andrea Alfonsi
"""

import os
import numpy as np
from collections import defaultdict
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..Generic import GenericParser


class HTPIPE(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to HTPIPE (HEAT PIPE) code
    Woloshun, K A, et al. 'HTPIPE: A steady-state heat pipe
    analysis program: A user's manual.' , Nov. 1988.
    
    The name of this class represents the type in the RAVEN input file
    e.g.
    <Models>
      <Code name="myName" subType="HTPIPE">
      ...
      </Code>
      ...
    </Models>

  """
  def __init__(self):
    """
      Initialize some variables
    """
    super().__init__()
    # Calculation type: 1) pressure/temperature profile 2) temperature vs q_max
    # The calculation type is inferred by the input file (see initialize)
    self.calcType = None
  
  def findInputFile(self, inputFiles):
    """
      Method to find the HTPIPE input file
      @ In, inputFiles, list, list of the original input files
      @ Out, inputFile, FileType, the input file
    """
    found = False
    # Find the first file in the inputFiles that is an HTPIPE compatible, which is what we need to work with.
    for inputFile in inputFiles:
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('No correct input file has been found. HTPIPE input must be of type "htpipe". Got: '+' '.join(inputFiles))
    return inputFile

  def getInputExtension(self):
    """
      Return input extensions
      for HTPIPE, not extensions are used
      @ In, None
      @ Out, getInputExtension, tuple(str), the ext of the code input file (empty string here)
    """
    return ("",)
  
  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    inputFile = self.findInputFile(oriInputFiles)
    with open(inputFile.getAbsFile(), "r") as filestream:
      # we check the first line
      line = filestream.readline()
      try:
        c = line.split()[0].strip()
        self.calcType = int(c)
      except ValueError as ae:
        raise IOError(f"Input File '{inputFile.getAbsFile()}' type 'htpipe' is not a valid HTPIPE input file. "
                      f"First line (first column) should indicate the calculation type (1 or 2)! Got '{c}'!")

  def generateCommand(self, inputFiles, executable, clargs=None,fargs=None,preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have
            been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input
            (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify
            in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to
            run the code (string), returnCommand[1] is the name of the output root
    """
    inputFile = self.findInputFile(inputFiles)
    #Creates the output file that saves information that is outputted to the command prompt
    #The output file name of the Neutrino results
    outputfile = f"o{inputFile.getBase()}"
    # since HTPIPE uses an interactive approach to inquire for the input file,
    #  we are using the run_htpipe.py script to run it with pre-defined options
    pathOfscript = os.path.dirname(os.path.abspath(__file__))
    runHTPIPEscript = os.path.join(pathOfscript,'run_htpipe.py')
    commandToRun = f'python {runHTPIPEscript} -e {executable} -i {inputFile.getFilename()}'
    returnCommand = [('parallel',commandToRun)], outputfile
    return returnCommand

  def _isValidInput(self, inputFile):
    """
      Check if an input file is a Neutrino input file.
      @ In, inputFile, string, the file name to be checked
      @ Out, valid, bool, 'True' if an input file has an extension of '.nescene', otherwise 'False'.
    """
    valid = False
    if inputFile.getType().lower() == 'htpipe':
      valid = True
    return valid

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generate a new HTPIPE input file from the original, changing parameters
      as specified in Kwargs['SampledVars']
      @ In , currentInputFiles, list,  list of current input files (input files of this iteration)
      @ In , oriInputFiles, list, list of the original input files
      @ In , samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In , Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another
             dictionary called "SampledVars" where RAVEN stores the variables that got sampled
             (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    parser = GenericParser.GenericParser(currentInputFiles)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputFiles,oriInputFiles)
    return currentInputFiles

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      In the case of HTPIPE, we look for the expression "failed". If it is present, the run is considered to be failed.
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    badWords  = ['failure', 'halted']

    try:
      outputToRead = open(os.path.join(workingDir,'plotf'),"r")
    except:
      return True
    lines = outputToRead.readlines()
    if len(lines) < 2:
      outputToRead.close()
      return True
    outputToRead.close()
    try:
      outputToRead = open(os.path.join(workingDir,output),"r")
    except:
      outputToRead.close()
      return True
    readLines = outputToRead.readlines()
    for badMsg in badWords:
      if any(badMsg in x.lower() for x in readLines):
        failure = True
    outputToRead.close()
    return failure

  def finalizeCodeOutput(self, command, output, workingDir):
    """
    Called by RAVEN to modify output files (if needed) so that they are in a proper form.
    In this case, we read the plotf file that is generated and contains the table with the results
    @ In, command, string, the command used to run the ended job
    @ In, output, string, the Output name root
    @ In, workingDir, string, current working dir
    @ Out, results, dict, the dictionary with the results
    """
    results = defaultdict(list)
    outputPath = os.path.join(workingDir, "plotf")
    # open original output file
    with open(outputPath,"r+") as outputFile:
      lines = outputFile.readlines()
      variables = [var.strip() for var in lines.pop(0).strip().split()]
      for line in lines:
        values = [float(val.strip()) for val in line.strip().split()]
        for index, var in enumerate(variables):
          results[var].append(values[index])
      for var in variables:
        results[var] = np.atleast_1d(results[var])
    return results
