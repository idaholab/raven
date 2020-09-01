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
Created on August 31, 2020

@author: Andrea Alfonsi

comments: Interface for AccelerateCFD
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import math
import csv
import re
import copy
import numpy

from CodeInterfaceBaseClass import CodeInterfaceBase
from GenericCodeInterface import GenericParser

class AccelerateCFD(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to AccelerateCFD
  """
  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
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
    # find the input file (check that one input is provided)
    if (len(inputFiles) > 1):
      raise Exception('Projectile INTERFACE ERROR -> Only one input file is accepted!')
    # create output file root
    outputfile = 'out~' + inputFiles[0].getBase()
    # create command (python "executable" -i "input file" -o "output file root")
    executeCommand = [('parallel', executable +' -i '+ inputFiles[0].getFilename() +' -o '+ outputfile + ' -text')]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def getInputExtension(self):
    """
      Return a tuple of possible file extensions for a simulation initialization file (e.g., input.i).
      @ In, None
      @ Out, validExtensions, tuple, tuple of valid extensions
    """
    validExtensions = ('')
    return validExtensions

def findInps(self,inputFiles):
    """
      Locates the input files required by AcellerateCDF Interface
      @ In, inputFiles, list, list of Files objects
      @ Out, inputDict, list, list containing AcellerateCFD required input files
    """
    inputDict = {}
    podDictInput = []

    for inputFile in inputFiles:
      if inputFile.getType().strip().lower() == "podDict":
        podDictInput.append(inputFile)
    if len(podDictInput) == 0:
      raise IOError('no podDict type input file has been found!')
    # Check if the input requested by the sequence has been found
    if self.sequence.count('triton') != len(triton):
      raise IOError('triton input file has not been found. Files type must be set to "triton"!')
    if self.sequence.count('origen') != len(origen):
      raise IOError('origen input file has not been found. Files type must be set to "origen"!')
    return podDictInput

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generate a new AccelerateCFD input file (txt format) from the original, changing parameters
      as specified in Kwargs['SampledVars'].
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """

    if 'dynamiceventtree' in str(samplerType).lower():
      raise IOError("Dynamic Event Tree-based samplers not supported by Scale interface yet!")
    currentInputsToPerturb = self.findInps(currentInputFiles)
    originalInputs         = self.findInps(origInputFiles)
    parser = GenericParser.GenericParser(currentInputsToPerturb)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputsToPerturb,originalInputs)
    return currentInputFiles

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Called by RAVEN to modify output files (if needed) so that they are in a proper form.
      In this case, the default .mat output needs to be converted to .csv output, which is the
      format that RAVEN can communicate with.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    # open output file
    outfileName = os.path.join(workingDir,output+".txt" )
    print(outfileName)
    with open(outfileName, 'r') as src:
      headers = [x.strip() for x in  src.readline().split() ]
      data = []
      line = ""
      # starts reading
      while not line.strip().startswith("--"):
        line = src.readline()
        if not line.strip().startswith("--"):
          data.append(",".join( line.split())+"\n")
      # write the output file
      with open(os.path.join(workingDir,output+".csv" ),"w") as outputFile:
        outputFile.writelines(",".join( headers ) +"\n")
        for i in range(len(data)):
          outputFile.writelines(data[i])

