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
Created on July 31, 2018

@author: Andrea Alfonsi

comments: Interface for Projectile Code
"""
import os
import math
import csv
import re
import copy
import numpy
from collections import defaultdict

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase

class Projectile(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to Projectile
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
    executeCommand = [('parallel', "python " +executable +' -i '+ inputFiles[0].getFilename() +' -o '+ outputfile + ' -text')]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def getInputExtension(self):
    """
      Return a tuple of possible file extensions for a simulation initialization file (e.g., input.i).
      @ In, None
      @ Out, validExtensions, tuple, tuple of valid extensions
    """
    validExtensions = ('i', 'I')
    return validExtensions

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, rlz):
    """
      Generate a new Projectile input file (txt format) from the original, changing parameters
      as specified in Kwargs['SampledVars']. In addition, it creaes an additional input file including the vector data to be
      passed to Dymola.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    # find the input file (check that one input is provided)
    if (len(currentInputFiles) > 1):
      raise IOError('Projectile INTERFACE ERROR -> Only one input file is accepted!')
    # read the original input file lines (read mode "r")
    # NOTE we originally read in the currentInputFiles[0] and read in all its entries,
    #      then modified them with the requested change dict. However, this was changed,
    #      and now we only write the key-value pairs sampled by RAVEN instead.
    # we are ready to write the new input file (open in write mode "w")
    with open(currentInputFiles[0].getAbsFile(), 'w', encoding='utf-8') as src:
      for var, value in rlz.items():
        src.writelines(var+ " = "+ str(value)+"\n")
    return currentInputFiles

  def _readOutputData(self, outfileName):
    """
      Simple method to read the projectile output and return it ad a dictionary
      @ In, outfileName, string, the Output file name
      @ Out, headers, list, the list of variables' names
      @ Out, data, list, the list (timesteps) of list of data [[],[],[],etc]
    """
    with open(outfileName, 'r') as src:
      headers = [x.strip() for x in  src.readline().split() ]
      data = []
      line = ""
      # starts reading
      while not line.strip().startswith("--"):
        line = src.readline()
        if not line.strip().startswith("--"):
          data.append(line.split())
    return headers, data

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
    headers, data = self._readOutputData(outfileName)
    # write the output file
    with open(os.path.join(workingDir,output+".csv" ),"w") as outputFile:
      outputFile.writelines(",".join( headers ) +"\n")
      for i in range(len(data)):
        outputFile.writelines(",".join( data[i] )+"\n")
