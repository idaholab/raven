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
Modified by Alp Tezbasaran @ INL (6/21/2018)
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
from ctfdata import ctfdata
from CodeInterfaceBaseClass import CodeInterfaceBase
from GenericCodeInterface import GenericParser

import os
class CTF(CodeInterfaceBase):
  """
    this class is used a part of a code dictionary to specialize Model.Code for Cobra-TF (CTF)
  """
  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each code run to create CSV files containing the code output results.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    outfile  = os.path.join(workingDir,output+'.out')
    outputobj= ctfdata(outfile)
    outputobj.writeCSV(os.path.join(workingDir,output+'.csv'))
  def findInps(self,inputFiles):
      """
        Locates the input files required by CTF (Cobra-TF) Interface
        @ In, inputFiles, list, list of Files objects
        @ Out, inputDict, dict, dictionary containing Scale required input files
      """
      inputDict = {}
      CTF = []
      vuq_param = []
      vuq_mult = []

      for inputFile in inputFiles:
          if inputFile.getType().strip().lower() == "ctf":
              CTF.append(inputFile)
          elif inputFile.getType().strip().lower() == "vuq_param":
              vuq_param.append(inputFile)
          elif inputFile.getType().strip().lower() == "vuq_mult":
              vuq_mult.append(inputFile)

      # raise error if there are multiple input files (not allowed)
      if len(CTF) > 1:
        raise IOError('multiple CTF input files have been found. Only one is allowed!')
      if len(vuq_param) > 1:
        raise IOError('multiple vuq_param.txt input files have been found. Only one is allowed!')
      if len(vuq_mult) > 1:
        raise IOError('multiple vuq_mult.txt input files have been found. Only one is allowed!')

      # raise error if the names of the vuq files are different (not allowed)
      # TO BE CODED

      # add inputs
      if len(CTF) > 0:
         inputDict['CTF'] = CTF
      if len(vuq_param) > 0:
         inputDict['vuq_param'] = vuq_param
      if len(vuq_param) > 0:
         inputDict['vuq_mult'] = vuq_mult
      return inputDict

  def createNewInput(self, currentInputFiles, origInputFiles, samplerType, **Kwargs):
    """
      Generates new perturbed input files for Scale sequences
      @ In, currentInputFiles, list,  list of current input files
      @ In, origInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dict, dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
        where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of new input files (modified or not)
    """
    #print(currentInputFiles)
    if 'dynamiceventtree' in str(samplerType).lower():
      raise IOError("Dynamic Event Tree-based samplers not supported by Scale interface yet!")
    currentInputsToPerturb = [item for subList in self.findInps(currentInputFiles).values() for item in subList]
    originalInputs         = [item for subList in self.findInps(origInputFiles).values() for item in subList]
    parser = GenericParser.GenericParser(currentInputsToPerturb)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputsToPerturb,originalInputs)
    return currentInputFiles


  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      Collects the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Command is for code execution and is currently available only for serial simulation.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (Not needed)
      @ In, fargs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (Not needed)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """

    inputDict = self.findInps(inputFiles)

    found = False
    for inputFile in inputFiles:
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('Input file missing: Check if the input file (.inp) is located in the working directory!' )
    commandToRun = executable + ' ' + inputFile.getFilename()
    commandToRun = commandToRun.replace("\n"," ")
    output = inputFile.getBase() + '.ctf'
    returnCommand = [('parallel', commandToRun)], output
    return returnCommand
