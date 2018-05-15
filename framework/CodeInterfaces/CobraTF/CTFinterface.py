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

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
from ctfdata import ctfdata
from CodeInterfaceBaseClass import CodeInterfaceBase
import os
class CobraTF(CodeInterfaceBase):
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

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType ,**Kwargs):
    """
      this generates a new CTF input file based on the sampled values from RAVEN 
      # @ In, Kwargs
      # @ Out, current input files 
    """
    from CTFparser import CTFparser
    if 'dynamicevent' in samplerType.lower():
      raise IOError("Sampler type error: Dynamic Even Tree-based sampling is not implemented yet!")
    found = False
    for inputFile in currentInputFiles:
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('Input file missing: Check if the input file (.inp) is located in the working directory!' )
    ctfParser = CTFparser(inputFile.getAbsFile())  
    modifDict = Kwargs["SampledVars"]
    ctfParser.changeVariable(modifDict)  
    modifiedDictionary = ctfParser.modifiedDictionary
    ctfParser.printInput(inputFile.getAbsFile())  
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

