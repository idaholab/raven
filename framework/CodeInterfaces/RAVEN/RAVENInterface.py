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
Created on Sept 10, 2017

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
from CodeInterfaceBaseClass import CodeInterfaceBase
#import RavenData
import csvUtilities

class RAVEN(CodeInterfaceBase):
  """
    this class is used as part of a code dictionary to specialize Model.Code for RAVEN
  """
  def __init__(self):
    CodeInterfaceBase.__init__(self)
    self.printTag  = 'RAVEN INTERFACE'
    self.outputPrefix = 'out~'
    self.setInputExtension(['xml'])

  def __findInputFile(self,inputFiles):
    """
      Method to return the index of the RAVEN input file (error out in case it is not found)
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ Out, index, int, index of the RAVEN input file
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getType().lower() == 'raven':
        inputFileIndex = index
        if found:
          raise IOError(self.printTag+" ERROR: Currently the RAVEN interface allows only one input file (xml). ExternalXML and Merging Files will be added in the future!")
        found = True
    if not found:
      raise IOError(self.printTag+' ERROR: None of the input files are tagged with the "type" "raven" (e.g. <Input name="aName" type="raven">inputFileName.xml</Input>)')
    return index

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    index = self.__findInputFile(inputFiles)
    outputfile = self.outputPrefix+inputFiles[index].getBase()
    executeCommand = [('parallel',executable+ ' '+inputFiles[index].getFilename())]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      this generates a new input file depending on which sampler has been chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    import RAVENparser
    if 'dynamiceventtree' in str(samplerType).strip().lower():
      raise IOError(self.printTag+' ERROR: DynamicEventTree-based sampling not supported!')
    index = self.__findInputFile(currentInputFiles)
    outName = self.outputPrefix+currentInputFiles[index].getBase()
    parser = RAVENparser.RAVENparser(currentInputFiles[index].getAbsFile())
    modifDict = Kwargs['SampledVars']
    # we set the workind directory to the current working dir
    modifDict['RunInfo|WorkingDir'] = '.'
    #make tree
    modifiedRoot = parser.modifyOrAdd(modifDict,False)
    #make input
    parser.printInput(modifiedRoot,currentInputFiles[index].getAbsFile())

    return currentInputFiles

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      this method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output formats into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, returnOut, string, optional, present in case the root of the output file gets changed in this method.
    """
    returnOut = output
    if self.vectorPPFound:
      returnOut = self.__mergeTime(output,workingDir)[0]
    return returnOut


