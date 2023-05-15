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
Created May 4, 2016

@author: wangc
"""

import os
import copy
from subprocess import Popen
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..MooseBasedApp.MooseBasedAppInterface import MooseBasedApp
#perturb the Yak multigroup library
from . import YakMultigroupLibraryParser
from . import YakInstantLibraryParser

class Rattlesnake(CodeInterfaceBase):
  """
    This class is used to couple raven with rattlesnake input and yak cross section xml input files to generate new
    cross section xml input files.
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.MooseInterface = MooseBasedApp()
    self.MooseInterface.addDefaultExtension()

  def findInps(self,inputFiles):
    """
      Locates the input files for Rattlesnake and Yak
      @ In, inputFiles, list, list of Files objects
      @ Out, inputDict, dict, dictionary containing Rattlesnake and Yak input files
    """
    inputDict = {}
    inputDict['FoundYakXSInput'] = False
    inputDict['FoundRattlesnakeInput'] = False
    inputDict['FoundYakXSAliasInput'] = False
    inputDict['FoundInstantXSInput'] = False
    inputDict['FoundInstantXSAliasInput'] = False
    yakInput = []
    rattlesnakeInput = []
    aliasInput = []
    instantInput = []
    instantAlias = []
    for inputFile in inputFiles:
      if inputFile.getType().strip().lower() == "yakxsinput":
        inputDict['FoundYakXSInput'] = True
        yakInput.append(inputFile)
      elif inputFile.getType().strip().lower().split("|")[-1] == "rattlesnakeinput":
        inputDict['FoundRattlesnakeInput'] = True
        rattlesnakeInput.append(inputFile)
      elif inputFile.getType().strip().lower() == "yakxsaliasinput":
        inputDict['FoundYakXSAliasInput'] = True
        aliasInput.append(inputFile)
      elif inputFile.getType().strip().lower() == "instantxsaliasinput":
        inputDict['FoundInstantXSAliasInput'] = True
        instantAlias.append(inputFile)
      elif inputFile.getType().strip().lower() == "instantxsinput":
        inputDict['FoundInstantXSInput'] = True
        instantInput.append(inputFile)
    if inputDict['FoundYakXSInput']:
      inputDict['YakXSInput'] = yakInput
    if inputDict['FoundInstantXSInput']:
      inputDict['InstantXSInput'] = instantInput
    if inputDict['FoundRattlesnakeInput']:
      inputDict['RattlesnakeInput'] = rattlesnakeInput
    if inputDict['FoundYakXSAliasInput']:
      inputDict['YakAliasInput'] =  aliasInput
    if inputDict['FoundInstantXSAliasInput']:
      inputDict['InstantAliasInput'] =  instantAlias
    if not inputDict['FoundRattlesnakeInput']:
      raise IOError('None of the input files has the type "RattlesnakeInput"! This is required by Rattlesnake interface.')
    return inputDict

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      Generate a command to run Rattlesnake using an input with sampled variables
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have
        been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input
        (e.g. under the node < Code >< clargstype = 0 input0arg = 0 i0extension = 0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify
        in the input (e.g. under the node < Code >< fargstype = 0 input0arg = 0 aux0extension = 0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the
        code (string), returnCommand[1] is the name of the output root
    """
    inputDict = self.findInps(inputFiles)
    rattlesnakeInput = inputDict['RattlesnakeInput']
    if len(rattlesnakeInput) != 1:
      raise IOError('The user should only provide one rattlesnake input file, but found ' + str(len(rattlesnakeInput)) + '!')
    mooseCommand, mooseOut = self.MooseInterface.generateCommand(rattlesnakeInput,executable,clargs,fargs)
    returnCommand = mooseCommand, mooseOut
    return returnCommand

  def createNewInput(self, currentInputFiles, origInputFiles, samplerType, **Kwargs):
    """
      Generates new perturbed input files for both Rattlesnake input file and Yak multigroup group cross section input files.
      @ In, currentInputFiles, list,  list of current input files
      @ In, origInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dict, dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
        where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of new input files (modified or not)
    """
    inputDict = self.findInps(currentInputFiles)
    rattlesnakeInputs = inputDict['RattlesnakeInput']
    foundXS = False
    foundAlias = False
    if inputDict['FoundYakXSInput']:
      foundXS = True
      yakInputs = inputDict['YakXSInput']
    if inputDict['FoundYakXSAliasInput']:
      foundAlias = True
      aliasFiles = inputDict['YakAliasInput']
    if foundXS and foundAlias:
      #perturb the yak cross section inputs
      parser = YakMultigroupLibraryParser.YakMultigroupLibraryParser(yakInputs)
      parser.initialize(aliasFiles)
      #perturb the xs files
      parser.perturb(**Kwargs)
      #write the perturbed files
      parser.writeNewInput(yakInputs,**Kwargs)
    foundInstantXS = False
    foundInstantAlias = False
    if inputDict['FoundInstantXSInput']:
      foundInstantXS = True
      instantInputs = inputDict['InstantXSInput']
    if inputDict['FoundInstantXSAliasInput']:
      foundInstantAlias = True
      instantAlias = inputDict['InstantAliasInput']
    if foundInstantXS and foundInstantAlias:
      #perturb the yak cross section inputs
      parser = YakInstantLibraryParser.YakInstantLibraryParser(instantInputs)
      parser.initialize(instantAlias)
      #perturb the xs files
      parser.perturb(**Kwargs)
      #write the perturbed files
      parser.writeNewInput(instantInputs,**Kwargs)
    #Moose based app interface
    origRattlesnakeInputs = copy.deepcopy(rattlesnakeInputs)
    newMooseInputs = self.MooseInterface.createNewInput(rattlesnakeInputs,origRattlesnakeInputs,samplerType,**Kwargs)
    return currentInputFiles

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      this method is called by the RAVEN code at the end of each run (if the method is present).
      Cleans up files in the working directory that are not needed after the run
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method (in this case None)
    """
    #may need to implement this method, such as remove unused files, ...
    pass
