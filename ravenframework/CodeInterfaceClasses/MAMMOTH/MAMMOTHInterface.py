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
Created on Apr 26, 2016

@author: wangc
"""

import os
import copy
from subprocess import Popen
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..MooseBasedApp.MooseBasedAppInterface import MooseBasedApp
from ..Rattlesnake.RattlesnakeInterface import Rattlesnake
from ..RELAP7.RELAP7Interface import RELAP7

class MAMMOTH(CodeInterfaceBase):
  """
    This class is used to couple raven with MAMMOTH (A moose based application, can call Rattlesnake, Bison and Relap-7)
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.MooseInterface = MooseBasedApp()       #used to perturb MAMMOTH input files
    self.MooseInterface.addDefaultExtension()
    self.BisonInterface = MooseBasedApp()       #used to perturb Bison input files
    self.BisonInterface.addDefaultExtension()
    self.RattlesnakeInterface  = Rattlesnake()  #used to perturb Rattlesnake and Yak input files
    #FIXME Would like to use RELAP7() as interface, but Distributions block appears to be out of date when running Mammoth
    #self.Relap7Interface = RELAP7()             #used to perturb RELAP7 input files
    self.Relap7Interface = MooseBasedApp()
    self.Relap7Interface.addDefaultExtension()

  def findInps(self,inputFiles):
    """
      Locates the input files required by MAMMOTH
      @ In, inputFiles, list, list of Files objects
      @ Out, inputDict, dict, dictionary containing MAMMOTH required input files
    """
    inputDict = {}
    inputDict['MammothInput'] = []
    inputDict['BisonInput'] = []
    inputDict['RattlesnakeInput'] = []
    inputDict['Relap7Input'] = []
    inputDict['AncillaryInput'] = []
    allowedDriverAppInput = ['bisoninput','rattlesnakeinput','relap7input']
    for inputFile in inputFiles:
      fileType = inputFile.getType()
      if fileType.strip().lower().split('|')[0] == "mammothinput":
        inputDict['MammothInput'].append(inputFile)
        inputDict['DriverAppInput'] = fileType.strip().lower().split('|')[-1]
      if fileType.strip().lower().split('|')[-1] == "bisoninput":
        inputDict['BisonInput'].append(inputFile)
      elif fileType.strip().lower().split('|')[-1] == "rattlesnakeinput" or \
           fileType.strip().lower() == "yakxsinput" or                      \
           fileType.strip().lower() == "yakxsaliasinput" or                 \
           fileType.strip().lower() == "instantxsinput" or                  \
           fileType.strip().lower() == "instantxsaliasinput":
        inputDict['RattlesnakeInput'].append(inputFile)
      elif fileType.strip().lower().split('|')[-1] == "relap7input":
        inputDict['Relap7Input'].append(inputFile)
      elif fileType.strip().lower() == "ancillaryinput":
        inputDict['AncillaryInput'] = []
    # Mammoth input is not found
    if len(inputDict['MammothInput']) == 0:
      errorMessage = 'No MAMMOTH input file specified! Please prepend "MAMMOTHInput|" to the driver App input \n'
      errorMessage += 'file\'s type in the RAVEN input file.'
      raise IOError(errorMessage)
    # Multiple mammoth files are found
    elif len(inputDict['MammothInput']) > 1:
      raise IOError('Multiple MAMMOTH input files are provided! Please limit the number of input files to one.')
    # Mammoth input found, but driverAppInput is not in the allowedDriverAppInput list
    elif len(inputDict['MammothInput']) == 1 and inputDict['DriverAppInput'] not in allowedDriverAppInput:
      errorMessage = 'A MAMMOTH input file was specified, but the driver app is not currently supported by this\n'
      errorMessage += 'interface. The MAMMOTH input file can only be specified as one of the following types:'
      for goodDriverAppInput in allowedDriverAppInput:
        errorMessage += '\nMAMMOTHInput|' + goodDriverAppInput
      raise IOError(errorMessage)
    return inputDict

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      Generate a command to run Mammoth using an input with sampled variables
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
    mammothInput = inputDict['MammothInput']
    mooseCommand, mooseOut = self.MooseInterface.generateCommand(mammothInput,executable,clargs,fargs)
    returnCommand = mooseCommand, mooseOut
    return returnCommand

  def createNewInput(self, currentInputFiles, origInputFiles, samplerType, **Kwargs):
    """
      Generates new perturbed input files for Mammoth and associated Moose based applications.
      @ In, currentInputFiles, list,  list of current input files
      @ In, origInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dict, dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
        where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of new input files (modified or not)
    """
    #split up sampledAars in Kwargs between Bison, Rattlesnake and Relap-7
    bisonArgs = copy.deepcopy(Kwargs)
    bisonArgs['SampledVars'] = {}
    perturbBison = False
    rattlesnakeArgs = copy.deepcopy(Kwargs)
    rattlesnakeArgs['SampledVars'] = {}
    perturbRattlesnake = False
    relap7Args = copy.deepcopy(Kwargs)
    relap7Args['SampledVars'] = {}
    perturbRelap7 = False
    foundAlias = False
    for varName,varValue in Kwargs['SampledVars'].items():
      # get the variable's full name
      if len(varName.split('@')) == 2:
        appName = varName.split('@')[0].lower()
        baseVarName = varName.split('@')[-1]
      elif len(varName.split('@')) == 1:
        appName = None
        baseVarName = varName
      else:
        errorMessage = 'Variable names passed to the MAMMOTH Code Interface must either\n'
        errorMessage += 'specifiy to which App input they belong by prepending the App\'s name\n'
        errorMessage += 'followed by "@" to the base variable\'s name or alias or have no App\n'
        errorMessage += 'name to signify a passthrough variable. Please check that\n'
        errorMessage += varName+'\n'
        errorMessage += 'fits within this syntax specification.'
        raise IOError(errorMessage)
      # Identify which app's input the variable goes into and separate appArgs
      if appName == 'bison':
        bisonArgs['SampledVars'][baseVarName] = varValue
        perturbBison = True
      elif appName == 'rattlesnake':
        rattlesnakeArgs['SampledVars'][baseVarName] = varValue
        perturbRattlesnake = True
      elif appName == 'relap7':
        relap7Args['SampledVars'][baseVarName] = varValue
        perturbRelap7 = True
      elif appName == None:
        # It's a dummy variable. Doesn't need to be added to any argument lists, just continue.
        pass
      else:
        errorMessage = appName+' is not an App supported by the MAMMOTH Code Interface!\n'
        errorMessage += 'Please specify a supported App in which to send \n'
        errorMessage += baseVarName+'\n'
        errorMessage += 'or add the desired App to the MAMMOTH Code Interface.'
        raise IOError(errorMessage)
    # Check if the user wants to perturb yak xs libraries
    for inputFile in currentInputFiles:
      fileType = inputFile.getType()
      if fileType.strip().lower() == "yakxsaliasinput":
        foundAlias = True
        break
      elif fileType.strip().lower() == "instantxsaliasinput":
        foundAlias = True
        break
    inputDicts = self.findInps(currentInputFiles)

    # Bison Interface
    if perturbBison:
      bisonInps = inputDicts['BisonInput']
      bisonInTypes = []
      for bisonIn in bisonInps:
        bisonInTypes.append(bisonIn.getType().strip().lower().split('|')[-1])
      if 'bisoninput' not in bisonInTypes:
        errorMessage = 'Variable(s):\n'
        for bisonVarName in bisonArgs['SampledVars'].keys():
          errorMessage += bisonVarName + '\n'
        errorMessage += 'are specified as Bison parameters, but no Bison input file is listed!'
        raise IOError(errorMessage)
      elif bisonInTypes.count('bisoninput') > 1:
        errorMessage = 'Multiple Bison input files specified! This interface currently only\n'
        errorMessage += 'supports one input for each App utilized.'
        raise IOError(errorMessage)
      origBisonInps = origInputFiles[currentInputFiles.index(bisonInps[0])]
      bisonInps = self.BisonInterface.createNewInput(bisonInps,[origBisonInps],samplerType,**bisonArgs)

    # Rattlesnake Interface
    if perturbRattlesnake or foundAlias:
      rattlesnakeInps = inputDicts['RattlesnakeInput']
      rattlesnakeInTypes = []
      for rattlesnakeIn in rattlesnakeInps:
        rattlesnakeInTypes.append(rattlesnakeIn.getType().strip().lower().split('|')[-1])
      if 'rattlesnakeinput' not in rattlesnakeInTypes:
        errorMessage = 'Variable(s):\n'
        for rattlesnakeVarName in rattlesnakeArgs['SampledVars'].keys():
          errorMessage += rattlesnakeVarName + '\n'
        errorMessage += 'are specified as Rattlesnake parameters, but no Rattlesnake input file is listed!'
        raise IOError(errorMessage)
      elif rattlesnakeInTypes.count('rattlesnakeinput') > 1:
        errorMessage = 'Multiple Rattlesnake input files specified! This interface currently only\n'
        errorMessage += 'supports one input for each App utilized.'
        raise IOError(errorMessage)
      origRattlesnakeInps = origInputFiles[currentInputFiles.index(rattlesnakeInps[0])]
      rattlesnakeInps = self.RattlesnakeInterface.createNewInput(rattlesnakeInps,
                              [origRattlesnakeInps],samplerType,**rattlesnakeArgs)

    # Relap7 Interface
    if perturbRelap7:
      relap7Inps = inputDicts['Relap7Input']
      relap7InTypes = []
      for relap7In in relap7Inps:
        relap7InTypes.append(relap7In.getType().strip().lower().split('|')[-1])
      if 'relap7input' not in relap7InTypes:
        errorMessage = 'Variable(s):\n'
        for relap7VarName in relap7Args['SampledVars'].keys():
          errorMessage += relap7VarName + '\n'
        errorMessage += 'are specified as Relap7 parameters, but no Relap7 input file is listed!'
        raise IOError(errorMessage)
      elif relap7InTypes.count('relap7input') > 1:
        errorMessage = 'Multiple Relap7 input files specified! This interface currently only\n'
        errorMessage += 'supports one input for each App utilized.'
        raise IOError(errorMessage)
      origRelap7Inps = origInputFiles[currentInputFiles.index(relap7Inps[0])]
      relap7Inps = self.Relap7Interface.createNewInput(relap7Inps,[origRelap7Inps],samplerType,**relap7Args)

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
