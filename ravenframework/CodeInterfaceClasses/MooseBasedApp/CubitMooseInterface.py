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
Created July 14, 2015

@author: talbpaul
"""
import copy
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..MooseBasedApp.MooseBasedAppInterface import MooseBasedApp
from .CubitInterface         import Cubit

class CubitMoose(CodeInterfaceBase): #MooseBasedAppInterface,CubitInterface):
  """
    This class provides the means to generate a stochastic-input-based mesh using the MOOSE
    standard Cubit python script in addition to uncertain inputs for the MOOSE app.
    @ In, None
    @ Out, None
  """

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.MooseInterface = MooseBasedApp()
    self.CubitInterface = Cubit()
    self.MooseInterface.addDefaultExtension()
    self.CubitInterface.addDefaultExtension()

  def findInps(self,inputFiles):
    """
      Locates the input files for Moose, Cubit
      @ In, inputFiles, list of Files objects
      @ Out, (File,File), Moose and Cubit input files
    """
    foundMooseInp = False
    foundCubitInp = False
    for inFile in inputFiles:
      if inFile.getType() == 'MooseInput':
        foundMooseInp = True
        mooseInp = inFile
      elif inFile.getType() == 'CubitInput':
        foundCubitInp = True
        cubitInp = inFile
    if not foundMooseInp:
      raise IOError('None of the input Files has the type "MooseInput"! CubitMoose interface requires one.')
    if not foundCubitInp:
      raise IOError('None of the input Files has the type "CubitInput"! CubitMoose interface requires one.')
    return mooseInp,cubitInp

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None, preExec = None):
    """
      Generate a multi-line command that runs both the Cubit mesh generator and then the desired MOOSE run.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (lenght of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    if preExec is None:
      raise IOError('No preExec listed in input!  Use MooseBasedAppInterface if mesh is not perturbed.  Exiting...')
    mooseInp,cubitInp = self.findInps(inputFiles)
    #get the cubit part
    cubitCommand,_ = self.CubitInterface.generateCommand([cubitInp],preExec,clargs,fargs)
    #get the moose part
    mooseCommand,mooseOut = self.MooseInterface.generateCommand([mooseInp],executable,clargs,fargs)
    #combine them
    returnCommand = cubitCommand + mooseCommand, mooseOut #can only send one...#(cubitOut,mooseOut)
    print('ExecutionCommand:',returnCommand[0],'\n')
    return returnCommand

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """
      Generates new perturbed input files.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    mooseInp,cubitInp = self.findInps(currentInputFiles)
    origMooseInp = origInputFiles[currentInputFiles.index(mooseInp)]
    origCubitInp = origInputFiles[currentInputFiles.index(cubitInp)]
    #split up sampledvars in kwargs between moose and Cubit script
    #  NOTE This works by checking the '@' split for the keyword Cubit at first!
    margs = copy.deepcopy(Kwargs)
    cargs = copy.deepcopy(Kwargs)
    for vname,var in Kwargs['SampledVars'].items():
      fullName = vname
      if fullName.split('@')[0]=='Cubit':
        del margs['SampledVars'][vname]
      else:
        del cargs['SampledVars'][vname]
    # Generate new cubit input files and extract exodus file name to add to SampledVars going to moose
    newCubitInputs = self.CubitInterface.createNewInput([cubitInp],[origCubitInp],samplerType,**cargs)
    margs['SampledVars']['Mesh|file'] = 'mesh~'+newCubitInputs[0].getBase()+'.e'#"".join(os.path.split(newCubitInputs[0])[1].split('.')[:-1])+'.e'
    newMooseInputs = self.MooseInterface.createNewInput([mooseInp],[origMooseInp],samplerType,**margs)
    #  if order doesn't matter, can loop through and check for type else copy directly
    newMooseInp,newCubitInp = self.findInps(currentInputFiles)
    newMooseInp.setAbsFile(newMooseInputs[0].getAbsFile())
    newCubitInp.setAbsFile(newCubitInputs[0].getAbsFile())
    return currentInputFiles

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Calls finalizeCodeOutput from Cubit Interface to clean up files
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method. (not in this case)
    """
    self.CubitInterface.finalizeCodeOutput(command, output, workingDir)
