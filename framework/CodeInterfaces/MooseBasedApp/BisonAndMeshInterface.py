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
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import copy
from CodeInterfaceBaseClass   import CodeInterfaceBase
from MooseBasedAppInterface   import MooseBasedApp
from BisonMeshScriptInterface import BisonMeshScript

class BisonAndMesh(CodeInterfaceBase):#MooseBasedAppInterface,BisonMeshScriptInterface):
  """
    This class provides the means to generate a stochastic-input-based mesh using the MOOSE
    standard Cubit python script in addition to uncertain inputs for the MOOSE app.
  """

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.MooseInterface     = MooseBasedApp()
    self.BisonMeshInterface = BisonMeshScript()
    self.MooseInterface    .addDefaultExtension()
    self.BisonMeshInterface.addDefaultExtension()

  def findInps(self,inputFiles):
    """
      Locates the input files for Moose, Cubit
      @ In, inputFiles, list, list of Files objects
      @ Out, (mooseInp,cubitInp), tuple, tuple containing Moose and Cubit input files
    """
    foundMooseInp = False
    foundCubitInp = False
    for inFile in inputFiles:
      if inFile.getType() == 'MooseInput':
        foundMooseInp = True
        mooseInp = inFile
      elif inFile.getType() == 'BisonMeshInput':
        foundCubitInp = True
        cubitInp = inFile
    if not foundMooseInp:
      raise IOError('None of the input Files has the type "MooseInput"! BisonAndMesh interface requires one.')
    if not foundCubitInp:
      raise IOError('None of the input Files has the type "BisonMeshInput"! BisonAndMesh interface requires one.')
    return mooseInp,cubitInp

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
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
    cubitCommand,cubitOut = self.BisonMeshInterface.generateCommand([cubitInp],preExec,clargs,fargs,preExec)
    #get the moose part
    mooseCommand,mooseOut = self.MooseInterface.generateCommand([mooseInp],executable,clargs,fargs,preExec)
    #combine them
    returnCommand = cubitCommand + mooseCommand, mooseOut #can only send one...#(cubitOut,mooseOut)
    print('Execution commands from JobHandler:')
    for r,c in returnCommand[0]:
      print('  in',r+':',c)
    return returnCommand

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """
      Generates new perturbed input files.
      This method is used to generate an input based on the information passed in.
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
    newCubitInputs = self.BisonMeshInterface.createNewInput([cubitInp],[origCubitInp],samplerType,**cargs)
    margs['SampledVars']['Mesh|file'] = 'mesh~'+newCubitInputs[0].getBase()+'.e'
    newMooseInputs = self.MooseInterface.createNewInput([mooseInp],[origMooseInp],samplerType,**margs)
    #make carbon copy of original input files
    for f in currentInputFiles:
      if f.isOpen():
        f.close()
    #replace old with new perturbed files, in place
    newMooseInp,newCubitInp = self.findInps(currentInputFiles)
    newMooseInp.setAbsFile(newMooseInputs[0].getAbsFile())
    newCubitInp.setAbsFile(newCubitInputs[0].getAbsFile())
    return currentInputFiles

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Calls finalizeCodeOutput from Bison Mesh Script Interface to clean up files
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method (not present in this case)
    """
    self.BisonMeshInterface.finalizeCodeOutput(command, output, workingDir)
