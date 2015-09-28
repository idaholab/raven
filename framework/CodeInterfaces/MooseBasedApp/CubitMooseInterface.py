"""
Created July 14, 2015

@author: talbpaul
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import Files
from CodeInterfaceBaseClass import CodeInterfaceBase
from MooseBasedAppInterface import MooseBasedAppInterface
from CubitInterface         import CubitInterface

class CubitMooseInterface(CodeInterfaceBase): #MooseBasedAppInterface,CubitInterface):
  """This class provides the means to generate a stochastic-input-based mesh using the MOOSE
     standard Cubit python script in addition to uncertain inputs for the MOOSE app."""

  def __init__(self):
    CodeInterfaceBase.__init__(self)
    self.MooseInterface = MooseBasedAppInterface()
    self.CubitInterface = CubitInterface()
    self.MooseInterface.addDefaultExtension()
    self.CubitInterface.addDefaultExtension()

  def findInps(self,inputFiles):
    """Locates the input files for Moose, Cubit
    @ In, inputFiles, list of Files objects
    @Out, (File,File), Moose and Cubit input files
    """
    foundMooseInp = False
    foundCubitInp = False
    for index,inFile in enumerate(inputFiles):
      if inFile.getType() == 'MooseInput':
        foundMooseInp = True
        mooseInp = inFile
      elif inFile.getType() == 'CubitInput':
        foundCubitInp = True
        cubitInp = inFile
    if not foundMooseInp: raise IOError('None of the input Files has the type "MooseInput"! CubitMoose interface requires one.')
    if not foundCubitInp: raise IOError('None of the input Files has the type "CubitInput"! CubitMoose interface requires one.')
    return mooseInp,cubitInp

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None, preexec = None):
    """Generate a multi-line command that runs both the Cubit mesh generator and then the
       desired MOOSE run.
    See base class.  Collects all the clargs and the executable to produce the command-line call.
    Returns tuple of commands and base file name for run.
    Commands are a list of tuples, indicating parallel/serial and the execution command to use.
    @ In, inputFiles, the input files to be used for the run
    @ In, executable, the executable to be run
    @ In, clargs, command-line arguments to be used
    @ In, fargs, in-file changes to be made
    @Out, tuple( list(tuple(serial/parallel, exec_command)), outFileRoot string)
    """
    if preexec is None:
      raise IOError('No preexec listed in input!  Use MooseBasedAppInterface if mesh is not perturbed.  Exiting...')
    mooseInp,cubitInp = self.findInps(inputFiles)
    #get the cubit part
    cubitCommand,cubitOut = self.CubitInterface.generateCommand([cubitInp],preexec,clargs,fargs)
    #get the moose part
    mooseCommand,mooseOut = self.MooseInterface.generateCommand([mooseInp],executable,clargs,fargs)
    #combine them
    #executeCommand = ' && '.join([cubitCommand,mooseCommand])
    executeCommand = [cubitCommand[0],
                      mooseCommand[0]]
    print('ExecutionCommand:',executeCommand,'\n')
    return executeCommand,mooseOut #can only send one...#(cubitOut,mooseOut)

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """Generates new perturbed input files.
    @ In, currrentInputFiles, list of Files objects, most recently perturbed files
    @ In, origInputFiles, the template input files originally shown
    @ In, samplerType, the sampler type used (not used in this algorithm)
    @ In, Kwargs, dictionary of key-val pairs
    @Out, list of perturbed Files
    """
    mooseInp,cubitInp = self.findInps(currentInputFiles)
    origMooseInp = origInputFiles[currentInputFiles.index(mooseInp)]
    origCubitInp = origInputFiles[currentInputFiles.index(cubitInp)]
    #split up sampledvars in kwargs between moose and Cubit script
    #  NOTE This works by checking the pipe split for the keyword Cubit at first!
    margs = copy.deepcopy(Kwargs)
    cargs = copy.deepcopy(Kwargs)
    for vname,var in Kwargs['SampledVars'].items():
      if 'alias' in Kwargs.keys():
        fullname = Kwargs['alias'].get(vname,vname)
      if fullname.split('|')[0]=='Cubit':
        del margs['SampledVars'][vname]
        if 'alias' in Kwargs.keys():
          if vname in Kwargs['alias']:
            del margs['alias'][vname]
      else:
        del cargs['SampledVars'][vname]
        if 'alias' in Kwargs.keys():
          if vname in Kwargs['alias']:
            del cargs['alias'][vname]
    # Generate new cubit input files and extract exodus file name to add to SampledVars going to moose
    newCubitInputs = self.CubitInterface.createNewInput([cubitInp],[origCubitInp],samplerType,**cargs)
    margs['SampledVars']['Mesh|file'] = 'mesh~'+newCubitInputs[0].getBase()+'.e'#"".join(os.path.split(newCubitInputs[0])[1].split('.')[:-1])+'.e'
    newMooseInputs = self.MooseInterface.createNewInput([mooseInp],[origMooseInp],samplerType,**margs)
    #make carbon copy of original input files
    for f in currentInputFiles:
      if f.isOpen(): f.close()
    newInputFiles = copy.deepcopy(currentInputFiles)
    #replace old with new perturbed files, in place TODO is that necessary, really? Does order matter?
    #  if order doesn't matter, can loop through and check for type else copy directly
    newMooseInp,newCubitInp = self.findInps(newInputFiles)
    newMooseInp.setAbsFile(newMooseInputs[0].getAbsFile())
    newCubitInp.setAbsFile(newCubitInputs[0].getAbsFile())
    return newInputFiles

  def finalizeCodeOutput(self, command, output, workingDir):
    """Calls finalizeCodeOutput from Cubit Interface to clean up files
       @ In, command, (string), command used to run the just ended job
       @ In, output, (string), the Output name root
       @ In, workingDir, (string), the current working directory
       @Out, None
    """
    self.CubitInterface.finalizeCodeOutput(command, output, workingDir)
