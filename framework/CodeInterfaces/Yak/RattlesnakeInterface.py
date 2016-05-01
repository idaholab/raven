from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
from subprocess import Popen
from CodeInterfaceBaseClass import CodeInterfaceBase
from MooseBasedAppInterface import MooseBasedAppInterface


class RattlesnakeInterface(CodeInterfaceBase):
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
    self.MooseInterface = MooseBasedAppInterface()
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
    yakInput = []
    rattlesnakeInput = []
    aliasInput = []
    for inputFile in inputFiles:
      if inputFile.getType().lower() == "yakxsinput":
        inputDict['FoundYakXSInput'] = True
        yakInput.append(inputFile)
      elif inputFile.getType().lower() == "rattlesnakeinput":
        inputDict['FoundRattlesnakeInput'] = True
        rattlesnakeInput.append(inputFile)
      elif inputFile.getType().lower() == "yakxsaliasinput":
        inputDict['FoundYakXSAliasInput'] = True
        aliasInput.append(inputFile)
    if inputDict['FoundYakXSInput']: inputDict['YakXSInput'] = yakInput
    if inputDict['FoundRattlesnakeInput']: inputDict['RattlesnakeInput'] = rattlesnakeInput
    if inputDict['FoundYakXSAliasInput']: inputDict['YakAliasInput'] =  aliasInput
    if not inputDict['FoundRattlesnakeInput']: raise IOError('None of the input files has the type "RattlesnakeInput"! This is required by Rattlesnake interface.')
    return inputDict

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None):
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
    #perturb the Yak multigroup library
    import YakMultigroupLibraryParser
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
      newYakInputs = copy.deepcopy(yakInputs)
      #write the perturbed files
      parser.writeNewInput(newYakInputs,**Kwargs)
    #Moose based app interface
    origRattlesnakeInputs = copy.deepcopy(rattlesnakeInputs)
    newMooseInputs = self.MooseInterface.createNewInput(rattlesnakeInputs,origRattlesnakeInputs,samplerType,**Kwargs)
    #replace the library name in the rattlesnake inputs
    if foundAlias and foundXS:
      self._updateRattlesnakeInputs(newMooseInputs,yakInputs,newYakInputs)
    #replace old with new perturbed files.
    for f in currentInputFiles:
      if f.isOpen(): f.close()
    newInputFiles = copy.deepcopy(currentInputFiles)
    #replace old with new perturbed files, in place
    newInputDict = self.findInps(newInputFiles)
    if foundXS and foundAlias:
      newYaks = newInputDict['YakXSInput']
      for fileIndex, newYakFile in enumerate(newYaks):
        newYakFile.setAbsFile(newYakInputs[fileIndex].getAbsFile())
    newRattlesnakes = newInputDict['RattlesnakeInput']
    for fileIndex, newFile in enumerate(newRattlesnakes):
      newFile.setAbsFile(newMooseInputs[fileIndex].getAbsFile())
    return newInputFiles

  def _updateRattlesnakeInputs(self,mooseInps, yakInps, newYakInps):
    """
      Update the rattlesnake inputs with the updated cross section library names
      @ In, mooseInps, list, list of rattlesnake input files
      @ In, yakInps, list, list of old yak cross section files
      @ In, newYakInps, list, list of new generated yak cross section files
      @ Out, None.
    """
    for mooseInp in mooseInps:
      if not os.path.isfile(mooseInp.getAbsFile()):
        raise IOError("Error on opening file, not a regular file: " + mooseInp.getFilename())
      mooseInp.open('r')
      mooseFileData = mooseInp.read()
      newData = copy.deepcopy(mooseFileData)
      mooseInp.close()
      for fileIndex, yakInp in enumerate(yakInps):
        oldYakName = yakInp.getFilename()
        newYakName = newYakInps[fileIndex].getFilename()
        newData = newData.replace(oldYakName,newYakName)
      mooseInp.open('w')
      mooseInp.write(newData)
      mooseInp.close()

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
