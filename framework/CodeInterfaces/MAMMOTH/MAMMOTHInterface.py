from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
from subprocess import Popen
from CodeInterfaceBaseClass import CodeInterfaceBase
from MooseBasedAppInterface import MooseBasedAppInterface
from RattlesnakeInterface   import RattlesnakeInterface
from BisonAndMeshInterface  import BisonAndMeshInterface


class MAMMOTHInterface(CodeInterfaceBase):
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
    self.MooseInterface = MooseBasedAppInterface() #used to perturb MAMMOTH input files
    self.MooseInterface.addDefaultExtension()
    self.RattlesnakeInterface  = RattlesnakeInterface() #used to perturb Rattlesnake and Yak input files
    self.BisonInterface = MooseBasedAppInterface() #used to perturb Bison input files

  def findInps(self,inputFiles):
    """
      Locates the input files required by MAMMOTH
      @ In, inputFiles, list, list of Files objects
      @ Out, inputDict, dict, dictionary containing MAMMOTH required input files
    """
    inputDict = {}
    inputDict['FoundMammothInput'] = False
    inputDict['FoundBisonInput'] = False
    inputDict['FoundRattlesnakeInput'] = False
    inputDict['FoundRelap-7Input'] = False #used in the future
    rattlesnakeInput = []
    mammothInput = []
    bisonInput = []
    relap7Input = []
    for inputFile in inputFiles:
      if inputFile.getType().lower() = "mammothinput":
        inputDict['FoundMammothInput'] = True
        mammothInput.append(inputFile)
      elif inputFile.getType().lower() = "bisoninput":
        inputDict['FoundBisonInput'] = True
        bisonInput.append(inputFile)
      elif inputFile.getType().lower() = "replap-7input":
        inputDict['FoundRelap-7Input'] = True
        relap7Input.append(inputFile)
      elif inputFile.getType().lower() = "rattlesnakeinput":
        inputDict['FoundRattlesnakeInput'] = True
        rattlesnakeInput.append(inputFile)

    if inputDict['FoundBisonInput']: inputDict['BisonInput'] = bisonInput
    if inputDict['FoundRelap-7Input']: inputDict['Relap-7Input'] = relap7Input
    if inputDict['FoundRattlesnakeInput']: inputDict['RattlesnakeInput'] = rattlesnakeInput
    if not inputDict['FoundMammothInput']:
      raise IOError('None of the input files has the type "MammothInput"! This is required by MAMMOTH interface.')
    else:
      inputDict['MammothInput'] = mammothInput
    return inputDict

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None):
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
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the
        code (string), returnCommand[1] is the name of the output root
    """
    inputDict = self.findInps(inputFiles)
    mammothInput = inputDict['MammothInput']
    if len(mammothInput) != 1:
      raise IOError('The user should only provide one mammoth input file, but found ' + str(len(mammothInput)) + '!')
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
    #split up sampledAars in Kwargs between MAMMOTH, Rattlesnake, Bison and Relap-7
    mammothArgs = copy.deepcopy(Kwargs)
    rattlesnakeArgs = copy.deepcopy(Kwargs)
    bisonArgs = copy.deepcopy(Kwargs)
    perturbMammoth = False
    perturbRattlesnake = False
    perturbBison = False
    for varName,varValue in Kwargs['SampledVars'].items():
      if 'alias' in Kwargs.keys():
        fullName = Kwargs['alias'].get(varName,varName)
      else:
        fullName = varName
      if fullName.split('|')[0].lower() == 'mammoth':
        del rattlesnakeArgs['SampledVars'][varName]
        del bisonArgs['SampledVars'][varName]
        perturbMammoth = True
        if varName in Kwargs['alias']:
          del rattlesnakeArgs['alias'][varName]
          del bisonArgs['alias'][varName]
      elif fullName.split('|')[0].lower() == 'rattlesnake':
        del mammothArgs['SampledVars'][varName]
        del bisonArgs['SampledVars'][varName]
        perturbRattlesnake = True
        if varName in Kwargs['alias']:
          del mammothArgs['alias'][varName]
          del bisonArgs['alias'][varName]
      elif fullName.split('|')[0].lower() == 'bison':
        del mammothArgs['SampledVars'][varName]
        del rattlesnakeArgs['SampledVars'][varName]
        perturbBison = True
        if varName in Kwargs['alias']:
          del mammothArgs['alias'][varName]
          del rattlesnakeArgs['alias'][varName]
    perturbYakXS = False
    for inputFile in currentInputFiles:
      if inputFile.getType().lower() == "yakxsaliasinput":
        perturbYakXS = True
        break
    #Rattlesnake interface
    if perturbYakXS or perturbRattlesnake:
      newUpdatedInputs = self.rattlesnakeInterface.createNewInput(currentInputFiles,origInputFiles,samplerType,**rattlesnakeArgs)
    else:
      newUpdatedInputs = copy.deepcopy(currentInputFiles)
    inputDicts = self.findInps(newUpdatedInputs)
    #Bison interface
    if perturbBison:
      if inputDicts['FoundBisonInput']:
        bisonInp = inputDicts['BisonInput']
        #FIXME this need to be changed if MAMMOTH can accept multiple Bision input files
        if len(bisonInp) != 1: raise IOError('Multiple Bison input files are found!')
        origBisonInp = origInputFiles[currentInputFiles.index(bisonInp)]
        newBisonInp = self.BisonInterface.createNewInput(bisonInp,[origBisonInp],samplerType,**bisonArgs)
      else:
        raise IOError('The user tried to perturb Bison input files, but no Bison input file is found!')
    #perturb Mammoth input files
    if perturbMammoth:
      mammothInp = inputDicts['MammothInput']
      if len(mammothInp) != 1: raise IOError('Multiple Mammoth input files are found!')
      origMammothInp = origInputFiles[currentInputFiles.index(mammothInp)]
      newMammothInp = self.MooseInterface.createNewInput(mammothInp,origMamothInp,samplerType,**mamothArgs)
    else:
      newMammothInp = copy.deepcopy(inputDicts['MammothInput'])
    #replace the input files names inside Mammoth input
    if perturbBison:
      self._updateMammothInputs(newMammothInp,bisonInp,newBisonInp)
      inputDicts['BisonInput'].setAbsFile(newBisonInp.getAbsFile())
    if perturbRattlesnake:
      newRattlesnakeInp = inputDicts['RattlesnakeInput']
      origRattlesnakeInp = origInputFiles[currentInputFiles.index(newRattlesnake)]
      self._updateMammothInputs(newMammothInp,origRattlesnakeInp,newRattlesnakeInp)
    inputDicts['MammothInput'].setAbsFile(newMammothInp.getAbsFile())

    return newUpdatedInputs

  def _updateMammothInputs(self,mammothInps, oldInps, newInps):
    """
      Update the rattlesnake inputs with the updated cross section library names
      @ In, mammothInps, list, list of MAMMOTH input files
      @ In, oldInps, list, list of old input files referenced via MAMMOTH
      @ In, newInps, list, list of new input files will be referenced via MAMMOTH
      @ Out, None.
    """
    for fileInp in mammothInps:
      if not os.path.isfile(fileInp.getAbsFile()):
        raise IOError("Error on opening file, not a regular file: " + fileInp.getFilename())
      fileInp.open('r')
      fileData = fileInp.read()
      for fileIndex, oldInp in enumerate(oldInps):
        oldName = oldInp.getFilename()
        newName = newInps[fileIndex].getFilename()
        newData = fileData.replace(oldName,newName)
      fileInp.close()
      fileInp.open('w')
      fileInp.write(newData)
      fileInp.close()

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
