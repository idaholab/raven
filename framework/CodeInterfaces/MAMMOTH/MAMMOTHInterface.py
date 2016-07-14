from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
from subprocess import Popen
from CodeInterfaceBaseClass import CodeInterfaceBase
from MooseBasedAppInterface import MooseBasedApp
from RattlesnakeInterface   import Rattlesnake

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
    self.MooseInterface = MooseBasedApp() #used to perturb MAMMOTH input files
    self.MooseInterface.addDefaultExtension()
    self.RattlesnakeInterface  = Rattlesnake() #used to perturb Rattlesnake and Yak input files
    self.BisonInterface = MooseBasedApp() #used to perturb Bison input files
    self.BisonInterface.addDefaultExtension()

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
    rattlesnakeInput = []
    mammothInput = []
    bisonInput = []
    for inputFile in inputFiles:
      fileType = inputFile.getType()
      if fileType.strip().lower() == "mammothinput|rattlesnakeinput":
        inputDict['FoundMammothInput'] = True
        inputDict['FoundRattlesnakeInput'] = True
        mammothInput.append(inputFile)
        rattlesnakeInput.append(inputFile)
      elif fileType.strip().lower() == "bisoninput":
        inputDict['FoundBisonInput'] = True
        bisonInput.append(inputFile)
    if inputDict['FoundBisonInput']: inputDict['BisonInput'] = bisonInput
    if inputDict['FoundRattlesnakeInput']: inputDict['RattlesnakeInput'] = rattlesnakeInput
    if not inputDict['FoundMammothInput']:
      errorMessage = 'This interface is only support the calculations via Rattlesnake coupled with Bison through MAMMOTH. \n'
      errorMessage = errorMessage + 'The type of MAMMOTH input file should be MammothInput|RattlesnakeInput. \n'
      errorMessage = errorMessage + 'But, none of the input files has this type. This is required by MAMMOTH interface.'
      raise IOError(errorMessage)
    elif len(mammothInput) != 1:
      raise IOError('Multiple MAMMOTH input files are provided! Please limit the number of input files to one!')
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
    #split up sampledAars in Kwargs between Mammoth:Rattlesnake and Bison
    rattlesnakeArgs = copy.deepcopy(Kwargs)
    bisonArgs = copy.deepcopy(Kwargs)
    perturbRattlesnake = False
    perturbBison = False
    for varName,varValue in Kwargs['SampledVars'].items():
      if 'alias' in Kwargs.keys():
        fullName = Kwargs['alias'].get(varName,varName)
      else:
        fullName = varName
      if fullName.split(':')[-1].lower() == 'rattlesnake':
        del bisonArgs['SampledVars'][varName]
        perturbRattlesnake = True
        if 'alias' in Kwargs.keys():
          if varName in Kwargs['alias']:
            del bisonArgs['alias'][varName]
      elif fullName.split(':')[-1].lower() == 'bison':
        del rattlesnakeArgs['SampledVars'][varName]
        perturbBison = True
        if 'alias' in Kwargs.keys():
          if varName in Kwargs['alias']: del rattlesnakeArgs['alias'][varName]
    #reset the type
    for inputFile in currentInputFiles:
      fileType = inputFile.getType()
      if fileType.strip().lower() == "mammothinput|rattlesnakeinput":
        inputFile.subtype = "rattlesnakeinput"
        break
    #check if the user want to perturb yak xs libraries
    for inputFile in currentInputFiles:
      fileType = inputFile.getType()
      if fileType.strip().lower() == "yakxsaliasinput":
        foundAlias= True
        break
    #Rattlesnake interface
    #if perturbRattlesnake or foundAlias:
    currentInputFiles = self.RattlesnakeInterface.createNewInput(currentInputFiles,origInputFiles,samplerType,**rattlesnakeArgs)
    #reset the type
    for inputFile in currentInputFiles:
      fileType = inputFile.getType()
      if fileType.strip().lower() == "rattlesnakeinput":
        inputFile.subtype = "mammothinput|rattlesnakeinput"
        break
    inputDicts = self.findInps(currentInputFiles)
    #Bison interface
    if perturbBison:
      if inputDicts['FoundBisonInput']:
        bisonInp = inputDicts['BisonInput']
        #FIXME this need to be changed if MAMMOTH can accept multiple Bision input files
        if len(bisonInp) != 1: raise IOError('Multiple Bison input files are found!')
        origBisonInp = origInputFiles[currentInputFiles.index(bisonInp[0])]
        bisonInp = self.BisonInterface.createNewInput(bisonInp,[origBisonInp],samplerType,**bisonArgs)
      else:
        raise IOError('The user tried to perturb Bison input files, but no Bison input file is found!')
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
