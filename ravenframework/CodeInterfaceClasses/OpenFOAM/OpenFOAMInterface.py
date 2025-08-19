# Copyright 2025 Battelle Energy Alliance, LLC
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
  Created on May 2, 2025
  @author: Andrea Alfonsi
"""

import os
import shutil
import pathlib

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..Generic.GenericCodeInterface import GenericParser
from ravenframework.Files import UserGenerated
from ravenframework.utils import utils
from . import OpenFOAMoutputParser as op


class OpenFOAM(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to OpenFOAM framework
    (https://www.openfoam.com)

    The name of this class represents the type in the RAVEN input file
    e.g.
    <Models>
      <Code name="myName" subType="OpenFOAM">
      ...
      </Code>
      ...
    </Models>

  """
  def __init__(self):
    """
      Initialize some variables
    """
    super().__init__()
    # check if foamlib is available, raise error it otherwise
    try:
      pyvista = __import__("pyvista")
    except ImportError:
      raise ImportError("python library 'pyvista' not found and OpenFOAM Interface has been invoked. "
                        "Install foamlib through pip/conda (conda-forge) or invoke RAVEN installation procedure (i.e. establish_conda_env.sh script) "
                        "with the additional command line option '--code-interface-deps'. "
                        "See User Manual for additiona details!")
    self._pyvista = pyvista
    self._caseDirectoryName = None
    self._caseFileName = None
    self.variables = None
    self.writeCentroids = False
    self.directoriesPerturbableInputs = None # eg. ['0.orig', 'constant', 'system', '0']

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    super()._readMoreXML(xmlNode)
    outputVariables = xmlNode.find("outputVariables")
    if outputVariables is not None:
      self.variables = [el.strip() for el in outputVariables.text.split(",")]
    writeCentroids = xmlNode.find("writeCentroids")
    if writeCentroids is not None:
      self.writeCentroids = utils.interpretBoolean(writeCentroids.text)
    directoriesPerturbableInputs = xmlNode.find("directoriesPerturbableInputs")
    if directoriesPerturbableInputs is None:
      raise IOError("ERROR OpenFOAM: list of perturbable directories must be provided in node <directoriesPerturbableInputs>")
    self.directoriesPerturbableInputs = [el.strip() for el in directoriesPerturbableInputs.text.split(",")]

  def findInputFile(self, inputFiles):
    """
      Method to find the OpenFOAM input file (that for OpenFOAM in this interface is the 'Allrun')
      @ In, inputFiles, list, list of the original input files
      @ Out, inputFile, FileType, the input file
    """
    found = False
    # Find the first file in the inputFiles that is an OpenFOAM compatible, which is what we need to work with.
    for inputFile in inputFiles:
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('ERROR OpenFOAM: No ".foam" file has been found. OpenFOAM input must be of type "openfoam" and have the extension ".foam". Got: '+' '.join(inputFiles))
    return inputFile

  def getInputExtension(self):
    """
      Return input extensions
      for OpenFOAM, not extensions are used
      @ In, None
      @ Out, getInputExtension, tuple(str), the ext of the code input file (.foam here)
    """
    return ("foam",)

  def _scanDirectoryAndListFiles(self, caseDir, directoryToScan = None):
    """
      Method to scan the directory 'caseDir' and create a list of files
      @ In, caseDir, str, the case directory path
      @ In, directoryToScan, list, optional, directory names to scan (eg. constant, system, etc.). If None, all directories will be scanned.
      @ Out, inputFiles, list, list of input files
    """
    from binaryornot.check import is_binary
    inputFiles = []
    for root, _, files in os.walk(caseDir):
      if directoryToScan is not None and os.path.split(root)[-1].strip() not in directoryToScan and root != caseDir:
        continue
      for file in files:
        if is_binary(os.path.join(root, file)):
          # we skip binary files
          continue
        fileObj = UserGenerated()
        fileObj.type = file.split(".")[0]
        fileObj.subDirectory = root.replace(caseDir, "")
        fileObj.setAbsFile(os.path.join(root, file))
        fileObj.alias = fileObj.getFilename()
        inputFiles.append(fileObj)
    return inputFiles

  def createNewInput(self, currentInputFiles, origInputFiles, samplerType, **Kwargs):
    """
      Generates new perturbed input files for OpenFOAM sequences
      @ In, currentInputFiles, list,  list of current input files
      @ In, origInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dict, dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
        where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of new input files (modified or not)
    """
    if 'dynamiceventtree' in str(samplerType).lower():
      raise IOError("Dynamic Event Tree-based samplers not supported by OpenFOAM interface yet!")

    caseFile = self.findInputFile(origInputFiles)
    # 1 get directory of original file, scan for all the files and clone it
    caseDir = os.path.join(Kwargs['BASE_WORKING_DIR'], caseFile.subDirectory)
    # 2 copy all the case files
    shutil.copytree(caseDir, self.findInputFile(currentInputFiles).getPath(), dirs_exist_ok=True)
    # 3 get original inputs
    originalInputs = self._scanDirectoryAndListFiles(caseDir, self.directoriesPerturbableInputs)
    # 4 cache the “signature” of every original input once
    origSignatures = {(pathlib.Path(el.getPath()).name, el.getFilename()) for el in originalInputs}
    # 5 find current inputs whose (dirname, filename) signature exists in originals
    candidateCurrent = self._scanDirectoryAndListFiles(self.findInputFile(currentInputFiles).getPath(), self.directoriesPerturbableInputs)
    # 6 construct current input
    currentInputsToPerturb = [ci for ci in candidateCurrent
        if (pathlib.Path(ci.getPath()).name, ci.getFilename()) in origSignatures]
    # 7 perturb the files using the Generic Parser
    parser = GenericParser.GenericParser(currentInputsToPerturb)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputsToPerturb,originalInputs)
    return currentInputFiles

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    super().initialize(runInfo, oriInputFiles)
    # the initialize here makes sure that the input file is the "Allrun"
    # we just run a Dry run of the method 'findInputFile'
    caseFile = self.findInputFile(oriInputFiles)
    if len(caseFile.subDirectory.strip()) == 0:
      raise Exception(f"OpenFOAM input file (.foam) '{str(caseFile)}' MUST have a 'subDirectory' "
                      "associated with it (containing all the input files required by OpenFOAM)")
    self._caseDirectoryName = caseFile.subDirectory.strip()
    self._caseFileName = caseFile.getFilename()

  def generateCommand(self, inputFiles, executable, clargs=None,fargs=None,preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have
            been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input
            (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify
            in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to
            run the code (string), returnCommand[1] is the name of the output root
    """
    caseFile = self.findInputFile(inputFiles)
    executableName = pathlib.Path(executable).name
    executableAbsPath = str(pathlib.Path(caseFile.getPath()) / pathlib.Path(executableName))
    if not os.path.exists(executableAbsPath):
      raise RuntimeError(f"the executable {executableName} not found in OpenFOAM "
                         f"data/input directory (indicated by the .foam file '{str(caseFile)}'): {executableAbsPath}")

    #Creates the output file that saves information that is outputted to the command prompt
    #The output file name of the Neutrino results
    outputfile = f"log_openfoam"
    # since OpenFOAM uses, in most of the workflows, several executables to
    # run preprocessors (e.g. meshing), kernel, postprocessing, we run the bash script (in most of the case, Allrun)
    commandToRun = f'{executableAbsPath}'
    commandElement = []
    if preExec is not None:
      commandToRun = f"openfoam bash -lc {executableAbsPath}"
    commandElement.append(('parallel', commandToRun))
    returnCommand = commandElement, outputfile
    return returnCommand

  def _isValidInput(self, inputFile):
    """
      Check if an input file is a Neutrino input file.
      @ In, inputFile, string, the file name to be checked
      @ Out, valid, bool, 'True' if an input file has a type == htpipe, otherwise 'False'.
    """
    valid = False
    if inputFile.getType().lower() == 'openfoam':
      inputFile.getExt()
      if inputFile.getExt() in self.getInputExtension():
        valid = True
    return valid

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      In the case of HTPIPE, we look for the expression "failed". If it is present, the run is considered to be failed.
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    badWords  = ['fatal error']
    try:
      outputToRead = open(os.path.join(workingDir,'log_openfoam'),"r")
    except:
      return True
    readLines = outputToRead.readlines()
    for badMsg in badWords:
      if any(badMsg in x.lower() for x in readLines):
        failure = True
    outputToRead.close()
    return failure

  def finalizeCodeOutput(self, command, output, workDir):
    """
      This function parses through the output files OpenFOAM.
      @ In, command, string, command to call serpent executable
      @ In, output, string, output file path
      @ In, workDir, string, working directory path
      @ Out, None
    """
    outputParser = op.openfoamOutputParser(os.path.join(workDir, self._caseDirectoryName), self._caseFileName, variables=self.variables, writeCentroids=self.writeCentroids)
    results = outputParser.processOutputs()
    return results

  def onlineStopCriteria(self, command, output, workDir):
    """
      This method is called by RAVEN during the simulation.
      It is intended to provide means for the code interface to monitor the execution of a run
      and stop it if certain criteria are met (defined at the code interface level)
      For example, the underlying code interface can check for a figure of merit in the output file
      produced by the driven code and stop the simulation if that figure of merit is outside a certain interval
      (e.g. Pressure > 2 bar, stop otherwise, continue).
      If the simulation is stopped because of this check, the return code is set artificially to 0 (normal termination) and
      the 'checkForOutputFailure' method is not called. So the simulation is considered to be successful.

      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, stopSim, bool, True if the job needs to stop being executed, False if it needs to continue to be executed
    """
    stopSim = False
    if self.stoppingCriteriaFunction is not None:
      if os.path.exists(os.path.join(workDir,"0")):
        outputParser = op.openfoamOutputParser(workDir, variables=self.variables,
                                              checkAccessAndWait = True)
        results = outputParser.processOutputs()
        stopSim = self.stoppingCriteriaFunction.evaluate(self.stoppingCriteriaFunction.name,results)
    return stopSim
