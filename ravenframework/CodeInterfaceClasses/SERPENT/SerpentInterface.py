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
Created May 9th, 2019

@author: alfoa
"""
#External Modules--------------------begin
import os
#External Modules--------------------end

#Internal Modules--------------------begin
from ravenframework.utils import utils
from ..Generic.GenericCodeInterface import GenericCode
from . import serpentOutputParser as op
#Internal Modules--------------------end

class SERPENT(GenericCode):
  """
    Provides code to interface RAVEN to SERPENT
    The class has been upgraded since its first version to allow
    the usage of the interface for both steadystate and depletion calculations.
    The output parsing is performed leveraging the library
    serpentTools (https://serpent-tools.readthedocs.io/en/master/index.html)
    Multiple output formats are now processable (both for steady state and depletion)
  """
  def __init__(self):
    """
      Initializes the SERPENT Interface.
      @ In, None
      @ Out, None
    """
    # check if serpentTools is available, raise error it otherwise
    try:
      import serpentTools
    except ImportError:
      raise ImportError("serpentTools not found and SERPENT Interface has been invoked. "
                        "Install serpentTools through pip or invoke RAVEN installation procedure (i.e. establish_conda_env.sh script) "
                        "with the additional command line option '--code-interface-deps'. "
                        "See User Manual for additiona details!")
    # intialize code interface
    super().__init__()
    self.printTag         = 'SERPENT'         # Print Tag
    self._fileTypesToRead = ['ResultsReader'] # container of file types to read
    # in case of burnup calc, the interface can compute the time at which FOMs (e.g. keff) crosses
    # a target. For example (default), we can compute the time (burnDays) at which absKeff crosses 1.0
    self.eolTarget = {}
    # volume calculation?
    self.volumeCalc = False
    self.nVolumePoints = None

  def _findInputFile(self, inputFiles):
    """
      Method to return the input file
      @ In, inputFiles, list, the input files of the step
      @ Out, inputFile, string, the input file
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getType().strip().lower() == "serpent":
        found = True
        break
    if not found:
      raise IOError(self.printTag+' ERROR: input type "serpent" not found in the Step!')
    return inputFile

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    super()._readMoreXML(xmlNode)
    preVolumeCalc = xmlNode.find("volumeCalculation")
    if preVolumeCalc is not None:
      self.volumeCalc = utils.interpretBoolean(preVolumeCalc.text)
      nPoints = preVolumeCalc.attrib.get("nPoints")
      if nPoints is not None:
        self.nVolumePoints = utils.intConversion(utils.floatConversion(nPoints))
        if self.nVolumePoints is None:
          raise ValueError(self.printTag+' ERROR: "nPoints" attribute in <volumeCalculation> must be present (and integer) if <volumeCalculation> node is inputted')

    eolNodes = xmlNode.findall("EOL")
    for eolNode in eolNodes:
      if eolNode is not None:
        target = eolNode.attrib.get('target')
        if target is None:
          raise ValueError(self.printTag+' ERROR: "target" attribute in <EOL> must be present if <EOL> node is inputted')
        value = float(eolNode.text)
        self.eolTarget[target] = value

    # by default only the "_res.m" file is read.
    # if additional files are required, the user should request them here
    addFileTypes = xmlNode.find("additionalFileTypes")
    if addFileTypes is not None:
      serpentFileTypes = [ft.strip() for ft in addFileTypes.text.split(",")]
      if 'ResultsReader' in  serpentFileTypes:
        # we pop this because it is the default
        serpentFileTypes.pop(serpentFileTypes.index('ResultsReader'))
      for ft in serpentFileTypes:
        if ft not in op.serpentOutputAvailableTypes:
          raise ValueError(self.printTag+f' ERROR: <Serpent File Type> {ft} not supported! Available types are "'
                        f'{", ".join(op.serpentOutputAvailableTypes)}!!')
      self._fileTypesToRead += serpentFileTypes

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    super().initialize(runInfo, oriInputFiles)
    inputFile = self._findInputFile(oriInputFiles)
    # check if all the output files will be actually generated by the provided input
    op.checkCompatibilityFileTypesAndInputFile(inputFile, self._fileTypesToRead)
    # set the extension
    self.setInputExtension([inputFile.getExt()])

  def generateCommand(self,inputFiles,executable,clargs=None, fargs=None, preExec=None):
    """
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
    inputFile = self._findInputFile(inputFiles)
    if clargs is not None:
      addflags = clargs['text']
    else:
      addflags = ''
    executeCommand = [('parallel',executable+' '+inputFile.getFilename()+' '+addflags)]
    if self.volumeCalc:
      executeCommand.insert(0, ('parallel',executable+' '+inputFile.getFilename()+f' -checkvolumes {self.nVolumePoints}') )
    returnCommand = executeCommand, inputFile.getFilename()+"_res"
    return returnCommand

  def finalizeCodeOutput(self, command, output, workDir):
    """
      This function parses through the output files SERPENT creates into a csv.
      @ In, command, string, command to call serpent executable
      @ In, output, string, output file path
      @ In, workDir, string, working directory path
      @ Out, None
    """
    inputRoot = output.replace("_res","")
    outputParser = op.SerpentOutputParser(self._fileTypesToRead, os.path.join(workDir,inputRoot), eol = self.eolTarget)
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
      inputRoot = output.replace("_res","")
      if os.path.exists(os.path.join(workDir,output+".m")):
        outputParser = op.SerpentOutputParser(self._fileTypesToRead,
                                              os.path.join(workDir,inputRoot),
                                              eol = self.eolTarget,
                                              checkAccessAndWait = True)
        results = outputParser.processOutputs()
        stopSim = self.stoppingCriteriaFunction.evaluate(self.stoppingCriteriaFunction.name,results)
    return stopSim

