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
Created on Jan 28, 2015

@author: alfoa
"""
#External Modules------------------------------------------------------------------------------------
import abc
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from . import CsvLoader
from .BaseClasses import BaseInterface
from .Functions import factory as functionFactory
from .Functions import returnInputParameter
from .utils.TreeStructure import InputNode
from .utils import utils
#Internal Modules End--------------------------------------------------------------------------------

class CodeInterfaceBase(BaseInterface):
  """
  Code Interface base class. This class should be the base class for all the code interfaces.
  In this way some methods are forced to be implemented and some automatic checking features
  are available (checking of the inputs if no executable is available), etc.
  NOTE: As said, this class SHOULD be the base class of the code interfaces. However, the developer
        of a newer code interface can decide to avoid to inherit from this class if he does not want
        to exploit the automatic checking of the code interface's functionalities
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.inputExtensions = []                # list of input extensions
    self._runOnShell = True                  # True if the specified command by the code interfaces will be executed through shell.
    self._ravenWorkingDir = None             # location of RAVEN's main working directory
    self._csvLoadUtil = 'pandas'             # utility to use to load CSVs
    self.printFailedRuns = True              # whether to print failed runs to the screen
    self._writeCSV = False                   # write CSV even if the data can be returned directly to raven (e.g. if the user requests them)
    # has the underlying code interface a method to stop the simulation if a set of criteria is met?
    onlineStopCriteria = getattr(self, "onlineStopCriteria", None)
    self.hasOnlineStopCriteriaCheckMethod = callable(onlineStopCriteria)
    # and, if so, is it active? (there is a function associated with it?)
    self.hasOnlineStopCriteriaCheck = False
    self.onlineStopCriteriaTimeInterval = 5.0  # 5 seconds interval by default (but it can be overwritten in the input file)
    # function to be evaluated for the stopping condition if any
    self.stoppingCriteriaFunction = None
    self._stoppingCriteriaFunctionNode = None

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    # This method is not used for now, once we convert all code interfaces to use InputData,
    # this method will be used to process xml input.
    super()._handleInput(paramInput)

  def setRunOnShell(self, shell=True):
    """
      Method used to set the the execution of code command through shell if shell=True
      @ In, shell, Boolean, True if the users want to execute their code through shell
      @ Out, None
    """
    self._runOnShell = shell

  def getRunOnShell(self):
    """
      Method to return the status of self._runOnShell
      @ In, None
      @ Out, None
    """
    return self._runOnShell

  def getIfWriteCsv(self):
    """
      Returns self._writeCSV. True if a CSV is requested by the user even if
      the code interface returns the data to RAVEN directly
      @ In, None
      @ Out, getIfWriteCsv, bool, should we write the csv?
    """
    return self._writeCSV

  def getCsvLoadUtil(self):
    """
      Returns the string representation of the CSV loading utility to use
      @ In, None
      @ Out, getCsvLoadUtil, str, name of utility to use
    """
    # default to pandas, overwrite to 'numpy' if all of the following:
    # - all entries are guaranteed to be floats
    # - results CSV have a large number of headers (>1000)
    return self._csvLoadUtil

  def setCsvLoadUtil(self, util):
    """
      Returns the string representation of the CSV loading utility to use
      @ In, getCsvLoadUtil, str, name of utility to use
    """
    ok = CsvLoader.CsvLoader.acceptableUtils
    if util not in ok:
      raise TypeError(f'Unrecognized CSV loading utility: "{util}"! Expected one of: {ok}')
    self._csvLoadUtil = util

  def genCommand(self, inputFiles, executable, flags=None, fileArgs=None, preExec=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      This method checks a boolean environment variable called 'RAVENinterfaceCheck':
      if true, the subcodeCommand is going to be overwritten with an empty string. In this way we can check the functionality of the interface without having an executable.
      See Driver.py to understand how this Env variable is set
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, flags, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fileArgs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (e.g. under the node < Code >< fileargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    subcodeCommand,outputfileroot = self.generateCommand(inputFiles,executable,clargs=flags,fargs=fileArgs,preExec=preExec)

    if utils.stringIsTrue(os.environ.get('RAVENinterfaceCheck','False')):
      return [('parallel','echo')],outputfileroot
    returnCommand = subcodeCommand,outputfileroot
    return returnCommand

  def readXML(self, xmlNode, workingDir=None):
    """
      Function to read the portion of the xml input that belongs to this class and
      initialize some members based on inputs.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ In, workingDir, str, location of RAVEN's working directory
      @ Out, None
    """
    self._ravenWorkingDir = workingDir
    # read global options
    # should we print CSV even if the data can be directly returned to RAVEN?
    csvLog = xmlNode.find("csv")
    self._writeCSV = utils.stringIsTrue(csvLog.text if csvLog is not None else "False")
    # onlineStopCriteriaTimeInterval (if any)
    onlineStopCriteriaTimeInterval = xmlNode.find("onlineStopCriteriaTimeInterval")
    if onlineStopCriteriaTimeInterval is not None:
      self.onlineStopCriteriaTimeInterval = utils.floatConversion(onlineStopCriteriaTimeInterval.text)
      if not self.hasOnlineStopCriteriaCheckMethod:
        self.raiseAWarning(f"onlineStopCriteriaTimeInterval provided, but Code interface {self.type} "
                           "does not have 'onlineStopCriteriaCheck' method! IGNORED!")
    # we just store it for now because we need the runInfo to be applied before parsing it
    self._stoppingCriteriaFunctionNode = xmlNode.find("stoppingCriteriaFunction")
    super().readXML(xmlNode, workingDir=workingDir)

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and
      initialize some members based on inputs. This can be overloaded in specialized code interface in order
      to read specific flags
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    # Right now this method is overloaded so that we have minimal impacts on existing code Interfaces
    # When the InputSpecs is forced to use in the code interface, we can just call the super method
    # super()._readMoreXML(xmlNode)
    pass

  @abc.abstractmethod
  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    return

  @abc.abstractmethod
  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      This method is used to generate an input based on the information passed in.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    pass

        ####################
  ####### OPTIONAL METHODS #######
        ####################

  def getInputExtension(self):
    """
      This method returns a list of extension the code interface accepts for the input file (the main one)
      @ In, None
      @ Out, tuple, tuple of strings containing accepted input extension (e.g.[".i",".inp"]) )
    """
    return tuple(self.inputExtensions)

  def setInputExtension(self,exts):
    """
      This method sets a list of extension the code interface accepts for the input files
      @ In, exts, list, list or other array containing accepted input extension (e.g.[".i",".inp"])
      @ Out, None
    """
    self.inputExtensions = []
    self.addInputExtension(exts)

  def addInputExtension(self,exts):
    """
      This method adds a list of extension the code interface accepts for the input files
      @ In, exts, list, list or other array containing accepted input extension (e.g.[".i",".inp"])
      @ Out, None
    """
    for e in exts:
      self.inputExtensions.append(e)

  def addDefaultExtension(self):
    """
      This method sets a list of default extensions a specific code interface accepts for the input files.
      This method should be overwritten if these are not acceptable defaults.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['i','inp','in'])

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    # store working dir for future needs
    self._ravenWorkingDir = runInfo['WorkingDir']
    if self.hasOnlineStopCriteriaCheckMethod and self._stoppingCriteriaFunctionNode is not None:
      # get instance of function and apply run info
      self.stoppingCriteriaFunction = functionFactory.returnInstance('External')
      self.stoppingCriteriaFunction.applyRunInfo(runInfo)
      # create a functions input node to use the Functions reader
      functs = InputNode('Functions')
      # change tag name to External (to use the parser)
      # this is very ugly but works fine
      self._stoppingCriteriaFunctionNode.tag = 'External'
      functs.append(self._stoppingCriteriaFunctionNode)
      inputParams = returnInputParameter()
      inputParams.parseNode(functs)
      self.stoppingCriteriaFunction.handleInput(inputParams.subparts[0])
      if self.stoppingCriteriaFunction.name not in self.stoppingCriteriaFunction.availableMethods():
        self.raiseAnError(ValueError, f"<stoppingCriteriaFunction> named '{self.stoppingCriteriaFunction.name}' "
                          f"not found in file '{self.stoppingCriteriaFunction.functionFile}'!")
      self.hasOnlineStopCriteriaCheck = True

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string or dict, optional, if present and string:
                                                 in case the root of the output file gets changed in this method (and a CSV is produced);
                                               if present and dict:
                                                 in case the output of the code is directly stored in a dictionary and can be directly used
                                                 without the need that RAVEN reads an additional CSV
    """
    return output

  def checkForOutputFailure(self, output, workingDir):
    """
      This method is called by RAVEN at the end of each run if the return code is == 0.
      This method needs to be implemented for the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the output file looking for a special keyword that testifies that a particular job got failed
      (e.g. in RELAP5 would be the keyword "********")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    return failure

  def onlineStopCriteriaCheck(self, command, output, workingDir):
    """
      This method is called by RAVEN during the simulation.
      It is intended to provide means for the code interface to monitor the execution of a run
      and stop it if certain creteria are met (defined at the code interface level)
      For example, the underlying code interface can check for a figure of merit in the output file
      produced by the driven code and stop the simulation if that figure of merit is outside a certain interval
      (e.g. Pressure > 2 bar, stop otherwise, continue).
      If the simulation is stopped because of this check, the return code is set artificially to 0 (normal execution) and
      the 'checkForOutputFailure' method is not called. So the simulation is considered to be successful.

      NOTE: This method is only called if 'onlineStopCriteria' is implemented in the underlying code interface
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, continueSim, bool, True if the job needs to continue being executed, False if it needs to be stopped
    """
    continueSim = self.onlineStopCriteria(command, output, workingDir)
    assert isinstance(continueSim, bool), f"Return argument of 'onlineStopCriteria' (for code interface {self.type}) must be a boolean, got {type(continueSim)}!"
    return continueSim

  def getOnlineStopCriteriaTimeInterval(self):
    """
      Method to get the time interval (frequency) of how often the onlineStopCriteriaCheck needs to be inquired.
      @ In, None
      @ Out,
    """
    return self.onlineStopCriteriaTimeInterval

  # Function is required by the base class but not used in this and its subclass
  def run(self, *args, **kwargs):
    """
      Main method to "do what you do".
      @ In, args, list, positional arguments
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    pass
