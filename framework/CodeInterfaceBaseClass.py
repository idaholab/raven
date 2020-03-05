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
from __future__ import division, print_function, unicode_literals, absolute_import
#External Modules------------------------------------------------------------------------------------
import abc
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------

class CodeInterfaceBase(utils.metaclass_insert(abc.ABCMeta,object)):
  """
  Code Interface base class. This class should be the base class for all the code interfaces.
  In this way some methods are forced to be implemented and some automatic checking features
  are available (checking of the inputs if no executable is available), etc.
  NOTE: As said, this class SHOULD be the base class of the code interfaces. However, the developer
        of a newer code interface can decide to avoid to inherit from this class if he does not want
        to exploit the automatic checking of the code interface's functionalities
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self.inputExtensions = []    # list of input extensions
    self._runOnShell = True      # True if the specified command by the code interfaces will be executed through shell.
    self._ravenWorkingDir = None # location of RAVEN's main working directory

  def setRunOnShell(self, shell=True):
    """
      Method used to set the the executation of code command through shell if shell=True
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

    if os.environ.get('RAVENinterfaceCheck','False').lower() in utils.stringsThatMeanTrue():
      return [('parallel','echo')],outputfileroot
    returnCommand = subcodeCommand,outputfileroot
    return returnCommand

  def readMoreXML(self, xmlNode, ravenWorkingDir):
    """
      Function to read the portion of the xml input that belongs to this class and
      initialize some members based on inputs.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ In, ravenWorkingDir, str, location of RAVEN's working directory
      @ Out, None
    """
    self._ravenWorkingDir = ravenWorkingDir
    self._readMoreXML(xmlNode)

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and
      initialize some members based on inputs. This can be overloaded in specialized code interface in order
      to read specific flags
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
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
      @ Out, tuple, tuple of strings containing accepted input extension (e.g.(".i",".inp"]) )
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

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      this method is called by the RAVEN code at the end of each run (if the method is present).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    return output

  def checkForOutputFailure(self, output, workingDir):
    """
      This method is called by RAVEN at the end of each run if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the output file looking for a special keyword that testifies that a particular job got failed
      (e.g. in RELAP5 would be the keyword "********")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    return failure
