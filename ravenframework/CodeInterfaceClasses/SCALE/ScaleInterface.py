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
Created on April 04, 2018

@author: alfoa

comments: Interface for Scale Simulation (current Origen and Triton)
"""
import os
import copy
import shutil
from ravenframework.utils import utils
import xml.etree.ElementTree as ET

from ..Generic import GenericParser
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from .tritonAndOrigenData import origenAndTritonData

class Scale(CodeInterfaceBase):
  """
    Scale Interface. It currently supports Triton and Origen sequences only.
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.sequence = []   # this contains the sequence that needs to be run. For example, ['triton'] or ['origen'] or ['triton','origen']
    self.timeUOM  = 's'  # uom of time (i.e. s, h, m, d, y )
    self.outputRoot = {} # the root of the output sequences

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    CodeInterfaceBase._readMoreXML(self,xmlNode)
    sequence = xmlNode.find("sequence")
    if sequence is None:
      self.sequence = ['triton']
    else:
      self.sequence = [elm.strip() for elm in sequence.text.split(",")]
    if self.sequence.count('triton') > 1 or self.sequence.count('origen') > 1:
      raise IOError("Multiple triton or origen sequences are not supported yet!")
    timeUOM = xmlNode.find("timeUOM")
    if timeUOM is not None:
      self.timeUOM = timeUOM.text.strip()
      if self.timeUOM not in ['s','m','h','d','y']:
        raise IOError("timeUOM not recognized. Supported are:" +','.join(['s','m','h','d','y']))

  def findInps(self,inputFiles):
    """
      Locates the input files required by Scale Interface
      @ In, inputFiles, list, list of Files objects
      @ Out, inputDict, dict, dictionary containing Scale required input files
    """
    inputDict = {}
    origen = []
    triton = []

    for inputFile in inputFiles:
      if inputFile.getType().strip().lower() == "triton":
        triton.append(inputFile)
      elif inputFile.getType().strip().lower() == "origen":
        origen.append(inputFile)
    if len(triton) > 1:
      raise IOError('multiple triton input files have been found. Only one is allowed!')
    if len(origen) > 1:
      raise IOError('multiple origen input files have been found. Only one is allowed!')
    # Check if the input requested by the sequence has been found
    if self.sequence.count('triton') != len(triton):
      raise IOError('triton input file has not been found. Files type must be set to "triton"!')
    if self.sequence.count('origen') != len(origen):
      raise IOError('origen input file has not been found. Files type must be set to "origen"!')
    # add inputs
    if len(origen) > 0:
      inputDict['origen'] = origen
    if len(triton) > 0:
      inputDict['triton'] = triton
    return inputDict

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      Generate a command to run SCALE using an input with sampled variables
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
    inputDict = self.findInps(inputFiles)
    executeCommand = []
    for seq in self.sequence:
      self.outputRoot[seq.lower()] = inputDict[seq.lower()][0].getBase()
      executeCommand.append(('parallel',executable+' '+inputDict[seq.lower()][0].getFilename()))
    returnCommand = executeCommand, list(self.outputRoot.values())[-1]
    return returnCommand

  def createNewInput(self, currentInputFiles, origInputFiles, samplerType, **Kwargs):
    """
      Generates new perturbed input files for Scale sequences
      @ In, currentInputFiles, list,  list of current input files
      @ In, origInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dict, dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
        where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of new input files (modified or not)
    """
    if 'dynamiceventtree' in str(samplerType).lower():
      raise IOError("Dynamic Event Tree-based samplers not supported by Scale interface yet!")
    currentInputsToPerturb = [item for subList in self.findInps(currentInputFiles).values() for item in subList]
    originalInputs         = [item for subList in self.findInps(origInputFiles).values() for item in subList]
    parser = GenericParser.GenericParser(currentInputsToPerturb)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputsToPerturb,originalInputs)
    return currentInputFiles

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      In the case of SCALE, we look for the expression "terminated due to errors". If it is present, the run is considered to be failed.
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    badWords  = ['terminated due to errors']
    try:
      outputToRead = open(os.path.join(workingDir,output+'.out'),"r")
    except:
      return True
    readLines = outputToRead.readlines()

    for badMsg in badWords:
      if any(badMsg in x for x in readLines[-20:]):
        failure = True
    return failure

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      This method converts the Scale outputs into a RAVEN compatible CSV file
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, output csv file containing the variables of interest specified in the input
    """
    outputType   = 'combined' if len(self.sequence) > 1 else self.sequence[0]
    filesIn = {}
    for key in self.outputRoot.keys():
      if self.outputRoot[key] is not None:
        filesIn[key] = os.path.join(workingDir,self.outputRoot[key]+'.out')
    outputParser = origenAndTritonData(filesIn, self.timeUOM, outputType)
    outputParser.writeCSV(os.path.join(workingDir,output+".csv"))
