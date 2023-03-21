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
import sys
#External Modules--------------------end

#Internal Modules--------------------begin
from ..Generic.GenericCodeInterface import GenericCode
from . import serpentOutputParser as op
#Internal Modules--------------------end

class SERPENT(GenericCode):
  """
    Provides code to interface RAVEN to SERPENT
    This class is used as a code interface for SERPENT in RAVEN.  It expects
    input parameters to be specified by input file, input files be specified by either
    command line or in a main input file, and produce a csv output.
    It is based on the code interface developed by jbae11 and mostly re-written.
  """
  def __init__(self):
    """
      Initializes the SERPENT Interface.
      @ In, None
      @ Out, None
    """
    GenericCode.__init__(self)
    self.printTag         = 'SERPENT'# Print Tag
    self.isotope_list_f   = None     # isotope list file (if any)
    self.isotopes         = []       # isotopes to collect
    self.traceCutOff      = 1.e-7    # cutoff threshold for ignoring isotopes

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
    isotopeList = xmlNode.find("isotope_list")

    if isotopeList is None:
      raise IOError(self.printTag+' ERROR: <isotope_list> node not inputted!!')
    body = isotopeList.text.strip()
    if "," in body:
      self.isotopes = [iso.strip() for iso in body.split(",")]
    else:
      if not body.strip().endswith(".csv"):
        raise IOError(self.printTag+' ERROR: <isotope_list> file must be a CSV!!')
      self.isotope_list_f = body
    cutOff = xmlNode.find("traceCutOff")

    if cutOff is not None:
      self.traceCutOff = float(cutOff.text)

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    # read the isotopes
    if self.isotope_list_f is not None:
      self.isotopes = []
      if not os.path.isabs(self.isotope_list_f):
        self.isotope_list_f = os.path.join(runInfo['WorkingDir'], self.isotope_list_f)
      lines = open(self.isotope_list_f,"r").readlines()
      for line in lines:
        self.isotopes+= [elm.strip() for elm in line.split(",")]
    inputFile = self._findInputFile(oriInputFiles)
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
    # resfile would be 'input.serpent_res.m'
    # this is the file produced by RAVEN
    inputRoot = output.replace("_res","")
    resfile = os.path.join(workDir, output+".m")
    inputFile = os.path.join(workDir,inputRoot)
    inbumatfile = os.path.join(workDir, inputRoot+".bumat0")
    outbumatfile = os.path.join(workDir, inputRoot+".bumat1")
    # parse files into dictionary
    keffDict = op.searchKeff(resfile)
    # the second argument is the percent cutoff
    inBumatDict = op.bumatRead(inbumatfile, self.traceCutOff)
    outBumatDict = op.bumatRead(outbumatfile, self.traceCutOff)

    outputPath = os.path.join(workDir, output+'.csv')
    op.makeCsv(outputPath, inBumatDict, outBumatDict,
                keffDict, self.isotopes, inputFile)
