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
  Created on April 18, 2017
  @author: Matteo D'Onorio (Sapienza University of Rome)
           Andrea Alfonsi (INL)
"""

import os
from ...contrib.melcorTools import melcorTools
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..Generic import GenericParser
import pandas as pd


class Melcor(CodeInterfaceBase):
  """
    This class is used a part of a code dictionary to specialize Model. Code for different MELCOR versions
    like MELCOR 2.2x, MELCOR 1.86, MELCOR for fusion applications
  """

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """

    melNode = xmlNode.find('MelcorOutput')
    varNode = xmlNode.find('variables')
    plotNode = xmlNode.find('CodePlotFile')

    if varNode is None:
      raise IOError("Melcor variables not found, define variables to print")
    if plotNode is None:
      raise IOError("Please define the name of the MELCOR plot file in the CodePlotFile xml node")
    if melNode is None:
      raise IOError("Please enter MELCOR message file name")

    self.varList = [var.strip() for var in varNode.text.split("$,")]
    self.melcorPlotFile = [var.strip() for var in plotNode.text.split(",")][0]
    self.melcorOutFile = [var.strip() for var in melNode.text.split(",")][0]

    return self.varList, self.melcorPlotFile, self.melcorOutFile

  def findInps(self,currentInputFiles):
    """
      Locates the input files for Melgen, Melcor
      @ In, currentInputFiles, list, list of Files objects
      @ Out, (melgIn,melcIn), tuple, tuple containing Melgen and Melcor input files
    """
    foundMelcorInp = False
    for index, inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        foundMelcorInp = True
        melgIn = currentInputFiles[index]
        melcIn = currentInputFiles[index]
    if not foundMelcorInp:
      raise IOError("Unknown input extensions. Expected input file extensions are "+ ",".join(self.getInputExtension())+" No input file has been found!")
    return melgIn, melcIn


  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (e.g. under the node < Code >< fileargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False

    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    melcOut = 'OUTPUT_MELCOR'
    melcin,melgin = self.findInps(inputFiles)
    if clargs:
      precommand = executable + clargs['text']
    else:
      precommand = executable
    melgCommand = str(preExec)+ ' '+melcin.getFilename()
    melcCommand = precommand+ ' '+melcin.getFilename()
    returnCommand = [('serial',melgCommand + ' && ' + melcCommand +' ow=o ')],melcOut

    return returnCommand

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """
      This generates a new input file depending on which sampler is chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """

    if "dynamicevent" in samplerType.lower():
      raise IOError("Dynamic Event Tree-based samplers not implemented for MELCOR yet! But we are working on that.")
    indexes  = []
    inFiles  = []
    origFiles= []
    for index,inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        indexes.append(index)
        inFiles.append(inputFile)
    for index,inputFile in enumerate(origInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        origFiles.append(inputFile)
    parser = GenericParser.GenericParser(inFiles)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputFiles,origFiles)

    return currentInputFiles

  def writeDict(self,workDir):
    """
      Output the parsed results into a CSV file
      @ In, workDir, str, current working directory
      @ Out, dictionary, dict, dictioanry containing the data generated by MELCOR
    """
    fileDir = os.path.join(workDir,self.melcorPlotFile)
    time,data,varUdm = melcorTools.MCRBin(fileDir,self.varList)
    dfTime = pd.DataFrame(time, columns= ["Time"])
    dfData = pd.DataFrame(data, columns = self.varList)
    df = pd.concat([dfTime, dfData], axis=1, join='inner')
    df.drop_duplicates(subset="Time",keep='first',inplace=True)
    dictionary = df.to_dict(orient='list')
    return dictionary

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, response, dict, dictionary containing the data
    """
    response = self.writeDict(workingDir)
    return response


  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      (e.g. in RELAP5 would be the keyword "********")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = True
    goodWord  = "Normal termination"   # This is for MELCOR 2.2 (todo: list for other MELCOR versions)
    try:
      outputToRead = open(os.path.join(workingDir,self.melcorOutFile),"r")
    except FileNotFoundError:
      return failure
    readLines = outputToRead.readlines()
    lastRow = readLines[-1]
    if goodWord in lastRow:
      failure = False
    outputToRead.close()
    return failure
