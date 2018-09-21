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
Created on March 26, 2017

@author: alfoa

comments: Interface for simple Bateman code used in Workshop

"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import xml.etree.ElementTree as ET

# import CodeInterfaceBaseClass
from CodeInterfaceBaseClass import CodeInterfaceBase

class BatemanSimple(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to Bateman simple code
    The name of this class is represent the type in the RAVEN input file
    e.g.
    <Models>
      <Code name="myName" subType="BatemanSimple">
      ...
      </Code>
      ...
    </Models>
  """

  def generateCommand(self, inputFiles, executable, clargs=None,fargs=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False
    # Find the first file in the inputFiles that is an XML, which is what we need to work with.
    for index, inputFile in enumerate(inputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found: raise Exception('No correct input file has been found. Got: '+' '.join(oriInputFiles))
    # create the output root
    # in this case, we decided to have an output file root that is equal to "results~<InputFileNameRoot>"
    outputfile = 'results~' + inputFiles[index].getBase()
    # create run command tuple (['executionType','execution command'], output file root)
    # in this case the execution type is "parallel" => it means that multiple instances of this code can be run
    # Since, we have a python code, we prepend the "python" command before the executable
    returnCommand = [('parallel',"python "+executable+" "+inputFiles[index].getFilename() + " "+ outputfile + ".csv")], outputfile
    return returnCommand

  def _isValidInput(self, inputFile):
    """
      Check if an input file is a xml file, with an extension of .xml, .XML or .Xml .
      @ In, inputFile, string, the file name to be checked
      @ Out, valid, bool, 'True' if an input file has an extension of .'xml', 'XML' or 'Xml', otherwise 'False'.
    """
    valid = False
    if inputFile.getExt() in ('xml', 'XML', 'Xml'): valid = True
    return valid

  def getInputExtension(self):
    """
      Return a tuple of possible file extensions for a simulation initialization file (i.e., dsin.txt).
      @ In, None
      @ Out, validExtensions, tuple, tuple of valid extensions
    """
    validExtensions = ('xml', 'XML', 'Xml')
    return validExtensions

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generate a new OpenModelica input file (XML format) from the original, changing parameters
      as specified in Kwargs['SampledVars']
      @ In , currentInputFiles, list,  list of current input files (input files of this iteration)
      @ In , oriInputFiles, list, list of the original input files
      @ In , samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In , Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    # Look for the correct input file
    found = False
    for index, inputFile in enumerate(currentInputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found: raise Exception('No correct input file has been found. Got: '+' '.join(oriInputFiles))

    originalPath = currentInputFiles[index].getAbsFile()

    # Since the input file is XML we can load and edit it directly using etree
    # Load the XML into a tree:
    tree = ET.parse(originalPath)
    # get the root node
    root = tree.getroot()

    # grep the variables that got sampled
    varDict = Kwargs['SampledVars']
    # the syntax of the variables is decided by us
    # for this test we decide that the variable names determine the way to walk in the input file
    # level_1|level_2|...|level_n|variableName
    # e.g. nuclides|A|initialMass

    for var in varDict:
      # loop over the variables
      if "|" not in var:
        # the syntax of the variable is wrong
        raise Exception('Variable syntax is wrong. Expected level_1|level_2|...|level_n|variableName!!')
      # the syntax is correct => we move on.
      levels   = var.split("|")        # we split the variables
      # we create the path to the element in one single line
      # e.g. "nuclides/A/initialMass"
      pathToElement = "/".join(levels)
      # find the element
      varElement   = root.find(pathToElement)
      if varElement is None:
        # if None, no variable has been found
        raise Exception('Not found variable '+var+' in input file '+ originalPath)
      # set the new variable value
      varElement.text = str(varDict[var])

    # now we can re-write the input file
    tree.write(currentInputFiles[index].getAbsFile())
    return currentInputFiles


  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Called by RAVEN to modify output files (if needed) so that they are in a proper form.
      In this case, even if this simple code dumps the results into a CSV, we are going to read
      the .out file that is in ASCI format, just to show how to use this method
      @ In, command, string, the command used to run the ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, newOutputRoot, string, present in case the root of the output file gets changed in this method.
    """
    # create full path to the outputfile
    outputPath = os.path.join(workingDir, output+".out")
    # open original output file (the working directory is provided)
    outputFile = open(outputPath,"r+")
    # we change the output root name just to show that it can be changed and returned by this method
    newOutputRoot = output+"_converted"
    newOutputPath = os.path.join(workingDir,newOutputRoot+".csv")
    # open the new outputfile (the working directory is provided)
    newOutputFile = open(newOutputPath,"w+")
    # read all the lines from the original outputfile
    lines = outputFile.readlines()
    # search for results. In this code,
    # the results start with the string "RESULTS:"
    # and they end with the string "CPU TIME:"

    # control variable to indicate that we need to read
    weRead  = False
    # here we store the resutls
    results = []

    for lineNumber, line in enumerate(lines):
      if line.strip().startswith("CPU TIME"):
        # we finished reading
        break
      if weRead:
        # we store the results
        # results is a list of lists
        results.append(line.split())
      if line.strip().startswith("RESULTS:"):
        # we start reading the results
        weRead = True
    # check if the output has been found
    if len(results) == 0: raise Exception('No outputs have been found!!!')
    # write the results back
    for result in results:
      newOutputFile.write(",".join(result)+"\n")
    # close the new outputfile
    newOutputFile.close()
    # close original output file
    outputFile.close()
    # Return the name without the .csv on it...RAVEN will add it
    # note that if the "output" has not been chagend, this method should not return anything
    return newOutputRoot
