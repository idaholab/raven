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
Created on August 31, 2020

@author: Andrea Alfonsi

comments: Interface for AccelerateCFD
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import math
import csv
import re
import copy
import numpy as np
from OpenFoamPP import field_parser

from CodeInterfaceBaseClass import CodeInterfaceBase
from GenericCodeInterface import GenericParser

def findNearest(array, value):
  """
    Find nearest value
    @ In, array, numpy array, the array
    @ In, value, float/int, the pivot value
    @ Out, idx, int, the index
  """
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

class AcceleratedCFD(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to AcceleratedCFD
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
    self.locations = {'x':None,'y':None,'z':None}
    for child in xmlNode:
      if child.tag == 'outputLocations':
        x = child.find("x")
        y = child.find("y")
        z = child.find("z")
        if x is not None:
          self.locations['x'] = [val.strip() for val in x.text.split(",")]
        if y is not None:
          self.locations['y'] = [val.strip() for val in y.text.split(",")]
        if z is not None:
          self.locations['z'] = [val.strip() for val in z.text.split(",")]
    if None in list(self.locations.values()):
      raise IOError("outputLocations must be inputted! x, y z!")
    if not( len(self.locations['x']) == len(self.locations['y']) == len(self.locations['z']) ):
      raise IOError("outputLocations must have the same size! len(x) !=  len(y) != len(z)!")


  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    self.coords = {}
    self.romName = None
    self.romType = None
    self.locations["xyz"] = np.zeros((len(self.locations["x"]), 3), dtype=int)
    map = {'x':0,'y':1,'z':2}
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getType().startswith("mesh"):
        coord = inputFile.getType().split("-")[-1].strip()
        if coord not in ['x', 'y', 'z']:
          raise IOError('Mesh type not == to x, y or z. Got: ' + coord)
        self.coords[coord] = self.readFoamFile(inputFile.getAbsFile())
        for i in range(len(self.locations[coord])):
          if self.locations[coord][i] in ['min','max','average']:
            if self.locations[coord][i] == 'min':
              operator = np.min
            elif self.locations[coord][i] == 'max':
              operator = np.max
            else:
              operator = np.average
            v = operator(self.coords[coord][1][0])
          else:
            v = float(self.locations[coord][i])
          self.locations["xyz"][i, map[coord]] = findNearest(self.coords[coord][1][0],v)
      if inputFile.getType().lower() == 'input':
        with open(inputFile.getAbsFile(), "r") as inputObj:
          lines = inputObj.readlines()
          for line in lines:
            if line.strip().startswith("<romName>"):
              self.romName = line.strip().replace("<romName>","").split("<")[0]
            if line.strip().startswith("<romType>"):
              self.romType = line.strip().replace("<romType>","").split("<")[0]
            if self.romName is not None and self.romType is not None:
              break
    if not len(self.coords):
      raise IOError('Mesh type files must be inputed (mesh-x, mesh-y, mesh-z). Got None!')
    if self.romName is None or self.romType is None:
      raise IOError('<romName> or <romType> not found in input file!')

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (lenght of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    # find the input file (check that one input is provided)
    inputToPerturb = self.findInps(inputFiles,"input")

    # create output file root
    outputfile = 'out~' + inputToPerturb[0].getBase()

    # create command
    # the input file name is hardcoded in AccelerateCFD (podInputs.xml)
    executeCommand = [('parallel', executable )]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def getInputExtension(self):
    """
      Return a tuple of possible file extensions for a simulation initialization file (e.g., input.i).
      @ In, None
      @ Out, validExtensions, tuple, tuple of valid extensions
    """
    return ('')

  def findInps(self,inputFiles, inputType):
    """
      Locates the input files required by AcellerateCDF Interface
      @ In, inputFiles, list, list of Files objects
      @ In, inputType, str, inputType to find (e.g. mesh-x, input, etc)
      @ Out, podDictInput, list, list containing AcellerateCFD required input files
    """
    podDictInput = []
    for inputFile in inputFiles:
      if inputFile.getType().strip().lower() == inputType.lower():
        podDictInput.append(inputFile)
    if len(podDictInput) == 0:
      raise IOError('no "'+inputType+'" type file has been found!')
    return podDictInput

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generate a new AccelerateCFD input file (txt format) from the original, changing parameters
      as specified in Kwargs['SampledVars'].
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    if 'dynamiceventtree' in str(samplerType).lower():
      raise IOError("Dynamic Event Tree-based samplers not supported by AccelerateCFD interface yet!")
    currentInputsToPerturb = self.findInps(currentInputFiles,"input")
    originalInputs         = self.findInps(oriInputFiles,"input")
    parser = GenericParser.GenericParser(currentInputsToPerturb)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputsToPerturb,originalInputs)
    return currentInputFiles

  def readFoamFile(self, filename):
    """
      This method is aimed to read a Open Faom file for accelerated CFD
      @ In, filename, str, the file name
      @ Out, content, dict, the open foam output content
    """
    with open(filename, "r") as foam:
      lines = foam.readlines()
      settings = {}
      for row, line in enumerate(lines):
        if line.strip().startswith("FoamFile"):
          info = lines[row+2:row+7]
          for var in info:
            inf, val = [v.strip().replace(";","").replace('"','') for v in var.split()]
            settings[inf] = val
        if line.strip().startswith("dimensions"):
          settings["dimensions"] = [int(v.replace("[","").replace("]","")) for v in line.split()[-1].replace(";","").split()]
        if len(settings) > 1:
          del lines
          break
    field = field_parser.parse_field_all(filename)
    return settings, field

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Called by RAVEN to modify output files (if needed) so that they are in a proper form.
      In this case, the default .mat output needs to be converted to .csv output, which is the
      format that RAVEN can communicate with.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    # open output file
    resultFolder = os.path.join(workingDir,"rom",self.romType +"_"+ self.romName)
    resultingDirs = [os.path.join(resultFolder, o) for o in os.listdir(resultFolder)
                        if os.path.isdir(os.path.join(resultFolder,o))]
    resultingDirs.sort()
    timeList = []
    results = {}
    for ts in resultingDirs:
      time = float(ts.split(os.path.sep)[-1])
      timeList.append(time)
    timeList.sort()
    results["time"] = np.zeros(len(timeList))
    for ts in resultingDirs:
      settingsVector, fieldVector = self.readFoamFile(os.path.join(ts, "Urom"))
      settingsScalar, fieldScalar = self.readFoamFile(os.path.join(ts, "srom"))
      time =  float(settingsVector['location'])
      indx = findNearest(timeList, time)
      results["time"][indx] = time
      for i in range(len(self.locations["xyz"])):
        variableName = ""
        for j, coord in enumerate(['x', 'y', 'z']):
          variableName = settingsVector['class']+"_"+coord + "_" + self.locations[coord][i]
          val = fieldVector[0][self.locations["xyz"][i][j], j]
          if variableName not in results:
            results[variableName] = np.zeros(len(timeList))
          results[variableName][indx] = val
      for i in range(len(self.locations["xyz"])):
        variableName = ""
        for j, coord in enumerate(['x']):
          variableName = settingsScalar['class']+"_"+coord + "_" + self.locations[coord][i]
          val = fieldScalar[0][self.locations["xyz"][i][j]]
          if variableName not in results:
            results[variableName] = np.zeros(len(timeList))
          results[variableName][indx] = val
    return results
