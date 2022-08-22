# Copyright 2017 University of Rome La Sapienza and Battelle Energy Alliance, LLC
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
  Created on Jul 25, 2017
  @author: Emerald Ryan
"""

from __future__ import division, print_function, absolute_import

import os
import lxml.etree as ET

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase

class Neutrino(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to Neutrino code
    The name of this class represents the type in the RAVEN input file
    e.g.
    <Models>
      <Code name="myName" subType="Neutrino">
      ...
      </Code>
      ...
    </Models>

  """

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
    found = False
    # Find the first file in the inputFiles that is an XML, which is what we need to work with.
    for index, inputFile in enumerate(inputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('No correct input file has been found. Got: '+' '.join(inputFiles))

    #Determines the path to the input file
    path = inputFiles[0].getAbsFile()

    #Creates the output file that saves information that is outputted to the command prompt
    #The output file name of the Neutrino results
    outputfile = 'results'

    #Creates run command tuple (['executionType','execution command'], output file root)
    #The path to the Neutrino executable is specified in the RAVEN input file as the executable
    # since it must change directories to run
    executablePath = executable.replace("Neutrino.exe","")
    returnCommand = [('serial','cd ' + executablePath + ' && ' + executable + ' --nogui --file ' + str(path) \
    + ' --run')], outputfile

    return returnCommand

  def _isValidInput(self, inputFile):
    """
      Check if an input file is a Neutrino input file.
      @ In, inputFile, string, the file name to be checked
      @ Out, valid, bool, 'True' if an input file has an extension of '.nescene', otherwise 'False'.
    """
    valid = False
    if inputFile.getExt() in ('nescene'):
      valid = True
    return valid

  def getInputExtension(self):
    """
      Return a tuple of possible file extensions for a simulation initialization file (i.e., dsin.txt).
      @ In, None
      @ Out, validExtensions, tuple, tuple of valid extensions
    """
    validExtensions = ('nescene')
    return validExtensions

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generate a new OpenModelica input file (XML format) from the original, changing parameters
      as specified in Kwargs['SampledVars']
      @ In , currentInputFiles, list,  list of current input files (input files of this iteration)
      @ In , oriInputFiles, list, list of the original input files
      @ In , samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In , Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another
             dictionary called "SampledVars" where RAVEN stores the variables that got sampled
             (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    # Look for the correct input file
    found = False
    for index, inputFile in enumerate(currentInputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('No correct input file has been found. Got: '+' '.join(oriInputFiles))

    originalPath = currentInputFiles[index].getAbsFile()
    originalPath = os.path.abspath(originalPath)

    # Since the input file is XML we can load and edit it directly using etree
    # Load the XML into a tree:
    tree = ET.parse(originalPath, ET.XMLParser(encoding='utf-8'))
    # get the root node
    root = tree.getroot()

    # grep the variables that got sampled
    varDict = Kwargs['SampledVars']

    # Go through sampled variables
    for var in varDict:
      #Search for the SPH solver properties
      #NIISphSolver_1 name may need to be changed based on Neutrino input file
      #Can add other properties to change beside the solver properties
      for element in root.findall('./properties/Scene/NIISphSolver_1/'):
        #Search for the Radius property
        if element.get('name') == 'ParticleSize':
          #Set the radius value to the sampled value
          element.set('val',str(varDict[var]))

        #Change where the measurements and the output data is stored in the input file to match RAVEN location
        #Search for the Base properties
        for elementBase in root.findall('./properties/Base/'):
          #Search for the SceneFilePath property
          if elementBase.get('name') == 'SceneFilePath':
            #Set the SceneFilePath
            elementBase.set('val', str(originalPath))

          #Search for the SaveDir property
          if elementBase.get('name') == 'SaveDir':
            #Create and set SaveDir
            #NeutrinoInput.nescene needs to be changed to the Neutrino input file name
            savePath = originalPath.replace("NeutrinoInput.nescene","",1)
            elementBase.set('val',str(savePath))

          if elementBase.get('name') == 'CacheDir':
            #Create and set CacheDir
            #NeutrinoInput.nescene needs to be changed to the Neutrino input file name
            cachePath = originalPath.replace("NeutrinoInput.nescene","",1)
            elementBase.set('val',str(cachePath))

        #Search for the Measurement field properties
        #MeasurementField_1 name may need to be changed based on Neutrino input file
        for elementMeas in root.findall('./properties/Scene/MeasurementField_1/'):
          #Search for the exportPath property
          if elementMeas.get('name') == 'exportPath':
            #Create and set the exportPath
            #NeutrinoInput.nescene needs to be changed to the Neutrino input file name
            exportPath = originalPath.replace("NeutrinoInput.nescene","",1)
            exportPath = exportPath + r"\Measurements\results.csv"
            elementMeas.set('val',str(exportPath))


    # Now we can re-write the input file
    tree.write(originalPath)

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
    # NeutrinoInput needs to be the name of the Neutrino Input file
    # Name of results file name needs to be the same as in the createNewInput function
    outputPath = os.path.join(workingDir, "NeutrinoInput", "Measurements", "results.csv")

    #Change the output path so RAVEN can read the output
    newOutputPath = os.path.join(workingDir, output)

    # check that the output file exists
    '''if not os.path.exists(outputPath):
      print('Results file does not exist. OK if during test.')
      return newOutputPath'''

    # open original output file (the working directory is provided)
    with open(outputPath,"r+") as outputFile:
      #Open the new output file so the results can be written to it and put in the form for RAVEN to read
      with open(newOutputPath + ".csv", 'w') as resultsFile:

        lines = outputFile.readlines()

        #Needed for RAVEN to read output
        #These need to match RAVEN input file output names
        resultsFile.write('time,result\n')

        #Write Neutrino results to a new file for RAVEN
        for line in lines:
          resultsFile.write(line)



    return newOutputPath
