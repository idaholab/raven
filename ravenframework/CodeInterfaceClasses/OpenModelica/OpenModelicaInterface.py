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
r"""
Created on May 22, 2015

@author: bobk

comments: Interface for OpenModelica Simulation

OpenModelica (http://www.openmodelica.org) is an open souce implementation of the Modelica simulation language.
This module provides an interface that allows RAVEN to utilize models built using OpenModelica.

General flow:

A Modelica model is specified in a text file.  For example (BouncingBall.mo):

--- BEGIN MODEL FILE ---
model BouncingBall
  parameter Real e=0.7 "coefficient of restitution";
  parameter Real g=9.81 "gravity acceleration";
  Real h(start=1) "height of ball";
  Real v "velocity of ball";
  Boolean flying(start=true) "true, if ball is flying";
  Boolean impact;
  Real v_new;
  Integer foo;

equation
  impact = h <= 0.0;
  foo = if impact then 1 else 2;
  der(v) = if flying then -g else 0;
  der(h) = v;

  when {h <= 0.0 and v <= 0.0,impact} then
    v_new = if edge(impact) then -e*pre(v) else 0;
    flying = v_new > 0;
    reinit(v, v_new);
  end when;

end BouncingBall;
--- END MODEL FILE ---

When OpenModelica simulates this file it is read, and from it C code is generated and then built into a platform-specific
executable that does the calculations.  The parameters from the model are written into an XML file (by default
BouncingBall_init.xml).  After the executable is generated it may be run multiple times.  There are several way to vary
input parameters:

  1) Modify the model file and re-build the simulation executable.
  2) Change the value(s) in the input XML generated as part of the model build process.
  3) Use a command-line parameter '-override <var>=<value>' to substitute something for the value in the XML input
  4) Use a command-line parameter '-overrideFile=<file>' to use a completely different XML input file.
  5) Use a command-line parameter '-iif=<file>' to specify initial conditions using a file in the .MAT format used
     for output.
  6) Paramters in the model file may also be overriden when the simulation executable is built using an OpenModelica
     shell command of the form: simulate(<model>, simflags="-override <var>=<value>)

For RAVEN purposes, this interface code will use option (2).  Variation of parameters may be done by editing the init
file and then re-running the model.  The OpenModelica shell provides a method that may be used to change a parameter:

  setInitXmlStartValue(<input file>, <parameter>, <new value>, <output file>)

To change the initial height of the bouncing ball to 5.0 in the above model, and write it back to a different input
file BouncingBall_new_init.xml.  It is also possible to write the output over the original file:

  setInitXmlStartValue("BouncingBall_init.xml", "h", "5.0", "BouncingBall_new_init.xml")

The output of the model may be configured to a number of output formats.  The default is a binary file <Model Name>_res.mat
(BouncingBall_res.mat for this example).  CSV is also an option, which we will use because that is what RAVEN likes best.
The output type may be set when generating the model executable.

To generate the executable, use the OM Shell:
  The generate phase builds C code from the modelica file and then builds an executable.  It also generates an initial
  init file <model>_init.xml for <model>.mo.  This xml can then be modified and used to re-run the simulation.

        (Using the OpenModelica Shell, load the base Modelica library)
        >> loadModel(Modelica)
        (Load the model to build)
        >> loadFile("BouncingBall.mo")
        (Build the model into an executable and generate the initial XML input file specifying CSV output)
        >> buildModel(BouncingBall, outputFormat="csv")
        (Copy the input file to BouncingBall_new_init.xml, changing the initial value of h to 5.0)
  >> setInitXmlStartValue("BouncingBall_init.xml", "h", "5.0", "BouncingBall_new_init.xml")

Alternatively, the python OM Shell interface may be used:

  >>> from OMPython import OMCSession                # Get the library with OMCSession
  >>> omc = OMCSession()                             # Creates a new shell session
  >>> omc.execute(<OpenModelica Shell Command>)      # General form
  >>> omc.execute("loadModel(Modelica)")             # Load base Modelica library
  >>> omc.execute("loadFile(\"BouncingBall.mo\")")   # Load BouncingBall.mo model
        >>> omc.execute("buildModel(BouncingBall, outputFormat=\"csv\")")  # Build the model (but not run it), setting for csv file output
  >>> omc.execute("setInitXmlStartValue(\"BouncingBall_init.xml\",         # Make a new input file with h = 5.0
    \"h\", \ "5.0\", \"BouncingBall_new_init.xml\")")
  >>> omc.execute("system(\"BouncingBall.exe\")")    # Run the model executable
  >>> omc.execute("simulate(BouncingBall, stopTime=10.0)")                 # Run simulation, changing stop time to 10.0

An alternative would be to take the default .mat output type and use the open source package based on SciPy called DyMat
(https://pypi.python.org/pypi/DyMat) may be used to convert these output files to human-readable forms (including CSV).  For example:

  <Python Code>
  import DyMat, DyMat.Export                      # Import necessary modules
  d = DyMat.DyMatFile("BouncingBall_res.mat")     # Load the result file
  d.names()                                       # Prints out the names in the result file
  DyMat.Export.export("CSV", d, ["h", "flying"])  # Export variables h and flying to a CSV file

Example of multiple parameter override (option 3 above): BouncingBall.exe -override "h=7,g=7,v=2"

To use RAVEN, we need to be able to perturb the input and output files from the defaults.  The command line
form of this is: (Where the output file will be of the type originally configured)

  <executable> -f <init file xml> -r <outputfile>
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import copy
import shutil
import tempfile
from ravenframework.utils import utils
import xml.etree.ElementTree as ET
#from OMPython import OMCSession    # Get the library with Open Modelica Session (needed to run OM stuff)

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase

class OpenModelica(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to OpenModelica
  """
  def __init__(self):
    """
      Initializes the GenericCode Interface.
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    #  Generate the command to run OpenModelica.  The form of the command is:
    #
    #    <executable> -f <init file xml> -r <outputfile>
    #
    #  Where:
    #     <executable>     The executable generated from the Modelica model file (.mo extension)
    #     <init file xml>  XML file containing the initial model parameters.  We will perturb this from the
    #                          one originally generated as part of the model build process, which is
    #                          typically called <model name>_init.xml.
    #     <outputfile>     The simulation output.  We will use the model generation process to set the format
    #                          of this to CSV, though there are other formats available.

  def generateCommand(self, inputFiles, executable, clargs=None,fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False
    # Find the first file in the inputFiles that is an XML, which is what we need to work with.
    for index, inputFile in enumerate(inputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('OpenModelica INTERFACE ERROR -> An XML file was not found in the input files!')
    # Build an output file name of the form: rawout~<Base Name>, where base name is generated from the
    #   input file passed in: /path/to/file/<Base Name>.ext.  'rawout' indicates that this is the direct
    #   output from running the OpenModelica executable.
    outputfile = 'rawout~' + inputFiles[index].getBase() #os.path.splitext(os.path.basename(inputFiles[index]))[0]
    returnCommand = [('parallel',executable+' -f '+inputFiles[index].getFilename() + ' -r '+ outputfile + '.csv')], outputfile
    return returnCommand

  def _isValidInput(self, inputFile):
    """
      Check if an input file is a xml file, with an extension of .xml, .XML or .Xml .
      @ In, inputFile, string, the file name to be checked
      @ Out, valid, bool, 'True' if an input file has an extension of .'xml', 'XML' or 'Xml', otherwise 'False'.
    """
    valid = False
    if inputFile.getExt() in ('xml', 'XML', 'Xml'):
      valid = True
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
      @ In , currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In , oriInputFiles, list, list of the original input files
      @ In , samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In , Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    # Since OpenModelica provides a way to do this (the setInitXmlStartValue described above), we'll
    #   use that.  However, since it can only change one value at a time we'll have to apply it multiple
    #   times.  Start with the original input file, which we have to find first.
    found = False
    for index, inputFile in enumerate(oriInputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('OpenModelica INTERFACE ERROR -> An XML file was not found in the input files!')

    # Figure out the new file name and put it into the proper place in the return list
    #newInputFiles = copy.deepcopy(currentInputFiles)
    originalPath = oriInputFiles[index].getAbsFile()
    #newPath = os.path.join(os.path.split(originalPath)[0],
    #                       "OM" + Kwargs['prefix'] + os.path.split(originalPath)[1])
    #newInputFiles[index].setAbsFile(newPath)

    # Since the input file is XML we can load and edit it directly using etree
    # Load the original XML into a tree:
    tree = ET.parse(originalPath)

    # Look at all of the variables in the XML and see if we have changes
    #   in our dictionary.
    varDict = Kwargs['SampledVars']
    for elem in tree.findall('.//ScalarVariable'):
      if (elem.attrib['name'] in varDict.keys()):
        # Should contain one sub-element called 'Real' 'Integer' or 'Boolean' (May be others)
        for subelem in elem:
          if 'start' in subelem.attrib.keys():
            # Change the start value to the provided one
            subelem.set('start', str(varDict[elem.attrib['name']]))
    # Now write out the modified file
    tree.write(currentInputFiles[index].getAbsFile())
    return currentInputFiles


  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Called by RAVEN to modify output files (if needed) so that they are in a proper form.
      In this case, OpenModelica CSV output comes with trailing commas that RAVEN doesn't
      like.  So we have to strip them.  Also, the first line (with the variable names)
      has those names enclosed in double quotes (which we have to remove)
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, destFileName, string, present in case the root of the output file gets changed in this method.
    """
    # Make a new temporary file in the working directory and read the lines from the original CSV
    #   to it, stripping trailing commas in the process.
    tempOutputFD, tempOutputFileName = tempfile.mkstemp(dir = workingDir, text = True)
    sourceFileName = os.path.join(workingDir, output)         # The source file comes in without .csv on it
    print('sourcefilename:',sourceFileName)
    destFileName = sourceFileName.replace('rawout~', 'out~')  # When fix the CSV, change rawout~ to out~
    sourceFileName += '.csv'
    with open(sourceFileName) as inputFile:
      for line in inputFile:
        # Line ends with a comma followed by a newline
        #XXX toBytes seems to be needed here in python3, despite the text = True
        os.write(tempOutputFD, utils.toBytes(line.replace('"','').strip().strip(',') + '\n'))
    os.close(tempOutputFD)
    shutil.move(tempOutputFileName, destFileName + '.csv')
    return destFileName   # Return the name without the .csv on it...RAVEN will add it
