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
Created on Nov 24, 2015

@author: Jong Suk Kim

comments: Interface for Dymola Simulation

Modelica is an object-oriented, equation-based language to conveniently model complex physical systems containing,
e.g., mechanical, electrical, electronic, hydraulic, thermal, control, electric power or process-oriented subcomponents.
Modelica models (with a file extension of .mo) are built, translated (compiled), and simulated in Dymola (http://www.modelon.com/products/dymola/),
which is a commercial modeling and simulation environment based on the Modelica modeling language.
This module provides an interface that allows RAVEN to utilize Modelica models built using Dymola.

General flow:

A Modelica model is built and implemented in Dymola. For example (BouncingBall.mo):

--- BEGIN MODEL FILE ---
model BouncingBall
  parameter Real e=0.7 "coefficient of restitution";
  parameter Real g=9.81 "gravity acceleration";
  parameter Real hstart = 10 "height of ball at time zero";
  parameter Real vstart = 0 "velocity of ball at time zero";
  Real h(start=hstart,fixed=true) "height of ball";
  Real v(start=vstart,fixed=true) "velocity of ball";
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

  annotation (uses(Modelica(version="3.2.1")),
    experiment(StopTime=10, Interval=0.1),
    __Dymola_experimentSetupOutput);

end BouncingBall;
--- END MODEL FILE ---

When a modelica model, e.g., BouncingBall model, is implemented in Dymola, the platform dependent C-code from a Modelica model
and the corresponding executable code (i.e., by default dymosim.exe on the Windows operating system) are generated for simulation.
A separate TEXT file (by default dsin.txt) containing model parameters and initial conditions are also generated as part of the build process.
After the executable is generated, it may be run multiple times (with Dymola license). There are several ways to vary input parameters:

  1) Modify the model file and re-build the simulation executable.
  2) Change the value(s) in the 'text' input file generated as part of the model build process.
  3) Use a completely different text input file for each run.
  4) Use the Matlab script file (m-file) or Python Dymola Shell interface to manipulate (perturb) input parameters.

For RAVEN purposes, this interface code will use option (2). Variation of parameters may be done by editing the input
file (dsin.txt) and then re-running the model executable (by default dymosim.exe).

An executable (dymosim.exe) and a simulation initialization file (dsin.txt) can be generated after either translating or simulating the Modelica
model (BouncingBall.mo) using the Dymola Graphical User Interface (GUI) or Dymola Application Programming Interface (API)-routines.
The output of the model is a binary file 'BouncingBall.mat' if the simulation is run in Dymola GUI or by using Dymola API-routines.
If the generated executable code is trigged directly from a command prompt, the output file is always named as 'dsres.mat'.

To change the initial height of the bouncing ball to 5.0 in the above model, we need to read and modify its value
from the 'dsin.txt,' and write it back to a different input file, e.g., DMdsin.txt. This .txt file can then be used
to re-run the simulation.

The default .mat output type needs to be converted to human-readable forms, i.e., .csv output. Note that the Python
interface that comes with the Dymola distribution cannot be used, especially when running the model with RAVEN on
cluster, as the Python interface is currently only supported on Windows.

To use RAVEN, we need to be able to perturb the input and output files from the defaults. The command line form
of this is:

  <executable> -s <dsin file text> <outputfile>
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import math
import scipy.io
import csv
import re
import copy
import numpy
import pandas as pd

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ravenframework.utils import mathUtils

class Dymola(CodeInterfaceBase):
  """
    Provides code to interface RAVEN to Dymola
  """

  def __init__(self):
    """
      Initializes the GenericCode Interface.
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.variablesToLoad = [] # the variables that should be loaded from the mat file (by default, all of them)

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    child = xmlNode.find("outputVariablesToLoad")
    if child is not None:
      self.variablesToLoad = [var.strip() for var in child.text.split()]

  #  Generate the command to run Dymola. The form of the command is:
  #
  #     <executable> -s <dsin file txt> <outputfile>
  #
  #  where:
  #     <executable>     The executable generated from the Modelica model file (.mo extension)
  #     <dsin file txt>  Text file containing the initial model parameters (as well as the start
  #                          values of variables.  We will perturb this from the one originally
  #                          generated as part of the model build process, which is called dsin.txt.
  #     <outputfile>     The simulation output, which is .mat file.

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

    # Find the file in the inputFiles that has the type "DymolaInitialisation", which is what we need to work with.
    foundInit = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getType() == "DymolaInitialisation":
        foundInit = True
        indexInit = index
    if not foundInit:
      raise Exception('Dymola INTERFACE ERROR -> None of the input files has the type "DymolaInitialisation"!')
    # Build an output file name of the form: rawout~<Base Name>, where base name is generated from the
    #   input file passed in: /path/to/file/<Base Name>.ext. 'rawout' indicates that this is the direct
    #   output from running the Dymola executable.
    outputfile = 'rawout~' + inputFiles[indexInit].getBase()
    executeCommand = [('parallel', executable +' -s '+ inputFiles[indexInit].getFilename() +' '+ outputfile+ '.mat')]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def getInputExtension(self):
    """
      Return a tuple of possible file extensions for a simulation initialization file (i.e., dsin.txt).
      @ In, None
      @ Out, validExtensions, tuple, tuple of valid extensions
    """
    validExtensions = ('txt', 'TXT')
    return validExtensions

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      Generate a new Dymola input file (txt format) from the original, changing parameters
      as specified in Kwargs['SampledVars']. In addition, it creaes an additional input file including the vector data to be
      passed to Dymola.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    # Start with the original input file, which we have to find first.
    # The types have to be "DymolaInitialisation" and "DymolaVectors"
    foundInit = False
    foundVect = False
    for index, inputFile in enumerate(oriInputFiles):
      if inputFile.getType() == "DymolaInitialisation":
        foundInit = True
        indexInit = index
      if inputFile.getType() == "DymolaVectors":
        foundVect = True
        indexVect = index
    if not foundInit:
      raise Exception('Dymola INTERFACE ERROR -> None of the input files has the type "DymolaInitialisation"!')
    if not foundVect:
      print('Dymola INTERFACE WARNING -> None of the input files has the type "DymolaVectors"! ')
    # Figure out the new file name and put it into the proper place in the return list
    #newInputFiles = copy.deepcopy(currentInputFiles)
    originalPath = oriInputFiles[indexInit].getAbsFile()
    #newPath = os.path.join(os.path.split(originalPath)[0], "DM" + Kwargs['prefix'] + os.path.split(originalPath)[1])
    #currentInputFiles[index].setAbsFile(newPath)
    # Define dictionary of parameters and pre-process the values.
    # Each key is a parameter name (including the full model path in Modelica_ dot notation) and
    #   each entry is a parameter value. The parameter name includes array indices (if any) in
    #   Modelica_ representation (1-based indexing). The values must be representable as scalar
    #   numbers (integer or floating point). *True* and *False* (not 'true' and 'false') are
    #   automatically mapped to 1 and 0. Enumerations must be given explicitly as the unsigned integer
    #   equivalent. Strings, functions, redeclarations, etc. are not supported.
    varDict = Kwargs['SampledVars']

    vectorsToPass= {}
    for key, value in list(varDict.items()):
      if isinstance(value, bool):
        varDict[key] = 1 if value else 0
      if isinstance(value, numpy.ndarray):
        # print warning here (no access to RAVEN Message Handler)
        print("Dymola INTERFACE WARNING -> Dymola interface found vector data to be passed. If %s" %key)
        print("                            is supposed to go into the simulation initialisation file of type")
        print("                            'DymolaInitialisation' the array must be split into scalars.")
        print("                            => It is assumed that the array goes into the input file with type 'DymolaVectors'")
        if not foundVect:
          raise Exception('Dymola INTERFACE ERROR -> None of the input files has the type "DymolaVectors"! ')
        # extract dict entry
        vectorsToPass[key] = varDict.pop(key)
      assert not type(value).__name__ in ['str','bytes','unicode'], ("Strings cannot be "
        "used as values in the simulation initialization file.")

    # create aditional input file for vectors if needed
    if bool(vectorsToPass):
      with open(currentInputFiles[indexVect].getAbsFile(), 'w') as Fvect:
        Fvect.write("#1\n")
        for key, value in sorted(vectorsToPass.items()) :
          inc = 0
          Fvect.write("double %s(%s,2) #Comments here\n" %(key, len(value)))
          for val in value:
            Fvect.write("%s\t%s\n" %(inc,val))
            inc += 1

    # Do the search and replace in input file "DymolaInitialisation"
    # Aliases for some regular sub-expressions.
    u = '\\d+' # Unsigned integer
    i = '[+-]?' + u # Integer
    f = i + '(?:\\.' + u + ')?(?:[Ee][+-]' + u + ')?' # Floating point number

    # Possible regular expressions for a parameter specification (with '%s' for
    #   the parameter name)
    patterns = [# Dymola 1- or 2-line parameter specification
                (r'(^\s*%s\s+)%s(\s+%s\s+%s\s+%s\s+%s\s*#\s*%s\s*$)'
                 % (i, f, f, f, u, u, '%s')),
                (r'(^\s*)' + i + r'(\s*#\s*%s)'),
                (r'(^\s*)' + f + r'(\s*#\s*%s)'),
                # From Dymola:
                # column 1: Type of initial value
                #           = -2: special case: for continuing simulation
                #                               (column 2 = value)
                #           = -1: fixed value   (column 2 = fixed value)
                #           =  0: free value, i.e., no restriction
                #                               (column 2 = initial value)
                #           >  0: desired value (column 1 = weight for
                #                                           optimization
                #                                column 2 = desired value)
                #                 use weight=1, since automatic scaling usually
                #                 leads to equally weighted terms
                # column 2: fixed, free or desired value according to column 1.
                # column 3: Minimum value (ignored, if Minimum >= Maximum).
                # column 4: Maximum value (ignored, if Minimum >= Maximum).
                #           Minimum and maximum restrict the search range in
                #           initial value calculation. They might also be used
                #           for scaling.
                # column 5: Category of variable.
                #           = 1: parameter.
                #           = 2: state.
                #           = 3: state derivative.
                #           = 4: output.
                #           = 5: input.
                #           = 6: auxiliary variable.
                # column 6: Data type of variable.
                #           = 0: real.
                #           = 1: boolean.
                #           = 2: integer.
               ]
    # These are tried in order until there is a match. The first group or pair
    #   of parentheses contains the text before the parameter value and the second
    #   contains the text after it (minus one space on both sides for clarity).

    # Read the file.
    with open(originalPath, 'r') as src:
      text = src.read()

    # Set the parameters.
    for name, value in varDict.items():
      # skip in the special key for the index mapper
      if name == '_indexMap':
        continue
      namere = re.escape(name) # Escape the dots, square brackets, etc.
      for pattern in patterns:
        text, n = re.subn(pattern % namere, r'\g<1>%s\2' % value, text, 1,
                          re.MULTILINE)
        if n == 1:
          break
      else:
        raise AssertionError(
          "Parameter %s does not exist or is not formatted as expected "
          "in %s." % (name, originalPath))

    # Re-write the file.
    with open(currentInputFiles[indexInit].getAbsFile(), 'w') as src:
      src.write(text)

    return currentInputFiles

  def checkForOutputFailure(self, output, workingDir):
    """
      Sometimes (e.g. when the license file is missing) the command returns 0 despite failing.
      Check for creation of the "success" file as a determination of success for Dymola runs.
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    try:
      open(os.path.join(workingDir, 'success'), 'r')
    except FileNotFoundError:
      return True
    return False

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
    _vars = {}
    _blocks = []
    _namesData1 = []
    _namesData2 = []
    _timeSeriesData1 = []
    _timeSeriesData2 = []

    # Load the output file (.mat file) that have been generated by running Dymola executable and
    #   store the data in this file to variable 'mat'.
    matSourceFileName = os.path.join(workingDir, output)
    matSourceFileName += '.mat'
    ###################################################################
    #FIXME: LOADMAT HAS A DIFFERENT BEHAVIOR IN SCIPY VERSION >= 0.18 #
    #if int(scipy.__version__.split(".")[1])>17:
    #  warnings.warn("SCIPY version >0.17.xx has a different behavior in reading .mat files!")
    mat = scipy.io.loadmat(matSourceFileName, chars_as_strings=False)

    ###################################################################

    # Define the functions that extract strings from the matrix:
    #  - strMatNormal: for parallel string
    #  - strMatTrans:  for vertical string
    # These functions join the strings together, resulting in one string in each row, and remove
    #   trailing whitespace.
    strMatNormal = lambda a: [''.join(s).rstrip() for s in a]
    strMatTrans  = lambda a: [''.join(s).rstrip() for s in zip(*a)]

    # Define the function that returns '1.0' with the sign of 'x'
    sign = lambda x: math.copysign(1.0, x)

    # Check the structure of the output file.
    try:
      fileInfo = strMatNormal(mat['Aclass'])
    except KeyError:
      raise Exception('File structure not supported!')

    # Check the version of the output file (version 1.1).
    if fileInfo[1] == '1.1' and fileInfo[3] == 'binTrans':
      names = strMatTrans(mat['name']) # names
      descr = strMatTrans(mat['description']) # descriptions
      for i in range(len(names)):
        d = mat['dataInfo'][0][i] # data block
        x = mat['dataInfo'][1][i] # column (original)
        c = abs(x)-1  # column (reduced)
        s = sign(x)   # sign
        if c:
          _vars[names[i]] = (descr[i], d, c, float(s))
          if not d in _blocks:
            _blocks.append(d)
        else:
          _absc = (names[i], descr[i])

      # Extract the trajectory for the variable 'Time' and store the data in the variable 'timeSteps'.
      timeSteps = mat['data_2'][0]

      # Compute the number of output points of trajectory (time series data).
      numOutputPts = timeSteps.shape[0]

      # Convert the variable type of 'timeSteps' from '1-d array' to '2-d array'.
      timeStepsArray = numpy.array([timeSteps])

      # Extract the names and output points of all variables and store them in the variables:
      #  - _namesData1: Names of parameters
      #  - _namesData2: Names of the variables that are not parameters
      #  - _timeSeriesData1: Trajectories (time series data) of '_namesData1'
      #  - _timeSeriesData2: Trajectories (time series data) of '_namesData2'
      for (k,v) in _vars.items():
        readIt = True
        if len(self.variablesToLoad) > 0 and k not in self.variablesToLoad:
          readIt = False
        if readIt:
          dataValue = mat['data_%d' % (v[1])][v[2]]
          if v[3] < 0:
            dataValue = dataValue * -1.0
          if v[1] == 1:
            _namesData1.append(k)
            _timeSeriesData1.append(dataValue)
          elif v[1] == 2:
            _namesData2.append(k)
            _timeSeriesData2.append(dataValue)
          else:
            raise Exception('File structure not supported!')
      timeSeriesData1 = numpy.array(_timeSeriesData1)
      timeSeriesData2 = numpy.array(_timeSeriesData2)

      # The csv writer places quotes arround variables that contain a ',' in the name, i.e.
      # a, "b,c", d would represent 3 variables 1) a 2) b,c 3) d. The csv reader in RAVEN does not
      # suport this convention.
      # => replace ',' in variable names with '@', i.e.
      # a, "b,c", d will become a, b@c, d
      for mylist in [_namesData1, _namesData2]:
        for i in range(len(mylist)):
          if ',' in mylist[i]:
            mylist[i]  = mylist[i].replace(',', '@')

      # Recombine the names of the variables and insert the variable 'Time'.
      # Order of the variable names should be 'Time', _namesData1, _namesData2.
      # Also, convert the type of the resulting variable from 'list' to '2-d array'.
      varNames = numpy.array([[_absc[0]] + _namesData1 + _namesData2])

      # Compute the number of parameters.
      sizeParams = timeSeriesData1.shape[0]

      # Create a 2-d array whose size is 'the number of parameters' by 'number of ouput points of the trajectories'.
      # Fill each row in a 2-d array with the parameter value.
      Data1Array =  numpy.full((sizeParams,numOutputPts),1.)
      for n in range(sizeParams):
        Data1Array[n,:] = timeSeriesData1[n,0]

      # Create an array of trajectories, which are to be written to CSV file.
      varTrajectories = numpy.matrix.transpose(numpy.concatenate((timeStepsArray,Data1Array,timeSeriesData2), axis=0))
      # create output response dictionary
      t = pd.Series(varTrajectories[:,0])
      m = t.duplicated()
      if len(t[m]):
        # duplicated values
        tIndex = None
        iIndex = 1
        for i in range(len(t[m])):
          index = t[m].index[i]
          if tIndex is None:
            tIndex = t[index]
          else:
            if mathUtils.compareFloats(tIndex, t[index], tol=1.0E-15):
              iIndex += 1
            else:
              iIndex = 1
              tIndex = t[index]
          t[index] = t[index] + numpy.finfo(float).eps*t[index]*iIndex
        varTrajectories[:,0] = t.to_numpy()
      response = {var:varTrajectories[:,i] for (i, var) in enumerate(varNames[0])}
    else:
      raise Exception('File structure not supported!')
    #release memory
    del _vars
    del _blocks
    del _namesData1
    del _namesData2
    del _timeSeriesData1
    del _timeSeriesData2
    del _absc
    del Data1Array
    del timeSeriesData1
    del timeSeriesData2
    return response
