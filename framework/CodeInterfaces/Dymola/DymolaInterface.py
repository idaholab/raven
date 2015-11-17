'''
Created on Nov 16, 2015

@author: Jong Suk Kim

comments: Interface for Dymola Simulation

Modelica is "a non-proprietary, object-oriented, equation-based language to conveniently model complex physical systems containing,
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
'''

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import sys
import math
import scipy.io
import csv
import re
import copy
import numpy

from six import string_types
from CodeInterfaceBaseClass import CodeInterfaceBase

class DymolaInterface(CodeInterfaceBase):
  '''Provides code to interface RAVEN to Dymola'''

  def __init__(self):
    '''Initializes the GenericCode Interface.
       @ In, None
       @Out, None
    '''

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

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None):
    """
    See base class.  Collects all the clargs and the executable to produce the command-line call.
    Returns tuple of commands and base file name for run.
    Commands are a list of tuples, indicating parallel/serial and the execution command to use.
    @ In, inputFiles, the input files to be used for the run
    @ In, executable, the executable to be run
    @ In, clargs, command-line arguments to be used
    @ In, fargs, in-file changes to be made
    @Out, tuple( list(tuple(serial/parallel, exec_command)), outFileRoot string)
    """
    # Find the first file in the inputFiles that is a text file, which is what we need to work with.
    found = False
    for index, inputFile in enumerate(inputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('Dymola INTERFACE ERROR -> A TEXT file was not found in the input files!')

    # Build an output file name of the form: rawout~<Base Name>, where base name is generated from the
    #   input file passed in: /path/to/file/<Base Name>.ext. 'rawout' indicates that this is the direct
    #   output from running the Dymola executable.
    outputfile = 'rawout~' + inputFiles[index].getBase()
    executeCommand = [('parallel', executable +' -s '+ inputFiles[index].getFilename() +' '+ outputfile+ '.mat')]

    return executeCommand, outputfile

  def _isValidInput(self, inputFile):
    if inputFile.getExt() in ('txt', 'TXT'):
      return True
    return False

  def getInputExtension(self):
    return ('txt', 'TXT')

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    '''Generate a new Dymola input file (txt format) from the original, changing parameters
       as specified in Kwargs['SampledVars']
    '''

    # Start with the original input file, which we have to find first.
    found = False
    for index, inputFile in enumerate(oriInputFiles):
      if self._isValidInput(inputFile):
        found = True
        break
    if not found:
      raise Exception('Dymola INTERFACE ERROR -> An text (.txt) file was not found in the input files!')

    # Figure out the new file name and put it into the proper place in the return list
    newInputFiles = copy.deepcopy(currentInputFiles)
    originalPath = oriInputFiles[index].getAbsFile()
    newPath = os.path.join(os.path.split(originalPath)[0],
                           "DM" + Kwargs['prefix'] + os.path.split(originalPath)[1])
    newInputFiles[index].setAbsFile(newPath)

    # Define dictionary of parameters and pre-process the values.
    # Each key is a parameter name (including the full model path in Modelica_ dot notation) and
    #   each entry is a parameter value. The parameter name includes array indices (if any) in
    #   Modelica_ representation (1-based indexing). The values must be representable as scalar
    #   numbers (integer or floating point). *True* and *False* (not 'true' and 'false') are
    #   automatically mapped to 1 and 0. Enumerations must be given explicitly as the unsigned integer
    #   equivalent. Strings, functions, redeclarations, etc. are not supported.
    varDict = Kwargs['SampledVars']
    for key, value in varDict.items():
      if isinstance(value, bool):
        varDict[key] = 1 if value else 0
      assert not isinstance(value, numpy.ndarray), ("Arrays must be split "
        "into scalars for the simulation initialization file.")
      assert not isinstance(value, string_types), ("Strings cannot be "
        "used as values in the simulation initialization file.")

    # Aliases for some regular sub-expressions.
    u = '\d+' # Unsigned integer
    i = '[+-]?' + u # Integer
    f = i + '(?:\.' + u + ')?(?:[Ee][+-]' + u + ')?' # Floating point number

    # Possible regular expressions for a parameter specification (with '%s' for
    #   the parameter name)
    patterns = [# Dymola 1- or 2-line parameter specification
                (r'(^\s*%s\s+)%s(\s+%s\s+%s\s+%s\s+%s\s*#\s*%s\s*$)'
                 % (i, f, f, f, u, u, '%s')),
                (r'(^\s*)' + i + '(\s*#\s*%s)'),
                (r'(^\s*)' + f + '(\s*#\s*%s)'),
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
    with open(newPath, 'w') as src:
      src.write(text)

    return newInputFiles

  def finalizeCodeOutput(self, command, output, workingDir):
    '''Called by RAVEN to modify output files (if needed) so that they are in a proper form.
       In this case, the default .mat output needs to be converted to .csv output, which is the
       format that RAVEN can communicate with.
       @ In, currentInputFiles, list, the current input files
       @ In, output, string, the Output name root
       @ In, workingDir, string, actual working dir
       @ Return is optional, in case the root of the output file gets changed as in this method.
    '''
    self._vars = {}
    self._blocks = []
    self._namesData1 = []
    self._namesData2 = []
    self._timeSeriesData1 = []
    self._timeSeriesData2 = []

    # Load the output file (.mat file) that have been generated by running Dymola executable and
    #   store the data in this file to variable 'mat'.
    matSourceFileName = os.path.join(workingDir, output)
    matSourceFileName += '.mat'
    mat = scipy.io.loadmat(matSourceFileName, chars_as_strings=False)

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
          self._vars[names[i]] = (descr[i], d, c, s)
          if not d in self._blocks:
            self._blocks.append(d)
        else:
          self._absc = (names[i], descr[i])

      # Extract the trajectory for the variable 'Time' and store the data in the variable 'timeSteps'.
      timeSteps = mat['data_2'][0]

      # Compute the number of output points of trajectory (time series data).
      numOutputPts = timeSteps.shape[0]

      # Convert the variable type of 'timeSteps' from '1-d array' to '2-d array'.
      timeStepsArray = numpy.array([timeSteps])

      # Extract the names and output points of all variables and store them in the variables:
      #  - self._namesData1: Names of parameters
      #  - self._namesData2: Names of the variables that are not parameters
      #  - self._timeSeriesData1: Trajectories (time series data) of 'self._namesData1'
      #  - self._timeSeriesData2: Trajectories (time series data) of 'self._namesData2'
      for (k,v) in self._vars.items():
        dataValue = mat['data_%d' % (v[1])][v[2]]
        if v[3] < 0:
          dataValue = dataValue * -1
        if v[1] == 1:
          self._namesData1.append(k)
          self._timeSeriesData1.append(dataValue)
        elif v[1] == 2:
          self._namesData2.append(k)
          self._timeSeriesData2.append(dataValue)
        else:
          raise Exception('File structure not supported!')
      timeSeriesData1 = numpy.array(self._timeSeriesData1)
      timeSeriesData2 = numpy.array(self._timeSeriesData2)

      # Recombine the names of the variables and insert the variable 'Time'.
      # Order of the variable names should be 'Time', self._namesData1, self._namesData2.
      # Also, convert the type of the resulting variable from 'list' to '2-d array'.
      varNames = numpy.array([[self._absc[0]] + self._namesData1 + self._namesData2])

      # Compute the number of parameters.
      sizeParams = timeSeriesData1.shape[0]

      # Create a 2-d array whose size is 'the number of parameters' by 'number of ouput points of the trajectories'.
      # Fill each row in a 2-d array with the parameter value.
      Data1Array =  numpy.full((sizeParams,numOutputPts),1)
      for n in range(sizeParams):
        Data1Array[n,:] = timeSeriesData1[n,0]

      # Create an array of trajectories, which are to be written to CSV file.
      varTrajectories = numpy.matrix.transpose(numpy.concatenate((timeStepsArray,Data1Array,timeSeriesData2), axis=0))

      # Define the name of the CSV file.
      sourceFileName = os.path.join(workingDir, output)         # The source file comes in without extension on it
      print('sourcefilename:',sourceFileName)
      destFileName = sourceFileName.replace('rawout~', 'out~')  # When write the CSV file, change rawout~ to out~
      destFileName += '.csv' # Add the file extension .csv

      # Write the CSV file.
      with open(destFileName,"wb") as csvFile:
        resultsWriter = csv.writer(csvFile, delimiter=str(u','), quotechar=str(u'"'))
        resultsWriter.writerows(varNames)
        resultsWriter.writerows(varTrajectories)
    else:
      raise Exception('File structure not supported!')

    return os.path.splitext(destFileName)[0]   # Return the name without the .csv on it as RAVEN will add it later.
