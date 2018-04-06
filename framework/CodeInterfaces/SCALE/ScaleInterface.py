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
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import shutil
from utils import utils
import xml.etree.ElementTree as ET

from GenericCodeInterface import GenericParser
from CodeInterfaceBaseClass import CodeInterfaceBase

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
    
  
  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    CodeInterfaceBase._readMoreXML(self,xmlNode)
    self.sequence = [] # this contains the sequence that needs to be run. For example, ['triton'] or ['origen'] or ['triton','origen']
    self.include=''
    self.tilastDict={} #{'folder_name':'tilast'} - this dictionary contains the last simulation time of each branch, this is necessary to define the correct restart time
    self.branch = {} #{'folder_name':['variable branch','variable value']} where variable branch is the variable sampled for the current branch e.g. {det_1:[timeloca, 200]}
    self.values = {} #{'folder_name':[['variable branch_1','variable value_1'],['variable branch_2','variable value_2']]} for each DET sampled variables
    self.printInterval = ''  #value of the print interval
    self.boolOutputVariables=[] #list of MAAP5 boolean variables of interest
    self.contOutputVariables=[] #list of MAAP5 continuous variables of interest
###########
    self.multiBranchOccurred=[]
###########
    for child in xmlNode:
      if child.tag == 'includeForTimer':
        if child.text != None:
          self.include = child.text
      if child.tag == 'boolMaapOutputVariables':
        #here we'll store boolean output MAAP variables to look for
        if child.text != None:
          self.boolOutputVariables = child.text.split(',')
      if child.tag == 'contMaapOutputVariables':
        #here we'll store boolean output MAAP variables to look for
        if child.text != None:
          self.contOutputVariables = child.text.split(',')
      if child.tag == 'stopSimulation':
        self.stop=child.text #this node defines if the MAAP5 simulation stop condition is: 'mission_time' or the occurrence of a given event e.g. 'IEVNT(691)'
    if (len(self.boolOutputVariables)==0) and (len(self.contOutputVariables)==0):
      raise IOError('At least one of two nodes <boolMaapOutputVariables> or <contMaapOutputVariables> has to be specified')

  def finalizeCodeOutput(self, command, output, workingDir):
    """
      finalizeCodeOutput checks MAAP csv files and looks for iEvents and
      continous variables we specified in < boolMaapOutputVariables> and
      contMaapOutputVairables> sections of RAVEN_INPUT.xml file. Both
      < boolMaapOutputVariables> and <contMaapOutputVairables> should be
      contained into csv MAAP csv file
      In case of DET sampler, if a new branching condition is met, the
      method writes the xml for creating the two new branches.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, output csv file containing the variables of interest specified in the input
    """
    csvSimulationFiles=[]
    realOutput=output.split("out~")[1] #rootname of the simulation files
    inp = os.path.join(workingDir,realOutput + ".inp") #input file of the simulation with the full path
    filePrefixWithPath=os.path.join(workingDir,realOutput) #rootname of the simulation files with the full path
    csvSimulationFiles=glob.glob(filePrefixWithPath+".d"+"*.csv") #list of MAAP output files with the evolution of continuous variables
    mergeCSV=csvU.csvUtilityClass(csvSimulationFiles,1,";",True)
    dataDict={}
    dataDict=mergeCSV.mergeCsvAndReturnOutput({'variablesToExpandFrom':['TIME'],'returnAsDict':True})
    timeFloat=dataDict['TIME']
    #Here we'll read evolution of continous variables """
    contVariableEvolution=[] #here we'll store the time evolution of MAAP continous variables
    if len(self.contOutputVariables)>0:
      for variableName in self.contOutputVariables:
        try:
          (dataDict[variableName])
        except:
          raise IOError('define the variable within MAAP5 plotfil: ',variableName)
        contVariableEvolution.append(dataDict[variableName])

     #here we'll read boolean variables and transform them into continous"""
     # if the discrete variables of interest are into the csv file:
    if len(self.boolOutputVariables)>0:
      boolVariableEvolution=[]
      for variable in self.boolOutputVariables:
        variableName=str(variable)
        try:
          (dataDict[variableName])
        except:
          raise IOError('define the variable within MAAP5 plotfil: ',variableName)
        boolVariableEvolution.append(dataDict[variableName])

    allVariableTags=[]
    allVariableTags.append('TIME')
    if (len(self.contOutputVariables)>0):
      allVariableTags.extend(self.contOutputVariables)
    if (len(self.boolOutputVariables)>0):
      allVariableTags.extend(self.boolOutputVariables)

    allVariableValues=[]
    allVariableValues.append(dataDict['TIME'])
    if (len(self.contOutputVariables)>0):
      allVariableValues.extend(contVariableEvolution)
    if (len(self.boolOutputVariables)>0):
      allVariableValues.extend(boolVariableEvolution)

    RAVENoutputFile=os.path.join(workingDir,output+".csv") #RAVEN will look for  output+'.csv'file but in the workingDir, so we need to append it to the filename
    outputCSVfile=open(RAVENoutputFile,"w+")
    csvwriter=csv.writer(outputCSVfile,delimiter=b',')
    csvwriter.writerow(allVariableTags)
    for i in range(len(allVariableValues[0])):
      row=[]
      for j in range(len(allVariableTags)):
        row.append(allVariableValues[j][i])
      csvwriter.writerow(row)
    outputCSVfile.close()
    #os.chdir(workingDir) NEVER CHANGE THE WORKING DIRECTORY

    if 'DynamicEventTree' in self.samplerType:
      dictTimer={}
      for timer in self.timer.values():
        timer='TIM'+str(timer)
        try:
          (dataDict[timer])
        except:
          raise IOError('Please ensure that the timer is defined into the include file and then it is contained into MAAP5 plotfil: ',timer)
        if 1.0 in dataDict[timer].tolist():
          index=dataDict[timer].tolist().index(1.0)
          timerActivation= timeFloat.tolist()[index]
        else:
          timerActivation=-1
        dictTimer[timer]=timerActivation

      #
      #  NOTE THAT THIS ERROR CAN BE WRONG SINCE IT IS POSSIBLE (BRANCHES ON DEMAND) THAT TWO BRANCHES (OR MORE) HAPPEN AT THE SAME TIME! Andrea
      #
      dictTimeHappened = []
      for value in dictTimer.values():
        if value != -1:
          dictTimeHappened.append(value)
      print('DictTimer =', dictTimer)
      print('Events occur at: ', dictTimeHappened)
      #if any([dictTimeHappened.count(value) > 1 for value in dictTimer.values()]): raise IOError('Branch must occur at different times')
      key1 = max(dictTimer.values())
      d1 = dict((v, k) for k, v in dictTimer.iteritems())
      timerActivated = d1[key1]
      key2 = timerActivated.split('TIM')[-1]
      d2 = dict((v, k) for k, v in self.timer.iteritems())
      varActivated = d2[key2]
      currentFolder = os.path.split(workingDir)[-1]
      for key, value in self.values[currentFolder].items():
        if key == varActivated:
          self.branch[currentFolder]=(key,value)

      self.dictVariables(inp)
      if self.stop.strip()!='mission_time':
        event=False
        userStop='IEVNT('+str(self.stop)+')'
        if dataDict[userStop][-1]==1.0:
          event=True

      condition=False
      tilast=str(timeFloat[-1])
      self.tilastDict[currentFolder]=tilast
      if self.stop.strip()=='mission_time':
        condition=(math.floor(float(tilast)) >= math.floor(float(self.endTime)))
      else:
        condition=(event or (math.floor(float(tilast)) >= math.floor(float(self.endTime))))
      if not condition:
        DictBranchCurrent='Dict'+str(self.branch[currentFolder][0])
        DictChanged=self.DictAllVars[DictBranchCurrent]
        self.branchXml(tilast, DictChanged,inp,dataDict)

