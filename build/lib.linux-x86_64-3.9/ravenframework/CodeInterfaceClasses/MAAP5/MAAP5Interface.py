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
'''
Created on April 14, 2016
@created: rychvale (EDF)
@modified: picoco (The Ohio State University)
'''

from ..Generic.GenericCodeInterface import GenericCode
import numpy as np
from  ..Utilities import csvUtilities as csvU
from ..Utilities import dynamicEventTreeUtilities as detU
import csv
import glob
import os
import copy
import re
import math
import sys

class MAAP5(GenericCode):
  """
  Class for MAAP5 interface with RAVEN
  """

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    GenericCode._readMoreXML(self,xmlNode)
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

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      This method is used to generate an input based on the information passed in.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
            where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    self.samplerType=samplerType
    if 'dynamiceventtree' in str(samplerType).lower():
      if Kwargs['RAVEN_parentID'] == 'None':
        self.oriInput(oriInputFiles) #original input files are checked only the first time
      self.stopSimulation(currentInputFiles, Kwargs)
###########
      if Kwargs['RAVEN_parentID'] != 'None':
        print('Kwargs',Kwargs)
        self.restart(currentInputFiles, Kwargs['RAVEN_parentID'])
###########
        if len(self.multiBranchOccurred)>0:
          self.multiBranchMethod(currentInputFiles, Kwargs)
###########
        if str(Kwargs['prefix'].split('-')[-1]) != '1':
          self.modifyBranch(currentInputFiles, Kwargs)
    return GenericCode.createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs)

  def oriInput(self, oriInputFiles):
    """
      ONLY IN CASE OF DET SAMPLER!
      This function read the original input file and the include file
      specified by the user into 'includeForTimer' block. In the input file,
      the method looks for the MAAP5 'END TIME' and 'PRINT INTERVAL'
      variables, it lists the DET sampled variables and the HYBRID ones,
      their default values. It checks that at least one branching exists, and
      that one branching condition exists for each of the DET sampled variables.
      The include file is also read looking for the timer defined for each DET sampled
      variable.
      @ In, oriInputFiles, list, list of all the original input files
      @ Out, None
    """
    self.lineTimerComplete=[] #list of all the timer (one for each DET sampled variables)
    self.DETsampledVars = [] #list of RAVEN DET sampled variables
    self.HYBRIDsampledVars = [] #list of RAVEN HYBRID sampled variables
    self.DETDefaultValue={} #record default value for DET sampled variables: e.g. {'TIMELOCA':-1, 'AFWOFF':-1}
    self.HYBRIDdefaultValue={} #record default value for HYBRID sampled variables

    HYBRIDVar = False #is True, when for loop is in the block defining the variables which are HYBRID sampling
    DETVar = False #is True, when for loop is in the block defining the variables which are DET sampling
    foundDET = False #is True, when one branching condition is found

    for filename in oriInputFiles:
      if '.inp' in str(filename):
        inp=filename.getAbsFile() #input file name with full path

    #read the original input file in order to find DET Sampled variables and their default values, save the print interval
    fileobject = open(inp, "r") #open .inp file
    lines=fileobject.readlines()
    fileobject.close()
    branching=[]

    for line in lines:
      if 'PRINT INTERVAL' in line:
        for digit in line.split():
          if digit.isdigit() or ('.' in digit):
            self.printInterval=float(digit)
      if 'C DET Sampled Variables' in line:
        #to distinguish between DET sampling  and Hybrid sampling (in case of Hybrid DET)
        DETVar = True
      if 'END TIME' in line:
        for digit in line.split():
          if digit.isdigit() or ('.' in digit):
            self.endTime=float(digit)
      if line.find('$RAVEN') != -1 and DETVar:
        #MAAP Variable for RAVEN is e.g. AFWOFF = $RAVEN-AFWOFF$ (string.find('x') = -1 when 'x' is not in the string)
        var = line.split('=')[0]
        if ':' in line:
          line=line.split(':')[1]
          line=line.strip('$\n')
          self.DETDefaultValue[str(var.strip())]=str(line)
        else:
          self.DETDefaultValue[str(var.strip())]=''
        self.DETsampledVars.append(var.strip()) #var.strip() deletes whitespace
      if 'C End DET Sampled Variables' in line:
        DETVar = False
      if 'C HYBRID Sampled Variables' in line and not DETVar:
        HYBRIDVar = True
      if line.find('$RAVEN') != -1 and HYBRIDVar:
        var = line.split('=')[0]
        if ':' in line:
          line=line.split(':')[1]
          line=line.strip('$\n')
          self.HYBRIDdefaultValue[str(var.strip())]=str(line)
          self.HYBRIDsampledVars.append(var.strip())
        else:
          self.HYBRIDdefaultValue[str(var.strip())]=''
          self.HYBRIDsampledVars.append(var.strip()) #var.strip() deletes whitespace
      if 'C End HYBRID Sampled Variables' in line:
        HYBRIDVar= False
      if 'C Branching ' in line:
        foundDET = True
        var=line.split()[-1]
        branching.append(var)

    print('DET sampled Variables =',self.DETsampledVars)
    print('Hybrid sampled Variables =',self.HYBRIDsampledVars)

    for var in self.DETsampledVars:
      if not var in branching:
        raise IOError('Please define a branch/add a branching marker for the variable: ', var)

    if foundDET:
      print ('There is at least one branching condition for DET analysis')
    else:
      raise IOError('No branching defined in the input file')

    if self.printInterval == '':
      raise IOError('Define a PRINT INTERVAL for the writing of the restart file')
    else:
      print ('Print interval =', self.printInterval)

    #read the include file in order to check that for all the branching there is a TIMER defined
    self.stopTimer=[]
    print('self.stop',self.stop)
    self.timer = {} #this dictionary contains for each DETSampledVar the number of the corrisponding TIMER set
    fileobject = open(self.include, "r") #open include file, this file contains all the user-defined TIMER
    lines=fileobject.readlines()
    fileobject.close()
    lineNumber=0
    branchingMarker=''
    stopMarker='C End Simulation'
    found = [False]*len(self.DETsampledVars) #TIMER condition
    block = [False]*len(self.DETsampledVars) #timer 'block'
    stop=False
    for line in lines:
      if str(self.stop).strip() != 'mission_time':
        if str(stopMarker) in line:
          stop=True
          continue
        if (str('SET TIMER') in line) and stop:
          self.stopTimer = line.split()[-1]
          print('stopTimer =', self.stopTimer)
          stop=False
      for cont, var in enumerate(self.DETsampledVars):
        var = var
        branchingMarker=str('C Branching '+var)
        if branchingMarker in line:
          #branching timer marker
          block[cont] = True
        if (str('SET TIMER') in line) and block[cont]:
          found[cont] = True
          self.timer[var]= line.split()[-1]
          self.lineTimerComplete.append('TIMER '+str(self.timer[var])) #this list contains all the TIMER associated with the branching e.g. [TIMER 100, TIMER 101, TIMER 102]
          print ('TIMER found for', var)
          block[cont]=False
        if (str('END') in line) and block[cont] and not found[cont]:
          print ('TIMER not found for', var)
          block[cont] = False
    for cont, val in enumerate(found):
      if not val:
        raise IOError('Please define a TIMER for', self.DETsampledVars[cont])

  def restart(self,currentInputFiles, parentName):
    """
      ONLY IN CASE OF DET SAMPLER!
      This method reads the input file and, for each branch, changes
      the value of the RESTART TIME and RESTART FILE to be used
      @ In, currentInputFiles, list, list of all the current input files
      @ In, parentName, string, name of the parent branch
      @ Out, input file modified according to the correct restart set
    """
    correctRestart='' #is the correct line for the restart time definition
    newLine=''
    restarFileCorrect=''
    for filename in currentInputFiles:
      if '.inp' in str(filename):
        break
    inp=filename.getAbsFile()
    currentFolder, base = os.path.split(inp)
    base=base.split('.')[0] #test
    parentFolder= '../'+parentName
    tilast=self.tilastDict[parentName]

    #given tilast, then correct RESTART TIME is tilast-self.printInterval
    fileobject = open(inp, "r") #open MAAP .inp for the current run
    lines=fileobject.readlines()
    fileobject.close()

    #verify the restart file and restart time is defined into the input file
    restartTime = False #is True, if the restart time defined is correct
    foundRestart = False #is True, if the restart file declaration is found
    correct = False #is True, if the restart file found is correct
    restarFileCorrect=os.path.join(parentFolder, base+".res")
    print('correct restart file is',restarFileCorrect)
    lineNumber=0
    restartTimeNew=0
    for line in lines:
      lineNumber=lineNumber+1
      if 'START TIME' in line:
        restartTimeNew=max(0,math.floor(float(tilast)-float(self.printInterval)))
        correctRestart= 'RESTART TIME IS '+str(restartTimeNew)+'\n'
        if line == correctRestart:
          restartTime=True
        else:
          lines[lineNumber-1]=correctRestart
          fileobject = open(inp, "w")
          linesNewInput = "".join(lines)
          fileobject.write(linesNewInput)
          fileobject.close()
        break
      # once checked for the correct restart time, he interface looks for the restart file definition
      if 'RESTART FILE ' in line:
        foundRestart = True
        restartFile =  line.split(" ")[-1].strip()
        if restartFile == restarFileCorrect:
          correct = True
          print ('RESTART FILE is correct: ',restartFile)
        break

    if not foundRestart:
      print ('NO RESTART FILE declared in the input file')
      #the restart file declaration need to be added to the input file
      lineInclude='INCLUDE '+str(self.include)+'\n'
      index=lines.index(lineInclude)
      lines.insert(index+1, 'RESTART FILE '+restarFileCorrect+'\n')
      fileobject = open(inp, "w")
      linesNewInput = "".join(lines)
      fileobject.write(linesNewInput)
      fileobject.close()
      print ('RESTART FILE declared in the input file')
    elif foundRestart and not correct:
      print ('RESTART FILE declared is not correct', restartFile)
      lineInclude='INCLUDE '+str(self.include)+'\n'
      index=lines.index(lineInclude)
      newLine='RESTART FILE '+restarFileCorrect+'\n'
      lines[index+1]=newLine
      fileobject = open(inp, "w")
      linesNewInput = "".join(lines)
      fileobject.write(linesNewInput)
      fileobject.close()
      print ('RESTART FILE name has been corrected',restarFileCorrect)

########################
  def modifyBranch(self,currentInputFiles,Kwargs):
    """
      This method is aimed to modify the branch in order to reflect the info
      coming from the DET-based sampler
      @ In, currentInputFiles, list, list of input files
      @ In, Kwargs, dict, dictionary of kwarded values
      @ Out, None
    """
    block=False
    lineNumber=0
    n=0
    newLine=''
    for filename in currentInputFiles:
      if '.inp' in str(filename):
        break
    inp=filename.getAbsFile()
    #given tilast, then correct RESTART TIME is tilast-self.printInterval
    fileobject = open(inp, "r") #open MAAP .inp for the current run
    lines=fileobject.readlines()
    fileobject.close()
    for line in lines:
      lineNumber=lineNumber+1
      if 'C Branching '+str((self.branch[Kwargs['RAVEN_parentID']])[0]) in line:
        block=True
      if n==len(Kwargs['branchChangedParam']):
        block=False
        break
      if block:
        for cont,var in enumerate(Kwargs['branchChangedParam']):
          if (var in line) and ('=' in line):
            newLine=' '+str(var)+'='+str(Kwargs['branchChangedParamValue'][cont])+'\n'
            print('Line correctly modified. New line is: ',newLine)
            lines[lineNumber-1]=newLine
            fileobject = open(inp, "w")
            linesNewInput = "".join(lines)
            fileobject.write(linesNewInput)
            fileobject.close()
            n=n+1

########################

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
    csvwriter=csv.writer(outputCSVfile,delimiter=',')
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
      d1 = dict((v, k) for k, v in dictTimer.items())
      timerActivated = d1[key1]
      key2 = timerActivated.split('TIM')[-1]
      d2 = dict((v, k) for k, v in self.timer.items())
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

######################################################"
  def branchXml(self, tilast,Dict,inputFile,dataDict):
    """
    ONLY FOR DET SAMPLER!
    This method writes the xml files used by RAVEN to create the two branches at each stop condition reached
    @ In, tilast, string, end time of the current simulation run
    @ In, Dict, dict, dictionary containing the name and the value of the variables modified by the branch occurrence
    @ In, inputFile, string, name of the current input file
    @ In, dataDict, dict, dictionary containing the time evolution of the MAAP5 output variables contained in the csv output file
    @ Out, None
    """
    self.multiBranch=[]
    try:
      inpPath=inputFile
      workingDir='/'.join(inpPath.split('/')[:-3])
      sys.path.append(os.path.dirname(workingDir))
      import multibranch as multi
      for method in dir(multi):
        method=str(method)
        if method in self.DETsampledVars:
          self.multiBranch.append(method)
      print('MultiBranch found for the following variables: ', (','.join(self.multiBranch)))
    except:
      print('No multi-branch found')
    base=os.path.basename(inputFile).split('.')[0]
    path=os.path.dirname(inputFile)
    filename=os.path.join(path,'out~'+base+"_actual_branch_info.xml")
    stopInfo={'end_time':tilast}
    listDict=[]
    variableBranch=''
    branchName = os.path.split(path)[-1]
    variableBranch=self.branch[str(branchName)][0]

    if variableBranch in self.DETsampledVars and variableBranch not in self.multiBranch:
      DictName='Dict'+str(variableBranch)
      dict1=self.DictAllVars[DictName]
      variables = list(dict1.keys())

      for var in variables:
        #e.g. for TIMELOCA variables are ABBN(1), ABBN(2), ABBN(3)
        if var==(self.branch[branchName])[0]:
          #ignore if the variable coincides with the trigger
          continue
        else:
          newValue=str(dict1[var])
          oldValue=str(dataDict[var][0])
          branch={'name':var, 'type':'auxiliar','old_value': oldValue, 'new_value': newValue.strip('\n')}
          listDict.append(branch)
        detU.writeXmlForDET(filename,variableBranch,listDict,stopInfo) #this function writes the xml file for DET
    else:
###########
      self.multiBranchOccurred.append(variableBranch)
###########
      methodToCall=getattr(multi,variableBranch)
      listDict=methodToCall(self,dataDict)
      detU.writeXmlForDET(filename,variableBranch,listDict,stopInfo) #this function writes the xml file for DET

######################################################"
#  def branchXml(self, tilast,Dict,inputFile,dataDict):
#    """
#      ONLY FOR DET SAMPLER!
#      This method writes the xml files used by RAVEN to create the two branches at each stop condition reached
#      @ In, tilast, string, end time of the current simulation run
#      @ In, Dict, dict, dictionary containing the name and the value of the variables modified by the branch occurrence
#      @ In, inputFile, string, name of the current input file
#      @ In, dataDict, dict, dictionary containing the time evolution of the MAAP5 output variables contained in the csv output file
#      @ Out, None
#    """
#    base=os.path.basename(inputFile).split('.')[0]#
#    path=os.path.dirname(inputFile)
#    filename=os.path.join(path,'out~'+base+"_actual_branch_info.xml")
#    stopInfo={'end_time':tilast}
#    listDict=[]
#    variableBranch=''
#    branchName=path.split('/')[-1]
#    variableBranch=self.branch[str(branchName)][0]
#    DictName='Dict'+str(variableBranch)
#    dict1=self.DictAllVars[DictName]
#    variables = list(dict1.keys())
#    for var in variables: #e.g. for TIMELOCA variables are ABBN(1), ABBN(2), ABBN(3)
#      if var==(self.branch[branchName])[0]: #ignore if the variable coincides with the trigger
#        continue
#      else:
#        newValue=str(dict1[var])
#        oldValue=str(dataDict[var][0])
#        branch={'name':var, 'type':'auxiliar','old_value': oldValue, 'new_value': newValue.strip('\n')}
#        listDict.append(branch)
#      detU.writeXmlForDET(filename,variableBranch,listDict,stopInfo) #this function writes the xml file for DET

  def dictVariables(self,currentInp):
    """
      ONLY FOR DET SAMPLER!
      This method creates a dictionary for the variables determining a branch and the values of the
      variables changed due to the branch occurrence.
      @ In, currentInp, string, name of the current input file
      @ Out, self.DictAllVars, dict, dictionary containing the value of all the variables that could branching and the value of the corresponding variables that would be modified in the branches
    """
    fileobject = open(currentInp, "r") #open MAAP .inp
    lines=fileobject.readlines()
    fileobject.close()
    self.DictAllVars= {} #this dictionary will contain all the dicitonary, one for each DET sampled variable
    for var in self.DETsampledVars:
      block = False
      DictVar = {}
      for line in lines:
        if (var in line) and ('=' in line) and not ('WHEN' or 'IF') in line:
        #conditions on var and '=' ensure that there is an assignment,while condition on 'WHEN'/'IF' excludes e.g.'WHEN TIM<=AFWOFF' line
          sampledVar = line.split('=')[0].strip()
          sampledValue = str(line.split('=')[1])
          DictVar[sampledVar] = sampledValue
          continue
        if ('C Branching '+var) in line:
          #branching marker
          block = True
        if ('=' in line) and block and not ('WHEN' or 'IF') in line:
          #there is a 'Branching block'
          modifiedVar = line.split('=')[0].strip()
          modifiedValue = line.split('=')[1]
          DictVar[modifiedVar] = modifiedValue
        if ('END' in line) and block:
          block = False
      self.DictAllVars["Dict{0}".format(var)]= DictVar #with Dict{0}.format(var) dictionary referred to each single sampled variable is called DictVar (e.g., 'DictTIMELOCA', 'DictAFWOFF')
      print('self.DictAllVars',self.DictAllVars)

  def stopSimulation(self,currentInputFiles, Kwargs):
    """
      ONLY FOR DET SAMPLER!
      This method update the stop simulation condition into the MAAP5 input
      to stop the run when the new branch occurs
      @ In, currentInputFiles, list, list of the current input files
      @ Out, Kwargs, dict,kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
           where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
    """
    for filename in currentInputFiles:
      if '.inp' in str(filename):
        inp=filename.getAbsFile() #input file name with full path

    fileobject = open(inp, "r")
    lines=fileobject.readlines()
    fileobject.close()

    currentFolder=os.path.dirname(inp)
    currentFolder = os.path.split(currentFolder)[-1]
    parents=[]
    self.values[currentFolder]=Kwargs['SampledVars']
    lineStop = int(lines.index(str('C Stop Simulation condition\n')))+1
########################
    lineStopList=[lineStop]
    while lines[lineStopList[-1]+1].split(' ')[0]=='OR':
      lineStopList.append(lineStopList[-1]+1)
########################
    if Kwargs['RAVEN_parentID']== 'None':
      if self.stop!='mission_time':
        self.lineTimerComplete.append('TIMER '+str(self.stopTimer))
####################
      found=[False]*len(self.lineTimerComplete)
      for cont,timer in enumerate(self.lineTimerComplete):
        for line in lineStopList:
          if timer in lines[line]:
            found[cont]=True
      if all(i for i in found) == False:
####################
        raise IOError('All TIMER must be considered for the first simulation') #in the original input file all the timer must be mentioned
    else:
      #Kwargs['RAVEN_parentID'] != 'None'
      parent=currentFolder[:-2]
      parents.append(parent)
      while len(parent.split('_')[-1])>2: #collect the name of all the parents, their corresponding timer need to be deleted from the stop condition (when the parents has already occurred)
        parent=parent[:-2]
        parents.append(parent)
      lineTimer= list(self.lineTimerComplete)
      for parent in parents:
        varTimerParent=self.branch[parent][0]
        valueVarTimerParent=self.branch[parent][1]
        for key,value in self.values[currentFolder].items():
          if key == varTimerParent and value != valueVarTimerParent:
            continue
          elif key == varTimerParent:
            timerToBeRemoved=str('TIMER '+ self.timer[self.branch[parent][0]])
            if timerToBeRemoved in lineTimer:
              lineTimer.remove(timerToBeRemoved)

      listTimer=['IF']
      if len(lineTimer)>0:
        timN=0
        for tim in lineTimer:
          timN=timN+1
          listTimer.append('(' + str(tim))
          listTimer.append('>')
          listTimer.append('0)')
          if (int(timN) % 4)==0:
            listTimer.append('\n')
          listTimer.append('OR')

        listTimer.pop(-1)
        while len(lineStopList)-1 > (listTimer.count('\n')):
          listTimer.append('\n')
        newLine=' '.join(listTimer)+'\n'
#        lines[lineStop]=newLine
        lines[lineStopList[0]:lineStopList[-1]+1]=newLine
      else:
        lines[lineStopList[0]-1:lineStopList[-1]+3]='\n'
#      else: lines[lineStop:lineStop+3]='\n'
      fileobject = open(inp, "w")
      linesNewInput = "".join(lines)
      fileobject.write(linesNewInput)
      fileobject.close()

###########
  def multiBranchMethod(self,currentInputFiles,Kwargs):
    """
      This method is aimed to handle the multi branch strategy
      @ In, currentInputFiles, list, list of input files
      @ In, Kwargs, dict, dictionary of kwarded values
      @ Out, None
    """
    for filename in currentInputFiles:
      if '.inp' in str(filename):
        inp=filename.getAbsFile() #input file name with full path
    fileobject = open(inp, "r")
    linesCurrent=fileobject.readlines()
    fileobject.close()

    parentInput=str(inp).replace(str(Kwargs['prefix']),str(Kwargs['RAVEN_parentID']))
    fileobject = open(parentInput, "r")
    linesParent=fileobject.readlines()
    fileobject.close()

    indexStartParent=linesParent.index('C End HYBRID Sampled Variables\n')
    indexEndParent=linesParent.index('C Stop Simulation condition\n')
    diff=indexEndParent-indexStartParent

    indexStartCurrent=linesCurrent.index('C End HYBRID Sampled Variables\n')
    indexEndCurrent=indexStartCurrent+diff

    linesCurrent[indexStartCurrent:indexEndCurrent]=linesParent[indexStartParent:indexEndParent]

    fileobject = open(inp, "w")
    linesNewInput = "".join(linesCurrent)
    fileobject.write(linesNewInput)
    fileobject.close()
###########
