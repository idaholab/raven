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
  MAAP5 code interface. This interface is able to drive MAAP5 calculations for any type of sampling strategy (Forward and DET-based)
  Created on April 14, 2016
  @author  : Valenting Rychkov - rychvale (EDF)
  @modified: Claudia Picoco    - picoco   (The Ohio State University)
"""

from __future__ import division, print_function, absolute_import

from GenericCodeInterface import GenericCode
import numpy as np
import csvUtilities as csvU
import dynamicEventTreeUtilities as detU
import csv
import glob
import os
import copy
import re
import math
import sys
import threading #Andrea suggestion as of 2018.08 to overcome dictionary error
import time #VR (to sleep for RESTART) 14.10.2018
import random #VR (to sleep for RESTART) 14.10.2018

class MAAP5(GenericCode):

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
    self.printDebug  = True
    self.tilastDict={} #{'folder_name':'tilast'} - this dictionary contains the last simulation time of each branch, this is necessary to define the correct restart time
    self.branch = {} #{'folder_name':['variable branch','variable value']} where variable branch is the variable sampled for the current branch e.g. {det_1:[timeloca, 200]}
    self.values = {} #{'folder_name':[['variable branch_1','variable value_1'],['variable branch_2','variable value_2']]} for each DET sampled variables
    self.boolOutputVariables=[] #list of MAAP5 boolean variables of interest
    self.contOutputVariables=[] #list of MAAP5 continuous variables of interest
    self.debugCommand = None
    self.restartUseVar=True
###########
    self.multiBranchOccurred=[]
###########
    for child in xmlNode:
      if child.tag == 'debugCommand':
        self.debugCommand = child.text
      if child.tag == 'printDebug':
        if child.text.lower() in ['t','true']:
          self.printDebug = True
        else:
          self.printDebug = False
      if child.tag == 'includeForTimer':
        if child.text != None:
          self.include = child.text
      if child.tag == 'boolMaapOutputVariables':
        #here we'll store boolean output MAAP variables to look for"
        if child.text != None:
          self.boolOutputVariables = child.text.split(',')
      if child.tag == 'contMaapOutputVariables':
        #here we'll store boolean output MAAP variables to look for
        if child.text != None:
          self.contOutputVariables = child.text.split(',')
      if child.tag == 'stopSimulation': self.stop=child.text #this node defines if the MAAP5 simulation stop condition is: 'mission_time' or the occurrence of a given event e.g. 'IEVNT(691)'
      if child.tag == 'restartUse':
        if child.text=='True': self.restartUseVar=True
        elif child.text=='False': self.restartUseVar=False
        else: raise IOError('restartUse needs to be defined either True or False')
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
    # We need to dump the branch dictionary kwargs to the file to use it by fnalizeCodeOutput to create correct multiBranch probabilities
    if 'DynamicEventTree' in samplerType:
      f=open(Kwargs['subDirectory']+'/Kwargs.txt','w')
      f.write(repr(Kwargs))
      f.close

      if Kwargs['RAVEN_parentID'] == 'None': self.oriInput(oriInputFiles) #original input files are checked only the first time
      self.stopSimulation(currentInputFiles, Kwargs)
###########
      if Kwargs['RAVEN_parentID'] != 'None':
        if self.printDebug : print('Kwargs',Kwargs,'\n')
        if self.restartUseVar==True: self.restart(currentInputFiles, Kwargs['RAVEN_parentID'])
#        self.includeUpdate(currentInputFiles)
###########
        if len(self.multiBranchOccurred)>0:
          self.multiBranchMethod(currentInputFiles, Kwargs)
        if self.printDebug : print ('Calling multiBranchMethod, priting Kwargs',Kwargs)

###########
        if str(Kwargs['prefix'].split('-')[-1]) != '1': self.modifyBranch(currentInputFiles, Kwargs)
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
    def _isNumber(strNumber,dtype="float"):
      """
        Simple method to check if the strNumber is a number
        @ In, strNumber, str, the string to check
        @ In, dtype, str, the type (either int or float)
        @ Out, _isNumber, bool, True if it is a number
      """
      try:
        float(strNumber) if dtype == 'float' else int(strNumber)
        return True
      except ValueError:
        return False
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
    for cnt, line in enumerate(lines):
      if 'C DET Sampled Variables' in line: #to distinguish between DET sampling  and Hybrid sampling (in case of Hybrid DET)
        DETVar = True
      if 'END TIME' in line and not line.strip().startswith("C"):
        for digit in line.split():
          if _isNumber(digit):
            self.endTime=float(digit)
      if line.find('$RAVEN') != -1 and DETVar: #MAAP Variable for RAVEN is e.g. AFWOFF = $RAVEN-AFWOFF$ (string.find('x') = -1 when 'x' is not in the string)
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
      if 'C End HYBRID Sampled Variables' in line: HYBRIDVar= False
      if 'C Branching ' in line:
        foundDET = True
        var=line.split()[-1]
        branching.append(var)
        # check if TIM variables have been inputted
        foundTIMgate = False
        subCnt = 1
        subLine = lines[cnt+subCnt]
        while subLine.split()[0].strip() != 'END':
          subCnt+=1
          subLine = lines[cnt+subCnt]
          if 'TIM' in subLine and "=" in subLine:
            splitted = subLine.split("=")
            digitTimer = splitted[0].replace("TIM","").strip()
            foundTIMgate = _isNumber(digitTimer,'int') and _isNumber(splitted[-1].strip(),'float')
            if _isNumber(splitted[-1].strip(),'float') and abs(float(splitted[-1]) - 1.0) > 1e-6:
              raise Exception('"TIM'+digitTimer.strip()+' variable must be set to 1.0 in the "C Branching" block!')
        if not foundTIMgate:
          raise Exception('TIM*** variable must be set in the "C Branching" block!')


    if self.printDebug : print('DET sampled Variables =',self.DETsampledVars)
    if self.printDebug : print('Hybrid sampled Variables =',self.HYBRIDsampledVars)

    for var in self.DETsampledVars:
      if not var in branching: raise IOError('Please define a branch/add a branching marker for the variable: ', var)

    if foundDET: print ('There is at least one branching condition for DET analysis')
    else: raise IOError('No branching defined in the input file')

#    if self.printInterval == '': raise IOError('Define a PRINT INTERVAL for the writing of the restart file')
#    else: print ('Print interval =', self.printInterval)

    #read the include file in order to check that for all the branching there is a TIMER defined
    self.stopTimer=[]
    if self.printDebug : print('self.stop',self.stop)
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
    #####
    parameterBlock=False
    self.paramDict={} #this dictionary contains for the user defined parameters and their corresponding initialization value
    #####
    for cnt,line in enumerate(lines):
    #####
      if 'PARAMETER CHANGE' in line: parameterBlock=True
      if parameterBlock and '=' in line:
        paramName = line.split('=')[0].strip()
        paramValue = str(line.split('=')[1])
        self.paramDict[paramName] = paramValue
      if 'END' in line and parameterBlock: parameterBlock=False
    #####
      if str(self.stop).strip() != 'mission_time':
        if str(stopMarker) in line:
          stop=True
          continue
        if (str('SET TIMER') in line) and stop:
          self.stopTimer = line.split()[-1]
          if self.printDebug : print('stopTimer =', self.stopTimer)
          stop=False
      for cont, var in enumerate(self.DETsampledVars):
        var = var
        branchingMarker=str('C Branching '+var)
        if branchingMarker.strip() == ' '.join(line.split()).strip():
          #branching timer marker
          block[cont] = True
        if (str('SET TIMER') in line) and block[cont]:
          found[cont] = True
          self.timer[var]= line.split()[-1]
          self.lineTimerComplete.append('TIMER '+str(self.timer[var])) #this list contains all the TIMER associated with the branching e.g. [TIMER 100, TIMER 101, TIMER 102]
          print (self.lineTimerComplete)
          print ('TIMER found for', var)
          block[cont]=False
        if (str('END') in line) and block[cont] and not found[cont]:
          print ('TIMER not found for', var)
          block[cont] = False
    for cont, val in enumerate(found):
      if not val: raise IOError('Please define a TIMER for', self.DETsampledVars[cont])

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
      if self.include in str(filename): inc=filename.getAbsFile() #current include file name with full path
      if '.inp' in str(filename): inp=filename.getAbsFile() #current include file name with full path

    currentFolder, baseInp = os.path.split(inp) #e.g., baseInp is test

    baseInp=baseInp.split('.')[0] #test
    parentFolder= '../'+parentName
####################################
# Correct definition of the restart time
    parentFolderPath=('/'.join(currentFolder.split('/')[:-1]))+'/'+parentName
    summ=os.path.join(parentFolderPath,baseInp+'.sum') #path of the summary file of the parent folder
    fileobject = open(summ, "r") #open MAAP .sum file of the parent branch
    linesSummary=fileobject.readlines()
    fileobject.close()
    timeRestartWrit=[]#list of all the time when restart file has been written in the parent branch
    for lineSumm in linesSummary:
      if 'RESTART FILE WRITTEN AT THIS TIME\n' in lineSumm: timeRestartWrit.append(lineSumm.split()[0])

    if (float(timeRestartWrit[-1])-float(timeRestartWrit[-2])) < 0.3: #modified by CP - 17/03/2018 - due to error with different restart file written in 0.1 s
      if self.printDebug :print('difference between last restart and first-to-last restart is lower than 0.3 s')
      if len(timeRestartWrit) > 2: restartTimeNew=timeRestartWrit[-3] #modified by CP - 17/03/2018 - to take the second-to-last is important to make sure that at least more than two restart are written
      else: raise IOError('Change print interval!')
    else: restartTimeNew=timeRestartWrit[-2] #restart time is the second-to-last time when a restart file has been written
####################################
# Parameters values from the parent branches are saved in the self.paramDict dictionary and the include file of the current branch is updated with these values

    csvSimulationFilesParam=[]
    filePrefixWithPathParam=os.path.join(parentFolderPath,baseInp) #path of the csv file of the parent folder containing the user defined variables

#    print('###--filePrefixWithPathParam',filePrefixWithPathParam)
    csvParam=glob.glob(filePrefixWithPathParam+".d"+"*.csv") #list of MAAP output files with the evolution of continuous variables
#    print('###--csvParam',csvParam)

    mergeCSV=csvU.csvUtilityClass(csvParam,1,";",True)
    dataParam={}
    dataParam=mergeCSV.mergeCsvAndReturnOutput({'variablesToExpandFrom':['TIME'],'returnAsDict':True})
    timeCsv=dataParam['TIME']

    for param in self.paramDict.keys(): #based on the values assumed by the user defined variables in the current branch the corresponding dictionary is updated.
      listTimeFloat=list([float(item) for item in timeCsv])
      timeValue=float(restartTimeNew)
      index=min(range(len(listTimeFloat)), key=lambda i: abs(listTimeFloat[i]-timeValue)) #since timeValue does not necessarily correspond to a time in TIME vector, this way the closest time is taken
      if listTimeFloat[index] > timeValue and index !=0:
        index -= 1
      self.paramDict[param]=str(dataParam[param][index])

    self.includeUpdate(currentInputFiles)
####################################
# Restart file and restart time info are updated

    #given tilast, then correct RESTART TIME is tilast-self.printInterval
    fileobject = open(inp, "r") #open MAAP .inp for the current run
    lines=fileobject.readlines()
    fileobject.close()

    #verify the restart file and restart time is defined into the input file
    restartTime = False #is True, if the restart time defined is correct
    foundRestart = False #is True, if the restart file declaration is found
    correct = False #is True, if the restart file found is correct
    restarFileCorrect=os.path.join(parentFolder, baseInp+".res")
    if self.printDebug :print('correct restart file is',restarFileCorrect)
    lineNumber=0
#    restartTimeNew=0
    for line in lines:
      lineNumber=lineNumber+1
      if 'START TIME' in line:
#        restartTimeNew=max(0,math.floor(float(tilast)-float(self.printInterval)))
        correctRestart= 'RESTART TIME IS '+str(restartTimeNew)+'\n'
        if line == correctRestart:
          restartTime=True
        else:
          lines[lineNumber-1]=correctRestart
          fileobject = open(inp, "w")
          linesNewInput = "".join(lines)
          fileobject.write(linesNewInput)
          fileobject.close()
        #break #CP It is necessary to comment these two 'break' since we want to use restart starting from the beginning (18.05)
      # once checked for the correct restart time, he interface looks for the restart file definition
      if 'RESTART FILE ' in line:
        foundRestart = True
        restartFile =  line.split(" ")[-1].strip()
        if restartFile == restarFileCorrect:
          correct = True
          #print ('RESTART FILE is correct: ',restartFile)
        #break #CP It is necessary to comment these two 'break' since we want to use restart starting from the beginning (18.05)

    if not foundRestart:
      #print ('NO RESTART FILE declared in the input file')
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
      if 'C Branching '+str((self.branch[Kwargs['RAVEN_parentID']])[0]) in line: block=True
      if Kwargs['branchChangedParam']==None: #modified by VR 17/03/18 due to an error in Adaptive DET sampler
        block=False
        break
      if n==len(Kwargs['branchChangedParam']):
        block=False
        break
      if block:
        for cont,var in enumerate(Kwargs['branchChangedParam']):
#          if (var in line) and ('=' in line):
#          if (var+' =' in line) or (var+'=' in line):
          if ((var+' =' in line) or (var+'=' in line)) and ('WHEN' not in line) and ('IF' not in line):
            newLine =' '+str(var)+'='+str(Kwargs['branchChangedParamValue'][cont])+'\n'
            newLine = newLine.replace("&$*"," ")
            if self.printDebug :
              print('Line correctly modified. New line is: ',newLine)
            lines[lineNumber-1]=newLine
            fileobject = open(inp, "w")
            linesNewInput = "".join(lines)
            fileobject.write(linesNewInput)
            fileobject.close()
            n=n+1

########################

  def _convertMAAP5asciiToCsv(self,filename):
    """
      If we are driving the standard release of MAAP5, the outputs are in ascii
      In order to make them compatible with the current MAAP interface, they need to be converted in CSVs
      @ In, filename, str, the filename to convert
      @ Out, comverted, bool, True if converted
    """
    converted = True
    outputMaapLines=open(filename,"r+").readlines()
    if len(outputMaapLines) == 0:
      return False
    # check number of records
    nVariables = int(outputMaapLines.pop(0).replace("-",""))
    # remove 3 trailing rows
    for _ in range(3):
      del outputMaapLines[0]
    headers = np.zeros((nVariables,),dtype=object)
    units   = np.zeros((nVariables,),dtype=object)
    # we skip the units
    units[:] = "n/a"
    storeUnits = False
    addedCounter = 0
    while True:
      line = outputMaapLines.pop(0)
      if len(line.strip()) == 0:
        break
      variables = [var.strip()  for var in line.split("   ") if len(var.strip()) >0]
      if not storeUnits:
        if 'TIMMID' in variables:
          variables[variables.index('TIMMID')] = 'TIME'
        headers[addedCounter:addedCounter+len(variables)] = variables[:]
      addedCounter+=len(variables)
      if addedCounter == nVariables:
        storeUnits = True
        addedCounter = 0
    del outputMaapLines[0]

    if 'TIME' not in headers:
      # we do not create the csv
      return False

    data = np.zeros((0,nVariables))
    addedCounter = 0
    tempValues = np.zeros(nVariables)
    while True:
      try:
        line = outputMaapLines.pop(0)
      except IndexError:
        break
      variables = [float(elm) for elm in line.split()]
      tempValues[addedCounter:addedCounter+len(variables)] = variables[:]
      addedCounter+=len(variables)
      if addedCounter == nVariables:
        data = np.vstack((data,tempValues))
        addedCounter = 0

    csvOutputFile = open(filename+".csv","w+")
    csvOutputFile.write( ";".join(headers)+"\n")
    csvOutputFile.write( ";".join(units)+"\n")
    np.savetxt(csvOutputFile, data,  delimiter=';')
    csvOutputFile.close()
    return converted

#######################

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      (e.g. in RELAP5 would be the keyword "********")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is fa:q:qiled, False otherwise
    """
    failure = False
    badWords  = ["Search for the word ERROR"]
    try:
      outputToRead = open(os.path.join(workingDir,output+'.o'),"r")
    except:
      return failure
    readLines = outputToRead.readlines()

    for badMsg in badWords:
      if any(badMsg in x for x in readLines[-20:]):
        failure = True
    return failure

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
      @ In, output, string, the Output name None
      @ In, workingDir, string, current working dir
      @ Out, output, string, output csv file containing the variables of interest specified in the input
    """
    print('workingDir:',workingDir) #02/08/2018
    print('output:',output) #02/08/2018
    print('exist-old:',os.path.exists('/'+workingDir+'/'+output+'~old')) #02/08/2018

    if os.path.exists('/'+workingDir+'/'+output+'~old'): #02/08/2018
      os.remove(workingDir+'/'+output) #02/08/2018
      os.rename(workingDir+'/'+output+'~old',workingDir+'/'+output) #02/08/2018
    #if '~dummy' in output: output='~'.join(output.split('~')[0:-1])  #02/08/2018

    csvSimulationFiles=[]
    realOutput=output.split("out~")[1] #rootname of the simulation files
    inp = os.path.join(workingDir,realOutput + ".inp") #input file of the simulation with the full path
    filePrefixWithPath=os.path.join(workingDir,realOutput) #rootname of the simulation files with the full path
    csvSimulationFiles=glob.glob(filePrefixWithPath+".d"+"*.csv") #list of MAAP output files with the evolution of continuous variables
    if len(csvSimulationFiles) == 0:
      # check if ASCII files are present. If not, error out
      simulationFiles=glob.glob(filePrefixWithPath+".d*")
      if len(simulationFiles) == 0:
        raise Exception('Neither CSV nor ASCII outputs have been found in directory :' +str(workingDir))
      # convert the ASCII files into CSVs
      for filename in simulationFiles:
        if self._convertMAAP5asciiToCsv(filename):
          csvSimulationFiles.append(filename+".csv")
    mergeCSV=csvU.csvUtilityClass(csvSimulationFiles,1,";",True)
    dataDict={}
    dataDict=mergeCSV.mergeCsvAndReturnOutput({'variablesToExpandFrom':['TIME'],'returnAsDict':True})
    timeFloat=dataDict['TIME']
    #Here we'll read evolution of continous variables """
    contVariableEvolution=[] #here we'll store the time evolution of MAAP continous variables
    if len(self.contOutputVariables)>0:
      for variableName in self.contOutputVariables:
        try:
          value = dataDict[variableName]
        except KeyError:
          raise IOError('Define the variable within MAAP5 plotfil: ',variableName)
        contVariableEvolution.append(value)

      #here we'll read boolean variables and transform them into continous"""
      # if the discrete variables of interest are into the csv file:
    if len(self.boolOutputVariables)>0:
      boolVariableEvolution=[]
      for variable in self.boolOutputVariables:
        variableName=str(variable)
        try: (dataDict[variableName])
        except: raise IOError('define the variable within MAAP5 plotfil: ',variableName)
        boolVariableEvolution.append(dataDict[variableName])

    allVariableTags=[]
    allVariableTags.append('TIME')
    if (len(self.contOutputVariables)>0): allVariableTags.extend(self.contOutputVariables)
    if (len(self.boolOutputVariables)>0): allVariableTags.extend(self.boolOutputVariables)

    allVariableValues=[]
    allVariableValues.append(dataDict['TIME'])
    if (len(self.contOutputVariables)>0): allVariableValues.extend(contVariableEvolution)
    if (len(self.boolOutputVariables)>0): allVariableValues.extend(boolVariableEvolution)

    RAVENoutputFile=os.path.join(workingDir,output+".csv") #RAVEN will look for  output+'.csv'file but in the workingDir, so we need to append it to the filename
    print('RAVENoutputFile=',RAVENoutputFile)  # 16.10.18
    print('RAVENoutputFile exist?',os.path.exists(RAVENoutputFile))   # 16.10.18
    #if os.path.exists(RAVENoutputFile): print('RAVEN csv output file already existing')  # 16.10.18
    if os.path.exists(RAVENoutputFile) and os.stat(RAVENoutputFile).st_size >=10000 : print('RAVEN csv output file already existing')  # 18.10.18
    else:  # 16.10.18
      print('Writing RAVEN csv output file')  # 16.10.18
      outputCSVfile=open(RAVENoutputFile,"w+")
      csvwriter=csv.writer(outputCSVfile,delimiter=',')
      csvwriter.writerow(allVariableTags)
      for i in range(len(allVariableValues[0])):
        row=[]
        for j in range(len(allVariableTags)):
          row.append(allVariableValues[j][i])
        csvwriter.writerow(row)
      outputCSVfile.close()

    if 'DynamicEventTree' in self.samplerType:
      dictTimer={} #
      for timer in self.timer.values():
        timer='TIM'+str(timer)
        try: (dataDict[timer])
        except: raise IOError('Please ensure that the timer is defined into the include file and then it is contained into MAAP5 plotfil: ',timer)
        index = np.nonzero(np.logical_and(dataDict[timer]>=0.99999, dataDict[timer]<=1.00001))
        dictTimer[timer]= timeFloat[index[0][0]] if len(index[0]) > 0 else -1

      #
      #  NOTE THAT THIS ERROR CAN BE WRONG SINCE IT IS POSSIBLE (BRANCHES ON DEMAND) THAT TWO BRANCHES (OR MORE) HAPPEN AT THE SAME TIME! Andrea
      #
      dictTimeHappened = []
      indexVector=[]
      for value in dictTimer.values():
        if value != -1: dictTimeHappened.append(value)
      if self.printDebug : print('dictTimer =', dictTimer)
      if self.printDebug : print('Events occur at: ', dictTimeHappened)
      ############
      maxTimer=max(dictTimer.values()) #maxTimer is the time of occurrence of the last branching condition
      for index in range(len(dictTimer.values())):
        if dictTimer.values()[index]==maxTimer and dictTimer.values()[index]!= -1 and dictTimer.values()[index]!=timeFloat.tolist()[0] :
          indexVector.append(index) #indexVector contain the indexes of the corresponding position in dictTimer of all the branching conditions ocurring at the same time
      if len(indexVector) > 1:
        warnings.warn('Branch must occur at different times. Branches occurring at the same time are those characterized by: ' + ', '.join(dictTimer.keys()) +'. We take the first one '+str(dictTimer.keys()[indexVector[0]]))
        indexVector      = [indexVector[0]]
        dictTimer        = {dictTimer.items()[indexVector[0]][0]: dictTimer.items()[indexVector[0]][1]}
        dictTimeHappened = [dictTimeHappened[0]]
      #if any([dictTimeHappened.count(value) > 1 for value in dictTimer.values()]): raise IOError('Branch must occur at different times')
      ############
      key1 = max(dictTimer.values())
      d1 = dict((v, k) for k, v in dictTimer.iteritems())
      timerActivated = d1[key1]
      key2 = timerActivated.split('TIM')[-1]
      d2 = dict((v, k) for k, v in self.timer.iteritems())
      varActivated = d2[key2]
      currentFolder=workingDir.split('/')[-1]
      for key, value in self.values[currentFolder].items():
        if key == varActivated: self.branch[currentFolder]=(key,value)

      #lock=threading.RLock() #03.08.2018 Andrea suggestion to lock threads while accessing this dictionary
      #with lock:        #03.08.2018 Andrea suggestion to lock threads while accessing this dictionary
      DictAllVars=self.dictVariables(inp)
      if self.stop.strip()!='mission_time':
        event=False
        userStop='IEVNT('+str(self.stop)+')'
        if dataDict[userStop][-1]==1.0: event=True

      condition=False
      tilast=str(timeFloat[-1])
      self.tilastDict[currentFolder]=tilast
      if self.stop.strip()=='mission_time':
        condition=(math.floor(float(tilast)) >= math.floor(float(self.endTime)))
      else:
        condition=(event or (math.floor(float(tilast)) >= math.floor(float(self.endTime))))
      if not condition:
        DictBranchCurrent='Dict'+str(self.branch[currentFolder][0])
        if self.printDebug : print('finalizeCodeOutput -- self.DictAllVars =', DictAllVars) #23/07/2018
        if self.printDebug : print('finalizeCodeOutput -- DictBranchCurrent =', DictBranchCurrent) #23/07/2018
        if self.printDebug : print('finalizeCodeOutput -- self.DictAllVars[DictBranchCurrent] =', DictAllVars[DictBranchCurrent]) #23/07/2018
        DictChanged=DictAllVars[DictBranchCurrent]
        if self.printDebug : print('finalizeCodeOutput -- DictChanged =', DictChanged) #23/07/2018
        self.branchXml(tilast, DictChanged,inp,dataDict)


  def branchXml(self,tilast,Dict,inputFile,dataDict):
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
        if method in self.DETsampledVars: self.multiBranch.append(method)
      if self.printDebug :print('MultiBranch found for the following variables: ', (','.join(self.multiBranch)))
    except: print('No multi-branch found')
    base=os.path.basename(inputFile).split('.')[0]
    path=os.path.dirname(inputFile)
    filename=os.path.join(path,'out~'+base+"_actual_branch_info.xml")
    stopInfo={'end_time':tilast}
    listDict=[]
    variableBranch=''
    branchName=path.split('/')[-1]
    variableBranch=self.branch[str(branchName)][0]

    if variableBranch in self.DETsampledVars and variableBranch not in self.multiBranch:
      #DictName='Dict'+str(variableBranch)
      dict1=Dict
      variables = list(dict1.keys())
      for var in variables: #e.g. for TIMELOCA variables are ABBN(1), ABBN(2), ABBN(3)
        if var==(self.branch[branchName])[0]: #ignore if the variable coincides with the trigger
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

      methodToCall=getattr(multi,variableBranch)
      # Here we call multibranch method defined in mutibranch.py
      listDict=methodToCall(self,dataDict)

      # Here we need to adjust mutibranch probabilities by the conditional branching point probability.
      # Triggered variables probabilities is contained in the Kwargs file written for each branch
      # print('path',path)
      f=open(path+'/Kwargs.txt','r')
      diction=f.read()
      f.close()
      Kwargs=eval(diction)
      condProb=float(Kwargs['SampledVarsPb'][variableBranch]) # CP - 03/20/2018 - here we read the probability value of the branch from the kwargs file
      # print('condProb',condProb)
      # print('listDict',listDict)
      for enum in range(len(listDict)): # CP - 03/20/2018 - there is one dictionary for each variable changed within the branch
        prob=listDict[enum]['associated_pb'].split() # in each dictionary probabilities for each branh are contained in a string
        #print('##',prob)
        changedProbs=[]

        for pb in prob:
          #print('--',pb)
          changedPb=str(float(pb)*condProb)
          changedProbs.append(changedPb)
        listDict[enum]['associated_pb']=' '.join(changedProbs) # CP - 03/20/2018 in the dictionary, multiple branches probabilities are updated to the conditional value
        #print('##----------listDict[enum][associated_pb]',listDict[enum]['associated_pb'])
        #print('##----------[.join(changedProbs)]',[' '.join(changedProbs)])

      #print('listDict',listDict)

      detU.writeXmlForDET(filename,variableBranch,listDict,stopInfo) #this function writes the xml file for DET

######################################################"
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
    DictAllVars= {} #this dictionary will contain all the dicitonary, one for each DET sampled variable
    for var in self.DETsampledVars:
      block = False
      DictVar = {}
      for line in lines:
        #VR if (var in line) and ('=' in line) and not ('WHEN' or 'IF') in line:
#        if (var in line) and ('=' in line) and ('WHEN' not in line) and ('IF' not in line):
        if ((var+' =' in line) or (var+'=' in line)) and ('WHEN' not in line) and ('IF' not in line):
        #conditions on var and '=' ensure that there is an assignment,while condition on 'WHEN'/'IF' excludes e.g.'WHEN TIM<=AFWOFF' line
          sampledVar = line.split('=')[0].strip()
          sampledValue = str(line.split('=')[1])
          DictVar[sampledVar] = sampledValue
          continue
        if ('C Branching '+var) in line: #branching marker
          block = True
        #if ('=' in line) and block and not ('WHEN' or 'IF') in line: #there is a 'Branching block'
        if '=' in line and "WHEN" not in line and "IF" not in line and block:
          #print ("####-222",line) #VR
          modifiedVar = line.split('=')[0].strip()
          modifiedValue = line.split('=')[1]
          if ' ' in modifiedValue:
            # there is a space (e.g. 0.1 S). We replace the whitespace with nothing
            modifiedValue = modifiedValue.replace(" ","&$*")
          DictVar[modifiedVar] = modifiedValue
        if ('END' in line) and block:
          block = False
      DictAllVars["Dict{0}".format(var)]= DictVar #with Dict{0}.format(var) dictionary referred to each single sampled variable is called DictVar (e.g., 'DictTIMELOCA', 'DictAFWOFF')
      if self.printDebug : print('self.DictAllVars',DictAllVars)
    return DictAllVars

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
    #print('lines=',lines)
    #for line in lines: print('---',line,'C Stop Simulation condition' in line) ##modified by CP - 24/1/2018
    currentFolder=os.path.dirname(inp)
    currentFolder=currentFolder.split('/')[-1]
    parents=[]
    self.values[currentFolder]=Kwargs['SampledVars']
    lineStop = int(lines.index(str('C Stop Simulation condition\n')))+1
########################
    lineStopList=[lineStop]
    while lines[lineStopList[-1]+1].split(' ')[0]=='OR': lineStopList.append(lineStopList[-1]+1)
########################
    if Kwargs['RAVEN_parentID']== 'None':
      if self.stop!='mission_time':
        self.lineTimerComplete.append('TIMER '+str(self.stopTimer))
####################
      found=[False]*len(self.lineTimerComplete)
      for cont,timer in enumerate(self.lineTimerComplete):
        for line in lineStopList:
          if timer in lines[line]: found[cont]=True
      if all(i for i in found) == False:
####################
        raise IOError('All TIMER must be considered for the first simulation') #in the original input file all the timer must be mentioned
    else: #Kwargs['RAVEN_parentID'] != 'None'
      parent=currentFolder[:-2]
      parents.append(parent)
      while len(parent.split('_')[-1])>2: #collect the name of all the parents, their corresponding timer need to be deleted from the stop condition (when the parents has already occurred)
        parent=parent[:-2]
        parents.append(parent)
      lineTimer= list(self.lineTimerComplete)
      if self.printDebug : print(lineTimer) #VR
      for parent in parents:
        varTimerParent=self.branch[parent][0]
        valueVarTimerParent=self.branch[parent][1]
        for key,value in self.values[currentFolder].items():
          if key == varTimerParent and value != valueVarTimerParent: continue
          elif key == varTimerParent:
            timerToBeRemoved=str('TIMER '+ self.timer[self.branch[parent][0]])
            #print("###--- Timer to be removed", timerToBeRemoved) #VR
            #print("###--- self.timer", self.timer) #VR
            #print("###--- self.branch",self.branch) #VR
            if timerToBeRemoved in lineTimer: lineTimer.remove(timerToBeRemoved)

      listTimer=['IF']
      if len(lineTimer)>0:
        timN=0
        for tim in lineTimer:
          timN=timN+1
          listTimer.append('(' + str(tim))
          listTimer.append('>')
          listTimer.append('0)')
          if (int(timN) % 4)==0: listTimer.append('\n')
          listTimer.append('OR')

        listTimer.pop(-1)
        while len(lineStopList)-1 > (listTimer.count('\n')): listTimer.append('\n')
        newLine=' '.join(listTimer)+'\n'
#        lines[lineStop]=newLine
        lines[lineStopList[0]:lineStopList[-1]+1]=newLine
      else: lines[lineStopList[0]-1:lineStopList[-1]+3]='\n'
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
      if '.inp' in str(filename): inp=filename.getAbsFile() #input file name with full path
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
  def includeUpdate(self, currentInputFiles):
    """
      This method is aimed to update the initialisation of the
      user defined variables and modify these in the include
      file with the values assumed at the end of the parent branch.
      This action is necessary since MAAP5 restart files do no save
      the values of user defined variables
      @ In, currentInputFiles, list, list of input files
      @ Out, None
    """
    for filename in currentInputFiles:
      if self.include in str(filename):
        break

    inc=filename.getAbsFile() #current include file name with full path

    if self.printDebug : print('include file =', inc)
    fileobject = open(inc, "r") #open include file
    lines=fileobject.readlines()
    fileobject.close()

    parameterBlock=False
    lineNumber=0

    for line in lines:
      lineNumber=lineNumber+1
      if 'PARAMETER CHANGE' in line: parameterBlock=True
      for param in self.paramDict.keys():
        if parameterBlock and ((param+' =' in line) or (param+'=' in line)):
          newLine=' '+param+'='+str(self.paramDict[param])+'\n'
          lines[lineNumber-1]=newLine
      if 'END' in line and parameterBlock:
        parameterBlock=False
        break

    fileobject = open(inc, "w")
    linesNewInput = "".join(lines)
    fileobject.write(linesNewInput)
    fileobject.close()
###########
  ##08/01/2018
  def generateCommand(self,inputFiles,executable,clargs=None, fargs=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (lenght of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    #02/08/2018 This modification is added to skip simulating MAAP5 run, if previousy run successfully
    caseNameM=''
    finished=False
    workingDir=str('/'.join(str(inputFiles[0]).split('/')[1:-1]))
    caseNameM = str(inputFiles[0].getBase())
    outFileM = 'out~'+caseNameM
    print('--------------',workingDir,outFileM)
    print('---out file existing---',os.path.exists('/'+workingDir+'/'+outFileM))
    if os.path.exists('/'+workingDir+'/'+outFileM):
      f=open('/'+workingDir+'/'+outFileM,'r')
      lines=f.readlines()
      for line in lines:
        if 'End Time:' in line: finished=True

    if finished:
      if os.path.exists('/'+workingDir+'/out~'+caseNameM+'_actual_branch_info.xml'):
        os.remove('/'+workingDir+'/out~'+caseNameM+'_actual_branch_info.xml')
        print('removed ','/'+workingDir+'/out~'+caseNameM+'_actual_branch_info.xml')
      print('finished=',finished)
      os.rename(('/'+workingDir+'/'+outFileM),('/'+workingDir+'/'+outFileM+'~old'))
      string='SIMULATION ALREADY RUN'
      print(string)
      self.caseName=caseNameM
      self.outFile=outFileM
      sleep_time=random.randint(5,20) #VR 17.10.18 add sleep to the scrip command instead of pausing raven
      #VR 17.10.18 todo=' echo ' + string + ' dummy.txt'
      todo=' echo ' + string + ' dummy.txt && sleep '+str(sleep_time)+'s' #VR 17.10.18
      returnCommand = [('parallel',todo)],outFileM
      #VR 17.10.18 sleep_time=random.randint(5,20) #VR 14.10.18
      print('Execution Command: '+str(returnCommand[0]))
      #VR 17.10.18 time.sleep(sleep_time) #VR 14.10.18
      return returnCommand

    else:     #02/08/2018 generateCommand as GenericCodeInterface
      if clargs==None:
        raise IOError('No input file was specified in clargs!')
      #check for output either in clargs or fargs
      #if len(fargs['output'])<1 and 'output' not in clargs.keys():
      #  raise IOError('No output file was specified, either in clargs or fileargs!')
      #check for duplicate extension use
      usedExt=[]
      for ext in list(clargs['input'][flag] for flag in clargs['input'].keys()) + list(fargs['input'][var] for var in fargs['input'].keys()):
        if ext not in usedExt:
          usedExt.append(ext)
        else:
          raise IOError('GenericCodeInterface cannot handle multiple input files with the same extension.  You may need to write your own interface.')

      #check all required input files are there
      inFiles=inputFiles[:]
      for exts in list(clargs['input'][flag] for flag in clargs['input'].keys()) + list(fargs['input'][var] for var in fargs['input'].keys()):
        for ext in exts:
          found=False
          for inf in inputFiles:
            if '.'+inf.getExt() == ext:
              found=True
              inFiles.remove(inf)
              break
          if not found:
            raise IOError('input extension "'+ext+'" listed in input but not in inputFiles!')
      #TODO if any remaining, check them against valid inputs

      #PROBLEM this is limited, since we can't figure out which .xml goes to -i and which to -d, for example.
      def getFileWithExtension(fileList,ext):
        """
        Just a script to get the file with extension ext from the fileList.
        @ In, fileList, the string list of filenames to pick from.
        @ Out, ext, the string extension that the desired filename ends with.
        """
        found = False
        for index,inputFile in enumerate(fileList):
          if inputFile.getExt() == ext:
            found=True
            break
        if not found:
          raise IOError('No InputFile with extension '+ext+' found!')
        return index,inputFile

      #prepend
      todo = ''
      todo += clargs['pre']+' '
      todo += executable
      index=None
      #inputs
      for flag,exts in clargs['input'].items():
        if flag == 'noarg':
          for ext in exts:
            idx,fname = getFileWithExtension(inputFiles,ext.strip('.'))
            todo+=' '+fname.getFilename()
            if index == None:
              index = idx
          continue
        todo += ' '+flag
        for ext in exts:
          idx,fname = getFileWithExtension(inputFiles,ext.strip('.'))
          todo+=' '+fname.getFilename()
          if index == None:
            index = idx
      #outputs
      #FIXME I think if you give multiple output flags this could result in overwriting
      self.caseName = inputFiles[index].getBase()
      outFile = 'out~'+self.caseName
      if 'output' in clargs:
        todo+=' '+clargs['output']+' '+outFile
      if self.fixedOutFileName is not None:
        outFile = self.fixedOutFileName
      todo+=' '+clargs['text']
      #postpend
      todo+=' '+clargs['post']
      returnCommand = [('parallel',todo)],outFile
      print('Execution Command: '+str(returnCommand[0]))
      return returnCommand






