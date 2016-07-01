"""
MAAP5 Code interface
Created on April 14, 2016
@created andrea (INL), rychvale (EDF) and picoco (The Ohio State University)
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

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

class MAAP5_GenericV7(GenericCode):

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and
      initialize some members based on inputs. This can be overloaded in specialized code interface in order
      to read specific flags
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
      if child.tag == 'stopSimulation': self.stop=child.text #this node defines if the MAAP5 simulation stop condition is: 'mission_time' or the occurrence of a given event e.g. 'IEVNT(691)'
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
    if 'DynamicEventTree' in samplerType:
      self.parent_ID=Kwargs['parentID']
      if Kwargs['parentID'] == 'root': self.oriInput(oriInputFiles) #original input files are checked only the first time
      self.stopSimulation(currentInputFiles, Kwargs)
      if Kwargs['parentID'] != 'root': self.restart(currentInputFiles, Kwargs['parentID'])
    return GenericCode.createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs)

  def oriInput(self, oriInputFiles):
    """
     ONLY IN CASE OF DET SAMPLER!
     this function read the original input file and the include file
     specified by the use into 'includeForTImer'block. In the input file,
     the method looks for the MAAP5 'END TIME' and 'PRINT INTERVAL'
     variables, it lists the DET sampled variables and the HYBRID ones,
     their default values. It checks that at least one branching exists, and
     that one branching condition exists for each of the DET sampled variables.
     The include file is also read looking for the timer defined for each DET sampled
     variable.
     @ In, oriInputFiles, list,
    """
    self.lineTimerComplete=[] #list of all the timer (one for each DET sampled variables)
    self.DETsampledVars = [] #list of RAVEN DET sampled variables
    self.HYBRIDsampledVars = [] #list of RAVEN HYBRID sampled variables
    self.DETdefaultValue={} #record default value for DET sampled variables: e.g. {'TIMELOCA':-1, 'AFWOFF':-1}
    self.HYBRIDdefaultValue={} #record default value for HYBRID sampled variables

    HYBRIDVar = False #is True, when for loop is in the block defining the variables which are HYBRID sampling
    DETVar = False #is True, when for loop is in the block defining the variables which are DET sampling
    foundDET = False #is True, when one branching condition is found

    for i in oriInputFiles:
      if '.inp' in str(i): inp=i.getAbsFile() #input file name with full path

    #read the original input file in order to find DET Sampled variables and their default values, save the print interval
    fileobject = open(inp, "r") #open .inp file
    lines=fileobject.readlines()
    fileobject.close()
    branching=[]
    for line in lines:
      if 'PRINT INTERVAL' in line:
        for i in line.split():
          if i. isdigit() or ('.' in i): self.printInterval=i
      if 'C DET Sampled Variables' in line: #to distinguish between DET sampling  and Hybrid sampling (in case of Hybrid DET)
        DETVar = True
      if 'END TIME' in line:
        for i in line.split():
          if i. isdigit() or ('.' in i): self.endTime=i
      if line.find('$RAVEN') != -1 and (DETVar == True): #MAAP Variable for RAVEN is e.g. AFWOFF = $RAVEN-AFWOFF$ (string.find('x') = -1 when 'x' is not in the string)
        var = line.split('=')[0]
        if ':' in line:
          line=line.split(':')[1]
          line=line.strip('$\n')
          self.DETdefaultValue[str(var.strip())]=str(line)
        else: self.DETdefaultValue[str(var.strip())]=''
        self.DETsampledVars.append(var.strip()) #var.strip() deletes whitespace
      if 'C End DET Sampled Variables' in line:
        DETVar = False
      if 'C HYBRID Sampled Variables' in line and (DETVar == False):
        HYBRIDVar = True
      if line.find('$RAVEN') != -1 and (HYBRIDVar == True):
        var = line.split('=')[0]
        if ':' in line:
          line=line.split(':')[1]
          line=line.strip('$\n')
          self.HYBRIDdefaultValue[str(var.strip())]=str(line)
        else: self.DETdefaultValue[str(var.strip())]=''
        self.HYBRIDsampledVars.append(var.strip()) #var.strip() deletes whitespace
      if 'C End HYBRID Sampled Variables' in line: HYBRIDVar= False
      if 'C Branching ' in line:
        foundDET = True
        var=line.split()[-1]
        branching.append(var)
    print('DET sampled Variables =',self.DETsampledVars)
    print('Hybrid sampled Variables =',self.HYBRIDsampledVars)
    print('branching',branching)

    for var in self.DETsampledVars:
      if not var in branching: raise IOError('Please define a branch/add a branching marker for the variable: ', var)

    if foundDET == True: print ('There is at least one branching condition for DET analysis')
    else               : raise IOError('No branching defined in the input file')

    if self.printInterval == '': raise IOError('Define a PRINT INTERVAL for the writing of the restart file')
    else                       : print ('Print interval =', self.printInterval)

    #read the include file in order to check that for all the branching there is a TIMER defined
    self.stopTimer=[]
    print('self.stop',self.stop)
    self.timer = {} #this dictionary contains for each DETSampledVar the number of the corrisponding TIMER set
    fileobject = open(self.include, "r") #open include file, this file contains all the user-defined TIMER
    lines=fileobject.readlines()
    fileobject.close()
    lineNumber=0
    branchingMarker=''
    stop_marker='C End Simulation'
    found = [False]*len(self.DETsampledVars) #TIMER condition
    block = [False]*len(self.DETsampledVars) #timer 'block'
    stop=False
    for line in lines:
        if str(self.stop).strip() != 'mission_time':
            if str(stop_marker) in line:
                stop=True
                continue
            if (str('SET TIMER') in line) and (stop == True):
                print('line',line)
                self.stopTimer = line.split()[-1]
                print('stopTimer =', self.stopTimer)
                stop=False
        for cont, var in enumerate(self.DETsampledVars):
            var = var.encode('utf8')
            branchingMarker=str('C Branching '+var)
            if branchingMarker in line: #branching timer marker
                 block[cont] = True
            if (str('SET TIMER') in line) and (block[cont] == True):
                 found[cont] = True
                 self.timer[var]= line.split()[-1]
                 self.lineTimerComplete.append('TIMER '+str(self.timer[var])) #this list contains all the TIMER associated with the branching e.g. [TIMER 100, TIMER 101, TIMER 102]
                 print ('TIMER found for', var)
                 block[cont]=False
            if (str('END') in line) and (block[cont] == True) and (found[cont] == False):
                 print ('TIMER not found for', var)
                 block[cont] = False
    for cont, val in enumerate(found):
        if (val == False): raise IOError('Please define a TIMER for', self.DETsampledVars[cont])

  def restart(self,currentInputFiles, parent_name):
      """
       ONLY IN CASE OF DET SAMPLER!
       this function reads the input file and, for each branch, changes
       rhe value of the RESTART TIME and RESTART FILE to be used
      """
      correct_restart='' #is the correct line for the restart time definition
      new_line=''
      restartFileCorrect=''
      for i in currentInputFiles:
          if '.inp' in str(i):
              break
      inp=i.getAbsFile() #current MAAP5 input file with path e.g. /Bureau/V6/TEST_V6_DET/TIMELOCA_grid/maap5input_generic/testDummyStep/DET_1_1/test.inp
      currentFolder, base = os.path.split(inp)
      base=base.split('.')[0] #test
      parent_folder= '../'+parent_name
      tilast=self.tilastDict[parent_name]

#given tilast, then correct RESTART TIME is tilast-self.printInterval
      fileobject = open(inp, "r") #open MAAP .inp for the current run
      lines=fileobject.readlines()
      fileobject.close()

      #verify the restart file and restart time is defined into the input file
      restart_time = False #is True, if the restart time defined is correct
      found_restart = False #is True, if the restart file declaration is found
      correct = False #is True, if the restart file found is correct
      restartFileCorrect=os.path.join(parent_folder, base+".res")
      print('correct restart file is',restartFileCorrect)
      lineNumber=0
      restart_time_new=0
      for line in lines:
          lineNumber=lineNumber+1
          if 'START TIME' in line:
                restart_time_new=math.floor(float(tilast)-float(self.printInterval))
                correct_restart= 'RESTART TIME IS '+str(restart_time_new)+'\n'
                if line == correct_restart:
                    restart_time=True
                    break
                else:
                    lines[lineNumber-1]=correct_restart
                    fileobject = open(inp, "w")
                    linesNewInput = "".join(lines)
                    fileobject.write(linesNewInput)
                    fileobject.close()
                    break
          # once checked for the correct restart time, he interface looks for the restart file definition
          if 'RESTART FILE ' in line:
              found_restart = True
              restartFile =  line.split(" ")[-1].strip()
              if restartFile == restartFileCorrect:
                  correct = True
              print ('RESTART FILE is correct: ',restartFile)
              break

      if found_restart == False:
          print ('NO RESTART FILE declared in the input file')

          #the restart file declaration need to be added to the input file
          line_include='INCLUDE '+str(self.include)+'\n'
          index=lines.index(line_include)
          lines.insert(index+1, 'RESTART FILE '+restartFileCorrect+'\n')
          fileobject = open(inp, "w")
          linesNewInput = "".join(lines)
          fileobject.write(linesNewInput)
          fileobject.close()
          print ('RESTART FILE declared in the input file')
      elif found_restart == True and correct == False:
          print ('RESTART FILE declared is not correct', restartFile)
          line_include='INCLUDE '+str(self.include)+'\n'
          index=lines.index(line_include)
          new_line='RESTART FILE '+restartFileCorrect+'\n'
          lines[index+1]=new_line
          fileobject = open(inp, "w")
          linesNewInput = "".join(lines)
          fileobject.write(linesNewInput)
          fileobject.close()
          print ('RESTART FILE name has been corrected',restartFileCorrect)

  def finalizeCodeOutput(self, command, output, workingDir):
      """
       finalizeCodeOutput checks MAAP csv files and looks for iEvents and
       continous variables we specified in < boolMaapOutputVariables> and
       <contMaapOutputVairables> sections of RAVEN_INPUT.xml file. Both
       < boolMaapOutputVariables> and <contMaapOutputVairables> should be
       contained into csv MAAP csv file
       In case of DET sampler, if a new branching condition is met, the
       method writes the xml for creating the two new branches.
      """
      csvSimulationFiles=[]
      realOutput=output.split("out~")[1] #rootname of the simulation files
      inp = os.path.join(workingDir,realOutput + ".inp") #input file of the simulation with the full path
      filePrefixWithPath=os.path.join(workingDir,realOutput) #rootname of the simulation files with the full path
      csvSimulationFiles=glob.glob(filePrefixWithPath+".d"+"*.csv") #list of MAAP output files with the evolution of continuos variables
      mergeCSV=csvU.csvUtilityClass(csvSimulationFiles,1,";",True)
      dataDict={}
      dataDict=mergeCSV.mergeCsvAndReturnOutput({'variablesToExpandFrom':['TIME'],'returnAsDict':True})
      timeFloat=dataDict['TIME']
      """Here we'll read evolution of continous variables """
      contVariableEvolution=[] #here we'll store the time evolution of MAAP continous variables
      if len(self.contOutputVariables)>0:
          for variableName in self.contOutputVariables:
              try: (dataDict[variableName])
              except: raise IOError('define the variable within MAAP5 plotfil: ',variableName)
              contVariableEvolution.append(dataDict[variableName])

      """here we'll read boolean variables and transform them into continous"""
      # if the discrete variables of interest are into the csv file:
      if len(self.boolOutputVariables)>0:
          boolVariableEvolution=[]
          for variable in self.boolOutputVariables:
              variableName='IEVNT('+str(variable)+')'
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

      RAVEN_outputfile=os.path.join(workingDir,output+".csv") #RAVEN will look for  output+'.csv'file but in the workingDir, so we need to append it to the filename
      outputCSVfile=open(RAVEN_outputfile,"w+")
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
          dict_timer={}
          print(self.timer.values())
          for timer in self.timer.values():
              timer='TIM'+str(timer) #in order to pass from 'TIMER 100' to 'TIMER_100'
              try: (dataDict[timer])
              except: raise IOError('Please ensure that the timer is defined into the include file and then it is contained into MAAP5 plotfil: ',timer)
              print(dataDict[timer].tolist(),timer)
              if 1.0 in dataDict[timer].tolist():
                  index=dataDict[timer].tolist().index(1.0)
                  timer_activation= timeFloat.tolist()[index]
              else: timer_activation=-1
              dict_timer[timer]=timer_activation

          print('dict_timer',dict_timer)

          key_1 = max(dict_timer.values())
          d1 = dict((v, k) for k, v in dict_timer.iteritems())
          timer_activated = d1[key_1]
          key_2 = timer_activated.split('TIM')[-1]
          d2 = dict((v, k) for k, v in self.timer.iteritems())
          var_activated = d2[key_2]
          currentFolder=workingDir.split('/')[-1]

          for key, value in self.values[currentFolder].items():
              if key == var_activated: self.branch[currentFolder]=(key,value)

          self.dictVariables(inp)
          if self.stop.strip()!='mission_time':
              event=False
              user_stop='IEVNT('+str(self.stop)+')'
              if dataDict[user_stop][-1]==1.0: event=True

          condition=False
          tilast=str(timeFloat[-1])
          self.tilastDict[currentFolder]=tilast
          if self.stop.strip()=='mission_time': condition=(math.floor(float(tilast)) >= math.floor(float(self.endTime)))
          else: condition=event
          if not condition:
              dictBranchCurrent='Dict'+str(self.branch[currentFolder][0])
              dictChanged=self.dictAllVars[dictBranchCurrent]
              self.branch_xml(tilast, dictChanged,inp,dataDict)

  def branch_xml(self, tilast,Dict,inputFile,dataDict):
      """
      ONLY FOR DET SAMPLER!
      Branch_xml writes the xml files used by RAVEN to create the two branches at each stop condition reached
      """
      base=os.path.basename(inputFile).split('.')[0]
      path=os.path.dirname(inputFile)
      filename=os.path.join(path,'out~'+base+"_actual_branch_info.xml")
      stopInfo={'end_time':tilast}
      listDict=[]
      variableBranch=''
      branchName=path.split('/')[-1]
      variableBranch=self.branch[str(branchName)][0]
      DictName='Dict'+str(variableBranch)
      dict1=self.dictAllVars[DictName]
      variables = list(dict1.keys())
      for var in variables: #e.g. for TIMELOCA variables are ABBN(1), ABBN(2), ABBN(3)
          if var==self.branch[branchName]: #ignore if the variable coincides with the trigger
              continue
          else:
              new_value=str(dict1[var])
              old_value=str(dataDict[var][0])
              branch={'name':var, 'type':'auxiliar','old_value': old_value, 'new_value': new_value.strip('\n')}
              listDict.append(branch)
          detU.writeXmlForDET(filename,variableBranch,listDict,stopInfo) #this function writes the xml file for DET

  def dictVariables(self,current_inp):
      """
      ONLY FOR DET SAMPLER!
      This method creates a dictionary for the variables determining and branch and the values of the
      variables changed due to the branch occurrence. Then is used to create the xml necessary to
      have the two branches.
      """
      fileobject = open(current_inp, "r") #open MAAP .inp
      lines=fileobject.readlines()
      fileobject.close()
      self.dictAllVars= {} #this dictionary will contain all the dicitonary, one for each DET sampled variable
      for var in self.DETsampledVars:
          block = False
          dictVar = {}
          for line in lines:
              if (var in line) and ('=' in line) and not ('WHEN' or 'IF') in line:
              #conditions on var and '=' ensure that there is an assignment,while condition on 'WHEN'/'IF' excludes e.g.'WHEN TIM<=AFWOFF' line
                  sampledVar = line.split('=')[0].strip()
                  sampled_value = str(line.split('=')[1])
                  dictVar[sampledVar] = sampled_value
                  continue
              if ('C Branching '+var) in line: #branching marker
                  block = True
              if ('=' in line) and (block == True) and not ('WHEN' or 'IF') in line: #there is a 'Branching block'
                  modified_var = line.split('=')[0].strip()
                  modifiedValue = line.split('=')[1]
                  dictVar[modified_var] = modifiedValue
              if ('END' in line) and (block == True):
                  block = False
          self.dictAllVars["Dict{0}".format(var)]= dictVar #with Dict{0}.format(var) dictionary referred to each single sampled variable is called dictVar (e.g., 'DictTIMELOCA', 'DictAFWOFF')

  def stopSimulation(self,currentInputFiles, Kwargs):
      """
      ONLY FOR DET SAMPLER!
      This method update the stop simulation condition into the MAAP5 input
      to stop the run when the new branch occurs
      """

      for i in currentInputFiles:
          if '.inp' in str(i):
              inp=i.getAbsFile() #input file name with full path

      fileobject = open(inp, "r")
      lines=fileobject.readlines()
      fileobject.close()

      currentFolder=os.path.dirname(inp)
      currentFolder=currentFolder.split('/')[-1]
      parents=[]
      self.values[currentFolder]=Kwargs['SampledVars']
      lineStop = int(lines.index(str('C Stop Simulation condition\n')))+1

      if Kwargs['parentID']== 'root':
          if self.stop!='mission_time':
              self.lineTimerComplete.append('TIMER '+str(self.stopTimer))
          if all(str(i) in lines[lineStop] for i in self.lineTimerComplete) == False:
             print(' self.lineTimerComplete', self.lineTimerComplete)
             raise IOError('All TIMER must be considered for the first simulation') #in the original input file all the timer must be mentioned
      else: #Kwargs['parentID'] != 'root'
          parent=currentFolder[:-2]
          parents.append(parent)
          print('parent.split(_)', parent.split('_'))
          while len(parent.split('_')[-1])>2: #collect the name of all the parents, their corresponding timer need to be deleted from the stop condition (when the parents has already occurred)
              parent=parent[:-2]
              parents.append(parent)
          lineTimer= list(self.lineTimerComplete)
          for parent in parents:
              varTimerParent=self.branch[parent][0]
              valueVarTimerParent=self.branch[parent][1]
              for key,value in self.values[currentFolder].items():
                  if key == varTimerParent and value != valueVarTimerParent: continue
                  elif key != varTimerParent: continue
                  else:
                      timerToBeRemoved=str('TIMER '+ self.timer[self.branch[parent][0]])
                      print('timerToBeRemoved',timerToBeRemoved)
                      if timerToBeRemoved in lineTimer: lineTimer.remove(timerToBeRemoved)

          listTimer=['IF']
          print('lineTimer',lineTimer)
          if len(lineTimer)>0:
              for tim in lineTimer:
                  listTimer.append('(' + str(tim))
                  listTimer.append('>')
                  listTimer.append('0)')
                  listTimer.append('OR')
              listTimer.pop(-1)
              new_line=' '.join(listTimer)+'\n'
              lines[lineStop]=new_line
          else: lines[lineStop-1:lineStop+3]='\n'

          fileobject = open(inp, "w")
          linesNewInput = "".join(lines)
          fileobject.write(linesNewInput)
          fileobject.close()


