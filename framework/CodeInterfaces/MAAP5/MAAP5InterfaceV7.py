'''
Created on April 14, 2016
@created andrea (INL)
@modified: rychvale (EDF)
@modified: picoco (The Ohio State University)
'''

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
        GenericCode._readMoreXML(self,xmlNode)
        self.include=''
        self.tilast_dict={} #{'folder_name':'tilast'} - this dictionary contains the last simulation time of each branch, this is necessary to define the correct restart time
        self.branch = {} #{'folder_name':['variable branch','variable value']} where variable branch is the variable sampled for the current branch e.g. {det_1:[timeloca, 200]}
        self.values = {} #{'folder_name':[['variable branch_1','variable value_1'],['variable branch_2','variable value_2']]} for each DET sampled variables
        self.print_interval = ''  #value of the print interval
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
        self.line_timer_complete=[] #list of all the timer (one for each DET sampled variables)
        self.DETsampledVars = [] #list of RAVEN DET sampled variables
        self.HYBRIDsampledVars = [] #list of RAVEN HYBRID sampled variables
        self.DETdefault_value={} #record default value for DET sampled variables: e.g. {'TIMELOCA':-1, 'AFWOFF':-1}
        self.HYBRIDdefault_value={} #record default value for HYBRID sampled variables

        HYBRID_var = False #is True, when for loop is in the block defining the variables which are HYBRID sampling
        DET_var = False #is True, when for loop is in the block defining the variables which are DET sampling
        found_DET = False #is True, when one branching condition is found

        for i in oriInputFiles: 
            if '.inp' in str(i):    
                inp=i.getAbsFile() #input file name with full path

        #read the original input file in order to find DET Sampled variables and their default values, save the print interval
        fileobject = open(inp, "r") #open .inp file
        lines=fileobject.readlines()
        fileobject.close()
        branching=[]
        for line in lines:
            if 'PRINT INTERVAL' in line:
                for i in line.split():
                    if i. isdigit() or ('.' in i): self.print_interval=i            
                continue
            if 'C DET Sampled Variables' in line: #to distinguish between DET sampling  and Hybrid sampling (in case of Hybrid DET)
                DET_var = True
                continue
            if 'END TIME' in line:  
                for i in line.split():
                    if i. isdigit() or ('.' in i): self.end_time=i            
                continue
            if line.find('$RAVEN') != -1 and (DET_var == True): #MAAP Variable for RAVEN is e.g. AFWOFF = $RAVEN-AFWOFF$ (string.find('x') = -1 when 'x' is not in the string)
                var = line.split('=')[0]
                if ':' in line:
                    line=line.split(':')[1]
                    line=line.strip('$\n')
                    self.DETdefault_value[str(var.strip())]=str(line)
                else: self.DETdefault_value[str(var.strip())]=''
                self.DETsampledVars.append(var.strip()) #var.strip() deletes whitespace
                continue
            if 'C End DET Sampled Variables' in line: 
                DET_var = False
                continue            
            if 'C HYBRID Sampled Variables' in line and (DET_var == False): 
                HYBRID_var = True
                continue
            if line.find('$RAVEN') != -1 and (HYBRID_var == True): 
                var = line.split('=')[0]
                if ':' in line:
                    line=line.split(':')[1]
                    line=line.strip('$\n')
                    self.HYBRIDdefault_value[str(var.strip())]=str(line)
                else: self.DETdefault_value[str(var.strip())]=''
                self.HYBRIDsampledVars.append(var.strip()) #var.strip() deletes whitespace
                continue
            if 'C End HYBRID Sampled Variables' in line: HYBRID_var= False           
            if 'C Branching ' in line:
                found_DET = True
                var=line.split()[-1]
                branching.append(var)
                continue

        print('DET sampled Variables =',self.DETsampledVars)
        print('branching',branching)

        for var in self.DETsampledVars:
            if not var in branching: raise IOError('Please define a branch/add a branching marker for the variable: ', var)

        if found_DET == True: print ('There is at least one branching condition for DET analysis')
        else: raise IOError('No branching defined in the input file')

        if self.print_interval == '': raise IOError('Define a PRINT INTERVAL for the writing of the restart file')   
        else: print ('Print interval =', self.print_interval)

        #read the include file in order to check that for all the branching there is a TIMER defined
        self.stop_timer=[]
        print('self.stop',self.stop)
        self.timer = {} #this dictionary contains for each DETSampledVar the number of the corrisponding TIMER set
        fileobject = open(self.include, "r") #open include file, this file contains all the user-defined TIMER
        lines=fileobject.readlines()
        fileobject.close()
        lineNumber=0
        branching_marker=''
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
                    self.stop_timer = line.split()[-1]
                    print('stop_timer =', self.stop_timer)
                    stop=False
            for cont, var in enumerate(self.DETsampledVars):
                var = var.encode('utf8')
                branching_marker=str('C Branching '+var)
                if branching_marker in line: #branching timer marker   
                     block[cont] = True    
                if (str('SET TIMER') in line) and (block[cont] == True):
                     found[cont] = True 
                     self.timer[var]= line.split()[-1]
                     self.line_timer_complete.append('TIMER '+str(self.timer[var])) #this list contains all the TIMER associated with the branching e.g. [TIMER 100, TIMER 101, TIMER 102]
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
        restart_file_correct=''
        for i in currentInputFiles: 
            if '.inp' in str(i):
                break 
        inp=i.getAbsFile() #current MAAP5 input file with path e.g. /Bureau/V6/TEST_V6_DET/TIMELOCA_grid/maap5input_generic/testDummyStep/DET_1_1/test.inp
        current_folder, base = os.path.split(inp) 
        base=base.split('.')[0] #test
        parent_folder= '../'+parent_name 
        tilast=self.tilast_dict[parent_name]

	#given tilast, then correct RESTART TIME is tilast-self.print_interval
        fileobject = open(inp, "r") #open MAAP .inp for the current run
        lines=fileobject.readlines()
        fileobject.close()
        
        #verify the restart file and restart time is defined into the input file
        restart_time = False #is True, if the restart time defined is correct
        found_restart = False #is True, if the restart file declaration is found
        correct = False #is True, if the restart file found is correct
        restart_file_correct=os.path.join(parent_folder, base+".res")
        print('correct restart file is',restart_file_correct)
        line_number=0
        restart_time_new=0
        for line in lines:
            line_number=line_number+1
            if 'START TIME' in line: 
                  restart_time_new=math.floor(float(tilast)-float(self.print_interval))
                  correct_restart= 'RESTART TIME IS '+str(restart_time_new)+'\n'
                  if line == correct_restart:
                      restart_time=True
                      break
                  else:
                      lines[line_number-1]=correct_restart
                      fileobject = open(inp, "w") 
                      lines_new_input = "".join(lines)
                      fileobject.write(lines_new_input)
                      fileobject.close()
                      break
            # once checked for the correct restart time, he interface looks for the restart file definition     
            if 'RESTART FILE ' in line:
                found_restart = True
                restart_file =  line.split(" ")[-1].strip()
                if restart_file == restart_file_correct:
                    correct = True
	            print ('RESTART FILE is correct: ',restart_file) 
                break

        if found_restart == False: 
            print ('NO RESTART FILE declared in the input file')

            #the restart file declaration need to be added to the input file
            line_include='INCLUDE '+str(self.include)+'\n'
            index=lines.index(line_include)
            lines.insert(index+1, 'RESTART FILE '+restart_file_correct+'\n')
            fileobject = open(inp, "w") 
            lines_new_input = "".join(lines)
            fileobject.write(lines_new_input)
            fileobject.close()
            print ('RESTART FILE declared in the input file')
        elif found_restart == True and correct == False: 
            print ('RESTART FILE declared is not correct', restart_file)
            line_include='INCLUDE '+str(self.include)+'\n'
            index=lines.index(line_include)
            new_line='RESTART FILE '+restart_file_correct+'\n'
            lines[index+1]=new_line
            fileobject = open(inp, "w") 
            lines_new_input = "".join(lines)
            fileobject.write(lines_new_input)
            fileobject.close()
            print ('RESTART FILE name has been corrected',restart_file_correct)

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
        os.chdir(workingDir)

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
            current_folder=workingDir.split('/')[-1]

            for key, value in self.values[current_folder].items():
                if key == var_activated: self.branch[current_folder]=(key,value)

            self.dictVariables(inp)
            if self.stop.strip()!='mission_time': 
                event=False
                user_stop='IEVNT('+str(self.stop)+')'
                if dataDict[user_stop][-1]==1.0: event=True

            condition=False
            tilast=str(timeFloat[-1])
            self.tilast_dict[current_folder]=tilast
            if self.stop.strip()=='mission_time': condition=(math.floor(float(tilast)) >= math.floor(float(self.end_time)))
            else: condition=event
            if not condition:
                Dict_branch_current='Dict'+str(self.branch[current_folder][0])
                Dict_changed=self.Dict_allVars[Dict_branch_current]
                self.branch_xml(tilast, Dict_changed,inp,dataDict)

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
        variable_branch=''
        branch_name=path.split('/')[-1]
        variable_branch=self.branch[str(branch_name)][0]
        DictName='Dict'+str(variable_branch) 
        dict1=self.Dict_allVars[DictName]
        variables = list(dict1.keys())
        for var in variables: #e.g. for TIMELOCA variables are ABBN(1), ABBN(2), ABBN(3)
            if var==self.branch[branch_name]: #ignore if the variable coincides with the trigger
                continue
            else:
                new_value=str(dict1[var])
                old_value=str(dataDict[var][0])
                branch={'name':var, 'type':'auxiliar','old_value': old_value, 'new_value': new_value.strip('\n')}
                listDict.append(branch)
            detU.write_xml_for_DET(filename,variable_branch,listDict,stopInfo) #this function writes the xml file for DET       

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
        self.Dict_allVars= {} #this dictionary will contain all the dicitonary, one for each DET sampled variable
        for var in self.DETsampledVars:
            block = False
            Dict_var = {}
            for line in lines:
                if (var in line) and ('=' in line) and not ('WHEN' or 'IF') in line:
                #conditions on var and '=' ensure that there is an assignment,while condition on 'WHEN'/'IF' excludes e.g.'WHEN TIM<=AFWOFF' line
                    sampled_var = line.split('=')[0].strip()
                    sampled_value = str(line.split('=')[1])
                    Dict_var[sampled_var] = sampled_value
                    continue
                if ('C Branching '+var) in line: #branching marker 
                    block = True 
                if ('=' in line) and (block == True) and not ('WHEN' or 'IF') in line: #there is a 'Branching block'
                    modified_var = line.split('=')[0].strip()
                    modified_value = line.split('=')[1]
                    Dict_var[modified_var] = modified_value
                if ('END' in line) and (block == True):            
                    block = False
            self.Dict_allVars["Dict{0}".format(var)]= Dict_var #with Dict{0}.format(var) dictionary referred to each single sampled variable is called Dict_var (e.g., 'DictTIMELOCA', 'DictAFWOFF')

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

        current_folder=os.path.dirname(inp)
        current_folder=current_folder.split('/')[-1]
        parents=[]
        self.values[current_folder]=Kwargs['SampledVars']
        line_stop = int(lines.index(str('C Stop Simulation condition\n')))+1
  
        if Kwargs['parentID']== 'root':
            if self.stop!='mission_time': 
                self.line_timer_complete.append('TIMER '+str(self.stop_timer))
            if all(str(i) in lines[line_stop] for i in self.line_timer_complete) == False:
               print(' self.line_timer_complete', self.line_timer_complete)
               raise IOError('All TIMER must be considered for the first simulation') #in the original input file all the timer must be mentioned 
        else: #Kwargs['parentID'] != 'root'
            parent=current_folder[:-2]
            parents.append(parent)
            print('parent.split(_)', parent.split('_'))
            while len(parent.split('_')[-1])>2: #collect the name of all the parents, their corresponding timer need to be deleted from the stop condition (when the parents has already occurred)
                parent=parent[:-2]
                parents.append(parent)
            line_timer= list(self.line_timer_complete)
            for parent in parents:
                var_timer_parent=self.branch[parent][0]
                value_var_timer_parent=self.branch[parent][1]
                for key,value in self.values[current_folder].items():
                    if key == var_timer_parent and value != value_var_timer_parent: continue
                    elif key != var_timer_parent: continue
                    else: 
                        timer_to_be_removed=str('TIMER '+ self.timer[self.branch[parent][0]])
                        print('timer_to_be_removed',timer_to_be_removed)
                        if timer_to_be_removed in line_timer: line_timer.remove(timer_to_be_removed)

            list_timer=['IF']
            print('line_timer',line_timer)
            if len(line_timer)>0:
                for tim in line_timer:
                    list_timer.append('(' + str(tim))
                    list_timer.append('>')
                    list_timer.append('0)')
                    list_timer.append('OR')
                list_timer.pop(-1)
                new_line=' '.join(list_timer)+'\n'
                lines[line_stop]=new_line
            else: lines[line_stop-1:line_stop+3]='\n'

            fileobject = open(inp, "w") 
            lines_new_input = "".join(lines)
            fileobject.write(lines_new_input)
            fileobject.close()


