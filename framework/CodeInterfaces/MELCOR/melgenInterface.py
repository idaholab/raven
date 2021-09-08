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
  Created on April 18, 2017
  @author: Matteo Donorio (University of Rome La Sapienza)
  @author: Tommaso Glingler (University of Rome La Sapienza)
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
import os
import MELCORdata
import GenericParser
import copy				
import shutil
from utils import utils
from CodeInterfaceBaseClass import CodeInterfaceBase
import re
from collections import defaultdict
from math import *

class MelgenApp(CodeInterfaceBase):
  """
    This class is the CodeInterface for MELGEN (a sub-module of Melcor)
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.inputExtensions = ['i','inp']
    self.detVars = []

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
      Generate a command to run MELGEN (a sub-module of Melcor)
      Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    import melcorCombinedInterface
    melcorInpFile = melcorCombinedInterface.MelcorApp.melcorInpFile
    isDet = melcorCombinedInterface.MelcorApp.melcorDetNode
    outputfile = 'OUTPUT'
    if isDet != None:
      outputfile = 'out~'+ melcorInpFile[:-2]#prefix ~out needed to create branchinfo.xml
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError("Unknown input extension. Expected input file extensions are "+ ",".join(self.getInputExtension()))
    if clargs:
      precommand = executable + clargs['text']
    else:
      precommand = executable
    executeCommand = [('serial',precommand + ' '+inputFiles[index].getFilename())]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def createNewInput(self,currentInputFiles,origInputFiles,samplerType,**Kwargs):
    """
      This generates a new input file depending on which sampler has been chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    print(Kwargs)
    import melcorCombinedInterface
    indexes = []
    inFiles = []
    origFiles = []
    workingDir = Kwargs['subDirectory']
    self._samplersDictionary = {}
    self.det = 'dynamiceventtree' in str(samplerType).lower()
    isDet = melcorCombinedInterface.MelcorApp.melcorDetNode
    # find input file index
    index = self._findInputFileIndex(currentInputFiles)
    # instanciate the parser
    subworkingDir = Kwargs['WORKING_DIR']
    firstDir = subworkingDir + "/DET_1"
    if self.det:
      melcorInpFile = melcorCombinedInterface.MelcorApp.melcorInpFile
      if workingDir != firstDir:
        self.inputAliases = Kwargs.get('alias').get('input')
        self._samplersDictionary[samplerType] = self.dynamicEventTreeForMELCOR
        self.detVars = Kwargs.get('DETVariables')
        if 'None' not in str(samplerType):
          Kwargs['currentPath'] = currentInputFiles[index].getPath()
          modifDict = self._samplersDictionary[samplerType](**Kwargs)
        if isDet != 'olderMelcor':				
          if modifDict['happenedEvent']:
            newInput = self.writeNewInput('happenedEvent', melcorInpFile, workingDir)
          else:
            newInput = self.writeNewInput('nothappenedEvent', melcorInpFile, workingDir)
        if not self.detVars:
          raise IOError('ERROR in "MELCOR Code Interface": NO DET variables with DET sampler!!!')
      else:
        print("FIRST RUN")
    self.__transferMetadata(Kwargs.get("metadataToTransfer",None), currentInputFiles[index].getPath())

    for index,inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        indexes.append(index)
        inFiles.append(inputFile)
      for index,inputFile in enumerate(origInputFiles):
        if inputFile.getExt() in self.getInputExtension():
          origFiles.append(inputFile)
    parser = GenericParser.GenericParser(inFiles)
    parser.modifyInternalDictionary(**Kwargs)
    parser.writeNewInput(currentInputFiles,origFiles)
    if isDet == 'olderMelcor': #MELCOR 1.86
      for var in Kwargs['SampledVars']:
        if workingDir != firstDir:
          tripVar = "%" + self.messageReader(workingDir[:-2])[0] + "%"
          if modifDict['happenedEvent'] and var == tripVar:
            newInput = self.writeEdf('happenedEvent' , melcorInpFile , workingDir , var)
          else:
            newInput = self.writeEdf('nothappenedEvent' , melcorInpFile , workingDir , var)
        else:
          newInput = self.writeEdf('nothappenedEvent' , melcorInpFile , workingDir, var)
    return currentInputFiles

  def stopDET (self, workingDir):
    """
      @ In, happenedorNot, string, info on whether event happend or not
			@ In, inputFile, string, name of melcor input file
			@ In, workingDir, string, actual working Directory
			@ In, var, string, name of sampled variables present written as '%var%'
      @ Out, None
    """
    found = False
    for var in self.detVars:
      variable = var[1:-1] #sampled variable without prefix and suffix %
      filename = "EDF_" + str( variable ) + ".txt"
      zeroOrNot = self.findZero( filename, workingDir )
      if zeroOrNot == 1.0:
        found = False
        break
      else:
        found = True
    return found

  def writeEdf (self, happenedorNot , inputFile , workingDir , var):
    """
      @ In, happenedorNot, string, info on whether event happend or not
			@ In, inputFile, string, name of melcor input file
			@ In , workingDir, string, actual working Directory
			@ In, var, string, name of sampled variables present written as '%var%'
      @ Out, None
    """
    tripVariable = var[1:-1] #sampled variable without prefix and suffix %
    filename = "EDF_" + str(tripVariable) + ".txt"
    Edf = open(os.path.join( workingDir , filename), 'a')
    sampledVar = self.ravenSample(inputFile , workingDir , tripVariable)
    if workingDir.endswith("/DET_1"):
      if happenedorNot == 'happenedEvent':
        Edf.write("%.5E" %0.0 + " " + "%.5E" %0.0 + " " + "%.5E" %sampledVar + "\n")
      else:
        Edf.write("%.5E" %0.0 + " " + "%.5E" %1.0 + " " + "%.5E" %sampledVar + "\n")
    else:
      restartTime = self.restartTime(workingDir[:-2])
      restartTimeStep = self.restartTimeStep(workingDir[:-2])
      self.writeNewRestart(inputFile, workingDir, restartTimeStep)
      zeroOrNot = self.findZero(filename, workingDir[:-2])
      if happenedorNot == 'happenedEvent':
        Edf.write("%.5E" %restartTime + " " + "%.5E" %0.0 + " " + "%.5E" %sampledVar + "\n")
      else:
        Edf.write("%.5E" %restartTime + " " + "%.5E" %zeroOrNot + " " + "%.5E" %sampledVar + "\n")
    Edf.close()

  def findZero(self, filename, workingDir):
    """
      @ In, filename, string, EDF_var.txt file present in previous working Directory
			@ In, workingDir, string, previous working Directory where EDF.txt files are present
      @ Out, zeroOrNot, int, returns 0.0 if last line in EDF file has a 0.0 in second column (column numeration starting from 1 while in MELCOR EDF column numeration starts from 0)
    """
    file = open(os.path.join(workingDir,filename),"r")
    for lastLine in file:
      pass
    lastLine = lastLine.rstrip()
    lastLine = lastLine.split()
    if float(lastLine[1]) == 0.0:
      zeroOrNot = 0.0
    else:
      zeroOrNot = 1.0
    file.close()
    return zeroOrNot
		
  def changeRestart (self, inputFile, workingDir, restartTimeStep):
    """
      @ In, inputFile, string, name of melcor input file
			@ In , workingDir, string, actual working Directory
			@ In, restartTimeStep, int, last restart cycle present in previous .MES file
      @ Out, newInput, list, container of inputFile with changed RESTART cycle
    """	  
    file = open(os.path.join(workingDir,inputFile),"r")
    newInput = []
    for index, line in enumerate(file, 1):
      line = line.rstrip()
      line = line.split()
      if line[0] == 'RESTART':
        newLine = [line[0], int(restartTimeStep)]
        newInput.append(newLine)
      else:
        newInput.append(line)
    return newInput

  def writeNewRestart(self, inputFile, workingDir, restartTimeStep):
    """
      This writes a new input file
      @ In, workingDir, string, workingDir where MELCOR output message in generated
      @ In, inputFile, string, MELCOR input
      @ Out, newInput , file, modified MELCOR input
    """
    newInput = self.changeRestart(inputFile, workingDir, restartTimeStep)
    newFile = open(os.path.join(workingDir, inputFile),"w")
    for line in newInput:
      for words in line:
        newFile.write(str(words) + " ")
      newFile.write('\n')
    newFile.close()

  def restartTimeStep(self, workingDir):  #OK   stoppedCF = {} and tripVariable = 'string'
    """
      @ In, workingDir, string, previous working Directory where .MES file is present
      @ Out, restartInfo, list, container of previous restart time (restartInfo[0]) and restart time step (restartInfo[1])
    """ 
    import melcorCombinedInterface
    melcorOutFile = melcorCombinedInterface.MelcorApp.melcorOutFile
    outputToRead = open(os.path.join(workingDir,melcorOutFile),"r")
    tripTime = self.messageReader(workingDir)[1][0]
    oldValue = 1
    restartTimeStep = 0
    for index, line in enumerate(outputToRead, 1):
      line = line.rstrip()
      line = line.split()
      if line[0] == 'Restart':
        if float(line[3]) == tripTime:
          break
        if int(line[5]) > oldValue:
          restartTimeStep = int(line[5])
          oldValue = int(line[5])
          continue
        else:
          continue
    if restartTimeStep == 0: #if previous .MES file doesn't contain info on last restart, therefore restart info searches in two times (or more) previous working directory
      try:
        restartTimeStep = self.restartTimeStep(workingDir[:-2])
      except:
        print("ERROR in time step. In order to restart MELCOR needs a restart dump written in .MES previous to the trip time step. Integrate the new correction in the MELCOR input")
    outputToRead.close()
    return restartTimeStep

  def restartTime(self, workingDir):  #OK   stoppedCF = {} and tripVariable = 'string'
    """
      @ In, workingDir, string, previous working Directory where .MES file is present
      @ Out, restartInfo, list, container of previous restart time (restartInfo[0]) and restart time step (restartInfo[1])
    """ 
    import melcorCombinedInterface
    melcorOutFile = melcorCombinedInterface.MelcorApp.melcorOutFile
    outputToRead = open(os.path.join(workingDir,melcorOutFile),"r")
    tripTime = self.messageReader(workingDir)[1][0]
    oldValue = 1
    restartTime = 0
    for index, line in enumerate(outputToRead, 1):
      line = line.rstrip()
      line = line.split()
      if line[0] == 'Listing':
        if float(line[3]) == tripTime:
          break
        if int(line[5]) > oldValue:
          restartTime = float(line[3])
          oldValue = int(line[5])
          continue
        else:
          continue
    if restartTime == 0: #if previous .MES file doesn't contain info on last restart, therefore restart info searches in two times (or more) previous working directory
      try:
        restartTime = self.restartTime(workingDir[:-2])
      except:
        print("ERROR in time step. In order to restart MELCOR needs a restart dump written in .MES previous to the trip time step. Integrate the new correction in the MELCOR input")
    restartTime = float (restartTime) + 0.001
    outputToRead.close()
    return restartTime
			
  def ravenSample (self, inputFile , workingDir , variable):
    """
      @ In, inputFile, string, name of MELCOR input 
			@ In, workingDir, string, actual working Directory
			@ In, variable, string, sampled variables without prefix and suffix %
      @ Out, sampledVar, float, corresponding sampling of Raven
    """
    file = open(os.path.join(workingDir,inputFile),"r")
    sampledVar = None
    for index, line in enumerate(file, 1):
      line = line.rstrip()
      line = line.split()
      if line[0] == '*ravenSample' + variable:
        sampledVar = float(line[1])
        break
    if sampledVar == None:
      raise IOError("Issue with wrong input or wrong sampling")
    return sampledVar
		
  def dynamicEventTreeForMELCOR(self, **Kwargs):
    """
      This generates a new input file depending on which tripCF are found in messageReader
      @ In, Kwargs, dict, container of different infos
      @ Out, modifDict, dict, container 
    """
    import melcorCombinedInterface
    modifDict = {}
    deckList = {1:{}}
    workingDir = Kwargs['subDirectory'][:-2]
    modifDict['subDirectory'] = Kwargs['subDirectory']
    melcorInpFile = melcorCombinedInterface.MelcorApp.melcorInpFile
    if self.det:
      modifDict['happenedEvent'] = Kwargs['happenedEvent']
      modifDict['excludeTrips'] = []
      modifDict['DETvariables'] = self.detVars
      parentID = Kwargs.get("RAVEN_parentID", "none")
      if parentID.lower() != "none":
        # now we can copy the restart file
        sourcePath = Kwargs['subDirectory'][:-2]
        self.__copyRestartFile(sourcePath, Kwargs['currentPath'])
        # now we can check if the event happened and if so, remove the variable fro the det variable list
        if modifDict['happenedEvent']:
          for var in Kwargs['happenedEventVarHistory']:
            aliased = self._returnAliasedVariable(var, False)
            tripVariable = self.messageReader(workingDir)[0]
            modifDict['excludeTrips'] = var
    for keys in Kwargs['SampledVars']:
      tripVariable = self.messageReader(workingDir)[0]
      if tripVariable not in deckList:
        deckList[tripVariable] = {}
      if tripVariable not in deckList[tripVariable]:
        deckList[tripVariable] = [{'value':Kwargs['SampledVars'][keys]}]
      else:
        deckList[tripVariable].append({'value':Kwargs['SampledVars'][keys]})
    modifDict['decks']=deckList
    return modifDict
  
  def modifyInput(self, melcorInpFile, workingDir):
    """
      This generates a new input file depending on which tripCF are found in messageReader
      @ In, file, string, MELCOR input file
      @ In, workingDir, string, workingDir where MELCOR output message in generated
      @ Out, newInputForHappenedEvent, list, container of newInputFile to run with changed CF that tripped
      @ Out, newInputForNotHappenedEvent, list, container of newInputFile to run with changed CF that tripped
    """
    import melcorCombinedInterface
    isDet = melcorCombinedInterface.MelcorApp.melcorDetNode
    workingDir = workingDir[:-2]
    tripVariable = self.messageReader(workingDir)[0]
    file = open(os.path.join(workingDir,melcorInpFile),"r")
    newInputForHappenedEvent = []
    newInputForNotHappenedEvent = []
    CF_linechange = -1    #line of the value, of a CF that tripped that needs to be changed
    CF_number = 'notfound'
    CF_indexLine = -1     #line of the CF that tripped
    for index, line in enumerate(file, 1):
      line = line.rstrip()
      line = line.split()
      if line[0] == 'CF_ID':
        check = line[1][1:-1]
        if check == tripVariable:
          CF_indexLine = index
          newInputForHappenedEvent.append(line)
          newInputForNotHappenedEvent.append(line)
          continue
        else:
          newInputForHappenedEvent.append(line)
          newInputForNotHappenedEvent.append(line)
          continue
      if line[0] == 'CF_ARG' and index - CF_indexLine < 5: #controllare per essere generalizzato
        CF_linechange = index + int(line[1])
        CF_firstValue = index + 1
        newInputForHappenedEvent.append(line)
        newInputForNotHappenedEvent.append(line)
        continue
      if CF_linechange - index >= 0:
        if index == CF_firstValue: #qui cambiamo la variabile nel caso di event happened
          newLine = [line[0], line[1], 0.0] #forse da cambiare posizione di newData
          newInputForHappenedEvent.append(newLine)
          newInputForNotHappenedEvent.append(line)
        else: #qui cambiamo la variabile nel caso di event NOT happened
          newLine = [line[0],line[1],'$RAVEN-%' + str(tripVariable) + '%:-1$']																  
          newInputForHappenedEvent.append(line)
          newInputForNotHappenedEvent.append(newLine)
          CF_linechange = -1
      else:
        newInputForHappenedEvent.append(line)
        newInputForNotHappenedEvent.append(line)
    file.close()
    return newInputForHappenedEvent, newInputForNotHappenedEvent
  
  def writeNewInput(self, happenedorNot, melcorInpFile, workingDir):
    """
      This writes a new input file depending on what event is described
      @ In, happenedorNot, string, description of whether happenedEvent is True or False
      @ In, workingDir, string, workingDir where MELCOR output message in generated
      @ In, melcorInpFile, string, MELCOR input
      @ Out, newInput , file, modified MELCOR input
    """
    newInputForHappenedEvent = self.modifyInput(melcorInpFile, workingDir)[0]
    newInputForNotHappenedEvent = self.modifyInput(melcorInpFile, workingDir)[1]
    newInput = open(os.path.join(workingDir,melcorInpFile),"w")
    if happenedorNot == 'happenedEvent':
      for line in newInputForHappenedEvent:
        for words in line:
          newInput.write(str(words) + " ")
        newInput.write('\n')
    elif happenedorNot == 'nothappenedEvent':
      for line in newInputForNotHappenedEvent:
        for words in line:
          newInput.write(str(words) + " ")
        newInput.write('\n')
    else:
      raise IOError('ERROR in "MELCOR Code Interface": something is wrong with the writeNewInput')
    newInput.close()
    return newInput
    
  def messageReader(self, workingDir):  #OK   stoppedCF = {} and tripVariable = 'string'
    """
      This def. reades the MELCOR message output generated after a stop
      @ In, workingDir, string, workingDir where MELCOR output message in generated
      @ Out, stoppedCF , dictonary, container of all tripVariable wt corresponding triptime (stoppedCF[tripVariable][0]) and triptimestep (stoppedCF[tripVariable][1])
      @ Out, tripVariable , string, name of the trip variable
      @ Out, stoppedCF , Dict, container of key (string): tripVariable and value (list): [endTime , endTimeStep]
    """ 
    import melcorCombinedInterface
    isDet = melcorCombinedInterface.MelcorApp.melcorDetNode
    melcorOutFile = melcorCombinedInterface.MelcorApp.melcorOutFile
    outputToRead = open(os.path.join(workingDir,melcorOutFile),"r")
    CF_line = -1
    timeLine = 0
    stoppedCF = []
    for index, line in enumerate(outputToRead, 1):
      line = line.rstrip()
      line = line.split()
      if isDet == 'olderMelcor':
        if line[0] == '/SMESSAGE/':
          CF_line = index + 2
          stoppedCF.append(float(line[2])) #triptime
          stoppedCF.append(int(line[4])) #triptimestep
          continue
        if index == CF_line:
          try:
            if line[4] != 'CENTRAL':
              tripVariable = line[4]
              return tripVariable, stoppedCF
              break
          except:
            continue
        #if index == timeLine:
        #  try:
        #    stoppedCF.append(float(line[3])) #triptime
        #    stoppedCF.append(int(line[5])) #triptimestep
        #    return tripVariable, stoppedCF
        #    break
        #  except:
        #    stoppedCF.append(float(line[2])) #triptime
        #    stoppedCF.append(int(line[4])) #triptimestep
        #    return tripVariable, stoppedCF
        #    break
        else:
          continue
      else:
        if line[0] == 'MESSAGE':
          CF_line = index + 1
          continue
        if index == CF_line:
          try:
            if line[2] != 'CENTRAL':
              tripVariable = line[2]
              timeLine = index + 6
          except:
            continue
        if index == timeLine:
          stoppedCF.append(float(line[3])) #triptime
          stoppedCF.append(int(line[5])) #triptimestep
          return tripVariable, stoppedCF
          break
        else:
          continue
    outputToRead.close()

  def _writeBranchInfo(self, filename, endTime, endTimeStep, tripVariable):
 
    """
      Method to write the branchInfo
      @ In, filename, str, the file name
      @ In, endTime, float, the end time
      @ In, endTimeStep, float, the end time step
      @ In, tripVariable, str, the variable that caused the stop of the simulation (trip)
      @ Out, None
    """
    import dynamicEventTreeUtilities as detUtils
    tripVar = "%" + str(tripVariable) + "%"
    detUtils.writeXmlForDET(filename,tripVar,[],{'end_time': endTime, 'end_ts': endTimeStep})
  
  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      In this method the MELCOR outputfile is parsed and a CSV is created
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, response, dict, the data dictionary {var1:array,var2:array, etc}
    """
    outfile = os.path.join(workingDir,output+'.out')
    #outputobj=MELCORdata.MELCORdata(output)
    # to remove the following
    outputobj=MELCORdata.MELCORdata(os.path.join(workingDir,output+'.o'))
    response = outputobj.writeCsv(os.path.join(workingDir,output+'.csv'),workingDir)
    if self.det:
      stopDET = self.stopDET ( workingDir )
      if stopDET == True:
        print("RUN ended successfully!!!!!")
      else:
        tripVariable = self.messageReader(workingDir)[0]
        endTime = self.messageReader(workingDir)[1][0]
        endTimeStep = self.messageReader(workingDir)[1][1]
        filename = os.path.join(workingDir,output+"_actual_branch_info.xml")
        self._writeBranchInfo(filename, endTime, endTimeStep, tripVariable)
    return response

  def __copyRestartFile(self, sourcePath, currentPath, restartFileName = None):
    """
      Copy restart file
      @ In, sourcePath, str, the source path where the restart is at
      @ In, currentPath, str, the current location (where the restart will be copied)
      @ In, restartFileName, str, optional, the restart file name if present (otherwise try to find one to copy in sourcePath)
      @ Out, None
    """
    # search for restrt file
    import melcorCombinedInterface
    rstrtFile = melcorCombinedInterface.MelcorApp.melcorRstFile
    plotFile = melcorCombinedInterface.MelcorApp.MelcorPlotFile
    edfFile = []
    for fileToCheck in os.listdir(sourcePath):
      if fileToCheck.strip().endswith(".txt"):
        edfFile.append(fileToCheck)
    if rstrtFile is None:
      raise IOError("no restart file has been found!" + rstFileName + " not found!")
    sourceFile = os.path.join(sourcePath, rstrtFile)
    sourceFilePlot = os.path.join(sourcePath, plotFile)
    sourceFileEdf = []
    for files in edfFile:
      sourceFileEdf.append(os.path.join(sourcePath, files))
    try:
      shutil.copy(sourceFile, currentPath)
      shutil.copy(sourceFilePlot, currentPath)
      for files in sourceFileEdf:
        shutil.copy(files, currentPath)
    except:
      raise IOError('not able to copy restart file from "'+sourceFile+'" to "'+currentPath+'"')

  def __transferMetadata(self, metadataToTransfer, currentPath):
    """
      Method to tranfer metadata if present
      @ In, metadataToTransfer, dict, the metadata to transfer
      @ In currentPath, str, the current working path
      @ Out, None
    """
    if metadataToTransfer is not None:
      sourceID = metadataToTransfer.get("sourceID",None)
      if sourceID is not None:
        # search for restrt file
        sourcePath = os.path.join(currentPath,"../",sourceID)
        self.__copyRestartFile(sourcePath, currentPath)
      else:
        raise IOError('the only metadtaToTransfer that is available in MELCOR is "sourceID". Got instad: '+', '.join(metadataToTransfer.keys()))

  def _findInputFileIndex(self, currentInputFiles):
    """
      Find input file index
      @ In, currentInputFiles, list, list of current input files to search from
      @ Out, index, int, the index of the relap input
    """
    found = False
    for index, inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    return index
    
  def _returnAliasedVariable(self, var, fromCodeToRaven = True):
    """
      Return the alias for variable in
      @ In, var, str, the variable the alias should return for
      @ Out, aliasVar, str, the aliased variable if found
    """
    aliasVar = var
    if len(self.inputAliases):
      for ravenVar, codeVar in self.inputAliases.items():
        if fromCodeToRaven:
          if codeVar.strip().startswith(var.strip()):
            aliasVar = ravenVar
            break
        else:
          if ravenVar.strip().startswith(var.strip()):
            aliasVar = codeVar
            break
    return aliasVar
