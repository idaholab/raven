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
Created on March 8th 2018  
@author: rouxpn 
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
import os
import copy
import shutil
import re
import sys
import csv

class combine():
  """
    class that combines the phisics and relap csv output
  """ 
  def __init__(self,workingDir,relapCSV, phisicsCSV,depTimeDict,inpTimeDict):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    paramDict = {}
    self.workingDir = workingDir
    self.phisicsCSV = phisicsCSV
    self.relapCSV = relapCSV
    relapTimePosition = self.findTimePosition()
    timeList = self.selectPhisicsTime(depTimeDict)
    phiDict = self.putCSVinDict(self.phisicsCSV)
    relDict = self.putCSVinDict(self.relapCSV)
    numOfRelapLines = self.getNumOfLines(self.relapCSV)
    numOfPhisicsLines = self.getNumOfLines(self.phisicsCSV)
    
    paramDict['numOfRelapLines'] = numOfRelapLines
    paramDict['numOfPhisicsLines'] = numOfPhisicsLines
    paramDict['timeList'] = timeList
    paramDict['phiDict'] = phiDict
    paramDict['relDict'] = relDict
    paramDict['relapTimePosition'] = relapTimePosition
    paramDict['depTimeDict'] = depTimeDict
    paramDict['inpTimeDict'] = inpTimeDict
    self.joinCSV(paramDict)
  
  def findTimePosition(self):
    """
      find the relap time position in the CSV file
      @ In, None 
      @ Out, timePosition, integer
    """
    count = 0 
    with open(self.relapCSV, 'r') as relfile:
      for line in relfile:
        count = count + 1 
        for i in xrange(0,len(line.split(','))):
          if line.strip('\n').split(',')[i] == 'time':
            return i 
        if count >= 1: 
          raise ValueError("\n the keyword time is not listed in the relap CSV output \n")
    
  def selectPhisicsTime(self,depTimeDict):
    """
      selects the time (<=> line) that will be printed in the final raven output. Those times match with the relap one
      @ In, depTimeDict
      @ Out, None 
    """
    timeList = []
    self.timeStepSelected = []
    lineSelected = 0 
    for i in depTimeDict['timeSteps'].split(' '):
      lineSelected = lineSelected + int(i)
      self.timeStepSelected.append(lineSelected)
      lineNumber = 0 
      with open(self.phisicsCSV, 'r') as phifile:
        for line in phifile:
          if lineNumber == lineSelected:
            timeList.append(line.split(',')[0])
          lineNumber = lineNumber + 1
    return timeList
  
  def putCSVinDict(self,csvFile):
    """
      places each line of the csv into a dictionary. key: line number (integer), value: line (list)
      @ In, csvFile, string, file name of the csv file parsed
      @ Out, csvDict, dictionary 
    """
    countLine = 0
    csvDict = {}
    with open(csvFile, 'r') as file:
      for line in file:
        if countLine == 0 and csvFile == self.phisicsCSV:
          self.numOfParameters = self.getNumOfParameters(line)
        countLine = countLine + 1 
        csvDict[countLine] = line
    return csvDict
  
  def getNumOfParameters(self,line):
    """
      return the number of parameters that are on one line of the csv file
      @ In, line, string, csv file first line  
      @ Out, numOfParameters, integer
    """
    return len(line.split(','))
  
  def getNumOfLines(self,csvFile):
    """
      counts the number of lines in the phisics or relap csv. 
    """
    count = 0 
    with open(csvFile, 'r') as file:
      for line in file:
        count = count + 1
    return count 
  
  def joinLine(self,phisicsList,relapList):
    """
      joins the phisics and relap line into one list and removes the \n nd \raise
      @ In, phisicsList, list, list contianing a csv phisics line 
      @ In, relapList, list, list containing a csv relap line 
      @ Out, joinedList, list, joined list of parameters
    """
    joinedList = []
    joinedList.append(phisicsList.strip('\n').strip('\r'))
    joinedList.append(relapList.strip('\n').strip('\r'))
    return joinedList
  
  def joinCSV(self,paramDict):
    """
      join the relap csv and phisics csv based on the time lines selected from phisics
      @ In, paramDict, dictionary, dictionary of parameters
      @ Out, None 
    """
    with open(os.path.join(self.workingDir,'test.csv'), 'wb') as f:
      instantWriter = csv.writer(f, delimiter=str(u',').encode('utf-8'),quotechar=str(u' ').encode('utf-8'), quoting=csv.QUOTE_MINIMAL)
      instantWriter.writerow(self.joinLine(paramDict['phiDict'][1],paramDict['relDict'][1]))
      instantWriter.writerow([0.0] * self.numOfParameters + [paramDict['relDict'][2]]) 
      lineNumber,THbetweenBurn, mrTau = 2,0,0 
      while THbetweenBurn < len(paramDict['inpTimeDict']['TH_between_BURN'].split(' ')):
        lineNumber = lineNumber + 1
        if float(paramDict['relDict'][lineNumber].split(',')[paramDict['relapTimePosition']]) <= float(paramDict['inpTimeDict']['TH_between_BURN'].split(' ')[THbetweenBurn]):
          instantWriter.writerow(self.joinLine(paramDict['phiDict'][self.timeStepSelected[mrTau] + 1],paramDict['relDict'][lineNumber]))
        if paramDict['relDict'][lineNumber].split(',')[paramDict['relapTimePosition']] > paramDict['inpTimeDict']['TH_between_BURN'].split(' ')[THbetweenBurn]:
          THbetweenBurn = THbetweenBurn + 1 
          mrTau = mrTau + 1 
          if THbetweenBurn == len(paramDict['inpTimeDict']['TH_between_BURN'].split(' ')) :
            instantWriter.writerow(self.joinLine(paramDict['phiDict'][paramDict['numOfPhisicsLines']],paramDict['relDict'][paramDict['numOfRelapLines']]))
    with open(os.path.join(self.workingDir,'test.csv'), 'r') as infile:
      with open(os.path.join(self.workingDir,'final.csv'), 'wb') as outfile:
        for line in infile:
          cleanedLine = line.strip(' ')
          if re.match(r'^\s*$', line): pass 
          else: outfile.write(cleanedLine)
