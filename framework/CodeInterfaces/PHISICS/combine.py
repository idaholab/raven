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
import re
import csv

class combine():
  """
    Combines the PHISICS and RELAP csv output into one.
  """
  def __init__(self,workingDir,relapCSV,phisicsCSV,depTimeDict,inpTimeDict,relapPhisicsCsv):
    """
      Constructor.
      @ In, workingDir, string, absolute path to working directory
      @ In, relapCSV, string, csv file name generated after the execution of relapdata.py
      @ In, phisicsCSV, string, csv file name generated after the execution of phisicsdata.py
      @ In, depTimeDict, dictionary, information from the xml depletion file
      @ In, inpTimeDict, dictionary, information from the xml input file
      @ In, relapPhisicsCsv, string, filename of the combines PHISICS and RELAP5 files
      @ Out, None
    """
    paramDict = {}
    paramDict['relapTimePosition'] = self.findTimePosition(relapCSV)
    paramDict['timeList'] = self.selectPhisicsTime(depTimeDict,phisicsCSV)
    paramDict['phiDict'] = phiDict = self.putCSVinDict(phisicsCSV,phisicsCSV)
    paramDict['relDict'] = self.putCSVinDict(relapCSV,phisicsCSV)
    paramDict['numOfRelapLines'] = len(paramDict['relDict'])
    paramDict['numOfPhisicsLines'] = len(paramDict['phiDict'])
    paramDict['depTimeDict'] = depTimeDict
    paramDict['inpTimeDict'] = inpTimeDict
    paramDict['relapPhisicsCsv'] = relapPhisicsCsv
    self.joinCSV(paramDict,workingDir)

  def findTimePosition(self,relapCSV):
    """
      Finds the RELAP time position in the csv file.
      @ In, relapCSV, string, csv file name generated after the execution of relapdata.py
      @ Out, timePosition, integer
    """
    with open(relapCSV, 'r') as relfile:
      for line in relfile:
        try:
          timePosition = next(i for i,x in enumerate(line.split(',')) if re.match(r'time',x.strip()))
          return timePosition
        except StopIteration:
          raise ValueError("\n the keyword -time- is not listed in the RELAP csv output \n")

  def selectPhisicsTime(self,depTimeDict,phisicsCSV):
    """
      Selects the time (<=> line) that will be printed in the final RAVEN output. Those times match with the RELAP one.
      @ In, depTimeDict, dictionary, information from the xml depletion file, key: time parameter, value: xml node text
      @ In, phisicsCSV, string, csv file name generated after the execution of phisicsdata.py
      @ Out, timeList, list, list of parameter values at a given line number of the phisics csv file
    """
    timeList = []
    self.timeStepSelected = []
    lineSelected = 0
    for i in depTimeDict['timeSteps'].split(' '):
      lineSelected = lineSelected + int(i)
      self.timeStepSelected.append(lineSelected)
      with open(phisicsCSV, 'r') as phifile:
        for lineNumber,line in enumerate(phifile,1):
          if lineNumber == lineSelected:
            timeList.append(line.split(',')[0])
    return timeList

  def putCSVinDict(self,csvFile,phisicsCSV):
    """
      Places each line of the csv into a dictionary. key: line number (integer), value: line (list).
      @ In, csvFile, string, file name of the csv file parsed
      @ In, phisicsCSV, string, csv file name generated after the execution of phisicsdata.py
      @ Out, csvDict, dictionary, key: line number (integer), value, line containing the parameter values (string)
    """
    csvDict = {}
    with open(csvFile, 'r') as inFile:
      countLine = 0
      for line in inFile:
        if countLine == 0 and csvFile == phisicsCSV:
          self.numOfParameters = len(line.split(','))
        if re.match(r'^\s*$', line):
          pass
        else:
          csvDict[countLine] = line
          countLine += 1
    return csvDict

  def joinLine(self,phisicsList,relapList):
    """
      Joins the PHISICS and RELAP line into one list.
      @ In, phisicsList, string, contians a csv PHISICS line
      @ In, relapList, string, contains a csv RELAP line
      @ Out, joinedList, list, joined list of parameters
    """
    return (phisicsList.rstrip(),relapList.rstrip())

  def joinCSV(self,paramDict,workingDir):
    """
      Joins the RELAP csv and PHISICS csv based on the time lines selected from PHISICS.
      @ In, paramDict, dictionary, dictionary of parameters
      @ In, workingDir, string, absolute path to working directory
      @ Out, None
    """
    cleanUpFiles = ['dummy.csv','relapPhisics.csv']
    for cleanUpFile in cleanUpFiles:
      if os.path.exists(cleanUpFile):
        os.remove(cleanUpFiles) # remove the file if was already existing
    thBurnStep = paramDict['inpTimeDict']['TH_between_BURN'].split(' ')
    with open(os.path.join(workingDir,'dummy.csv'), 'w') as f:
      instantWriter = csv.writer(f, delimiter=str(','),quotechar=str(' '),
                                 quoting=csv.QUOTE_MINIMAL)
      instantWriter.writerow(self.joinLine(paramDict['phiDict'][0],paramDict['relDict'][0]))
      instantWriter.writerow([0.0] * self.numOfParameters + [paramDict['relDict'][1]])
      lineNumber = 1
      THbetweenBurn = 0
      mrTau = 0
      while THbetweenBurn < len(thBurnStep):
        lineNumber = lineNumber + 1
        # if the time on a relap line is <= than the TH_between_burn selected
        if float(paramDict['relDict'][lineNumber].split(',')[paramDict['relapTimePosition']]) <= float(thBurnStep[THbetweenBurn]):
          # print the relap line with the phisics line corresponding to last time step of a burnstep
          instantWriter.writerow(self.joinLine(paramDict['phiDict'][self.timeStepSelected[mrTau]],paramDict['relDict'][lineNumber]))
        # if the relap time on a line is larger the TH_between_burn selected
        if paramDict['relDict'][lineNumber].split(',')[paramDict['relapTimePosition']] > thBurnStep[THbetweenBurn]:
          # change the TH_between_burn selected
          THbetweenBurn = THbetweenBurn + 1
          # change the burn step in phisics
          mrTau = mrTau + 1
          # if this is the last TH_between_burn
          if THbetweenBurn == len(thBurnStep):
            # print the last line of phisics and relap.
            instantWriter.writerow(self.joinLine(paramDict['phiDict'][paramDict['numOfPhisicsLines'] - 1],
                                                 paramDict['relDict'][paramDict['numOfRelapLines'] - 1]))
    with open(os.path.join(workingDir,'dummy.csv'), 'r') as inFile:
      with open(os.path.join(workingDir,paramDict['relapPhisicsCsv']), 'w') as outFile:
        for line in inFile:
          cleanedLine = line.strip(' ')
          if re.match(r'^\s*$', line):
            pass
          else:
            outFile.write(cleanedLine)
    if os.path.exists('dummy.csv'):
      os.remove('dummy.csv')

