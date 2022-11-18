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
Created on May 5, 2016

@author: alfoa
"""
import re
import xml.etree.ElementTree as ET

class relapdata:
  """
    Class that parses output of relap5 output file and reads in trip, minor block and write a csv file
  """
  def __init__(self,filen, deckNumber=-1):
    """
      Constructor
      @ In, filen, string, file name to be parsed
      @ In, deckNumber, int, optional, the deckNumber from which the outputs need to be retrieved (default is the last)
      @ Out, None
    """
    # self.totNumberOfDecks is set in getTimeDeck method!
    self.lines           = open(filen,"r").readlines()
    self.deckEndTimeInfo = self.getTimeDeck(self.lines,deckNumber)
    self.deckNumberToTake= deckNumber if deckNumber != -1 else self.totNumberOfDecks
    startLine, endLine   = self.deckEndTimeInfo[self.deckNumberToTake]['sliceCoordinates'][0:2]
    self.trips           = self.returnTrip(self.lines[startLine:endLine])
    self.minordata       = self.getMinor(self.lines[startLine:endLine])
    self.readRaven()

  def hasAtLeastMinorData(self):
    """
      Method to check if at least the minor edits are present
      @ In, None
      @ Out, hasMinor, bool, True if it has minor data
    """
    hasMinor = self.minordata != None
    return hasMinor

  def getTime(self,lines):
    """
      Method to check ended time of the simulation
      @ In, lines, list, list of lines of the output file
      @ Out, time, float, Final time
    """
    return self.getTimeDeck(lines).values()[-1]['time']

  def getTimeDeck(self,lines, deckNumber=-1):
    """
      Method to check ended time of the simulation (multi-deck compatible)
      @ In, lines, list, list of lines of the output file
      @ In, deckNumber, int, optional, the deckNumber from which the outputs need to be retrieved (default is the last)
      @ Out, times, dict, dict containing the information {'deckNumber':{'time':float,'sliceCoordinates':tuple(startLine,EndLine)}}. Dict of final times and corresponding deck start and end line number
    """
    times = {}
    deckNum, startLineNumber, endLineNumber = 0, 0, 0
    for cnt, line in enumerate(lines):
      if re.match('^\\s*Final time=',line) or re.match('^\\s*0Final time=',line):
        deckNum+=1
        startLineNumber = endLineNumber
        endLineNumber   = cnt+1
        times[deckNum] = {'time':line.split()[2],'sliceCoordinates':(startLineNumber,endLineNumber)}
    if deckNum < deckNumber:
      raise IOError("the deck number requested is greater than the number found in the outputfiles! Found "+ str(deckNum) + " decks and requested are "+str(deckNumber))
    self.totNumberOfDecks = deckNum
    return times

  def returnTrip(self,lines):
    """
      Method to return the trip information
      @ In, lines, list, list of lines of the output file
      @ Out, tripArray, list, list of dictionaries containing the trip info
    """
    tripArray=[]
    for i in range(len(lines)):
      if re.match('^\\s*0Trip\\s*number',lines[i]):
        tripArray=[]
        i=i+1
        while not (re.match('^0System|^0\\s*Total',lines[i])):
          temp1 = lines[i].split();
          for j in range(len(temp1)//2):
            if (float(temp1[2*j+1])>-1.000):
              tripArray.append({temp1[2*j]:temp1[2*j+1]});
          i=i+1;
    return tripArray;

  def readMinorBlock(self, lines, i):
    """
      Method that reads in a block of minor edit data and returns a dictionary of lists
      @ In, lines, list, list of lines of the output file
      @ In, i, int, line number where to start the reading
      @ Out, minorDict, dict, dictionary containing the minor edit info
    """
    minorDict={}
    keepReading = True

    while(keepReading):
      headerKeys=[]
      varNames = re.split('\\s{2,}|\n',lines[i])[:-1]
      componetName = [lines[i+1][j:j+13].strip() for j in range(0, len(lines[i+1]), 13)][:-1]
      componetName = ['_'.join(key.split()) for key in componetName]
      dataArray=[]
      for j in range(len(varNames)):
        headerKeys.append(varNames[j]+'_'+componetName[j])
        dataArray.append([]);                  # allocates array for data block
      i=i+4
      while not re.match('^\\s*1 time|^1RELAP5|^\\s*\n|^\\s*1RELAP5|^\\s*MINOR EDIT',lines[i]):
        tempData=lines[i].split()
        # Here I check that none of the keywords contained in errorKeywords are contained in tempData
        if self.checkLine(tempData) and (len(dataArray)==len(tempData)):
          for k in range(len(dataArray)):
            dataArray[k].append(float(tempData[k]))
        i=i+1
        if (re.match('^\\s*1 time|^\\s*1\\s*R5|^\\s*\n|^1RELAP5',lines[i]) or
            re.match('^\\s*0Final time',lines[i]) or
            re.match('^\\s*Final time',lines[i]) or
            not self.checkLine(tempData)):
            # if we reached the END OF THE MINOR EDIT BLOCK
            # OR if the line is invalid (checkLine is false), it means that RELAP printed
            # out a warning/error at the end of the minor edit block => we can skip it
          break
      for l in range(len(headerKeys)):
        minorDict.update({headerKeys[l]:dataArray[l]})
      if (re.match('^\\s*1\\s*R5|^\\s*\n|^\\s*1RELAP5|^\\s*MINOR EDIT',lines[i]) or
          re.match('^\\s*1 time',lines[i]) or
          re.match('^\\s*0Final time',lines[i]) or
          not self.checkLine(tempData)):
          # if we reached the END OF THE MINOR EDIT BLOCK
          # OR if the line is invalid (checkLine is false), it means that RELAP printed
          # out a warning/error at the end of the minor edit block => we can skip it
        keepReading = False
    return minorDict

  def checkLine(self, lineList):
    """
      Method that checks the content of a list (i.e., a line); a list must contain only numbers
      @ In, list, lineList, list that contained values located in a single line
      @ Out, outcome, bool, boolean variable which is True if the list contains only numbers, False if the contains at list a string
    """
    outcome = True
    for element in lineList:
      try:
        float(element)
      except ValueError:
        outcome = outcome and False
    return outcome

  def getMinor(self,lines):
    """
      Method that looks for key word MINOR EDIT for reading minor edit block
      and calls readminor block to read in the block of minor edit data
      @ In, lines, list, list of lines of the output file
      @ Out, minorDict, dict, dictionary containing the minor edit info
    """
    count  = 0
    minorDict = None
    timeList = []
    totTs = 0
    for i in range(len(lines)):
      if re.match('^1 time',lines[i]):
        count=count+1
        minorBlock=self.readMinorBlock(lines,i)
        timeBlock = minorBlock.pop('1 time_(sec)')
        if (count==1):
          minorDict=minorBlock
          totTs = len(timeBlock)
          timeList.append(timeBlock)
        else:
          if set(timeBlock) != set(timeList[-1]):
            timeList.append(timeBlock)
            totTs += len(timeBlock)
          for k in minorBlock.keys():
            if k in minorDict.keys():
              values = minorBlock.get(k)
              totLen = len(minorDict[k]) + len(values)
              if totLen == totTs:
                minorDict[k].extend(minorBlock.get(k))
              else:
                print('RELAP5 Interface: WARNING: Redundant variable "{}". Keep it only one!'.format(k))
            else:
              minorDict[k] =  minorBlock[k]
    timeBlock = []
    for tBlock in timeList:
      timeBlock.extend(tBlock)
    minorDict['time'] = timeBlock
    return minorDict

  def readRaven(self):
    """
      Method that looks for the RAVEN keyword where the sampled vars are stored
      @ In, None
      @ Out, None
    """
    flagg=0
    self.ravenData={}
    ravenLines = []
    deckCounter = 0
    for i in range(len(self.lines)):
      if re.search('RAVEN',self.lines[i]):
        ravenLines.append([])
        deckNum = None
        i=i+1
        while flagg==0:
          if re.search('RAVEN',self.lines[i]):
            flagg=1
          else:
            splitted = self.lines[i].split()
            if   'deckNum:' in splitted:
              deckNum = splitted[-1].strip()
            elif 'card:'    in splitted:
              sampleVar = splitted[splitted.index('card:')+1].strip()+(":"+splitted[splitted.index('word:')+1].strip() if splitted[splitted.index('word:')+1].strip() != '0' else '')
              value     = splitted[splitted.index('value:')+1].strip()
              if deckNum is not None:
                sampleVar = str(deckNum)+'|'+sampleVar
              try:
                self.ravenData[sampleVar]=float(value)
              except ValueError:
                self.ravenData[sampleVar]=value
          i=i+1
        deckCounter+=1

    return

  def returnData(self):
    """
      Method to return the data in a dictionary
      @ In, None
      @ Out, data, dict, the dictionary containing the data {var1:array,var2:array,etc}
    """
    data = self.minordata
    return data
