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
import re
import xml.etree.ElementTree as ET
"""
Created on May 5, 2016

@author: alfoa
"""

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
      if re.match('^\s*Final time=',line) or re.match('^\s*0Final time=',line):
        deckNum+=1
        startLineNumber = endLineNumber
        endLineNumber   = cnt+1
        times[deckNum] = {'time':line.split()[2],'sliceCoordinates':(startLineNumber,endLineNumber)}
    if deckNum < deckNumber: raise IOError("the deck number requested is greater than the number found in the outputfiles! Found "+ str(deckNum) + " decks and requested are "+str(deckNumber))
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
      if re.match('^\s*0Trip\s*number',lines[i]):
        tripArray=[]
        i=i+1
        while not (re.match('^0System|^0\s*Total',lines[i])):
          temp1 = lines[i].split();
          for j in range(len(temp1)/2):
            if (float(temp1[2*j+1])>-1.000):
              tripArray.append({temp1[2*j]:temp1[2*j+1]});
          i=i+1;
    return tripArray;

  def readMinorBlock(self,lines,i):
    """
      Method that reads in a block of minor edit data and returns a dictionary of lists
      @ In, lines, list, list of lines of the output file
      @ In, i, int, line number where to start the reading
      @ Out, minorDict, dict, dictionary containing the minor edit info
    """
    minorDict={}
    edit_keys=[]
    flagg1 = 0
    flagg2 = 0
    block_count=0

    # The following object is a list of keywords that RELAP5 might generate in the minor edits which would
    # corrupt the .csv files. If more keywords are discovered add them here in the list
    errorKeywords = ['Reducing','Thermodynamic','ncount','0$$$$$$$$','written','block']

    while(flagg1==0 & flagg2==0):
      if flagg1==0:
        tempkeys=[]
        temp1 = re.split('\s{2,}|\n',lines[i])
        #temp2 = re.split('\s{2,}|\n',lines[i+1])
        temp2 = [lines[i+1][j:j+13].strip() for j in range(0, len(lines[i+1]), 13)]
        temp1.pop()
        temp2.pop()
        temp2 = ['_'.join(key.split()) for key in temp2]
        #temp2.pop(0)
        tempArray=[]
        for j in range(len(temp1)):
          tempkeys.append(temp1[j]+'_'+temp2[j])
          edit_keys.append(temp1[j]+'_'+temp2[j])
          tempArray.append([]);     #   allocates array for data block
        i=i+4
        while not re.match('^\s*1 time|^1RELAP5|^\s*\n|^\s*1RELAP5|^\s*MINOR EDIT',lines[i]):
          tempData=lines[i].split()
          #takeIt = False if re.match("^\d+?\.\d+?$", tempData[0]) is None else True
          #if takeIt:
          #  for k in range(len(tempArray)): tempArray[k].append(tempData[k])
          # Here I check that none of the keywords contained in errorKeywords are contained in tempData
          if (not list(set(tempData) & set(errorKeywords))) and (len(tempArray)==len(tempData)):
            for k in range(len(tempArray)): tempArray[k].append(tempData[k])
          i=i+1
          if re.match('^\s*1 time|^\s*1\s*R5|^\s*\n|^1RELAP5',lines[i]) or re.match('^\s*0Final time',lines[i]) or re.match('^\s*Final time',lines[i]): break
        for l in range(len(tempkeys)): minorDict.update({tempkeys[l]:tempArray[l]})
        if re.match('^\s*1\s*R5|^\s*\n|^\s*1RELAP5|^\s*MINOR EDIT',lines[i]): #or i+1 > len(lines) -1:
          flagg2=1
          flagg1=1
        elif re.match('^\s*1 time',lines[i]):
          block_count=block_count+1
          flagg=1
          flagg1=1
          flagg2=1
        elif re.match('^\s*0Final time',lines[i]):
          flagg=1
          flagg1=1
          flagg2=1
    return minorDict

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
    for i in range(len(lines)):
      if re.match('^1 time',lines[i]):
        count=count+1
        tempdict=self.readMinorBlock(lines,i)
        timeBlock = tempdict.pop('1 time_(sec)')
        if (count==1):
          minorDict=tempdict
          timeList.append(timeBlock)
        else:
          if set(timeBlock) != set(timeList[-1]):
            timeList.append(timeBlock)
          for k in tempdict.keys():
            if k in minorDict.keys():
              minorDict[k].extend(tempdict.get(k))
            else:
              minorDict[k] =  tempdict[k]
#             for k in minorDict.keys():
#               if k in tempdict.keys():
#                 minorDict[k].extend(tempdict.get(k))
    timeBlock = []
    for tBlock in timeList: timeBlock.extend(tBlock)
    minorDict['1 time_(sec)'] = timeBlock
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
          if re.search('RAVEN',self.lines[i]): flagg=1
          else:
            splitted = self.lines[i].split()
            if   'deckNum:' in splitted: deckNum = splitted[-1].strip()
            elif 'card:'    in splitted:
              sampleVar = splitted[splitted.index('card:')+1].strip()+(":"+splitted[splitted.index('word:')+1].strip() if splitted[splitted.index('word:')+1].strip() != '0' else '')
              value     = splitted[splitted.index('value:')+1].strip()
              if deckNum is not None: sampleVar = str(deckNum)+'|'+sampleVar
              self.ravenData[sampleVar]=value
          i=i+1
        deckCounter+=1

    return

  def writeCSV(self,filen):
    """
      Method that writes the csv file from minor edit data
      @ In, filen, string, input file name
      @ Out, None
    """
    IOcsvfile=open(filen,'w')
    if self.minordata != None:
      for i in range(len(self.minordata.keys())): IOcsvfile.write('%s,' %(self.minordata.keys()[i].strip().replace("1 time_(sec)","time").replace(' ', '_')))
    for j in range(len(self.ravenData.keys())):
      IOcsvfile.write('%s' %(self.ravenData.keys()[j]))
      if j+1<len(self.ravenData.keys()): IOcsvfile.write(',')
    IOcsvfile.write('\n')
    for i in range(len(self.minordata.get(self.minordata.keys()[0]))):
      for j in range(len(self.minordata.keys())): IOcsvfile.write('%s,' %(self.minordata.get(self.minordata.keys()[j])[i]))
      for k in range(len(self.ravenData.keys())):
        IOcsvfile.write('%s' %(self.ravenData[self.ravenData.keys()[k]]))
        if k+1<len(self.ravenData.keys()): IOcsvfile.write(',')
      IOcsvfile.write('\n')
