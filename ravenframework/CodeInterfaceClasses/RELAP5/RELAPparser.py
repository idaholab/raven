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
Created on July 11, 2013
@author: nieljw
@modified: alfoa
"""
import os
import fileinput
import re
import math
from collections import defaultdict
import copy

def _splitRecordAndRemoveComments(line, delimiter=None):
  """
    Method to split a record (relap5 input line) removing the comments if present
    @ In, line, str, the line to be splitted
    @ In, delimiter, str, optional, the delimiter to split by (default None -> white spaces)
    @ Out, splitted, list, the list containing the splitted line (without comments)
  """
  splitted = []
  for component in line.split(delimiter):
    stripped = component.strip()
    if stripped.startswith("*"):
      break
    splitted.append(stripped)
  return splitted

class RELAPparser():
  """
    Import the RELAP5 input as list of lines, provide methods to add/change entries and print it back
  """
  def __init__(self,inputFile,addMinorEdits=False):
    """
      Constructor
      @ In, inputFile, string, input file name
      @ In, addMinorEdits, bool, flag to add Minor Edits in case there are trips and the variables in the trip is not among the minor edits (generally for DET)
      @ Out, None
    """
    self.printTag = 'RELAP5 PARSER'
    if not os.path.exists(inputFile):
      raise IOError(self.printTag+'ERROR: not found RELAP input file')
    lines                 = open(inputFile,'r').readlines()
    self.inputfile        = inputFile
    self.inputTrips       = {}    #  {deckNumber:{'variableTrips':{},'logicalTrips':{}}}
    self.inputMinorEdits  = {}
    self.inputControlVars = {}
    self.hasLogicalTrips  = False
    self.hasVariableTrips = False
    self.inputStopTrips   = None
    self.lastTripLine     = {}
    self.lastCntrLine     = {}
    self.lastMinorEditLine= {}
    self.controlVarType   = {}    # {deckNum:1 CCC format or 2 CCCC format}
    self.deckLines        = {}
    self.maxNumberOfDecks = 0
    self.presentStopTrip = {}
    # stop trip nomenclature (either 600 or 0000600 => see below)
    self.stopTripNomenclature = '600'
    prevDeckLineNum       = 0
    self.addMinorEdits    = addMinorEdits
    pStop = None
    for lineNum, line in enumerate(lines):
      if line.strip().startswith("."):
        self.maxNumberOfDecks += 1
        self.deckLines[self.maxNumberOfDecks] = lines[prevDeckLineNum:lineNum]
        self.presentStopTrip[self.maxNumberOfDecks] = copy.deepcopy(pStop)
        pStop = None
        prevDeckLineNum = lineNum + 1
      elif line.strip().startswith("600 ") or line.strip().startswith("0000600"):
        pStop = [el.strip() for el in line.strip().split()[1:]]
        self.stopTripNomenclature = '600' if line.strip().startswith("600 ") else '0000600'
        lines[lineNum] = "*"

    if self.maxNumberOfDecks < 1:
      raise IOError(self.printTag+ "ERROR: the file "+inputFile+" does not contain a end case fullstop '.'!")

    if addMinorEdits:
      # add Minor Edits in case there are trips and the variables in the trip is not among the minor edits
      self.getTripsMinorEditsAndControlVars()

  def addControlVariablesForStoppingCoditions(self, monitoredTrips, excludeFromStop=None):
    """
      Method to add the control variables to make any trip in the input list to cause a stop of the code
      @ In, monitoredTrips, dict, dictionary of list of monitored trips {deckNum:[trips]}
      @ In, excludeFromStop, dict, optional, dictionary of list of monitored trips {deckNum:[trips]} that should be excluded from the stop condition
      @ Out, None
    """
    exclTrips = excludeFromStop if excludeFromStop is not None else dict.fromkeys(monitoredTrips.keys(), [])
    self.additionalControlVariables = {}
    for deckNum in self.inputControlVars.keys():
      originalStopping = [] if self.presentStopTrip[deckNum] is None else self.presentStopTrip[deckNum]
      if len(originalStopping) > 1:
        raise IOError("One slot for the stopping trip '{}' must be left out for the DET! Number of trips specified for card {} is: {}".format(self.stopTripNomenclature,self.stopTripNomenclature,str(len(originalStopping))))
      self.additionalControlVariables[deckNum] = {}
      (rangeLow, rangeUp, fill) = (1,999,3) if self.controlVarType[deckNum] == 1 else (1,9999,4)
      #if self.controlVarType[deckNum] == 1: rangeLow, rangeUp, fill = 1,999,3
      #else                                : rangeLow, rangeUp, fill = 1,9999,4
      availableControlVars  = [str(y).zfill(fill) for y in range(rangeLow,rangeUp)
                               if y not in [int(x) for x in self.inputControlVars[deckNum].keys()]]
      if len(availableControlVars) < len(monitoredTrips)+1:
        raise IOError("Not enough control variables' slots are available. We need at least "+str(len(monitoredTrips)+1)+" free slots!")

      presentTrips, cnt = self.inputTrips[deckNum]['variableTrips'].keys(), 501
      while any([str(cnt) in s for s in presentTrips]) and cnt < 600:
        cnt+=1
      if cnt == 600:
        raise IOError("All the Trip slots are used! We need at least " +str(len(monitoredTrips)+1)+" slots to specify the stop conditions!!!!")
      else:
        stopTripNumber, stopCntrVar = str(max(cnt,599)), availableControlVars.pop()
      self.deckLines[deckNum].append("* START -- CONTROL VARIABLES ADDED BY RAVEN *\n")
      # should we exclude all?
      excludeAll = len(exclTrips[deckNum]) == len(monitoredTrips[deckNum])
      if excludeAll:
        # 414  time  0  ge  null  0  1.00000e+01  l
        self.deckLines[deckNum].append(stopTripNumber + " time 0 le null 0 -1.0 l"+"\n")
      else:
        self.deckLines[deckNum].append(stopTripNumber + " cntrlvar "+stopCntrVar+" gt null 0 0.0 l "+"\n")

      self.deckLines[deckNum].append(self.stopTripNomenclature+" "+' '.join([stopTripNumber]+originalStopping)+" \n")
      # convert trips in tripunit for control variables
      controlledControlVars = {}
      for cnt, trip in enumerate(monitoredTrips[deckNum]):
        controlVar = availableControlVars.pop()
        controlledControlVars[trip] = controlVar
        self.deckLines[deckNum].append("205"+controlVar.strip()+"0".zfill(2 if self.controlVarType[deckNum] == 1 else 1 ) + " r_"+str(trip).lstrip("0")+" tripunit 1.0 0.0 0 \n")
        self.additionalControlVariables[deckNum][controlVar.strip()] = trip
        self.deckLines[deckNum].append("205"+controlVar.strip()+"1".zfill(2 if self.controlVarType[deckNum] == 1 else 1 ) + " " + str(list(monitoredTrips[deckNum])[cnt])+" \n")
      # to fix. the following can handle only 50 trips
      self.lastCntrLine[deckNum]+=2
      # if len(exclTrips[deckNum]) != len(monitoredTrips[deckNum]):
      if not excludeAll:
        self.deckLines[deckNum].append("205"+stopCntrVar.strip()+"0".zfill(2 if self.controlVarType[deckNum] == 1 else 1 )
                                     +" tripstop sum 1.0 0.0 0 \n")
      else:
        self.deckLines[deckNum].append("205"+stopCntrVar.strip()+"0".zfill(2 if self.controlVarType[deckNum] == 1 else 1 )
                                     +" tripstop constant 0.0 \n")
      if not excludeAll:
        tripCnt = 0
        for tripLine in range(int(math.ceil(float(len(monitoredTrips[deckNum]))/3.0))):
          toWrite="205"+stopCntrVar.strip()+str(tripLine+1).strip().zfill(2 if self.controlVarType[deckNum] == 1 else 1 )+ (" 0.0 " if tripLine==0 else "")
          stoppingTrips = sorted(list(set(monitoredTrips[deckNum]) - set(exclTrips[deckNum])))
          for x in range(3):
            if tripCnt+1<= len(stoppingTrips):
              toWrite += " 1.0 cntrlvar " + controlledControlVars[stoppingTrips[tripCnt]]
              tripCnt+=1
          toWrite += " \n"
          self.deckLines[deckNum].append(toWrite)
      self.deckLines[deckNum].append("* END -- CONTROL VARIABLES ADDED BY RAVEN *\n")

  def addTripsVarsInMinorEdits(self):
    """
      This method is aimed to check if the variables that can activate trips (variable trips), are part of the minor
      edits section. In case they are not, those variables are added automatically
      @ In, None
      @ Out, None
    """
    try:
      cntrVars =  self.additionalControlVariables
    except:
      cntrVars = None
    for deckNum in self.inputTrips.keys():
      alreadyAvailableEdits = self.inputMinorEdits[deckNum].values()
      availableMinorEditsCards = [x for x in range(301,399) if str(x) not in self.inputMinorEdits[deckNum].keys()]
      addedVars = []
      self.deckLines[deckNum].append("* START -- MINOR EDITS TRIPS ADDED BY RAVEN *\n")
      for tripID, trip in self.inputTrips[deckNum]['variableTrips'].items():
        found = False
        for edit in alreadyAvailableEdits:
          if trip['variable'] != 'time' and trip['component'] in edit['component'] and trip['variable'] in edit['component']:
            found = True
          elif trip['variable'] == 'time' and "timeof" in edit['variable'] and tripID in edit['component']:
            found = True
        if not found and trip['variable']+"_"+trip['component'] not in addedVars:
          if len(availableMinorEditsCards) == 0:
            raise IOError("Number of minor edits reached already the upper bound of RELAP5. There are no available cards to monitor the trips!")
          if trip['variable'] != 'time':
            addedVars.append(trip['variable']+"_"+trip['component'])
            self.deckLines[deckNum].append(str(availableMinorEditsCards.pop()) + " "+trip['variable']+" "+trip['component'] +"\n")
          else:
            addedVars.append("timeof"+"_"+tripID)
            self.deckLines[deckNum].append(str(availableMinorEditsCards.pop()) + " "+"timeof"+" "+tripID +"\n")
      if cntrVars is not None:
        for cntrVar in cntrVars[deckNum]:
          self.deckLines[deckNum].append(str(availableMinorEditsCards.pop()) + " "+"cntrlvar"+" "+cntrVar +"\n")
      self.deckLines[deckNum].append("* END --  MINOR EDITS TRIPS ADDED BY RAVEN *\n")

  def getTripsMinorEditsAndControlVars(self):
    """
      Method to store the Trips, Minor edits and Control Variables in RELAP5
      @ In, None
      @ Out, None
    """
    for deckNum in self.deckLines.keys():
      self.lastTripLine[deckNum]     = -1
      self.lastCntrLine[deckNum]     = -1
      self.lastMinorEditLine[deckNum]= -1
      for lineNum, line in enumerate(self.deckLines[deckNum]):
        splitted = _splitRecordAndRemoveComments(line)
        if len(splitted) > 0 and splitted[0].strip().isdigit():
          isMinor = self.storeMinorEdit(deckNum,splitted)
          isTrip  = self.storeTrip(deckNum,splitted)
          isCntr  = self.storeControlVars(deckNum,splitted)
          if isMinor:
            self.lastMinorEditLine[deckNum] = lineNum+1
          if isTrip :
            self.lastTripLine[deckNum]      = lineNum+1
          if isCntr :
            self.lastCntrLine[deckNum]      = lineNum+1

  def getMinorEditsInfo(self):
    """
      This method is aimed to retrieve and store (in self.inputMinorEdits)
      the information linked to minor edits (control variables, trips, etc.)
      @ In, None
      @ Out, None
    """
    for deckNum in self.deckLines.keys():
      for lineNum, line in enumerate(self.deckLines[deckNum]):
        splitted = _splitRecordAndRemoveComments(line)
        if len(splitted) > 0 and splitted[0].strip().isdigit():
          isMinor = self.storeMinorEdit(deckNum, splitted)
          if isMinor:
            self.lastMinorEditLine[deckNum] = lineNum+1

  def storeControlVars(self,deckNum, splitted):
    """
      This method is aimed to store the control variables info (id, operation, etc.)
      in self (i.e. self.inputControlVars, self.controlVarType)
      @ In, deckNum, int, the deck number that should be scanned and for which the info must be retrieved/stored
      @ In, splitted, list, list of line splitted (splitted[0] == card id, splitted[1:n_words] card words)
      @ Out, isCntr, bool, boolean True if this card is a control vars (and consequentially the info are stored), False otherwise
    """
    isCntr = False
    if int(splitted[0]) == 20500000:
      self.controlVarType[deckNum] = 1 if splitted[1].strip() == '999' else 2
      return isCntr
    if deckNum not in self.inputControlVars.keys():
      self.inputControlVars[deckNum] = {}
      self.controlVarType[deckNum] = 1
    if 20500010 <= int(splitted[0]) <= 20599990:
      isCntr = True
      if splitted[0].endswith('0'):
        controlVar = splitted[0][3:6] if self.controlVarType[deckNum] == 1 else splitted[0][3:7]
        if controlVar not in self.inputControlVars[deckNum].keys():
          self.inputControlVars[deckNum][controlVar] = {'alphaNumId': splitted[1], 'cardId':splitted[0]}
    return isCntr

  def storeMinorEdit(self,deckNum, splitted):
    """
      This method is aimed to store the minor edits info (which variables, etc)
      in self (i.e. self.inputMinorEdits)
      @ In, deckNum, int, the deck number that should be scanned and for which the info must be retrieved/stored
      @ In, splitted, list, list of line splitted (splitted[0] == card id, splitted[1:n_words] card words)
      @ Out, isMinor, bool, boolean True if this card is a minor edit card(and consequentially the info are stored), False otherwise
    """
    isMinor = False
    if deckNum not in self.inputMinorEdits.keys():
      self.inputMinorEdits[deckNum] = {}
    if 301 <= int(splitted[0]) <= 399:

      if not len(splitted) == 3:
        raise IOError(self.printTag+ "ERROR: in RELAP5 input file the number of words in minor edits section needs to be 2. Edit= "+splitted[0])
      self.inputMinorEdits[deckNum][splitted[0].strip()], isMinor = {'variable':splitted[1].strip(),'component':splitted[2].strip()}, True
    return isMinor

  def storeTrip(self,deckNum, splitted):
    """
      This method is aimed to store the trips info (which variables, etc)
      in self (i.e. self.inputTrips)
      @ In, deckNum, int, the deck number that should be scanned and for which the info must be retrieved/stored
      @ In, splitted, list, list of line splitted (splitted[0] == card id, splitted[1:n_words] card words)
      @ Out, isTrip, bool, boolean True if this card is a trip card(and consequentially the info are stored), False otherwise
    """
    isTrip = False
    if deckNum not in self.inputTrips.keys():
      self.inputTrips[deckNum] = {'variableTrips':{},'logicalTrips':{}}
    if (401 <= int(splitted[0]) <= 599) or (20600010 <= int(splitted[0]) <= 20610000):
      if  not (8 <= len(splitted)<= 9):
        raise IOError(self.printTag+ "ERROR: in RELAP5 input file the number of words in variable trip section needs to be 7 or 8 . Trip= "+splitted[0])
      tripNumber = splitted[0].strip()
      isTrip     = True
      self.inputTrips[deckNum]['variableTrips'][tripNumber] = {}
      self.inputTrips[deckNum]['variableTrips'][tripNumber]['variable'] = splitted[1].strip()
      self.inputTrips[deckNum]['variableTrips'][tripNumber]['component'] = splitted[2].strip()
      self.inputTrips[deckNum]['variableTrips'][tripNumber]['operator'] = splitted[3].strip()
      self.inputTrips[deckNum]['variableTrips'][tripNumber]['variableCode'] = splitted[4].strip()
      self.inputTrips[deckNum]['variableTrips'][tripNumber]['parameter'] = splitted[5].strip()
      self.inputTrips[deckNum]['variableTrips'][tripNumber]['additiveConstant'] = splitted[6].strip()
      self.inputTrips[deckNum]['variableTrips'][tripNumber]['latch'] = splitted[7].strip()
      self.hasVariableTrips = True
      if len(splitted) == 9:
        self.inputTrips[deckNum]['variableTrips'][tripNumber]['timeOf'] =  splitted[8].strip()
    if (601 <= int(splitted[0]) <= 799) or (20610010 <= int(splitted[0]) <= 20620000):
      if  not (5 <= len(splitted)<= 6):
        raise IOError(self.printTag+ "ERROR: in RELAP5 input file the number of words in logical trip section needs"
                      +" to be 4 or 5 . Trip= "+splitted[0])
      tripNumber = splitted[0].strip()
      isTrip     = True
      self.inputTrips[deckNum]['logicalTrips'][tripNumber] = {}
      self.inputTrips[deckNum]['logicalTrips'][tripNumber]['tripNumber1'] = splitted[1].strip()
      self.inputTrips[deckNum]['logicalTrips'][tripNumber]['operator'] = splitted[2].strip()
      self.inputTrips[deckNum]['logicalTrips'][tripNumber]['tripNumber2'] = splitted[3].strip()
      self.inputTrips[deckNum]['logicalTrips'][tripNumber]['latch'] = splitted[4].strip()
      if len(splitted) == 6:
        self.inputTrips[deckNum]['logicalTrips'][tripNumber]['timeOf'] =  splitted[5].strip()
      self.hasLogicalTrips = True
    if int(splitted[0]) == 600:
      # it already has a stop trip
      self.inputStopTrips = [splitted[1]] if len(splitted) == 2 else [splitted[1],splitted[2]]
    return isTrip

  def getTrips(self, deckNum=None):
    """
     Method to retrieve the read trips
     @ In, deckNum, int, optional, the deck number (if None, self.maxNumberOfDecks)
     @ Out, trips, dict, the dictionary of the trips for deck 'deckNum'
    """
    deckN = deckNum if deckNum is not None else self.maxNumberOfDecks
    trips = self.inputTrips[deckN]
    return trips

  def printInput(self,outfile=None):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    if outfile==None:
      outfile =self.inputfile
    outfile.open('w')
    for deckNum in self.deckLines.keys():
      for i in self.deckLines[deckNum]+['.\n']:
        outfile.write('%s' %(i))
    outfile.close()

  def retrieveCardValues(self, listOfCards):
    """
      This method is to retrieve the card values contained in the list
      @ In, listOfCards, list, list of cards ([deck,card,word])
      @ Out, cardValues, dict, dictionary containing the card and the value
    """
    foundAllCards = {}
    deckCards     = {}
    cardValues    = {}
    # check all the decks
    for deck,card,word in listOfCards:
      if deck not in self.deckLines:
        raise IOError("RELAP5 Interface: The number of deck found in the original input file is "
                      +str(self.maxNumberOfDecks)+" while the user requested to modify the deck number "+str(deck))
      if deck not in foundAllCards:
        foundAllCards[deck] = {}
        deckCards[deck] = defaultdict(list)
      foundAllCards[deck][card] = False
      deckCards[deck][card].append(word)
    for deck in deckCards:
      for lineNum, line in enumerate(self.deckLines[deck]):
        if all(foundAllCards[deck].values()):
          break
        if not re.match(r'^\s*\n',line):
          readCard = line.split()[0].strip()
          if readCard in deckCards[deck].keys():
            foundWord = False
            foundAllCards[deck][readCard] = True
            numberOfWords = self.countNumberOfWords(line)
            for word in deckCards[deck][readCard]:
              if int(word) <= numberOfWords:
                cardValues[(deck,readCard,word)] = line.split()[word]
                foundWord = True
              else:
                moveToNextLine            = True
                cnt                       = 1
                while moveToNextLine:
                  if self.deckLines[deck][lineNum+cnt].strip().startswith("+"):
                    currentNumberWords = self.countNumberOfWords(self.deckLines[deck][lineNum+cnt])
                    if int(word) <= numberOfWords+currentNumberWords:
                      cardValues[(deck,readCard,word)] = line.split()[word-currentNumberWords]
                      foundWord = True
                      moveToNextLine = False
                    numberOfWords+=currentNumberWords
                  else:
                    moveToNextLine=False
              if not foundWord:
                raise IOError("RELAP5 Interface: The number of words found for card "+str(readCard)
                              +" is "+str(numberOfWords)+"while the user requested to modify the word number "+str(word))
      # check if all cards have been found
      if not all(foundAllCards[deck].values()):
        cardsNotFound = ""
        for card,found in foundAllCards[deck].items():
          if not found:
            cardsNotFound+= card +" "
        raise IOError("RELAP5 Interface: The following cards have not been found in the original input files: "+cardsNotFound)
    return cardValues

  def modifyOrAdd(self,modifyDict,save=True):
    """
      dictionaryList is a list of dictionaries of the required addition or modification
      the method looks in self.lines for a card number matching the card in modiDictionaryList
      and modifies the word from dictionaryList at needed
      @ In, modifyDict, dict, dictionary containing the info to modify the XML tree
      @ In, save, bool, optional, True if the original tree needs to be saved
      @ Out, lines, list, list of modified lines (of the original input)
    """
    decks              = {}
    toAdd              = {} # for DET
    lines              = []
    if 'decks' not in modifyDict:
      raise IOError(self.printTag+"ERROR: no card inputs found!!")
    else:
      decks = modifyDict['decks']
    for deckNum in decks.keys():
      if deckNum not in self.deckLines.keys():
        raise IOError("RELAP5 Interface: The number of deck found in the original input file is "
                      +str(self.maxNumberOfDecks)+" while the user requested to modify the deck number "+str(deckNum))
      temp               = []
      modiDictionaryList = decks[deckNum]
      temp.append('*RAVEN INPUT VALUES\n')
      if self.maxNumberOfDecks > 1:
        temp.append('*'+' deckNum: '+str(deckNum)+'\n')
      for j in sorted(modiDictionaryList):
        for var in modiDictionaryList[j]:
          try:
            temp.append('* card: '+j+' word: '+str(var['position'])+' value: '+'{:.7e}'.format(var['value'])+'\n')
          except ValueError:
            temp.append('* card: '+j+' word: '+str(var['position'])+' value: '+var['value']+'\n')
      temp.append('*RAVEN INPUT VALUES\n')

      temp+=self.deckLines[deckNum]
      cardLines = {}
      foundAllCards = dict.fromkeys(modiDictionaryList.keys(),False)
      cnt = 0
      for lineNum, line in enumerate(temp):
        if all(foundAllCards.values()):
          break
        if not re.match(r'^\s*\n',line):
          splitted = _splitRecordAndRemoveComments(line)
          card = splitted[0].strip() if len(splitted) > 0 else '*-None-*'
          if card in modiDictionaryList.keys():
            cardLines[card] = {'lineNumber':lineNum,'numberOfLevels':1,'numberOfAvailableWords':self.countNumberOfWords(line)}
            foundAllCards[card] = True
            moveToNextLine      = True
            cnt                 = 1
            while moveToNextLine:
              if lineNum+cnt < len(temp) and temp[lineNum+cnt].strip().startswith("+"):
                cardLines[card]['numberOfLevels'        ]+=1
                cardLines[card]['numberOfAvailableWords']+=self.countNumberOfWords(temp[lineNum+cnt])
                cnt+=1
              else:
                moveToNextLine=False
      # modify the cards
      for card in cardLines.keys():
        for var in modiDictionaryList[card]:
          if cardLines[card]['numberOfAvailableWords'] >= var['position']:
            totalNumberOfWords = 0
            for i in range(cardLines[card]['numberOfLevels']):
              numberOfWords = self.countNumberOfWords(temp[cardLines[card]['lineNumber']+i])
              if totalNumberOfWords+numberOfWords>=var['position']:
                temp[cardLines[card]['lineNumber']+i] = self.replaceword(temp[cardLines[card]['lineNumber']+i],var['position']-totalNumberOfWords,var['value'])
                break
              totalNumberOfWords+=numberOfWords
          else:
            raise IOError("RELAP5 Interface: The word that needs to be sampled is in a position ("+str(var['position'])+") > then the actual number of words ("+str(cardLines[card]['numberOfAvailableWords'])+")!!")
      if save:
        self.deckLines[deckNum]=temp
        if self.addMinorEdits:
          self.lastCntrLine[deckNum]      +=cnt
          self.lastMinorEditLine[deckNum] +=cnt
          self.lastTripLine[deckNum]      +=cnt
          detVars = modifyDict.get('DETvariables', None)
          if detVars is not None:
            toAdd[deckNum] = []
            for var in detVars:
              splitted =  var.split(":")
              if "|" not in splitted[0] and len(decks) == deckNum:
                toAdd[deckNum].append(splitted[0].strip())
              elif "|" in splitted[0] and int(splitted[0].split("|")[0]) == deckNum:
                toAdd[deckNum].append(splitted[0].split("|")[1].strip())
            if toAdd[deckNum]:
              self.addControlVariablesForStoppingCoditions(toAdd, modifyDict.get('excludeTrips'))
            else:
              if self.presentStopTrip[deckNum] is not None:
                self.deckLines[deckNum].append(self.stopTripNomenclature+" "+" ".join(self.presentStopTrip[deckNum]) + "\n")
          # add trips and control variables
          self.addTripsVarsInMinorEdits()
        else:
          if self.presentStopTrip[deckNum] is not None:
            self.deckLines[deckNum].append(self.stopTripNomenclature+" "+" ".join(self.presentStopTrip[deckNum]) + "\n")
      lines = lines + temp
    return lines

  def countNumberOfWords(self,line,additionFactor=-1):
    """
      Method to count the number of words in a certain line
      @ In, line, string, line to be evaluated
      @ In, additionFactor, int, addition factor
      @ Out, number, int, the number of words
    """
    number = len(_splitRecordAndRemoveComments(line))+additionFactor
    return number

  def replaceword(self,line,position,value):
    """
      Method to replace a word value with the a new value
      @ In, line, string, line to be modified
      @ In, position, int, word position that needs to be changed
      @ In, value, float, new value
      @ Out, newline, string, modified line
    """
    temp=line.split()
    try:
      temp[int(position)]='{:.7e}'.format(value)
    except ValueError:
      temp[int(position)]=value
    newline=temp.pop(0)
    for i in temp:
      newline=newline+'  '+i
    newline=newline+'\n'
    return newline
