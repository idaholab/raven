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
from collections import defaultdict

class RELAPparser():
  """
    Import the RELAP5 input as list of lines, provide methods to add/change entries and print it back
  """
  def __init__(self,inputFile):
    """
      Constructor
      @ In, inputFile, string, input file name
      @ Out, None
    """
    self.printTag = 'RELAP5 PARSER'
    if not os.path.exists(inputFile):
      raise IOError(self.printTag+'ERROR: not found RELAP input file')
    IOfile = open(inputFile,'r')
    self.inputfile = inputFile
    self.deckLines = {}
    self.maxNumberOfDecks = 0
    prevDeckLineNum       = 0
    lines                 = IOfile.readlines()
    for lineNum, line in enumerate(lines):
      if line.strip().startswith("."):
        self.maxNumberOfDecks += 1
        self.deckLines[self.maxNumberOfDecks] = lines[prevDeckLineNum:lineNum+1]
        prevDeckLineNum = lineNum + 1
    if self.maxNumberOfDecks < 1:
      raise IOError(self.printTag+ "ERROR: the file "+inputFile+" does not contain a end case fullstop '.'!")

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
      for i in self.deckLines[deckNum]:
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
        raise IOError("RELAP5 Interface: The number of deck found in the original input file is "+str(self.maxNumberOfDecks)+" while the user requested to modify the deck number "+str(deck))
      if deck not in foundAllCards:
        foundAllCards[deck] = {}
        deckCards[deck] = defaultdict(list)
      foundAllCards[deck][card] = False
      deckCards[deck][card].append(word)
    for deck in deckCards:
      for lineNum, line in enumerate(self.deckLines[deck]):
        if all(foundAllCards[deck].values()):
          break
        if not re.match('^\s*\n',line):
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
                raise IOError("RELAP5 Interface: The number of words found for card "+str(readCard)+" is "+str(numberOfWords)+"while the user requested to modify the word number "+str(word))
      # check if all cards have been found
      if not all(foundAllCards[deck].values()):
        cardsNotFound = ""
        for card,found in foundAllCards[deck].items():
          if not found: cardsNotFound+= card +" "
        raise IOError("RELAP5 Interface: The following cards have not been found in the original input files: "+cardsNotFound)
    return cardValues

  def modifyOrAdd(self,dictionaryList,save=True):
    """
      dictionaryList is a list of dictionaries of the required addition or modification
      the method looks in self.lines for a card number matching the card in modiDictionaryList
      and modifies the word from dictionaryList at needed
      @ In, dictionaryList, list, list of dictionaries containing the info to modify the XML tree
      @ In, save, bool, optional, True if the original tree needs to be saved
      @ Out, lines, list, list of modified lines (of the original input)
    """
    decks              = {}
    lines              = []
    for i in dictionaryList:
      if 'decks' not in i.keys():
        raise IOError(self.printTag+"ERROR: no card inputs found!!")
      else:
        decks.update(i['decks'])
    for deckNum in decks.keys():
      if deckNum not in self.deckLines.keys():
        raise IOError("RELAP5 Interface: The number of deck found in the original input file is "+str(self.maxNumberOfDecks)+" while the user requested to modify the deck number "+str(deckNum))
      temp               = []
      modiDictionaryList = decks[deckNum]
      temp.append('*RAVEN INPUT VALUES\n')
      if self.maxNumberOfDecks > 1:
        temp.append('*'+' deckNum: '+str(deckNum)+'\n')
      for j in sorted(modiDictionaryList):
        for var in sorted(modiDictionaryList[j]):
          temp.append('* card: '+j+' word: '+str(var['position'])+' value: '+str(var['value'])+'\n')
      temp.append('*RAVEN INPUT VALUES\n')

      temp+=self.deckLines[deckNum]
      cardLines = {}
      foundAllCards = dict.fromkeys(modiDictionaryList.keys(),False)
      for lineNum, line in enumerate(temp):
        if all(foundAllCards.values()):
          break
        if not re.match('^\s*\n',line):
          card = line.split()[0].strip()
          if card in modiDictionaryList.keys():
            cardLines[card] = {'lineNumber':lineNum,'numberOfLevels':1,'numberOfAvailableWords':self.countNumberOfWords(line)}
            foundAllCards[card] = True
            moveToNextLine      = True
            cnt                 = 1
            while moveToNextLine:
              if temp[lineNum+cnt].strip().startswith("+"):
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
      lines = lines + temp
    return lines

  def countNumberOfWords(self,line,additionFactor=-1):
    """
      Method to count the number of words in a certain line
      @ In, line, string, line to be evaluated
      @ In, additionFactor, int, addition factor
      @ Out, number, int, the number of words
    """
    number = len(line.split())+additionFactor
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
    temp[int(position)]=str(value)
    newline=temp.pop(0)
    for i in temp:
      newline=newline+'  '+i
    newline=newline+'\n'
    return newline
