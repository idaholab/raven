"""
Created on July 11, 2013
@author: nieljw
@modified: alfoa
"""
import os
import fileinput
import re

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
    if not os.path.exists(inputFile): raise IOError(self.printTag+'ERROR: not found RELAP input file')
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
    if self.maxNumberOfDecks < 1: raise IOError(self.printTag+ "ERROR: the file "+inputFile+" does not contain a end case fullstop '.'!")

  def printInput(self,outfile=None):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    if outfile==None: outfile =self.inputfile
    outfile.open('w')
    for deckNum in self.deckLines.keys():
      for i in self.deckLines[deckNum]: outfile.write('%s' %(i))
    outfile.close()

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
      if 'decks' not in i.keys(): raise IOError(self.printTag+"ERROR: no card inputs found!!")
      else                      : decks.update(i['decks'])
    for deckNum in decks.keys():
      temp               = []
      modiDictionaryList = decks[deckNum]
      temp.append('*RAVEN INPUT VALUES\n')
      if self.maxNumberOfDecks > 1: temp.append('*'+' deckNum: '+str(deckNum)+'\n')
      for j in modiDictionaryList:
        for var in modiDictionaryList[j]:
          temp.append('* card: '+j+' word: '+str(var['position'])+' value: '+str(var['value'])+'\n')
      temp.append('*RAVEN INPUT VALUES\n')
      for line in self.deckLines[deckNum]: #     fileinput.input(self.inputfile, mode='r'):
        temp1=line
        if not re.match('^\s*\n',line):
          card = line.split()[0].strip()
          if card in modiDictionaryList.keys():
            temp2 = line
            for var in modidictionaryList[card]:
              temp1 = self.replaceword(temp2,var['position'],var['value'])
              temp2 = temp1
            #temp1 = self.replaceword(line,modiDictionaryList[card]['position'],modiDictionaryList[card]['value'])
        temp.append(temp1)
      if save: self.deckLines[deckNum]=temp
      lines = lines + temp
    return lines

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
    for i in temp: newline=newline+'  '+i
    newline=newline+'\n'
    return newline

