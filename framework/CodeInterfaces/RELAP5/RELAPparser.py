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
    if not os.path.exists(inputFile): raise IOError('not found RELAP input file')
    IOfile = open(inputFile,'r')
    self.inputfile = inputFile
    self.lines = IOfile.readlines()

  def printInput(self,outfile=None):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    if outfile==None: outfile =self.inputfile
    outfile.open('w')
    for i in self.lines: outfile.write('%s' %(i))
    outfile.close()

  def modifyOrAdd(self,DictionaryList,save=True):
    """
      DictionaryList is a list of dictionaries of the required addition or modification
      the method looks in self.lines for a card number matching the card in modiDictionaryList
      and modifies the word from DictionaryList at needed
      @ In, DictionaryList, list, list of dictionaries containing the info to modify the XML tree
      @ In, save, bool, optional, True if the original tree needs to be saved
      @ Out, lines, list, list of modified lines (of the original input)
    """
    temp=[]
    modiDictionaryList = {}
    for i in DictionaryList:
      if 'cards' in i.keys():  modiDictionaryList.update(i['cards'])
    temp.append('*RAVEN INPUT VALUES\n')
    for j in modiDictionaryList: temp.append('*'+j+'    '+str(modiDictionaryList[j]['position'])+'   '+str(modiDictionaryList[j]['value'])+'\n')
    temp.append('*RAVEN INPUT VALUES\n')
    for line in fileinput.input(self.inputfile, mode='r'):
      temp1=line
      if not re.match('^\s*\n',line):
        if line.split()[0] in modiDictionaryList: temp1 = self.replaceword(line,modiDictionaryList[line.split()[0]]['position'],modiDictionaryList[line.split()[0]]['value'])
      temp.append(temp1)
    if save: self.lines=temp
    return self.lines

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

