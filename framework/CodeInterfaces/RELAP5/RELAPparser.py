'''
Created on July 11, 2013

@author: nieljw
@modified: alfoa
'''
import os
import fileinput
import re

class RELAPparser():
  '''import the MOOSE input as xml tree, provide methods to add/change entries and print it back'''
  def __init__(self,inputFile):
    self.printTag = 'RELAP5 PARSER'
    if not os.path.exists(inputFile): raise IOError('not found RELAP input file')
    IOfile = open(inputFile,'r')
    self.inputfile = inputFile
    self.lines = IOfile.readlines()

  def printInput(self,outfile=None):
    if outfile==None: outfile =self.inputfile
    #IOfile = open(outfile,'w')
    outfile.open('w')
    #for i in self.lines: IOfile.write('%s' %(i))
    for i in self.lines: outfile.write('%s' %(i))

  def modifyOrAdd(self,DictionaryList,save=True):
    '''ModiDictionaryList is a list of dictionaries of the required addition or modification
    the method looks in self.lines for a card number matching the card in modiDictionaryList
    and modifies the word from modiDictionaryList at needed'''
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
    temp=line.split()
    temp[int(position)]=str(value)
    newline=temp.pop(0)
    for i in temp: newline=newline+'  '+i
    newline=newline+'\n'
    return newline

