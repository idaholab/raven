'''
Created on July 11, 2013

@author: nieljw
'''
import xml.etree.ElementTree as ET
import os
import copy
import fileinput
import re
class RELAPparser:
  '''import the MOOSE input as xml tree, provide methods to add/change entries and print it back'''
  def __init__(self,inputFile):
    if not os.path.exists(inputFile): raise IOError('not found RELAP input file')
    IOfile = open(inputFile,'r')
    self.inputfile = inputFile
    self.lines = IOfile.readlines()

  def printInput(self,outfile=None):
    if outfile==None: outfile =self.inputfile
    IOfile = open(outfile,'w')
    for i in self.lines:
      IOfile.write('%s' %(i))

  def modifyOrAdd(self,modiDictionaryList,save=True):
    '''ModiDictionaryList is a list of dictionaries of the required addition or modification'''
    '''the method looks in self.lines for a card number matching the card in modiDictionaryList'''
    '''and modifies the word from modiDictionaryList at needed'''      
    temp=[]
    temp.append('*RAVEN INPUT VALUES\n')
    for j in modiDictionaryList:
      temp.append('*'+j+'    '+modiDictionaryList[j]['position']+'   '+str(modiDictionaryList[j]['value'])+'\n')
    temp.append('*RAVEN INPUT VALUES\n')
    for line in fileinput.input(self.inputfile, mode='r'):
      temp1=line
      if not re.match('^\s*\n',line):
        if line.split()[0] in modiDictionaryList:
            temp1 = self.replaceword(line,modiDictionaryList[line.split()[0]]['position'],modiDictionaryList[line.split()[0]]['value'])
      temp.append(temp1)  
    if save: 
      self.lines=temp
    return self.lines     

  def replaceword(self,line,position,value):
    temp=line.split()
    temp[int(position)]=str(value)
    newline=temp.pop(0)
    for i in temp:
      newline=newline+'  '+i
    newline=newline+'\n'
    return newline

      
if __name__=='__main__':
  file=RELAPparser('restart.i')
  dictlist={}
  dictlist['531']={'position':6,'value':1.0E6}
  dictlist['525']={'position':6,'value':1.0E6}
  file.modifyOrAdd(dictlist,True)
  file.printInput('restart.n')


