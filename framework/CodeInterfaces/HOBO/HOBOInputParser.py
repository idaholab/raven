'''

Created on July 16, 2014

@author: alfoa

'''
import os
import copy
import fileinput

class HOBOInputParser:
  ''' provide methods to add/change entries and print it back'''
  def __init__(self,inputFile):
    # init function. It opens the original input file
    if type(inputFile) == list:
      self.lines = {}
      IOfile     = {}
      self.inputfile = {}
      for filin in inputFile:
        if not os.path.exists(filin): raise IOError('not found HOBO input file')
        IOfile[filin]  = open(filin,'r')
        self.inputfile[filin] =  filin 
        self.lines[filin]= IOfile[filin].readlines()      
    else:
      self.lines = []
      if not os.path.exists(inputFile): raise IOError('not found HOBO input file')
      IOfile         = open(inputFile,'r')
      self.inputfile = inputFile
      self.lines     = IOfile.readlines()
      
  def printInput(self,outfile=None):
    # print back the input file
    if type(self.lines) == dict:
      for cnt,key in enumerate(self.lines.keys()):
        if outfile==None: outfile2 =self.inputfile[key]
        else: 
          for aa in outfile:
            if os.path.basename(aa) == os.path.basename(key):
              outfile2 = aa    
        IOfile = open(outfile2,'w')
        for line in self.lines[key]:
          IOfile.write('%s' %(line))
        IOfile.close()
    else:
      if outfile==None: outfile =self.inputfile
      IOfile = open(outfile,'w')
      for line in self.lines:
        IOfile.write('%s' %(line))      

  def modifyOrAdd(self,modifDict,save=True):
    '''modifDict is a dict of the required addition or modification
    the method looks in self.lines for a row number matching the row in modifDict
    and modifies the word from modifDict at needed'''
    temp=copy.deepcopy(self.lines)
    for key,value in modifDict.items():
      if key == 'strategy1':
        for cnt,key in enumerate(self.inputfile.keys()):
          if os.path.basename(key) == '2_temperature.txt':
            self.lines[key] = []
            for cnt2,T in enumerate(modifDict['strategy1']['T']):
              self.lines[key].append(str(modifDict['strategy1']['time'][cnt2]) + "\t" + str(modifDict['strategy1']['T'][cnt2])+'\n')
          if os.path.basename(key) == '3_fissionrate.txt':
            self.lines[key] = []
            for cnt2,T in enumerate(modifDict['strategy1']['FissionRate']):
              self.lines[key].append(str(modifDict['strategy1']['time'][cnt2]) + "\t" + str(modifDict['strategy1']['FissionRate'][cnt2])+'\n')      
      else:
        rowNumber     = int(copy.deepcopy(value['row']))-1
        valueToChange = copy.deepcopy(value['value'])
        if rowNumber > len(temp): raise IOError("ExampleCodeInputParser: ERROR -> row number defined in sampler bigger than input lines. Got" + str(rowNumber) + '>' +str(len(temp)))
        temp[rowNumber] = str(valueToChange)+'\n'
    if save:
      if type(self.lines) != dict: 
        self.lines=copy.deepcopy(temp)
      else:
        self.lines=copy.deepcopy(self.lines)
    return self.lines
  
  

