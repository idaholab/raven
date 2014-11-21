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
    self.lines = []
    if not os.path.exists(inputFile): raise IOError('not found HOBO input file')
    IOfile         = open(inputFile,'r')
    self.inputfile = inputFile
    self.lines     = IOfile.readlines()

  def printInput(self,outfile=None):
    # print back the input file
    if outfile==None: outfile =self.inputfile
    IOfile = open(outfile,'w')
    for line in self.lines:
      IOfile.write('%s' %(line))

  def modifyOrAdd(self,modifDict,save=True):
    '''modifDict is a dict of the required addition or modification
    the method looks in self.lines for a row number matching the row in modifDict
    and modifies the word from modifDict at needed'''
    temp=copy.deepcopy(self.lines)
    for value in modifDict.values():
      rowNumber     = int(copy.deepcopy(value['row']))-1
      valueToChange = copy.deepcopy(value['value'])
      if rowNumber > len(temp): raise IOError("ExampleCodeInputParser: ERROR -> row number defined in sampler bigger than input lines. Got" + str(rowNumber) + '>' +str(len(temp)))
      temp[rowNumber] = str(valueToChange)+'\n'
    if save:
      self.lines=copy.deepcopy(temp)
    return self.lines

