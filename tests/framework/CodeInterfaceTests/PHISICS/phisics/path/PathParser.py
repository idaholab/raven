"""
Created on July 17th, 2017
@author: rouxpn
"""

import os
import sys
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal 

import time 


class PathParser():

  def matrix_printer(self, line, outfile):
  
    #print line 
    line = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',line)
    line = line.upper().split()
    #print line
    if line[0] in self.setOfPerturbedIsotopes:
      try :
        #print line[1]
        #print self.listedQValuesDict.get(line[0])
        line[1] = str(self.listedQValuesDict.get(line[0]))
        #print line[1]
      except : 
        raise Exception('Error. Check if the the actinides perturbed QValues have existing Qvalues in the unperturbed library')
    #print line 
    if len(line) > 1:
      line[0] = "{0:<7s}".format(line[0])
      line[1] = "{0:<11s}".format(line[1])
      line = ''.join(line[0]+line[1]+"\n")
      #outfile.writelines(line)
      #print line 
      #outfile.writelines(line)
      outfile.writelines(line) 
    if re.search(r'(.*?)END', line[0]):
      self.stopFlag = self.stopFlag + 1
      self.harcodingSection = 0 

  def modifyValues(self, **Kwargs):
    """
      This method needs to access to the dictionary self.isotopeDecay
      and modify the values based on the input dictionary dictOfValues
      In: isotopeDecay 
      Out: 
    """
    
    self.pertQValuesDict  = {'BETAXDECAY|U235':2.222, 'BETAXDECAY|U238':1.08E+02, 'BETAXDECAY|CF252':4.85, 'BETAXDECAY|HE4':8.915}
    for key, value in self.pertQValuesDict.iteritems(): 
      self.pertQValuesDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    #print self.pertDict
    #print Kwargs.get('SampledVars', {})

          
  def __init__(self):
    """
      Parse the PHISICS Decay data file and put the isotopes name as key and 
      the decay constant relative to the isotopes as values  
      In: decay.dat
      Out: self.decay (dictionary)
    """
    self.start_time = time.time()
    self.directory = "path"
    self.pathList = []
    self.pathListWithQvalue = []
    self.pathToFileDict = {}
    for file in os.listdir(self.directory):
      if file.endswith(".path"):
        #print os.path.join("path", file)
        filePath =  str(os.path.join(self.directory, file))  
        self.pathList.append(filePath)
      self.pathToFileDict[filePath] = file
    #print self.pathList
    #print self.pathToFileDict 
    for i in xrange (0, len(self.pathList)):
      qvalueFlag = 0 
      with open(self.pathList[i], 'r') as infile:
        for line in infile:
          #print line 
          if re.search(r'Qvalue',line):
            qvalueFlag = 1
            break 
        if qvalueFlag == 1 :
          self.pathListWithQvalue.append(self.pathList[i])
    #print self.pathListWithQvalue
    
    
  def fileReconstruction(self):
    """
      Method convert the formatted dictionary pertdict -> {'DECAY|ALPHA|U235':1.30} 
      into a dictionary of dictionaries that has the format -> {'U235':{'ALPHA':1.30}}
      In: Dictionary pertDict 
      Out: Dictionary of dictionaries listedDict 
    """
    #print self.genericDecayDict
    #print self.pertDict
    self.listedQValuesDict = {}
    perturbedIsotopes = []
    for i in self.pertQValuesDict.iterkeys() :
      splittedDecayKeywords = i.split('|')
      perturbedIsotopes.append(splittedDecayKeywords[1])
    #print perturbedIsotopes 
    for i in xrange (0,len(perturbedIsotopes)):
      self.listedQValuesDict[perturbedIsotopes[i]] = {}   # declare all the dictionaries
    #print self.listedQValuesDict
    for isotopeKeyName, QValue in self.pertQValuesDict.iteritems():
      isotopeName = isotopeKeyName.split('|')
      #print isotopeName
      self.listedQValuesDict[isotopeName[1]] = QValue
    #print self.listedQValuesDict
    self.setOfPerturbedIsotopes = set(self.listedQValuesDict.iterkeys())
    #print self.setOfPerturbedIsotopes
    
  def printInput(self):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    for i in xrange (0, len(self.pathListWithQvalue)):
      print self.pathListWithQvalue[i]
      print self.pathToFileDict.get(self.pathListWithQvalue[i]) 
      modifiedFile = 'modified'+self.pathToFileDict.get(self.pathListWithQvalue[i])  
      print modifiedFile
      print "\n"
      sectionCounter = 0
      self.stopFlag = 0 
      self.harcodingSection = 0 
      open(modifiedFile, 'w')
      with open(modifiedFile, 'a') as outfile:
        with open(self.pathListWithQvalue[i]) as infile:   # count the number of times QValue sections occur
          for line in infile :
            if re.search(r'(.*?)(\s?)[a-zA-Z](\s+Qvalue)',line.strip()):
              #print line 
              sectionCounter = sectionCounter + 1 
              #print sectionCounter
            #line = line.upper().split()
            if not line.split(): continue       # if the line is blank, ignore it 
            if sectionCounter == 1 and self.stopFlag == 0:     #actinide section
              self.harcodingSection = 1 
              path_parser.matrix_printer(line, outfile)
            if sectionCounter == 2 and self.stopFlag == 1:     #FP section
              self.harcodingSection = 2 
              path_parser.matrix_printer(line, outfile)
            #print line 
            if self.harcodingSection != 1 and self.harcodingSection !=2:
              outfile.writelines(line)
    print ("--- %s seconds --- " % (time.time() - self.start_time))

if __name__ == "__main__": 
  path_parser = PathParser()
  path_parser.modifyValues()
  path_parser.fileReconstruction()
  path_parser.printInput()

