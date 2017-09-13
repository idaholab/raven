"""
Created on June 19th, 2017
@author: rouxpn
"""

import os
import sys
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal 
from random import *


class DecayParser():

  def matrix_printer(self, infile, outfile, atomicNumber):
  
    #print atomicNumber
    for line in infile :
      line = line.upper().split()
      line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',line[0])
      #print line 
      for isotopeID in self.listedDict.iterkeys():
        if line[0] == isotopeID :
          #print isotopeID
          typeOfDecayPerturbed = []
          typeOfDecayPerturbed = self.listedDict.get(isotopeID,{}).keys()
          #print typeOfDecayPerturbed
          for i in xrange (0, len(typeOfDecayPerturbed)):
            #print line[self.decayModeNumbering[self.isotopeParsed[0]].get(typeOfDecayPerturbed[i])]  # should give the value of the decay type of interest in the original library. This is the value that has to be replaced
            #print self.listedDict.get(isotopeID).get(typeOfDecayPerturbed[i])    # should give the pertubed value      
            try :
              if self.isotopeClassifier.get(isotopeID) == self.isotopeParsed[0]:  # it means the isotope is an actinide 
                #print typeOfDecayPerturbed
                line[self.decayModeNumbering.get(atomicNumber).get(typeOfDecayPerturbed[i])] = str(self.listedDict.get(isotopeID).get(typeOfDecayPerturbed[i]))
              elif self.isotopeClassifier.get(isotopeID) == self.isotopeParsed[1]:  # it means the isotope is a FP 
                #print self.FPNumbering
                #print typeOfDecayPerturbed
                #print self.FPNumbering.get(typeOfDecayPerturbed[i])
                line[self.decayModeNumbering.get(atomicNumber).get(typeOfDecayPerturbed[i])] = str(self.listedDict.get(isotopeID).get(typeOfDecayPerturbed[i]))
            except : 
              raise Exception('you used the decay mode'+str(typeOfDecayPerturbed)+'Check if the decay mode '+str(typeOfDecayPerturbed)+'exist in the decay library. You can also check if you perturbed dictionary is under the format |DECAY|DECAYMODE|ISOTOPEID.')
      #print line
      # FORMATING OF THE TABLE
      if   any('ACTINIDES' in s for s in line)  : flag = self.isotopeParsed[0]
      elif any('FPRODUCTS' in s for s in line)  : flag = self.isotopeParsed[1]
      try :
        #print self.isotopeClassifier
        if self.isotopeClassifier[line[0]] == atomicNumber :
          #print line 
          line[0] = "{0:<7s}".format(line[0])
          i = 1
          while i <= len(self.decayModeNumbering[atomicNumber]) : 
            line[i] = "{0:<11s}".format(line[i])
            i = i + 1 
          outfile.writelines(' '+''.join(line[0:len(self.decayModeNumbering[atomicNumber]) + 1])+"\n") 
      except KeyError: 
        pass

  def hardcopy_printer(self, input, atomicNumber, modifiedFile):
     
    flag = 0 
    #print atomicNumber
    with open(modifiedFile, 'a') as outfile:
      with open(input) as infile:
        for line in infile:
          #line = line.split()
          if not line.split(): continue   # if the line is blank, ignore it 
          #print line 
          #if re.match(r'(.*?)'+atomicNumber+'s'r'(s?)\s+\w',line.strip()) and atomicNumber == 'Actinides':
          if re.match(r'(.*?)'+atomicNumber+'s',line.strip()) and atomicNumber == self.isotopeParsed[0]:
            flag = 2
            #print flag 
          if flag == 2 :
            if re.match(r'(.*?)\s+\w+(\W)\s+\w+(\W)',line) and any(s in 'BETA' for s in line.split()) and atomicNumber == self.isotopeParsed[0] : 
              #print line
              outfile.writelines(line)
              break
            outfile.writelines(line)
          #print atomicNumber
          if any(s in atomicNumber+'roducts' for s in line.split()):
          #if any(s in atomicNumber+'roducts' for s in line.split()) and atomicNumber == self.isotopeParsed[1]):
            flag = 1 
            #print line
            #print flag 
          if  flag == 1 :
            if re.match(r'(.*?)\s+\w+(\W)\s+\w+(\W)',line) and any(s in 'BETA' for s in line.split()) and atomicNumber == self.isotopeParsed[1] :
              #print line 
              outfile.writelines(line)
              break
            outfile.writelines(line)
          
        self.matrix_printer(infile, outfile, atomicNumber)

  def __init__(self, inputFiles, **pertDict):
    """
      Parse the PHISICS Decay data file and put the isotopes name as key and 
      the decay constant relative to the isotopes as values  
      In: decay.dat
      Out: self.decay (dictionary)
    """
    self.pertDict = pertDict
    for key, value in self.pertDict.iteritems(): 
      self.pertDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    #print self.pertDict
    numbering    = {}
    concatenateDecayList = []
    self.allDecayList = []
    self.inputFiles = inputFiles
    #print self.inputFiles
    #print self.outputFiles
    self.isotopeClassifier = {}   # FP or Actinide 
    #print self.inputFiles    
    OpenInputFile = open (inputFiles, "r")
    lines = OpenInputFile.readlines()
    #print lines 
    OpenInputFile.close()
    self.isotopeParsed=['Actinide','FP']
    self.decayModeNumbering = {}
    for line in lines:
      #print line 
      if   re.match(r'(.*?)Actinides', line) : typeOfIsotopeParsed = self.isotopeParsed[0]
      elif re.match(r'(.*?)FProducts', line) : typeOfIsotopeParsed = self.isotopeParsed[1]
      # create dynamic column detector 
      if (re.match(r'(.*?)\w+(\W?)\s+\w+(\W?)\s+\w',line) and any(s in "BETA" for s in line)) :
        count = 0                            # reset the counter and the dictionary numbering if new colum sequence is detected
        numbering = {}
        decayList = []
        line = re.sub(r'(Yy?)ield',r'',line)          # Remove the word 'yield' in the decay type lines 
        SplitStringDecayType = line.upper().split()   # Split the words into individual strings 
        for i in SplitStringDecayType :               # replace + and * by strings
          decayList.append(i.replace('*', 'S').replace('+','PLUS').replace('_',''))
        #print decayList
        concatenateDecayList = concatenateDecayList + decayList  # concatenate all the possible decay type (including repetition among actinides and FP)
        self.allDecayList = list(set(concatenateDecayList))
        #print self.allDecayList 
        for i in xrange(len(decayList)) :
          count = count + 1
          numbering[decayList[i]] = count   # assign the column position of the given decay types
        if typeOfIsotopeParsed == self.isotopeParsed[0]: self.decayModeNumbering[self.isotopeParsed[0]] = numbering
        if typeOfIsotopeParsed == self.isotopeParsed[1]: self.decayModeNumbering[self.isotopeParsed[1]] = numbering
        #print typeOfIsotopeParsed
      #print self.decayModeNumbering
      if re.match(r'(.*?)\D+(-?)\d+(M?)\s+\d', line):
        #print numbering
        SplitString = line.upper().split()   # split the lines and transform all string into uper cases
        #print SplitString
        for i, x in enumerate(SplitString):
          try:
            SplitString[i] = float(x)         # convert strings into numbers (scientific notation)
          except ValueError:
            pass
        #print SplitString # to verify the numbers are not strings formatted anymore 
        #print typeOfIsotopeParsed
        SplitString[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',SplitString[0])   # remove the dash if it the key (isotope ID) contains it
        #print SplitString[0]
        if typeOfIsotopeParsed == self.isotopeParsed[0]: self.isotopeClassifier[SplitString[0]] = self.isotopeParsed[0]
        elif typeOfIsotopeParsed == self.isotopeParsed[1]: self.isotopeClassifier[SplitString[0]] = self.isotopeParsed[1]
    self.fileReconstruction()

  def fileReconstruction(self):
    """
      Method convert the formatted dictionary pertdict -> {'DECAY|ALPHA|U235':1.30} 
      into a dictionary of dictionaries that has the format -> {'U235':{'ALPHA':1.30}}
      In: Dictionary pertDict 
      Out: Dictionary of dictionaries listedDict 
    """
    #print self.genericDecayDict
    #print self.pertDict
    self.listedDict = {}
    perturbedIsotopes = []
    for i in self.pertDict.iterkeys() :
      splittedDecayKeywords = i.split('|')
      perturbedIsotopes.append(splittedDecayKeywords[2])
    #print perturbedIsotopes 
    #isotopeList = self.decay.get('BETA', {}).keys()  # get the complete isotope list
    #print isotopeList 
    for i in xrange (0,len(perturbedIsotopes)):
      self.listedDict[perturbedIsotopes[i]] = {}   # declare all the dictionaries
    #print self.listedDict
    for decayTypeKey, decayValue in self.pertDict.iteritems():
      decayKeyWords = decayTypeKey.split('|')
      for i in xrange (0, len(self.allDecayList)):
        self.listedDict[decayKeyWords[2]][decayKeyWords[1]] = decayValue
    #print self.listedDict
    
    self.printInput()
    
  def removeRandomlyNamedFiles(self, modifiedFile):
    """
      Remove the temporary file with a random name in the working directory
      In, modifiedFile, string
      Out, None 
    """
    os.remove(modifiedFile)
  
  def generateRandomName(self):
    """
      generate a random file name for the modified file
      @ in, None
      @ Out, string
    """
    return str(randint(1,1000000000000))+'.dat'
   
  def printInput(self):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    modifiedFile = self.generateRandomName() 
    open(modifiedFile, 'w')
    for atomicNumber in self.isotopeParsed:
      self.hardcopy_printer(self.inputFiles, atomicNumber, modifiedFile)
    with open(modifiedFile, 'a') as outfile:
      outfile.writelines(' end')
    #print self.inputFiles
    copyfile(modifiedFile, self.inputFiles)
    self.removeRandomlyNamedFiles(modifiedFile)
   

