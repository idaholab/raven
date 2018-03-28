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

class DecayParser():
  """
    Parses the PHISICS decay library and replaces the nominal values by the perturbed values 
  """
  def matrixPrinter(self,infile,outfile,atomicNumber):
    """
      Prints the perturbed decay matrix in the outfile 
      @ In, infile, string, input file name 
      @ In, outfile, string, output file name 
      @ In, atomicNumber, integer, indicates if the isotope parsed is an actinide (0) or a fission product (1)
      @ Out, None 
    """
    for line in infile :
      line = line.upper().split()
      line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',line[0])   # remove isotope dashes 
      for isotopeID in self.listedDict.iterkeys():
        if line[0] == isotopeID :
          typeOfDecayPerturbed = []
          typeOfDecayPerturbed = self.listedDict.get(isotopeID,{}).keys()
          for i in xrange (0, len(typeOfDecayPerturbed)):
            try :
              if self.isotopeClassifier.get(isotopeID) == self.isotopeParsed[0]:  # it means the isotope is an actinide 
                line[self.decayModeNumbering.get(atomicNumber).get(typeOfDecayPerturbed[i])] = str(self.listedDict.get(isotopeID).get(typeOfDecayPerturbed[i]))
              elif self.isotopeClassifier.get(isotopeID) == self.isotopeParsed[1]:  # it means the isotope is a FP 
                line[self.decayModeNumbering.get(atomicNumber).get(typeOfDecayPerturbed[i])] = str(self.listedDict.get(isotopeID).get(typeOfDecayPerturbed[i]))
            except : 
              raise Exception('you used the decay mode'+str(typeOfDecayPerturbed)+'Check if the decay mode '+str(typeOfDecayPerturbed)+'exist in the decay library. You can also check if you perturbed dictionary is under the format |DECAY|DECAYMODE|ISOTOPEID.')
      if   any('ACTINIDES' in s for s in line)  : 
        flag = self.isotopeParsed[0]
      elif any('FPRODUCTS' in s for s in line)  : 
        flag = self.isotopeParsed[1]
      try :
        if self.isotopeClassifier[line[0]] == atomicNumber :
          line[0] = "{0:<7s}".format(line[0])
          i = 1
          while i <= len(self.decayModeNumbering[atomicNumber]) : 
            line[i] = "{0:<11s}".format(line[i])
            i = i + 1 
          outfile.writelines(' '+''.join(line[0:len(self.decayModeNumbering[atomicNumber]) + 1])+"\n") 
      except KeyError: 
        pass

  def hardcopyPrinter(self,atomicNumber,modifiedFile):
    """
      Prints the hardcopied information at the begining of the xml file
      @ In, atomicNumber, integer, indicates if the isotope parsed is an actinide (0) or a fission product (1)
      @ In, modifiedFile, string, output temperary file name
      @ Out, None 
    """
    flag = 0 
    with open(modifiedFile, 'a') as outfile:
      with open(self.inputFiles) as infile:
        for line in infile:
          if not line.split(): 
            continue   # if the line is blank, ignore it 
          if re.match(r'(.*?)'+atomicNumber+'s',line.strip()) and atomicNumber == self.isotopeParsed[0]:
            flag = 2
          if flag == 2 :
            if re.match(r'(.*?)\s+\w+(\W)\s+\w+(\W)',line) and any(s in 'BETA' for s in line.split()) and atomicNumber == self.isotopeParsed[0] : 
              outfile.writelines(line)
              break
            outfile.writelines(line)
          if any(s in atomicNumber+'roducts' for s in line.split()):
            flag = 1 
          if  flag == 1 :
            if re.match(r'(.*?)\s+\w+(\W)\s+\w+(\W)',line) and any(s in 'BETA' for s in line.split()) and atomicNumber == self.isotopeParsed[1] :
              outfile.writelines(line)
              break
            outfile.writelines(line)
        self.matrixPrinter(infile, outfile, atomicNumber)

  def __init__(self,inputFiles,workingDir,**pertDict):
    """
      @ In, inputFiles, string, Decay library file 
      @ In, workingDir, string, path to the working directory
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """ 
    self.pertDict = pertDict
    self.inputFiles = inputFiles
    for key, value in self.pertDict.iteritems(): 
      self.pertDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    numbering    = {}
    concatenateDecayList = []
    self.allDecayList = []
    OpenInputFile = open (self.inputFiles, "r")
    lines = OpenInputFile.readlines()
    OpenInputFile.close()
    self.inputFiles = inputFiles
    self.isotopeClassifier = {}   # FP or Actinide   
    self.isotopeParsed=['Actinide','FP']
    self.decayModeNumbering = {}
    for line in lines:
      if   re.match(r'(.*?)Actinides', line) : 
        typeOfIsotopeParsed = self.isotopeParsed[0]
      elif re.match(r'(.*?)FProducts', line) : 
        typeOfIsotopeParsed = self.isotopeParsed[1]
      if (re.match(r'(.*?)\w+(\W?)\s+\w+(\W?)\s+\w',line) and any(s in "BETA" for s in line)) : # create dynamic column detector 
        count = 0                            # reset the counter and the dictionary numbering if new colum sequence is detected
        numbering = {}
        decayList = []
        line = re.sub(r'(Yy?)ield',r'',line)          # Remove the word 'yield' in the decay type lines 
        SplitStringDecayType = line.upper().split()   # Split the words into individual strings 
        for i in SplitStringDecayType :               # replace + and * by strings
          decayList.append(i.replace('*', 'S').replace('+','PLUS').replace('_',''))
        concatenateDecayList = concatenateDecayList + decayList  # concatenate all the possible decay type (including repetition among actinides and FP)
        self.allDecayList = list(set(concatenateDecayList))
        for i in xrange(len(decayList)) :
          count = count + 1
          numbering[decayList[i]] = count   # assign the column position of the given decay types
        if typeOfIsotopeParsed == self.isotopeParsed[0]: 
          self.decayModeNumbering[self.isotopeParsed[0]] = numbering
        if typeOfIsotopeParsed == self.isotopeParsed[1]: 
          self.decayModeNumbering[self.isotopeParsed[1]] = numbering
      if re.match(r'(.*?)\D+(-?)\d+(M?)\s+\d', line):
        SplitString = line.upper().split()  
        for i, x in enumerate(SplitString):
          try:
            SplitString[i] = float(x)   
          except ValueError:
            pass
        SplitString[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',SplitString[0])   # remove the dash if it the key (isotope ID) contains it
        if typeOfIsotopeParsed == self.isotopeParsed[0]: 
          self.isotopeClassifier[SplitString[0]] = self.isotopeParsed[0]
        elif typeOfIsotopeParsed == self.isotopeParsed[1]: 
          self.isotopeClassifier[SplitString[0]] = self.isotopeParsed[1]
    self.fileReconstruction()
    self.printInput(workingDir)

  def fileReconstruction(self):
    """
      Converts the formatted dictionary pertdict -> {'DECAY|ALPHA|U235':1.30} 
      into a dictionary of dictionaries that has the format -> {'DECAY':{'ALPHA':{'U235'1.30}}}
      @ In, None  
      @ Out, None 
    """
    self.listedDict = {}
    perturbedIsotopes = []
    for i in self.pertDict.iterkeys() :
      splittedDecayKeywords = i.split('|')
      perturbedIsotopes.append(splittedDecayKeywords[2])
    for i in xrange (0,len(perturbedIsotopes)):
      self.listedDict[perturbedIsotopes[i]] = {}  
    for decayTypeKey, decayValue in self.pertDict.iteritems():
      decayKeyWords = decayTypeKey.split('|')
      for i in xrange (0, len(self.allDecayList)):
        self.listedDict[decayKeyWords[2]][decayKeyWords[1]] = decayValue 
   
  def printInput(self,workingDir):
    """
      Prints out the pertubed Qvalues library into a file
      @ In, workingDir, string, path to working directory
      @ Out, None
    """
    modifiedFile = os.path.join(workingDir,'test.dat')
    open(modifiedFile, 'w')
    for atomicNumber in self.isotopeParsed:
      self.hardcopyPrinter(atomicNumber, modifiedFile)
    with open(modifiedFile, 'a') as outfile:
      outfile.writelines(' end')
    os.rename(modifiedFile,self.inputFiles)
   

