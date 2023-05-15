"""
Created on June 19th, 2017
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import re
from decimal import Decimal


class DecayParser():
  """
    Parses the PHISICS xml decay file and replaces the nominal values by the perturbed values.
  """

  def __init__(self, inputFiles, workingDir, **pertDict):
    """
      Constructor
      @ In, inputFiles, string, .dat Decay file.
      @ In, workingDir, string, absolute path to the working directory
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.allDecayList = []  # All possible types of decay for actinides and FP
    self.isotopeClassifier = {}  # String, FP or Actinide
    self.decayModeNumbering = {}  # Gives the column number of each decay type
    self.isotopeParsed = ['Actinide', 'FP']
    self.listedDict = {}  # Nested dictionary of perturbed variables

    self.inputFiles = inputFiles
    self.pertDict = self.scientificNotation(pertDict)
    # open the unperturbed file
    with open(self.inputFiles, "r") as openInputFile:
      lines = openInputFile.readlines()

    self.characterizeLibrary(lines)
    self.fileReconstruction()
    self.printInput(workingDir,lines)

  def matrixPrinter(self, line, outfile, atomicNumber):
    """
      The xml files is split into two categories: hardcopied lines (banner, column labels etc.) that cannot
      be modified by RAVEN variable definition, and matrix lines that can be modified by RAVEN variable definition.
      This method treats the matrix lines, and print them into the perturbed file.
      @ In, line, list, unperturbed input file line
      @ In, outfile, file object, output file in file object format
      @ In, atomicNumber, integer, indicates if the isotope parsed is an actinide (0) or a fission product (1)
      @ Out, None
    """
    line = line.upper().split()
    line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)', r'\1\2\4', line[0])  # remove isotope dashes
    for isotopeID in self.listedDict.keys():
      if line[0] == isotopeID:
        typeOfDecayPerturbed = []
        typeOfDecayPerturbed = list(self.listedDict.get(isotopeID, {}).keys())
        for i in range(len(typeOfDecayPerturbed)):
          try:
            if self.isotopeClassifier.get(isotopeID) == self.isotopeParsed[0]:  # it means the isotope is an actinide
              line[self.decayModeNumbering.get(atomicNumber).get(typeOfDecayPerturbed[i])] = str(self.listedDict.get(isotopeID).get(typeOfDecayPerturbed[i]))
            elif self.isotopeClassifier.get(isotopeID) == self.isotopeParsed[1]:  # it means the isotope is a FP
              line[self.decayModeNumbering.get(atomicNumber).get(typeOfDecayPerturbed[i])] = str(self.listedDict.get(isotopeID).get(typeOfDecayPerturbed[i]))
          except (KeyError, TypeError):
            raise KeyError('you used the decay mode' + str(typeOfDecayPerturbed) +'Check if the decay mode ' + str(typeOfDecayPerturbed) +'exist in the decay library. You can also check if you perturbed dictionary is under the format |DECAY|DECAYMODE|ISOTOPEID.')
    if any('ACTINIDES' in s for s in line):
      flag = self.isotopeParsed[0]
    elif any('FPRODUCTS' in s for s in line):
      flag = self.isotopeParsed[1]
    try:
      if self.isotopeClassifier[line[0]] == atomicNumber:
        line[0] = "{0:<7s}".format(line[0])
        i = 1
        while i <= len(self.decayModeNumbering[atomicNumber]):
          line[i] = "{0:<11s}".format(line[i])
          i = i + 1
        outfile.writelines(' ' + ''.join(
            line[0:len(self.decayModeNumbering[atomicNumber]) + 1]) + "\n")
    except KeyError:  # happens for all the unperturbed isotopes
      pass

  def hardcopyPrinter(self, atomicNumber, lines):
    """
      The files are split into two categories: hardcopied lines (banner, column labels etc.) that cannot
      be modified by RAVEN variable definition, and matrix lines that can be modified by RAVEN variable definition.
      This method treats the hardcopied lines, and then call the matrix line handler method.
      @ In, atomicNumber, integer, indicates if the isotope parsed is an actinide (0) or a fission product (1)
      @ In, lines, list, unperturbed input file lines
      @ Out, None
    """
    flag = 0
    count = 0
    with open(self.inputFiles, 'a+') as outfile:
      for line in lines:
        # if the line is blank, ignore it
        if not line.split():
          continue
        if re.match(r'(.*?)' + atomicNumber + 's', line.strip()) and atomicNumber == self.isotopeParsed[0]:
          flag = 2 # if the word Actinides is found
          outfile.writelines(line)
        if re.match(r'(.*?)' + atomicNumber + 'roducts', line.strip())and atomicNumber == self.isotopeParsed[1]:
          flag = 1 # if the word FProducts is found
          outfile.writelines(line)
        if flag == 2:
          # for the actinides decay section
          if re.match(r'(.*?)\s+\w+(\W)\s+\w+(\W)', line) and any(
              s in 'BETA' for s in
              line.split()) and atomicNumber == self.isotopeParsed[0] and count == 0:
            count = count + 1
            outfile.writelines(line)
          self.matrixPrinter(line, outfile, atomicNumber)
        if flag == 1:
          #for the FP decay section
          if re.match(r'(.*?)\s+\w+(\W)\s+\w+(\W)', line) and any(
              s in 'BETA' for s in
              line.split()) and atomicNumber == self.isotopeParsed[1]:
            outfile.writelines(line)
          self.matrixPrinter(line, outfile, atomicNumber)

  def characterizeLibrary(self,lines):
    """
      Characterizes the structure of the library. Teaches the type of decay available for the actinide family and FP family.
      @ In, lines, list, unperturbed input file lines
      @ Out, None
    """
    concatenateDecayList = []

    for line in lines:
      if re.match(r'(.*?)Actinides', line):
        typeOfIsotopeParsed = self.isotopeParsed[0]
      elif re.match(r'(.*?)FProducts', line):
        typeOfIsotopeParsed = self.isotopeParsed[1]
      if (
          re.match(r'(.*?)\w+(\W?)\s+\w+(\W?)\s+\w', line)
          and any(s in "BETA" for s in line)
      ):  # create dynamic column detector, the search for 'BETA' ensures this is the label line.
        count = 0  # reset the counter and the dictionary numbering if new colum sequence is detected
        numbering = {}
        decayList = []
        line = re.sub(r'(Yy?)ield', r'',
                      line)  # Remove the word 'yield' in the decay type lines
        splitStringDecayType = line.upper().split(
        )  # Split the words into individual strings
        for decayType in splitStringDecayType:  # replace + and * by strings
          decayList.append(
              decayType.replace('*', 'S').replace('+', 'PLUS').replace(
                  '_', ''))
        concatenateDecayList = concatenateDecayList + decayList  # concatenate all the possible decay type (including repetition among actinides and FP)
        self.allDecayList = list(set(concatenateDecayList))
        for i in range(len(decayList)):
          count = count + 1
          numbering[decayList[
              i]] = count  # assign the column position of the given decay types
        if typeOfIsotopeParsed == self.isotopeParsed[0]:
          self.decayModeNumbering[self.isotopeParsed[0]] = numbering
        if typeOfIsotopeParsed == self.isotopeParsed[1]:
          self.decayModeNumbering[self.isotopeParsed[1]] = numbering
      if re.match(r'(.*?)\D+(-?)\d+(M?)\s+\d', line):
        splitString = line.upper().split()
        for i, decayConstant in enumerate(splitString):
          try:
            splitString[i] = float(decayConstant)
          except ValueError:
            pass  # the element is a string (isotope tag). It can be ignored
        splitString[0] = re.sub(
            r'(.*?)(\w+)(-)(\d+M?)', r'\1\2\4', splitString[
                0])  # remove the dash if it the key (isotope ID) contains it
        if typeOfIsotopeParsed == self.isotopeParsed[0]:
          self.isotopeClassifier[splitString[0]] = self.isotopeParsed[0]
        elif typeOfIsotopeParsed == self.isotopeParsed[1]:
          self.isotopeClassifier[splitString[0]] = self.isotopeParsed[1]

  def scientificNotation(self, pertDict):
    """
      Converts the numerical values into a scientific notation.
      @ In, pertDict, dictionary, perturbed variables
      @ Out, pertDict, dictionary, perturbed variables in scientific format
    """
    for key, value in pertDict.items():
      pertDict[key] = '%.3E' % Decimal(str(value))
    return pertDict

  def fileReconstruction(self):
    """
      Converts the formatted dictionary pertdict -> {'DECAY|ALPHA|U235':1.30}.
      into a dictionary of dictionaries that has the format -> {'DECAY':{'ALPHA':{'U235'1.30}}}
      @ In, None
      @ Out, None
    """
    perturbedIsotopes = []
    for key in self.pertDict.keys():
      splittedDecayKeywords = key.split('|')
      perturbedIsotopes.append(splittedDecayKeywords[2])
    for i in range(len(perturbedIsotopes)):
      self.listedDict[perturbedIsotopes[i]] = {}
    for decayTypeKey, decayValue in self.pertDict.items():
      decayKeyWords = decayTypeKey.split('|')
      for i in range(len(self.allDecayList)):
        self.listedDict[decayKeyWords[2]][decayKeyWords[1]] = decayValue

  def printInput(self, workingDir,lines):
    """
      Prints out the pertubed decay library into a file. The workflow is:
      Open a new file with a dummy name; parse the unperturbed library; print the line in the dummy,
      replace with perturbed variables if necessary. Change the name of the dummy file.
      @ In, workingDir, string, path to working directory
      @ In, lines, list, unperturbed input file lines
      @ Out, None
    """
    if os.path.exists(self.inputFiles):
      os.remove(self.inputFiles) # remove the file if was already existing
    for atomicNumber in self.isotopeParsed:
      self.hardcopyPrinter(atomicNumber, lines)
    with open(self.inputFiles, 'a') as outfile:
      outfile.writelines(' end')
