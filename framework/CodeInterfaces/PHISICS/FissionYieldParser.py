"""
Created on June 25th, 2017
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
import os
import re
from decimal import Decimal


class FissionYieldParser():
  """
    Parses the PHISICS xml fission yield file and replaces the nominal values by the perturbed values.
  """

  def __init__(self, inputFiles, workingDir, **pertDict):
    """
      Constructor.
      @ In, inputFiles, string, xml fission yield file.
      @ In, workingDir, string, absolute path to the working directory
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.allYieldList = [
    ]  # Contains all the fission yield families in fast and thermal spectrum
    self.inputFiles = inputFiles
    self.spectrum = [
        'Thermal', 'Fast'
    ]  # Possible spectrum found in the library. Each of them have an independent library
    self.typeOfSpectrum = None  # Flag. Takes the value of one of the possible spectrum, depending what line of the file is parsed
    self.isotopeList = []  # Fission products having a fission yield defined
    self.spectrumNumbering = {
    }  # Keys: type of spectrum (fast or theraml), values: numbering dictionary
    self.listedYieldDict = {}  # Nested dictionary of perturbed variables

    self.pertYieldDict = self.scientificNotation(
        pertDict)  # Perturbed variables
    self.characterizeLibrary()
    self.isotopeList = list(set(
        self.isotopeList))  # Removes all the repetion in the isotope list
    self.numberOfIsotopes = len(self.isotopeList)
    self.fileReconstruction()  # Puts the perturbed variables in a dictionary
    self.printInput(
        workingDir
    )  # Replaces the the nominal values by the perturbed one and print in a file

  def scientificNotation(self, pertDict):
    """
      Converts the numerical values into a scientific notation.
      @ In, pertDict, dictionary, perturbed variables
      @ Out, pertDict, dictionary, perturbed variables in scientific format
    """
    for key, value in pertDict.iteritems():
      pertDict[key] = '%.3E' % Decimal(str(value))
    return pertDict

  def characterizeLibrary(self):
    """
      Characterizes the structure of the library. Teaches the type of decay available for the fast spectrum family and thermal family.
      @ In, None
      @ Out, None
    """
    concatenateYieldList = []
    OpenInputFile = open(self.inputFiles, "r")
    lines = OpenInputFile.readlines()
    OpenInputFile.close()
    for line in lines:
      if re.match(r'(.*?)Thermal Fission Yield', line):
        self.typeOfSpectrum = self.spectrum[0]
      elif re.match(r'(.*?)Fast Fission Yield', line):
        self.typeOfSpectrum = self.spectrum[1]
      if (re.match(r'(.*?)\w+(-?)\d+\s+\w+\s+\w(-?)\d+\s+\w', line)
          and any(s in "FY" for s in line)):  # create dynamic column detector
        count = 0
        FYgroup = [
        ]  # reset the counter and the dictionary numbering if new colum sequence is detected
        numbering = {}
        line = re.sub(r'FY', r'', line)
        splitStringYieldType = line.upper().split()
        for i in splitStringYieldType:
          FYgroup.append(
              i.replace('-', '')
          )  # get the fission yield group's names (U235, Pu239 etc.) and remove the dash in those IDs
        concatenateYieldList = concatenateYieldList + FYgroup  # concatenate all the possible yield type (including repetition among actinides and FP)
        self.allYieldList = list(set(concatenateYieldList))

        for i in range(
            len(splitStringYieldType
                )):  # assign the column position of the given yield types
          count = count + 1
          numbering[FYgroup[
              i]] = count  # assign the column position of the given Yield types
          splitStringYieldType[i] = re.sub(r'(.*?)(\w+)(-)(\d+M?)', r'\1\2\4',
                                           splitStringYieldType[i])
          if self.typeOfSpectrum == self.spectrum[0]:
            self.spectrumNumbering[self.spectrum[0]] = numbering
          if self.typeOfSpectrum == self.spectrum[1]:
            self.spectrumNumbering[self.spectrum[1]] = numbering
          numbering[splitStringYieldType[i]] = count
      if re.match(r'(.*?)\s+\D+(-?)\d+(M?)\s+\d+.\d', line) or re.match(
          r'(.*?)ALPHA\s+\d+.\d', line):
        isotopeLines = line.split()
        self.isotopeList.append(isotopeLines[0])

  def matrixPrinter(self, infile, outfile, spectra):
    """
      Prints the perturbed decay matrix in the outfile.
      @ In, infile, file object, input file in file object format
      @ In, outfile, file object, output file in file object format
      @ In, spectra, integer, indicates if the yields are related to a thermal spectrum (0) or a fast spectrum (1)
      @ Out, None
    """
    isotopeCounter = 0
    for line in infile:
      if re.search(r'END\s+', line):
        return
      line = line.strip()
      if not line:
        continue
      line = line.upper().split()
      line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)', r'\1\2\4',
                       line[0])  # remove the dashes in isotope names
      spectraUpper = spectra.upper()
      try:
        for fissionProductID in self.listedYieldDict[spectraUpper].iterkeys():
          for actinideID in self.listedYieldDict[spectraUpper][
              fissionProductID].iterkeys():
            if line[0] == fissionProductID:
              typeOfYieldPerturbed = []
              self.spectrumUpperCase = [x.upper() for x in self.spectrum]
              typeOfYieldPerturbed = self.listedYieldDict.get(
                  spectraUpper).get(fissionProductID).keys()
              for i in range(len(typeOfYieldPerturbed)):
                try:
                  if self.listedYieldDict.get(spectraUpper).get(
                      fissionProductID).get(typeOfYieldPerturbed[i]) != {}:
                    line[self.spectrumNumbering.get(spectra).get(
                        typeOfYieldPerturbed[i])] = str(
                            self.listedYieldDict.get(spectraUpper).get(
                                fissionProductID).get(typeOfYieldPerturbed[i]))
                except TypeError:
                  raise Exception(
                      'Make sure the fission yields you are perturbing have existing values in the unperturbed fission yield library'
                  )
      except KeyError:
        pass  # pass you pertub 'FAST': {u'ZN67': {u'U235': '5.659E+00'}} only, the case 'THERMAL': {u'ZN67': {u'U235': '5.659E+00'}} ignored in the line for fissionProductID in self.listedYieldDict[spectraUpper].iterkeys() (because non existent)
      try:
        isotopeCounter = isotopeCounter + 1
        line[0] = "{0:<7s}".format(line[0])
        i = 1
        while i <= len(
            self.spectrumNumbering.get(spectra)
        ):  # while i is smaller than the number of columns that represents the number of fission yield families
          try:
            line[i] = "{0:<11s}".format(line[i])
            i = i + 1
          except IndexError:
            i = i + 1
        outfile.writelines(' ' + ''.join(
            line[0:len(self.spectrumNumbering.get(spectra)) + 1]) + "\n")
        if isotopeCounter == self.numberOfIsotopes:
          for lineInput in infile:
            lineStripped = lineInput.strip()
      except KeyError:
        raise Exception(
            'Make sure the fission yields you are perturbing have existing values in the unperturbed fission yield library'
        )

  def hardcopyPrinter(self, spectra, modifiedFile):
    """
      Prints the hardcopied information at the begining of the xml file.
      @ In, spectra, integer, indicates if the yields are related to a thermal spectrum (0) or a fast spectrum (1)
      @ In, modifiedFile, string, output temperary file name
      @ Out, None
    """
    flag = 0
    with open(modifiedFile, 'a') as outfile:
      with open(self.inputFiles) as infile:
        for line in infile:
          if re.match(
              r'(.*?)END\s+\w+', line.strip()
          ) and spectra == self.spectrum[1]:  # find the line- END Fast Fission Yield (2)
            flag = 2
          if flag == 2:
            if re.match(r'(.*?)\w+(-?)\d+\s+\w+\s+\w(-?)\d+\s+\w',
                        line.strip()) and spectra == self.spectrum[1]:
              outfile.writelines(line)
              break
            outfile.writelines(line)
          if (re.match(r'(.*?)' + spectra, line.strip())
              and spectra == self.spectrum[0]
              ):  # find the line- Thermal Fission Yield (1)
            flag = 1
          if flag == 1:
            if re.match(
                r'(.*?)\w+(-?)\d+\s+\w+\s+\w(-?)\d+\s+\w', line.strip()
            ) and spectra == self.spectrum[0]:  # find the line U-235 FY U-238 FY (last hardcopied line)
              outfile.writelines(line)
              flag = 0
              break
            outfile.writelines(line)
        self.matrixPrinter(infile, outfile, spectra)
      outfile.close()

  def fileReconstruction(self):
    """
      Converts the formatted dictionary pertdict -> {'FY|THERMAL|U235|XE135':1.30}.
      into a dictionary of dictionaries that has the format -> {'FY':{'THERMAL':{'U235':{'XE135':1.30}}}}
      @ In, None
      @ Out, None
    """
    fissioningActinide = []
    resultingFP = []
    spectrumType = []
    for key in self.pertYieldDict.iterkeys():
      splittedYieldKeywords = key.split('|')
      spectrumType.append(splittedYieldKeywords[1])
      fissioningActinide.append(splittedYieldKeywords[2])
      resultingFP.append(splittedYieldKeywords[3])
    for i in range(len(spectrumType)):
      self.listedYieldDict[spectrumType[i]] = {}
      for j in range(len(resultingFP)):
        self.listedYieldDict[spectrumType[i]][resultingFP[j]] = {
        }  # declare all the dictionaries
        for k in range(len(fissioningActinide)):
          self.listedYieldDict[spectrumType[i]][resultingFP[j]][
              fissioningActinide[k]] = {}
    for yieldTypeKey, yieldValue in self.pertYieldDict.iteritems():
      self.listedYieldDict[yieldTypeKey.split('|')[1]][yieldTypeKey.split('|')[
          3]][yieldTypeKey.split('|')[2]] = yieldValue

  def printInput(self, workingDir):
    """
      Prints out the pertubed fission yield library into a .dat file. The workflow is:
      open a new file with a dummy name; parse the unperturbed library; print the line in the dummy and
      replace with perturbed variables if necessary, Change the name of the dummy file.
      @ In, workingDir, string, path to working directory
      @ Out, None
    """
    modifiedFile = os.path.join(workingDir, 'test.dat')
    open(modifiedFile, 'w')
    for spectra in self.spectrum:
      self.hardcopyPrinter(spectra, modifiedFile)
    with open(modifiedFile, 'a') as outfile:
      outfile.writelines(' end')
    os.rename(modifiedFile, self.inputFiles)
