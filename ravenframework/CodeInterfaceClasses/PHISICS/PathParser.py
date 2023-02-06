"""
Created on July 17th, 2017
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import re
from decimal import Decimal


class PathParser():
  """
    Parses the PHISICS decay Qvalue library located in the path folder (betadecay, alphadecay etc.)
    and replaces the nominal values by the perturbed values.
  """

  def __init__(self, inputFiles, workingDir, **pertDict):
    """
      Constructor.
      @ In, inputFiles, string, decay Qvalue library file
      @ In, workingDir, string, absolute path to working directory
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.endStringCounter = 0  # counts how many times 'END' occurs
    self.harcodingSection = 0  # 0 if portions of files that will not be perturbed, one if variables can be perturbed
    self.listedQValuesDict = {}
    self.inputFiles = inputFiles
    self.pertQValuesDict = self.scientificNotation(
        pertDict)  # Perturbed variables
    self.fileReconstruction()  # Puts the perturbed variables in a dictionary
    self.printInput(
        workingDir
    )  # Replaces the nom. values by the perturbed one and prints in a file

  def scientificNotation(self, pertDict):
    """
      Converts the numerical values into a scientific notation.
      @ In, pertDict, dictionary, perturbed variables
      @ Out, pertDict, dictionary, perturbed variables in scientific format
    """
    for key, value in pertDict.items():
      pertDict[key] = '%.3E' % Decimal(str(value))
    return pertDict

  def matrixPrinter(self, line, outfile):
    """
      Prints the perturbed decay matrix in the outfile.
      @ In, line, file object, input file in file object format
      @ In, outfile, file object, output file in file object format
      @ Out, None
    """
    line = re.sub(r'(.*?)(\w+)(-)(\d+M?)', r'\1\2\4', line)
    line = line.upper().split()
    if line[0] in self.setOfPerturbedIsotopes:
      try:
        line[1] = str(self.listedQValuesDict.get(line[0]))
      except KeyError:
        raise KeyError(
            'Error. Check if the unperturbed library has defined values relative to the requested perturbed isotopes'
        )
    if len(line) > 1:
      line[0] = "{0:<7s}".format(line[0])
      line[1] = "{0:<11s}".format(line[1])
      line = ''.join(line[0] + line[1] + "\n")
      outfile.writelines(line)
    if re.search(r'(.*?)END', line[0]):
      self.endStringCounter = self.endStringCounter + 1
      self.harcodingSection = 0

  def fileReconstruction(self):
    """
      Converts the formatted dictionary -> {'XXXDECAY|U235':1.30, XXXDECAY|FUEL2|U238':4.69}.
      into a dictionary of dictionaries that has the format -> {'XXXDECAY':{'U235':1.30}, 'XXXDECAY':{'U238':4.69}}
      @ In, None
      @ Out, reconstructedDict, nested dictionary
    """
    perturbedIsotopes = []
    for key in self.pertQValuesDict.keys():
      perturbedIsotopes.append(key.split('|')[1])
    for perturbedIsotope in perturbedIsotopes:
      self.listedQValuesDict[perturbedIsotope] = {
      }  # declare all the dictionaries
    for isotopeKeyName, QValue in self.pertQValuesDict.items():
      isotopeName = isotopeKeyName.split('|')
      self.listedQValuesDict[isotopeName[1]] = QValue
    self.setOfPerturbedIsotopes = set(self.listedQValuesDict.keys())

  def printInput(self, workingDir):
    """
      Prints out the pertubed decay qvalue file into a .dat file. The workflow is:
      open a new file with a dummy name; parse the unperturbed library; print the line in the dummy and
      replace with perturbed variables if necessary, Change the name of the dummy file.
      @ In, workingDir, string, path to working directory
      @ Out, None
    """
    # open the unperturbed file
    with open(self.inputFiles, "r") as openInputFile:
      lines = openInputFile.readlines()

    # remove the file if was already existing
    if os.path.exists(self.inputFiles):
      os.remove(self.inputFiles)
    sectionCounter = 0
    with open(self.inputFiles, 'a+') as outfile:
      for line in lines:
        if re.search(r'(.*?)(\s?)[a-zA-Z](\s+Qvalue)', line.strip()):
          sectionCounter = sectionCounter + 1
        if not line.split():
          continue  # if the line is blank, ignore it
        if sectionCounter == 1 and self.endStringCounter == 0:  #actinide section
          self.harcodingSection = 1
          self.matrixPrinter(line, outfile)
        if sectionCounter == 2 and self.endStringCounter == 1:  #FP section
          self.harcodingSection = 2
          self.matrixPrinter(line, outfile)
        if self.harcodingSection != 1 and self.harcodingSection != 2:
          outfile.writelines(line)
