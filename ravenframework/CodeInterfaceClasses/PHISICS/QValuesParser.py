"""
Created on June 19th, 2017
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import re
from decimal import Decimal


class QValuesParser():
  """
    Parses the PHISICS Qvalues library and replaces the nominal values by the perturbed values.
  """

  def __init__(self, inputFiles, workingDir, **pertDict):
    """
      Constructor.
      @ In, inputFiles, string, Qvalues library file.
      @ In, workingDir, string, path to working directory
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.listedQValuesDict = {}
    self.inputFiles = inputFiles
    self.pertQValuesDict = self.scientificNotation(pertDict)
    self.fileReconstruction()
    self.printInput(workingDir)

  def scientificNotation(self, pertDict):
    """
      Converts the numerical values into a scientific notation.
      @ In, pertDict, dictionary, perturbed variables
      @ Out, pertDict, dictionary, perturbed variables in scientific format
    """
    for key, value in pertDict.items():
      pertDict[key] = '%.3E' % Decimal(str(value))
    return pertDict

  def matrixPrinter(self, lines, outfile):
    """
      Prints the perturbed Qvalues matrix in the outfile.
      @ In, lines, list, unperturbed input file lines
      @ In, outfile, file object, output file in file object format
      @ Out, None
    """
    for line in lines:
      line = line.upper().split()
      line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)', r'\1\2\4', line[0])
      for isotopeID in self.listedQValuesDict.keys():
        if line[0] == isotopeID:
          try:
            line[1] = str(self.listedQValuesDict.get(isotopeID))
          except KeyError:
            raise KeyError(
                'Error. Check if the unperturbed library has defined values relative to the requested perturbed isotopes'
            )
      if len(line) > 1:
        outfile.writelines(
            ' ' + "{0:<7s}".format(line[0]) + "{0:<7s}".format(line[1] + "\n"))

  def hardcopyPrinter(self, lines):
    """
      Prints the hardcopied information at the begining of the xml file.
      @ In, lines, list, unperturbed input file lines
      @ Out, None
    """
    with open(self.inputFiles, 'a+') as outfile:
      for line in lines:
        if not line.split():
          continue  # if the line is blank, ignore it
        if re.match(r'(.*?)\s+\w+(-?)\d+\s+\d+.\d+', line):
          outfile.writelines(
              ' ' + "{0:<7s}".format(line.upper().split()[0]) +
              "{0:<7s}".format(line.upper().split()[1]) +
              "\n")  # print the first fission qvalue line of the matrix
          break
        outfile.writelines(line)
      self.matrixPrinter(lines, outfile)
      outfile.writelines(' END')

  def fileReconstruction(self):
    """
      Converts the formatted dictionary -> {'QVALUES|U235':1.30}
      into a dictionary of dictionaries that has the format -> {'QVALUES':{'U235':1.30}}
      @ In, None
      @ Out, None
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

  def printInput(self, workingDir):
    """
      Prints out the pertubed fission qvalue file into a .dat file. The workflow is:
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
    self.hardcopyPrinter(lines)
