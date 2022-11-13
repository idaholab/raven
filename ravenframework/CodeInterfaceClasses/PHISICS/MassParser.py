"""
Created on May 10th 2018
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import re
from decimal import Decimal


class MassParser():
    """
    Parses the MRTAU mass input file and replaces the masses with the perturbed values. This class is only used in MRTAU standalone cases.
  """

    def __init__(self, inputFiles, workingDir, **pertDict):
        """
      Constructor.
      @ In, inputFiles, string, mass file.
      @ In, workingDir, string, path to working directory
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
        self.listedDict = {}
        self.inputFiles = inputFiles
        self.pertDict = self.scientificNotation(pertDict)
        self.listedDict = self.fileReconstruction(self.pertDict)
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

    def matrixPrinter(self, line, outfile):
      """
        Prints the perturbed mass matrix in the outfile.
        @ In, infile, file object, input file in file object format
        @ In, outfile, file object, output file in file object format
        @ Out, None
      """
      line = line.upper().split()
      line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)', r'\1\2\4', line[0])
      for isotopeID in self.listedDict['MASS'].keys():
        if line[0] == isotopeID:
          try:
            line[2] = str(
                self.listedDict.get('MASS').get(isotopeID))
          except KeyError:
            raise KeyError(
                'Error. Check if the unperturbed library has defined values relative to the requested perturbed isotopes'
            )
      if len(line) > 1:
        outfile.writelines(
            ' ' + "{0:<7s}".format(line[0]) + "{0:<7s}".format(
                line[1]) + ' ' + "{0:<7s}".format(line[2]) + "\n")

    def hardcopyPrinter(self, lines):
        """
      Prints the hardcopied information at the begining of the file.
      @ In, modifiedFile, string, output temperary file name
      @ Out, None
    """
        with open(self.inputFiles, 'a+') as outfile:
          for line in lines:
            # if the line is blank, ignore it
            if not line.split():
              continue
            if re.match(r'(.*?)\s+\w+(-?)\d+\s+\d+.\d+', line):
              self.matrixPrinter(line, outfile)
            else:
              outfile.writelines(line)

    def fileReconstruction(self, deconstructedDict):
        """
      Converts the formatted dictionary -> {'MASS|U235':1.30, MASS|PU241':4.69}.
      into a dictionary of dictionaries that has the format -> {'MASS':{'U235':1.30, 'PU241':4.69}}
      @ In, deconstructedDict, dictionary, dictionary of perturbed variables
      @ Out, reconstructedDict, dictionary, nested dictionary of perturbed variables
    """
        reconstructedDict = {}
        perturbedIsotopes = []
        perturbedMaterials = []
        perturbedPhysicalParameters = []
        for key in deconstructedDict.keys():
            perturbedIsotopes.append(key.split('|')[1])
            perturbedPhysicalParameters.append(key.split('|')[0])
        for i in range(len(perturbedPhysicalParameters)):
            reconstructedDict[perturbedPhysicalParameters[i]] = {}
            for j in range(len(perturbedIsotopes)):
                reconstructedDict[perturbedPhysicalParameters[i]][
                    perturbedIsotopes[j]] = {}
        for typeKey, value in deconstructedDict.items():
            keyWords = typeKey.split('|')
            reconstructedDict[keyWords[0]][keyWords[1]] = value
        return reconstructedDict

    def printInput(self, workingDir):
      """
        Prints out the pertubed masses file into a .dat file. The workflow is:
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
