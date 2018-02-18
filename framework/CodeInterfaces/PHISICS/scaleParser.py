"""
Created on October 23rd 2017
@author: rouxpn
"""

import os
import sys
import re
from shutil import copyfile
import fileinput
from decimal import Decimal
import time
import xml.etree.ElementTree as ET
from random import *
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom
from xml.dom.ext import PrettyPrint
from StringIO import StringIO

class scaleParser():
  """
    parses the NEWT scale output. The parser looks for the disadvantage factors
    and the mixture scalar fluxes. It then prints those values in the Xs-Library.xml, which is
    one of the PHISICS input.
  """

  def getNumberOfGroups(self):
    with open(self.inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'Number of broad groups in collapsed set',line.strip()):
          return line.split()[-1]

  def __init__(self, inputFiles):
    """
      Parse the scale.out data to get the disadvantage factor and mixture
      scalar fluxes
      IN: scale.out (scale/NEWT output file)
      OUT: None
    """
    disadvantageFactors = {}
    self.inputFile = inputFiles
    self.outputFile = 'Xs-Library.xml'
    #print self.inputFile
    self.numberOfGroups = self.getNumberOfGroups()
    disadvantageFactorsDict = self.getDisadvantageFactors()
    mixScalarFluxDict = self.getMixScalarFlux()
    self.generateXML(disadvantageFactorsDict, mixScalarFluxDict)

  def isNumber(self, line):
    """
      check if a string is an integer
    """
    try:
      int(line[0])
      return True
    except ValueError:
      return False

  def getDisadvantageFactors(self):
    """
      parses the scale/NEWT output to get the disadvantage factors
      IN: None
      OUT: disadvantageFactorsDict, dictionary
    """
    disadvantageFactorsDict = {}
    startFlag = 0
    endFlag = 0
    with open(self.inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'Flux Disadvantage Factors',line.strip()):
          startFlag = 1
        if startFlag == 1 and re.search(r'\s+\d+\s+', line):
          lineSplit = line.split()
          stringIsNumber = self.isNumber(lineSplit)
          if stringIsNumber is True:
            disadvantageFactorsDict[lineSplit[0]] = {}
            for i in xrange(1,int(self.numberOfGroups)+1):
              disadvantageFactorsDict[lineSplit[0]][i] = lineSplit[i]
        if startFlag == 1 and re.search(r'\s+Total\s+', line):
          break
    #print disadvantageFactorsDict
    return disadvantageFactorsDict

  def getMixScalarFlux(self):
    """
      parses the scale/NEWT output to get the scalar flux in each mixture
      IN: None
      OUT: mixScalarFluxDict, dictionary
    """
    mixScalarFluxDict = {}
    startFlag = 0
    endFlag = 0
    with open(self.inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'Cell Averaged Fluxes',line.strip()):
          startFlag = 1
        if startFlag == 1 and re.search(r'\s+\d+\s+', line):
          lineSplit = line.split()
          stringIsNumber = self.isNumber(lineSplit)
          if stringIsNumber is True:
            mixScalarFluxDict[lineSplit[0]] = {}
            for i in xrange(1,int(self.numberOfGroups)+1):
              mixScalarFluxDict[lineSplit[0]][i] = lineSplit[i]
        if startFlag == 1 and re.search(r'\s+Total\s+', line):
          break
    #print mixScalarFluxDict
    return mixScalarFluxDict

  def prettify(self, elem):
    """
      Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    uglyXml = reparsed.toprettyxml(indent="\t")
    pattern = re.compile('>\n\s+([^<>\s].*?)\n\s+</', re.DOTALL)
    return pattern.sub('>\g<1></', uglyXml)
    #return reparsed.toprettyxml(indent="  "

  def generateXML(self, disadvantageFactorsDict, mixScalarFluxDict):
    """
      print the values in an XML-formatted file
      IN: disadvantageFactorsDict, dictionary
      IN: mixScalarFluxDict, dictionary
      Out: mixture.xml (xml file)
    """
    copyOut = 'lib_copy.xml'
    copyfile(self.outputFile, copyOut)
    tree = ET.parse(copyOut)
    root = tree.getroot()
    for child in root.findall(".//POINT"):
      for mixtureNumber in disadvantageFactorsDict.iterkeys():
        #print mixtureNumber
        disadvantageFactorList = []
        topChildDF = SubElement(child, 'disadvantage_factors', {'mix':mixtureNumber})
        for groupNumber in xrange (1,int(self.numberOfGroups)+1):
          disadvantageFactorList.append(disadvantageFactorsDict.get(mixtureNumber).get(groupNumber))
          values = ' '.join(str(f) for f in disadvantageFactorList)
          if groupNumber == int(self.numberOfGroups):
            topChildDF.text = values
            #root.append(topChildDF)
      for mixtureNumber in mixScalarFluxDict.iterkeys():
        scalarFluxList = []
        topChildScalarFlux = SubElement(child, 'scalar_flux', {'mix':mixtureNumber})
        for groupNumber in xrange (1,int(self.numberOfGroups)+1):
          scalarFluxList.append(mixScalarFluxDict.get(mixtureNumber).get(groupNumber))
          values = ' '.join(str(f) for f in scalarFluxList)
          if groupNumber == int(self.numberOfGroups):
            topChildScalarFlux.text = values
    tree.write(copyOut)

    with open(copyOut, 'w') as output:
      output.write(self.prettify(root))
    # remove the annoying blank lines
    with open(copyOut) as infile:
      with open(self.outputFile, 'w') as outfile:
        for line in infile:
          if not line.strip(): continue
          outfile.write(line)

scaleParser = scaleParser('scale.out')
