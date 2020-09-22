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

class scaleParser():
  """
    parses the NEWT scale output. The parser looks for the disadvantage factors
    and the mixture scalar fluxes. It then prints those values in the Xs-Library.xml, which is
    one of the PHISICS input.
  """
  def __init__(self):
    """
      Parse the scale.out data to get the disadvantage factor and mixture
      scalar fluxes
      @ In, None
      @ Out, None
    """
    disadvantageFactorDict = {}
    # input flag
    inputFile
    # lib flag
    self.outputFile

    for inp in xrange (0, len(inputFile)):
      disadvantageFactorDict[inputFile[inp]] = {}
      self.numberOfGroups         = self.getNumberOfGroups(inputFile[inp])
      isItDoubleHet, doubleHetMix,txsecInp = self.isTheSystemDoubleHetIfYesGiveMeMixNumber(inputFile[inp])
      if isItDoubleHet is True and txsecInp is False:
        doubleHetMixList = self.doubleHetMixtures(doubleHetMix, inputFile[inp])
        print doubleHetMixList
      if isItDoubleHet is True and txsecInp is not False:  # if txsecInp is not false, it is equal to txsec file name
        doubleHetMixList = self.doubleHetMixtures(doubleHetMix, txsecInp)
      if isItDoubleHet is False:
        doubleHetMixList = None
      homogenizedMixtures         = self.findHomogBlock(inputFile[inp])
      mixtureVolumesDict          = self.getMixtureVolumes(homogenizedMixtures, inputFile[inp])
      volumeOfHomMix              = self.getTotalVolume(mixtureVolumesDict)
      mixScalarFluxDict           = self.getMixScalarFlux(inputFile[inp])
      disadvantageFactorsDict     = self.calculateDisadvantageFactors(homogenizedMixtures, mixtureVolumesDict, mixScalarFluxDict, volumeOfHomMix, inputFile[inp], disadvantageFactorDict)
    self.generateXML(disadvantageFactorsDict, mixScalarFluxDict, isItDoubleHet, doubleHetMix, doubleHetMixList, inputFile)

  def getTotalVolume(self, mixtureVolumesDict):
    """
      calculate the total volume of the mixtures utilized in the homogenized mixture
      In: mixtureVolumesDict, dictionary
      Out: totalVolume, float number
    """
    volumeOfHomMix = 0
    for volMix, volValue in mixtureVolumesDict.iteritems():
      volumeOfHomMix = float(volValue) + float(volumeOfHomMix)
    print 'Volume Cell (per unit length)'
    print volumeOfHomMix
    return volumeOfHomMix

  def getMixtureVolumes(self, homogenizedMixtures, inputFile):
    """
      find the mixture volume in the scale output. The mixture volumes are used to calculate the
      disadvantage factors relative to the region homogenizedMixtures. The volumes are in cubic cm
      per unit length (NEWT is a 2D code)
      In: None
      Out: mixtureVolumes
    """
    startFlag = 0
    endFlag = 0
    mixtureVolumesDict = {}
    with open(inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'Mixture\s+Volume\s+Volume',line.strip()):
          startFlag = 1
        if startFlag == 1:
          volumeBlock = filter(None, line.split(' '))
          if self.isNumber(volumeBlock[0]) is True :
            for i in xrange (0, len(homogenizedMixtures)):
                if volumeBlock[0] == homogenizedMixtures[i]:
                  mixtureVolumesDict[homogenizedMixtures[i]] = volumeBlock[1]
        if re.search(r'Total\s+\d',line.strip()) and startFlag == 1:
          break
    return mixtureVolumesDict

  def findHomogBlock(self,inputFile):
    """
      parses the NEWT scale output. The parser looks for the disadvantage factors
      and the mixture scalar fluxes. It then prints those values in the Xs-Library.xml, which is
      one of the PHISICS input.
      @ In, inputFile, string, name of the input file (which is the scale output)
      @ Out, homogenizedMixtures, list
    """
    startFlag = 0
    endFlag = 0
    keywords = ['homo', 'hmog']
    homogenizedMixtures = []
    with open(inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'read',line.strip()):
          for keyword in keywords:
            if re.search(r'read\s+'+keyword.lower(),line.strip()):
              startFlag = 1
              break
        if startFlag == 1 and endFlag == 0:
          homogenizationBlock = filter(None, line.split(' '))
          for i in xrange (0, len(homogenizationBlock)):
            if i > 1 and self.isNumber(homogenizationBlock[i]) is True:
              homogenizedMixtures.append(homogenizationBlock[i])
        if re.search(r'end',line.strip()) and startFlag == 1:
          for keyword in keywords:
            if re.search(r'end\s+'+keyword.lower(),line.strip()) and startFlag == 1:
              endFlag = 1
              break
        if endFlag == 1:
          break

    if homogenizedMixtures == []:
      raise ValueError\
      ('Your NEWT input does not have an homogenization block. '\
      'The homogenization block is necessary to specify which mixtures are homogenized')
    #print homogenizedMixtures
    return homogenizedMixtures

  def getNumberOfGroups(self, inputFile):
    """
      Parses the scale output to get the number of groups in the collapsed structure
      @ In, inputFile, string, name of the input file (which is the scale output)
      @ Out: NumberOfGroups, integer
    """
    with open(inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'Number of broad groups in collapsed set',line.strip()):
          return line.split()[-1]

  def doubleHetMixtures(self, doubleHetMix, inputFile):
    """
      if the system is in double het self-shielding treatment, finds the mixtures involved in the double het mix
      @ In, inputFile, string, name of the input file (which is the scale output)
      @ In, doubleHetMix, string, mixture number (integer) of the double het mix
      @ Out, doubleHetMixList, list, list of the mixture included in the double het mix
    """
    startFlag = 0
    doubleHetMixList = []
    with open(inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'read\s+celldata',line.strip(),re.IGNORECASE):
          startFlag = 1
        if re.search(r'matrix(\s?)=',line, re.IGNORECASE):
          line = re.sub('matrix(\s?)=',' ',line)
          line = re.sub('Matrix(\s?)=',' ',line)
        if startFlag == 1:
          for el in line.split():
              try:
                if int(el) != int(doubleHetMix):
                  doubleHetMixList.append(int(el))
              except ValueError:
                pass
        if re.search(r'end\s+grain',line, re.IGNORECASE):
          break
    #print  doubleHetMixList
    return doubleHetMixList

  def newtOrTnewt(self,line):
    """
      Searches if the module newt or tnewt is used. If Newt is used, the scale sequence does not
      know if the fuel is in single or double het. Hence the user has to say so, and which materials
      belong to the fuel,coatings and matrix. If tnewt is used, the material number of the double het can
      be found and no additional information from the user is neeeded.
      @ In, line, string, line from the scale output file
      @ Out, sequence, string, indicates the sequence used, can be 'newt' or 'tnewt'
    """
    if re.search(r'module\s+newt',line):
      sequence = 'newt'
      return sequence
    if re.search(r'module\s+t-newt',line):
      sequence = 'tnewt'
      return sequence

  def isTheSystemDoubleHetIfYesGiveMeMixNumber(self,inputFile):
    """
      Check if the self shielding sequence of NEWT calls the double het treatment
      @ In, inputFile, string, name of the input file (which is the scale output)
      @ Out, True if double het system, false otherwise
      @ Out, line.split()[-2], string, mixture number of the double het mixture (integer in string format)
      @ Out, True if Txsec ius necessary, False if not (in that case, verythin is in tnewt output)
    """

    startFlag = 0
    with open(inputFile, 'r') as infile:
      for line in infile:
        sequence = self.newtOrTnewt(line)
        if sequence == 'tnewt':
          if re.search(r'doublehet',line.strip(), re.IGNORECASE):
            if self.isNumber(line.split()[-2]) is True:
              return  True, line.split()[-2],False
            else:
              return True, line.split()[-2].split('=')[1],False
          if re.search(r'end\s+celldata',line, re.IGNORECASE):
            return False, None, False
    # if the code reaches this point without having returned something,
    # it means the sequence newt standalone is used. txsec has to be parsed
    txsecInp = 'txsec.inp'
    with open(txsecInp, 'r') as f:
      for line in f:
        if re.search(r'doublehet',line.strip(), re.IGNORECASE):
          if self.isNumber(line.split()[-2]) is True:
            return  True, line.split()[-2], txsecInp
          else:
            return True, line.split()[-2].split('=')[1], txsecInp
        if re.search(r'end\s+celldata',line, re.IGNORECASE):
          return False, None, None

  def isNumber(self, line):
    """
      check if a string is an integer
    """
    try:
      int(line[0])
      return True
    except ValueError:
      return False

  def readGroupNumbersOfMixFluxes(self,line):
    """
      in the cell averaged fluxe line, this mehtod reads which groups are being parsed.
      @ In, line, string, string looking like 'Mixture  Group  9      Group 10      Group 11'
      @ Out, groupsInLine, list, list of integers relative to the group numbers of interest
    """
    groupsInLine = []
    line = re.sub(r'Group',r'Group ',line) #add a space to group if there are none
    groupsInLine = line.split(' ')
    groupsInLine = map(lambda s: s.strip(), groupsInLine)
    groupsInLine = [x for x in groupsInLine if x != 'Mixture']
    groupsInLine = [x for x in groupsInLine if x != '']
    groupsInLine = [x for x in groupsInLine if x != 'Group']
    return groupsInLine

  def getMixScalarFlux(self, inputFile):
    """
      parses the scale/NEWT output to get the scalar flux in each mixture
      @ In, inputFile, string, name of the input file (which is the scale output)
      @ Out, mixScalarFluxDict, dictionary
    """
    mixScalarFluxDict = {}
    startFlag, count = 0, 0
    with open(inputFile, 'r') as infile:
      for line in infile:
        if re.search(r'Cell Averaged Fluxes',line.strip()):
          startFlag = 1
        if startFlag == 1:
          if re.search(r'Mixture\s+Group\s+\d',line) or re.search(r'Mixture\s+Group\d',line):
            groupsInLine = []
            groupsInLine = self.readGroupNumbersOfMixFluxes(line)
            count = count + 1
        if startFlag == 1 and re.search(r'\s+\d+\s+\d+', line):
          lineSplit = line.split()
          stringIsNumber = self.isNumber(lineSplit)
          if stringIsNumber is True:
            if count == 1:
              mixScalarFluxDict[lineSplit[0]] = {}
            for i in xrange(0,len(groupsInLine)):
              mixScalarFluxDict[lineSplit[0]][groupsInLine[i]] = lineSplit[i+1]
        if startFlag == 1 and re.search(r'Flux Disadvantage Factors', line): break
    #print mixScalarFluxDict
    return mixScalarFluxDict

  def calculateDisadvantageFactors(self, homogenizedMixtures, mixtureVolumesDict, mixScalarFluxDict, volumeOfHomMix, inputFile, disadvantageFactorDict):
    """
      calculate the disadvantage factors based on the mixture included in the homogenization scheme
      @ In, inputFile, string, name of the input file (which is the scale output)
      @ In, homogenizedMixtures, list
      @ In, mixtureVolumes, disctionary
      @ In, mixScalarFluxDict, dictionary
      @ Out, disadvantageFactorDict
    """
    totalGroupFlux = 0
    for mix in mixtureVolumesDict.iterkeys():
      disadvantageFactorDict[inputFile][mix] = {}
    for g in range (1,int(self.numberOfGroups)+1):
      product = []
      totalGroupFlux,  normalizedGroupFlux = 0, 0
      for fluxMix, flux in mixScalarFluxDict.iteritems():
        try:
          product.append(float(flux[str(g)])*float(mixtureVolumesDict[fluxMix]))
        except KeyError: pass
      for i in range(0,len(product)):
        totalGroupFlux = totalGroupFlux + product[i]
      normalizedGroupFlux = totalGroupFlux / volumeOfHomMix
      for fluxMix, flux in mixScalarFluxDict.iteritems():
        try:
          if normalizedGroupFlux == 0.0:
            normalizedGroupFlux = 1e-15
          disadvantageFactorDict[inputFile][fluxMix][str(g)] = float(flux[str(g)]) / normalizedGroupFlux
        except KeyError: pass
    #print disadvantageFactorDict
    return disadvantageFactorDict

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

  def generateXML(self, disadvantageFactorsDict, mixScalarFluxDict, isItDoubleHet, doubleHetMix, doubleHetMixList, inputFile):
    """
      print the values in an XML-formatted file
      @ In, disadvantageFactorsDict, dictionary
      @ In, mixScalarFluxDict, dictionary
      @ In, isItDoubleHet, boolean, Tells if the NEWT system has double het self shielding treatment
      @ In, doubleMix, if the system is double het, it give the fuel mix (integer), if not, gives None
      @ Out, mixture.xml (xml file)
    """
    copyOut = 'lib_copy.xml'
    copyfile(self.outputFile, copyOut)
    tree = ET.parse(copyOut)
    root = tree.getroot()
    countReflSlots, countSlots, count = 0, 0, 0
    for child in root.findall(".//filename"):
      countSlots = countSlots + 1
      if child.text == 'ft30f001_1200_26_r': countReflSlots = countReflSlots + 1
    for child in root.findall(".//POINT"):
      count = count + 1
      for lib in child.findall("filename"):
        for inp in xrange (0,len(inputFile)):
          if lib.text == 'ft30f001_'+str(inp):
            inputUsed = inputFile[inp]
      if count < (countSlots - countReflSlots + 1):
        for mixtureNumber in disadvantageFactorsDict[inputUsed].iterkeys():
          disadvantageFactorList = []
          if doubleHetMix is None:
            topChildDF = SubElement(child, 'disadvantage_factors', {'mix':mixtureNumber})
            for groupNumber in xrange (1,int(self.numberOfGroups)+1):
              disadvantageFactorList.append(disadvantageFactorsDict.get(inputUsed).get(mixtureNumber).get(str(groupNumber)))
              values = ' '.join(str(f) for f in disadvantageFactorList)
              if groupNumber == int(self.numberOfGroups):
                topChildDF.text = values
          else:
            if int(mixtureNumber) == int(doubleHetMix):
              for j in xrange (0,len(doubleHetMixList)):
                disadvantageFactorList = []
                subMixtureNumber = doubleHetMixList[j]
                topChildDF = SubElement(child, 'disadvantage_factors', {'mix':str(subMixtureNumber)})
                for groupNumber in xrange (1,int(self.numberOfGroups)+1):
                  disadvantageFactorList.append(disadvantageFactorsDict.get(inputUsed).get(mixtureNumber).get(str(groupNumber)))
                  values = ' '.join(str(f) for f in disadvantageFactorList)
                  if groupNumber == int(self.numberOfGroups):
                    topChildDF.text = values
            else:
              topChildDF = SubElement(child, 'disadvantage_factors', {'mix':mixtureNumber})
              for groupNumber in xrange (1,int(self.numberOfGroups)+1):
                disadvantageFactorList.append(disadvantageFactorsDict.get(inputUsed).get(mixtureNumber).get(str(groupNumber)))
                values = ' '.join(str(f) for f in disadvantageFactorList)
                if groupNumber == int(self.numberOfGroups):
                  topChildDF.text = values
    tree.write(copyOut)

    with open(copyOut, 'w') as output:
      output.write(self.prettify(root))
    # remove the annoying blank lines
    with open(copyOut) as infile:
      with open(self.outputFile, 'w') as outfile:
        for line in infile:
          if not line.strip(): continue
          outfile.write(line)

scaleParser = scaleParser()
