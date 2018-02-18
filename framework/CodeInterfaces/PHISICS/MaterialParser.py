"""
Created on July 11th, 2017
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

class MaterialParser():

  def replaceValues(self, genericXMLdict):
    """
    replace the values from the pertured dict and put them in the deconstructed original dictionary
    """
    setXML = set(genericXMLdict)
    #print setXML
    setPertDict = set(self.pertDict)
    #print setPertDict
    for key in setPertDict.intersection(setXML):
      genericXMLdict[key] = self.pertDict.get(key, {})
    #print genericXMLdict
    return genericXMLdict

  def dictFormating_from_perturbed_to_generic(self, XMLdict):
    """
    Transform the ditionary comning from the XML input into the templated dictionary.
    The templated format is {DENSITY|FUEL|ISOTOPE}
    """
    genericXMLdict = {}
    #print XMLdict
    for paramXML in XMLdict.iterkeys():
      for matXML in XMLdict.get(paramXML).iterkeys():
        for isotopeXML, densityValue in XMLdict.get(paramXML).get(matXML).iteritems():
          genericXMLdict[paramXML.upper()+'|'+matXML.upper()+'|'+isotopeXML.upper()] = densityValue
    #print genericXMLdict
    return genericXMLdict

  def dictFormating_from_XML_to_perturbed(self):
    """
    Transform the dictionary of dictionaries from the XML tree to a dictionary of dictionaries
    formatted identically as the perturbed dictionary
    the perturbed dictionary template is {'DENSITY':{'FUEL':{'ISOTOPE'}}}
    """
    # declare the dictionaries
    XMLdict = {}
    matList = []
    isotopeList = []
    XMLdict['density'] = {}
    for matXML in self.root.getiterator('mat'):
      #print matXML.attrib.get('id')
      for isotopeXML in self.root.getiterator('isotope'):
        matList.append(matXML.attrib.get('id'))
        isotopeList.append(isotopeXML.attrib.get('id'))
    #print matList
    #print isotopeList
    for i in xrange(0,len(matList)):
      XMLdict['density'][matList[i]] = {}
      for j in xrange(0,len(isotopeList)):
        XMLdict['density'][matList[i]][isotopeList[j]] = {}
    for matXML in self.root.getiterator('mat'):
      for isotopeXML in matXML.findall('isotope'):
        #print isotopeXML.attrib
        XMLdict['density'][matXML.attrib.get('id')][isotopeXML.attrib.get('id')] = isotopeXML.attrib.get('density')
        #print XMLdict
    return XMLdict

  def __init__(self, inputFiles, **pertDict):
    """
      Parse the Material.xml data file and put the isotopes name as key and
      the decay constant relative to the isotopes as values
    """
    self.pertDict = pertDict
    #print self.pertDict
    for key, value in self.pertDict.iteritems():
      self.pertDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    self.start_time = time.time()
    numbering    = {}
    concatenateDecayList = []
    self.allDecayList = []
    self.inputFiles = inputFiles
    self.tree = ET.parse(self.inputFiles)
    self.root = self.tree.getroot()
    #print self.pertDict
    #print "\n\n\n\n"
    self.listedDict = self.fileReconstruction(self.pertDict)
    self.printInput()

  def fileReconstruction(self, deconstructedDict):
    """
      Converts the formatted dictionary -> {'DENSITY|FUEL1|U235':1.30, DENSITY|FUEL2|U238':4.69}
      into a dictionary of dictionaries that has the format -> {'DENSITY':{'FUEL1':{'U235':1.30}, 'FUEL2':{'U238':4.69}}}
      In: Dictionary deconstructedDict
      Out: Dictionary of dictionaries reconstructedDict
    """
    #print deconstructedDict
    #print self.pertDict
    reconstructedDict           = {}
    perturbedIsotopes           = []
    perturbedMaterials          = []
    perturbedPhysicalParameters = []
    for i in deconstructedDict.iterkeys() :
      splittedKeywords = i.split('|')
      perturbedIsotopes.append(splittedKeywords[2])
      perturbedMaterials.append(splittedKeywords[1])
      perturbedPhysicalParameters.append(splittedKeywords[0])
    #print perturbedIsotopes
    #print perturbedMaterials
    #print perturbedPhysicalParameters

    for i in xrange (0,len(perturbedPhysicalParameters)):
      reconstructedDict[perturbedPhysicalParameters[i]] = {}
      for j in xrange (0,len(perturbedMaterials)):
        reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]] = {}
        for k in xrange (0,len(perturbedIsotopes)):
          reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]][perturbedIsotopes[k]] = {}

    for typeKey, value in deconstructedDict.iteritems():
      keyWords = typeKey.split('|')
      reconstructedDict[keyWords[0]][keyWords[1]][keyWords[2]] = value
    #print reconstructedDict

    return reconstructedDict

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
    return str(randint(1,1000000000000))+'.xml'

  def printInput(self):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    modifiedFile = self.generateRandomName()
    open(modifiedFile, 'w')
    XMLdict = {}
    genericXMLdict = {}
    newXMLdict = {}
    templatedNewXMLdict = {}
    mapAttribIsotope = {}

    XMLdict = self.dictFormating_from_XML_to_perturbed()
    #print XMLdict
    genericXMLdict = self.dictFormating_from_perturbed_to_generic(XMLdict)
    #print genericXMLdict
    newXMLDict = self.replaceValues(genericXMLdict)
    #print newXMLDict
    templatedNewXMLdict = self.fileReconstruction(newXMLDict)
    #print templatedNewXMLdict

    for matXML in self.root.getiterator('mat'):
      for isotopeXML in matXML.findall('isotope'):
        isotopeXML.attrib['density'] = templatedNewXMLdict.get(isotopeXML.attrib.keys()[1].upper()).get(matXML.attrib.get('id').upper()).get(isotopeXML.attrib.get('id').upper())
        self.tree.write(modifiedFile)
    copyfile(modifiedFile, self.inputFiles)
    self.removeRandomlyNamedFiles(modifiedFile)


