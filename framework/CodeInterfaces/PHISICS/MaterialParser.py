"""
Created on July 11th, 2017
@author: rouxpn
"""
import os
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal 
import time
import xml.etree.ElementTree as ET
from random import *

class MaterialParser():
  """
    Parses the PHISICS Material XML input and replaces the nominal values by the perturbed values 
  """
  def replaceValues(self, genericXMLdict):
    """
      Replaces the values from the pertured dict and puts them in the deconstructed original dictionary
      @ In, genericXMLdict, dictionary, dictionary under the format  {X|Y|Z:nominalValue}
      @ Out, genericXMLdict, dictionary, dictionary under the format  {X|Y|Z:perturbedValue}
    """
    setXML = set(genericXMLdict)
    setPertDict = set(self.pertDict)
    for key in setPertDict.intersection(setXML):
      genericXMLdict[key] = self.pertDict.get(key, {})
    return genericXMLdict

  def dictFormating_from_perturbed_to_generic(self, XMLdict):
    """
      Transforms the ditionary coming from the XML input into the templated dictionary.
      The templated format is {DENSITY|FUEL|ISOTOPE}
      @ In, XMLdict, dictionary, under the format {'DENSITY':{'FUEL1':{'U238':1.000}}}
      @ Out, genericXMLdict, dictionary, under the format {DENSITY|FUEL1|U238|1.000}
    """
    genericXMLdict = {}
    for paramXML in XMLdict.iterkeys():
      for matXML in XMLdict.get(paramXML).iterkeys():
        for isotopeXML, densityValue in XMLdict.get(paramXML).get(matXML).iteritems():
          genericXMLdict[paramXML.upper()+'|'+matXML.upper()+'|'+isotopeXML.upper()] = densityValue 
    return genericXMLdict

  def dictFormating_from_XML_to_perturbed(self):
    """
      Transforms the dictionary of dictionaries from the XML tree to a dictionary of dictionaries formatted identically as the perturbed dictionary. 
      @ In, None 
      @ Out, XMLdict, dictionary, under the format {'DENSITY':{'FUEL1':{'U238':1.000}}}
    """
    XMLdict = {}
    matList = []
    isotopeList = []
    XMLdict['density'] = {}
    for matXML in self.root.getiterator('mat'):
      for isotopeXML in self.root.getiterator('isotope'):
        matList.append(matXML.attrib.get('id'))
        isotopeList.append(isotopeXML.attrib.get('id'))
    matList = self.unifyElements(matList)
    isotopeList = self.unifyElements(isotopeList)
    for i in xrange(0,len(matList)):
      XMLdict['density'][matList[i]] = {}
      for j in xrange(0,len(isotopeList)):
        XMLdict['density'][matList[i]][isotopeList[j]] = {}  
    for matXML in self.root.getiterator('mat'): 
      for isotopeXML in matXML.findall('isotope'):
        XMLdict['density'][matXML.attrib.get('id')][isotopeXML.attrib.get('id')] = isotopeXML.attrib.get('density')
    return XMLdict
  
  def unifyElements(self,listWithRepetitions):
    """
      Takes a list as an entry and reduces the list into a list of unique elements. 
      @ In, listWithRepetitions, list 
      @ Out, listWithUniqueElements, list
    """
    valueSeen = set() 
    listWithUniqueElements = [x for x in listWithRepetitions if x not in valueSeen and not valueSeen.add(x)]
    return listWithUniqueElements
    
  def __init__(self, inputFiles, **pertDict):
    """
      @ In, inputFiles, string, Qvalues library file 
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.pertDict = pertDict
    for key, value in self.pertDict.iteritems(): 
      self.pertDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values 
    self.inputFiles = inputFiles
    self.tree = ET.parse(self.inputFiles)
    self.root = self.tree.getroot()
    self.listedDict = self.fileReconstruction(self.pertDict)
    self.printInput()

  def fileReconstruction(self, deconstructedDict):
    """
      Converts the formatted dictionary -> {'DENSITY|FUEL1|U235':1.30, DENSITY|FUEL2|U238':4.69} 
      into a dictionary of dictionaries that has the format -> {'DENSITY':{'FUEL1':{'U235':1.30}, 'FUEL2':{'U238':4.69}}}
      @ In, deconstructedDict, dictionary 
      @ Out, reconstructedDict, nested dictionary 
    """
    reconstructedDict           = {}
    perturbedIsotopes           = []
    perturbedMaterials          = []
    perturbedPhysicalParameters = []
    for i in deconstructedDict.iterkeys() :
      splittedKeywords = i.split('|')
      perturbedIsotopes.append(splittedKeywords[2])
      perturbedMaterials.append(splittedKeywords[1])
      perturbedPhysicalParameters.append(splittedKeywords[0])
    for i in xrange (0,len(perturbedPhysicalParameters)):
      reconstructedDict[perturbedPhysicalParameters[i]] = {} 
      for j in xrange (0,len(perturbedMaterials)):
        reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]] = {}
        for k in xrange (0,len(perturbedIsotopes)):
          reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]][perturbedIsotopes[k]] = {} 
    for typeKey, value in deconstructedDict.iteritems():
      keyWords = typeKey.split('|')
      reconstructedDict[keyWords[0]][keyWords[1]][keyWords[2]] = value
    return reconstructedDict

  def removeRandomlyNamedFiles(self, modifiedFile):
    """
      Removes the temporary file with a random name in the working directory
      @ In, modifiedFile, string
      @ Out, None 
    """
    os.remove(modifiedFile)    
       
  def generateRandomName(self):
    """
      Generates a random file name for the modified file
      @ In, None
      @ Out, string 
    """
    return str(randint(1,100000000000000))+'.xml'
    
  def printInput(self):
    """
      Prints out the pertubed Qvalues library into a file
      @ In, None 
      @ Out, None
    """
    modifiedFile = self.generateRandomName()      
    open(modifiedFile, 'w')
    XMLdict = self.dictFormating_from_XML_to_perturbed()
    genericXMLdict = self.dictFormating_from_perturbed_to_generic(XMLdict)    
    newXMLDict = self.replaceValues(genericXMLdict)    
    templatedNewXMLdict = self.fileReconstruction(newXMLDict)
    
    for matXML in self.root.getiterator('mat'): 
      for isotopeXML in matXML.findall('isotope'):
        isotopeXML.attrib['density'] = templatedNewXMLdict.get(isotopeXML.attrib.keys()[1].upper()).get(matXML.attrib.get('id').upper()).get(isotopeXML.attrib.get('id').upper())
        self.tree.write(modifiedFile)
    copyfile(modifiedFile, self.inputFiles)  
    self.removeRandomlyNamedFiles(modifiedFile)


