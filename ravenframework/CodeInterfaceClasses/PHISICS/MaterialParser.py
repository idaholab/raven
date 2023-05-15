"""
Created on July 11th, 2017
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
from decimal import Decimal
import xml.etree.ElementTree as ET
import re

class MaterialParser():
  """
    Parses the PHISICS xml material file and replaces the nominal values by the perturbed values.
  """
  def __init__(self,inputFiles,workingDir,**pertDict):
    """
      Constructor.
      @ In, inputFiles, string, Qvalues library file
      @ In, workingDir, string, path to working directory
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    noDashDict = {}
    for key, value in pertDict.items():
      noDashDict[self.noDash(key)] = value
    self.pertDict = self.scientificNotation(noDashDict) # Perturbed variables
    self.inputFiles = inputFiles                      # material inp file
    self.tree = ET.parse(self.inputFiles)             # xml tree
    self.root = self.tree.getroot()                   # xml root
    self.listedDict = self.fileReconstruction(self.pertDict)
    self.printInput(workingDir)

  def scientificNotation(self,pertDict):
    """
      Converts the numerical values into a scientific notation. Removes also the dashes from the isotopes name
      @ In, pertDict, dictionary, perturbed variables
      @ Out, pertDict, dictionary, perturbed variables in scientific format
    """
    for key, value in pertDict.items():
      pertDict[key] = '%.4E' % Decimal(str(value))
    return pertDict

  def replaceValues(self,genericXMLdict):
    """
      Replaces the values from the pertured dict and puts them in the deconstructed original dictionary
      @ In, genericXMLdict, dictionary, dictionary templated  {X|Y|Z:nominalValue}
      @ Out, genericXMLdict, dictionary, dictionary templated  {X|Y|Z:perturbedValue}
    """
    setXML = set(genericXMLdict)
    setPertDict = set(self.pertDict)
    for key in setPertDict.intersection(setXML):
      genericXMLdict[key] = self.pertDict.get(key, {})
    return genericXMLdict

  def noDash(self, dashStr):
    """
      Removes the dash from the isotope key. hence the variables are dash-independent.
      The isotope name in the material file and the isotope name in the raven variable do not have to match, in terms of dashes
      @ In, dashStr, string, isotope name with or without dash
      @ Out, noDashStr, string, without dash
    """
    noDashStr = dashStr.replace('-','')
    return noDashStr

  def dictFormatingFromPerturbedToGeneric(self,XMLdict):
    """
      Transforms the dictionary coming from the xml file into the templated dictionary.
      The templated format is {DENSITY|FUEL|ISOTOPE}
      @ In, XMLdict, dictionary, under the format {'DENSITY':{'FUEL1':{'U238':1.000}}}
      @ Out, genericXMLdict, dictionary, under the format {DENSITY|FUEL1|U238|1.000}
    """
    genericXMLdict = {}
    for paramXML in XMLdict.keys():
      for matXML in XMLdict.get(paramXML).keys():
        for isotopeXML, densityValue in XMLdict.get(paramXML).get(matXML).items():
          genericXMLdict[paramXML.upper()+'|'+matXML.upper()+'|'+self.noDash(isotopeXML.upper())] = densityValue
    return genericXMLdict

  def dictFormatingFromXmlToPerturbed(self):
    """
      Transforms the dictionary of dictionaries from the xml tree to a dictionary of dictionaries formatted identically as the perturbed dictionary.
      @ In, None
      @ Out, XMLdict, dictionary, under the format {'DENSITY':{'FUEL1':{'U238':1.000}}}
    """
    XMLdict = {}
    matList = []
    isotopeList = []
    XMLdict['density'] = {}
    for matXML in self.root.iter('mat'):
      for isotopeXML in self.root.iter('isotope'):
        matList.append(matXML.attrib.get('id'))
        isotopeList.append(self.noDash(isotopeXML.attrib.get('id')))
    matList = self.unifyElements(matList)
    isotopeList = self.unifyElements(isotopeList)
    for mat in matList:
      XMLdict['density'][mat] = {}
      for isotope in isotopeList:
        XMLdict['density'][mat][isotope] = {}
    for matXML in self.root.iter('mat'):
      for isotopeXML in matXML.findall('isotope'):
        XMLdict['density'][matXML.attrib.get('id')][self.noDash(isotopeXML.attrib.get('id'))] = isotopeXML.attrib.get('density')
    return XMLdict

  def unifyElements(self,listWithRepetitions):
    """
      removes any repetitions of elements in a list.
      @ In, listWithRepetitions, list, list made of non-unique elements
      @ Out, listWithUniqueElements, list, list made of unique elements
    """
    valueSeen = set()
    listWithUniqueElements = [x for x in listWithRepetitions if x not in valueSeen and not valueSeen.add(x)]
    return listWithUniqueElements

  def fileReconstruction(self,deconstructedDict):
    """
      Converts the formatted dictionary -> {'DENSITY|FUEL1|U235':1.30, DENSITY|FUEL2|U238':4.69}.
      into a dictionary of dictionaries that has the format -> {'DENSITY':{'FUEL1':{'U235':1.30}, 'FUEL2':{'U238':4.69}}}
      @ In, deconstructedDict, dictionary, dictionary of perturbed variables
      @ Out, reconstructedDict, dictionary, nested dictionary of perturbed variables
    """
    reconstructedDict = {}
    perturbedIsotopes = []
    perturbedMaterials = []
    perturbedPhysicalParameters = []
    for key in deconstructedDict.keys():
      perturbedIsotopes.append(key.split('|')[2])
      perturbedMaterials.append(key.split('|')[1])
      perturbedPhysicalParameters.append(key.split('|')[0])
    for i in range (len(perturbedPhysicalParameters)):
      reconstructedDict[perturbedPhysicalParameters[i]] = {}
      for j in range (len(perturbedMaterials)):
        reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]] = {}
        for k in range (len(perturbedIsotopes)):
          reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]][perturbedIsotopes[k]] = {}
    for typeKey, value in deconstructedDict.items():
      keyWords = typeKey.split('|')
      reconstructedDict[keyWords[0]][keyWords[1]][keyWords[2]] = value
    return reconstructedDict

  def printInput(self,workingDir):
    """
      Prints out the pertubed xml material file into a xml file.
      @ In, workingDir, string, path to working directory
      @ Out, None
    """
    XMLdict = self.dictFormatingFromXmlToPerturbed()
    genericXMLdict = self.dictFormatingFromPerturbedToGeneric(XMLdict)
    newXMLDict = self.replaceValues(genericXMLdict)
    templatedNewXMLdict = self.fileReconstruction(newXMLDict)
    open(self.inputFiles, 'w')
    for matXML in self.root.iter('mat'):
      for isotopeXML in matXML.findall('isotope'):
        isotopeXML.attrib['density'] = templatedNewXMLdict.get('DENSITY').get(matXML.attrib.get('id').upper()).get(self.noDash(isotopeXML.attrib.get('id')).upper())
        self.tree.write(self.inputFiles)
