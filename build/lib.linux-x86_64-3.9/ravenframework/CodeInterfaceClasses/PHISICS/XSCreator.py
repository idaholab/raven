"""
Created on September 1st, 2017
@author: rouxpn
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import os
from decimal import Decimal
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom

class XSCreator():
  """
    Creates a perturbed cross section xml file based.
  """
  def __init__(self,inputFiles,booleanTab,workingDir,tabMapFileName,**pertDict):
    """
      Parses the PHISICS scaled_xs data file.
      @ In, inputFiles, string, file name the perturbed cross sections are printed into
      @ In, booleanTab, bool, True if a tabulation mapping is provided in the problem input. False otherwise
                                  This variable is ontrolled by the xml node <tabulation> in the raven input
      @ In, workingDir, string, absolute path to working directory
      @ In, tabMapFileName, string, absolute path to xml tabulation file
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.pertDict = self.scientificNotation(pertDict) # Perturbed variables
    self.listedDict = self.fileReconstruction(self.pertDict)
    self.generateXML(workingDir,booleanTab,inputFiles,tabMapFileName)

  def scientificNotation(self,pertDict):
    """
      Converts the numerical values into a scientific notation.
      @ In, pertDict, dictionary, perturbed variables
      @ Out, pertDict, dictionary, perturbed variables in scientific format
    """
    for key, value in pertDict.items():
      pertDict[key] = '%.8E' % Decimal(str(value))
    return pertDict

  def tabMapping(self,tab,tabMapFileName):
    """
      Links the tabulation number to the actual tabulation points
      @ In, tab, string, refers to the tabulation number
      @ In, tabMapFileName, string, absolute path to xml tabulation file
      @ Out, tabList, list, lists of all the tabulation parameters
      @ Out, valueList, lists of all the tabulation values
    """
    mappingTree = ET.parse(tabMapFileName)
    mappingRoot = mappingTree.getroot()
    for tabulationXML in mappingRoot.iter('tabulation'):
      if tab == tabulationXML.attrib.get('set'):
        tabList = []
        valueList = []
        for tabXML in tabulationXML.iter('tab'):
          tabList.append(tabXML.attrib.get('name'))
          valueList.append(tabXML.text)
    return tabList, valueList

  def prettify(self,elem):
    """
      Returns a pretty-printed xml string for the Element.
      @ In, elem, xml.etree.ElementTree.Element
      @ Out, None
    """
    roughString = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(roughString)
    return reparsed.toprettyxml(indent="  ")

  def generateXML(self,workingDir,bool,inputFiles,tabMapFileName):
    """
      Creates an xml file from the interface.
      @ In, workingDir, string, absolute path to working directory
      @ In, bool, boolean, True if a tabulation mapping is provided, false otherwise
      @ In, inputFiles, string, file name the perturbed cross sections are printed into
      @ In, tabMapFileName, string, absolute path to xml tabulation file
      @ Out, modifiedFile, string, name of the xml file created (under a dummy name)
    """
    top = Element('scaling_library', {'print_xml':'t'})
    print (self.listedDict)
    for XS in self.listedDict.keys():
      for tabulation in self.listedDict.get('XS').keys():
        topChild = SubElement(top, 'set')
        if bool:
          tabList, valueList = self.tabMapping(tabulation,tabMapFileName)
          for tab,value in zip(tabList,valueList):
            tabChild = SubElement(topChild, 'tab', {'name':tab})
            tabChild.text = value
        for material in self.listedDict.get('XS').get(tabulation).keys():
          tabulationChild = SubElement(topChild, 'library', {'lib_name':material})
          for isotope in self.listedDict.get('XS').get(tabulation).get(material).keys():
            for typeOfXs in self.listedDict.get('XS').get(tabulation).get(material).get(isotope).keys():
              libraryChild = SubElement(tabulationChild, 'isotope', {'id':isotope, 'type':typeOfXs.lower()})
              for reaction in self.listedDict.get('XS').get(tabulation).get(material).get(isotope).get(typeOfXs).keys():
                for count,(group,value) in enumerate(self.listedDict.get('XS').get(tabulation).get(material).get(isotope).get(typeOfXs).get(reaction).items()):
                  reactionChild = SubElement(libraryChild, self.formatXS(reaction), {'g':group})
                  reactionChild.text = value
    with open(inputFiles, 'w') as fileObj:
      fileObj.write(self.prettify(top))

  def formatXS(self,reaction):
    """
      Formats the reaction type to the proper PHISICS template
      @ In, reaction, string, a reaction type, in capital letters 'FISSIONXS'
      @ Out, reactionTemplated, a reaction type, templated 'FissionXS'
    """
    if reaction == 'FISSIONXS':
      reactionTemplated = 'FissionXS'
    elif reaction == 'KAPPAXS':
      reactionTemplated = 'KappaXS'
    elif reaction == 'NUFISSIONXS':
      reactionTemplated = 'NuFissionXS'
    elif reaction == 'N2NXS':
      reactionTemplated = 'n2nXS'
    elif reaction == 'NPXS':
      reactionTemplated = 'npXS'
    elif reaction == 'NALPHAXS':
      reactionTemplated = 'nalphaXS'
    elif reaction == 'NGXS':
      reactionTemplated = 'ngXS'
    else:
      raise IOError('the type of cross section '+reaction+' cannot be processed. Refer to manual for available reactions.')
    return reactionTemplated

  def cleanEmpty(self,reconstructedDict):
    """
      Removes all the empty string in the nested dictionary reconstructedDict.
      @ In, reconstructedDict, dictionary or list,  nested dictionary or list
      @ Out, cleanEmpty, dictionary or list, nested dictionary or list without trailing blanks
    """
    if not isinstance(reconstructedDict,(dict,list)):
      return reconstructedDict
    if isinstance(reconstructedDict, list):
      return [v for v in (self.cleanEmpty(v) for v in reconstructedDict) if v]
    return {k: v for k, v in ((k, self.cleanEmpty(v)) for k, v in reconstructedDict.items()) if v}

  def fileReconstruction(self,deconstructedDict):
    """
      Converts the formatted dictionary -> {'XS|FUEL1|U235|FISSION|1':1.30, 'XS|FUEL2|U238|ABS|2':4.69}
      into a dictionary of dictionaries that has the format -> {'XS':{'FUEL1':{'U235':{'FISSION':{'1':1.30}}}}, 'FUEL2':{'U238':{'ABS':{'2':4.69}}}}
      @ In, deconstructedDict, dictionary, dictionary of perturbed variables
      @ Out, leanReconstructedDict, dictionary, nested dictionary of perturbed variables
    """
    reconstructedDict = {}
    perturbedPhysicalParameters = []
    perturbedTabulationPoint = []
    perturbedMaterials = []
    perturbedIsotopes = []
    perturbedTypes = []
    perturbedReactions = []
    perturbedGroups = []

    pertDictSet = set(self.pertDict)
    deconstructedDictSet = set(deconstructedDict)
    for key in pertDictSet.intersection(deconstructedDictSet):
      if len(key.split('|')) != 7:
        raise IOError("The cross section variable "+key+" is not properly formatted")
      perturbedPhysicalParameters.append(key.split('|')[0])
      perturbedTabulationPoint.append(key.split('|')[1])
      perturbedMaterials.append(key.split('|')[2])
      perturbedIsotopes.append(key.split('|')[3])
      perturbedTypes.append(key.split('|')[4])
      perturbedReactions.append(key.split('|')[5])
      perturbedGroups.append(key.split('|')[6])

    for pertPhysicalParam in perturbedPhysicalParameters:
      reconstructedDict[pertPhysicalParam] = {}
      for pertTabulationPoint in perturbedTabulationPoint:
        reconstructedDict[pertPhysicalParam][pertTabulationPoint] = {}
        for mat in perturbedMaterials:
          reconstructedDict[pertPhysicalParam][pertTabulationPoint][mat] = {}
          for isotope in perturbedIsotopes:
            reconstructedDict[pertPhysicalParam][pertTabulationPoint][mat][isotope] = {}
            for reactType in perturbedTypes:
              reconstructedDict[pertPhysicalParam][pertTabulationPoint][mat][isotope][reactType] = {}
              for react in perturbedReactions:
                reconstructedDict[pertPhysicalParam][pertTabulationPoint][mat][isotope][reactType][react] = {}
                for group in perturbedGroups:
                  reconstructedDict[pertPhysicalParam][pertTabulationPoint][mat][isotope][reactType][react][group] = {}
    for typeKey, value in deconstructedDict.items():
      if typeKey in pertDictSet:
        keyWords = typeKey.split('|')
        reconstructedDict[keyWords[0]][keyWords[1]][keyWords[2]][keyWords[3]][keyWords[4]][keyWords[5]][keyWords[6]] = value
    leanReconstructedDict = self.cleanEmpty(reconstructedDict)
    return leanReconstructedDict
