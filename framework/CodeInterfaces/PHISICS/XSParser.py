"""
Created on June 19th, 2017
@author: rouxpn
"""

import re 
from shutil import copyfile 
from decimal import Decimal
import xml.etree.ElementTree as ET 
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom

class XSParser():

  def replaceValues(self, genericXMLdict):
    """
      Replaces the values from the perturbed dict and put them in the deconstructed original dictionary
      @ In, genericXMLdict, dictionary, dictionary under the format  {V|W|X|Y|Z:nominalValue}
      @ Out, genericXMLdict, dictionary, dictionary under the format {V|W|X|Y|Z:perturbedValue} 
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
      @ In, XMLdict, dictionary, under the format {'XS':{'FUEL1':{'u238':{'FISSION':{'1':1.000}}}}}
      @ Out, genericXMLdict, dictionary, under the format {XS|FUEL1|U238|FISSION|1|1.000}
    """
    genericXMLdict = {}
    for paramXML in XMLdict.iterkeys():
      for tabXML in XMLdict.get(paramXML).iterkeys():
        for matXML in XMLdict.get(paramXML).get(tabXML).iterkeys():
          for isotopeXML in XMLdict.get(paramXML).get(tabXML).get(matXML).iterkeys():
            for reactionXML in XMLdict.get(paramXML).get(tabXML).get(matXML).get(isotopeXML).iterkeys():
              for groupXML, pertValue in XMLdict.get(paramXML).get(tabXML).get(matXML).get(isotopeXML).get(reactionXML).iteritems():
                genericXMLdict[paramXML.upper()+'|'+str(tabXML).upper()+'|'+matXML.upper()+'|'+isotopeXML.upper()+'|'+reactionXML.upper()+'|'+str(groupXML).upper()] = pertValue 
    return genericXMLdict

  def dictFormating_from_XML_to_perturbed(self):
    """
      Transforms the dictionary of dictionaries from the XML tree to a dictionary of dictionaries formatted identically as the perturbed dictionary. 
      @ In, None 
      @ Out, XMLdict, dictionary, under the format {'XS':{'FUEL1':{'u238':{'FISSION':{'1':1.000}}}}}
    """
    XMLdict = {}
    reactionList = []
    XMLdict['XS'] = {}
    reactionList = []
    count = 0
    for tabulationXML in self.root.getiterator('tabulation'):
      count = count + 1 
      XMLdict['XS'][count] = {}
      for libraryXML in tabulationXML.getiterator('library'):
        currentMat = libraryXML.attrib.get('lib_name')
        XMLdict['XS'][count][libraryXML.attrib.get('lib_name')] = {}
        for isotopeXML in libraryXML.getiterator('isotope'):
          currentIsotope = isotopeXML.attrib.get('id')
          currentType = isotopeXML.attrib.get('type')
          reactionList = [j.tag for j in isotopeXML]
          XMLdict['XS'][count][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')+isotopeXML.attrib.get('type')] = {}
          for k in xrange (0, len(reactionList)):
            XMLdict['XS'][count][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')+isotopeXML.attrib.get('type')][reactionList[k]] = {}
            for groupXML in isotopeXML.getiterator(reactionList[k]):
              individualGroup = [x.strip() for x in groupXML.attrib.get('g').split(',')]
              individualGroupValues = [y.strip() for y in groupXML.text.split(',')]
              for position in xrange(0,len(individualGroup)):
                XMLdict['XS'][count][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')+isotopeXML.attrib.get('type')][reactionList[k]][individualGroup[position]] = individualGroupValues[position]
    return XMLdict
  
  def prettify(self, elem):
    """
      Returns a pretty-printed XML string for the Element
      @ In, elem, xml.etree.ElementTree.Element
      @ Out, None 
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
      
  def __init__(self, inputFiles, **pertDict):
    """
      Parse the PHISICS XS.xml data file   
      @ In, inputFiles, 
      @ Out, None 
    """
    print 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
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
      Converts the formatted dictionary -> {'XS|FUEL1|U235|FISSION|1':1.30, 'XS|FUEL2|U238|ABS|2':4.69} 
      into a dictionary of dictionaries that has the format -> {'XS':{'FUEL1':{'U235':{'FISSION':{'1':1.30}}}}, 'FUEL2':{'U238':{'ABS':{'2':4.69}}}}
      @ In, deconstructedDict, dictionary 
      @ Out, reconstructedDict, dictionary of dictionaries 
    """
    reconstructedDict           = {}
    perturbedPhysicalParameters = []
    perturbedTabulationPoint    = []
    perturbedMaterials          = []
    perturbedReactions          = []
    perturbedGroups             = []
    perturbedIsotopes           = []
    
    pertDictSet = set(self.pertDict)
    deconstructedDictSet = set(deconstructedDict)
    for i in pertDictSet.intersection(deconstructedDictSet): 
      splittedKeywords = i.split('|')
      perturbedPhysicalParameters.append(splittedKeywords[0])
      perturbedTabulationPoint.append(splittedKeywords[1])
      perturbedMaterials.append(splittedKeywords[2])
      perturbedIsotopes.append(splittedKeywords[3])
      perturbedReactions.append(splittedKeywords[4])
      perturbedGroups.append(splittedKeywords[5])  
    
    for i in xrange (0,len(perturbedPhysicalParameters)):
      reconstructedDict[perturbedPhysicalParameters[i]] = {}
      for j in xrange (0,len(perturbedTabulationPoint)):
        reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]] = {} 
        for k in xrange (0,len(perturbedMaterials)):
          reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]] = {}
          for l in xrange (0,len(perturbedIsotopes)):
            reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]][perturbedIsotopes[l]] = {} 
            for m in xrange (0,len(perturbedReactions)):
              reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]][perturbedIsotopes[l]][perturbedReactions[m]] = {}
              for n in xrange (0,len(perturbedGroups)):
                reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]][perturbedIsotopes[l]][perturbedReactions[m]][perturbedGroups[n]] = {}
    for typeKey, value in deconstructedDict.iteritems():
      if typeKey in pertDictSet:
        keyWords = typeKey.split('|')
        reconstructedDict[keyWords[0]][keyWords[1]][keyWords[2]][keyWords[3]][keyWords[4]][keyWords[5]] = value
    return reconstructedDict
    
  def printInput(self):
    """
      Method to print out the new input
      @ In, None 
      @ Out, None
    """
    modifiedFile = 'modif.xml'     
    open(modifiedFile, 'w')
    templatedNewXMLdict = {} 
    mapAttribIsotope = {}
    
    XMLdict = self.dictFormating_from_XML_to_perturbed()
    genericXMLdict = self.dictFormating_from_perturbed_to_generic(XMLdict)
    newXMLDict = self.replaceValues(genericXMLdict)
    templatedNewXMLdict = self.fileReconstruction(newXMLDict)
    print templatedNewXMLdict
    templatedNewXMLdict = self.listedDict
    print 'qqqqqqqqqqqqq'
    print templatedNewXMLdict
    count = 0
    for tabulationXML in self.root.getiterator('tabulation'):
      count = count + 1 
      for libraryXML in tabulationXML.getiterator('library'):
        for isotopeXML in libraryXML.getiterator('isotope'):
          reactionList = [j.tag for j in isotopeXML]
          for k in xrange(0,len(reactionList)):
            for groupXML in isotopeXML.getiterator(reactionList[k]):
              individualGroup = [x.strip() for x in groupXML.attrib.get('g').split(',')]
              individualGroupValues = [y.strip() for y in groupXML.text.split(',')]
              for position in xrange(0,len(individualGroup)):
                groupXML.text = templatedNewXMLdict.get('XS').get(str(count).upper()).get(libraryXML.attrib.get('lib_name').upper()).get(isotopeXML.attrib.get('id').upper()+isotopeXML.attrib.get('type').upper()).get(reactionList[k].upper()).get(groupXML.attrib.get('g'))
          self.tree.write(modifiedFile)
    copyfile('modif.xml', self.inputFiles)  
   

