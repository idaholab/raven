"""
Created on June 19th, 2017
@author: rouxpn
"""

import os
import sys
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal
import xml.etree.ElementTree as ET 

class XSParser():

  def replaceValues(self, genericXMLdict):
    """
    replace the values from the perturbed dict and put them in the deconstructed original dictionary
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
        for isotopeXML in XMLdict.get(paramXML).get(matXML).iterkeys():
          for reactionXML in XMLdict.get(paramXML).get(matXML).get(isotopeXML).iterkeys():
            for groupXML, pertValue in XMLdict.get(paramXML).get(matXML).get(isotopeXML).get(reactionXML).iteritems():
              genericXMLdict[paramXML.upper()+'|'+matXML.upper()+'|'+isotopeXML.upper()+'|'+reactionXML.upper()+'|'+groupXML.upper()] = pertValue 
    #print genericXMLdict
    return genericXMLdict

  def dictFormating_from_XML_to_perturbed(self):
    """
    Transform the dictionary of dictionaries from the XML tree to a dictionary of dictionaries
    formatted identically as the perturbed dictionary 
    the perturbed dictionary template is {'XS':{'FUEL1':{'FISS':{'1':{'U238'}}}}}
    """
    # declare the dictionaries 
    XMLdict = {}
    matList = []
    isotopeList = []
    reactionList = []
    XMLdict['XS'] = {}
    
    for libraryXML in self.root.getiterator('library'):
      XMLdict['XS'][libraryXML.attrib.get('lib_name')] = {}
      for isotopeXML in libraryXML.findall('.//isotope'):
        reactionList = [j.tag for j in isotopeXML]
        XMLdict['XS'][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')] = {}
        for k in xrange(0,len(reactionList)):
          XMLdict['XS'][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')][reactionList[k]] = {}
          for groupXML in isotopeXML.findall(reactionList[k]):
            if groupXML.attrib.get('gg') == None: 
              XMLdict['XS'][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')][reactionList[k]][groupXML.attrib.get('g')] = {}
            else: 
              XMLdict['XS'][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')][reactionList[k]][str(groupXML.attrib.get('g'))+'->'+str(groupXML.attrib.get('gg'))] = {}
    #print XMLdict
    for libraryXML in self.root.getiterator('library'):
      for isotopeXML in libraryXML.findall('.//isotope'):
        reactionList = [j.tag for j in isotopeXML]
        for k in xrange(0,len(reactionList)):
          for groupXML in isotopeXML.findall(reactionList[k]):
            if groupXML.attrib.get('gg') == None: 
              XMLdict['XS'][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')][reactionList[k]][groupXML.attrib.get('g')] = groupXML.text
            else: 
              XMLdict['XS'][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')][reactionList[k]][str(groupXML.attrib.get('g'))+'->'+str(groupXML.attrib.get('gg'))] = groupXML.text
    #print XMLdict
    return XMLdict

  def __init__(self, inputFiles, **pertDict):
    """
      Parse the PHISICS XS.xml data file   
      In: XS.xml
      Out: None 
    """
    self.pertDict = pertDict
    #print self.pertDict
    #print inputFiles
    for key, value in self.pertDict.iteritems(): 
      self.pertDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values   
    self.inputFiles = inputFiles
    self.tree = ET.parse(self.inputFiles)
    self.root = self.tree.getroot()
    self.listedDict = self.fileReconstruction(self.pertDict)
    self.printInput()

  def fileReconstruction(self, deconstructedDict):
    """
      Converts the formatted dictionary -> {'XS|FUEL1|FISS|1|U235':1.30, XS|FUEL2|ABS|2|U238':4.69} 
      into a dictionary of dictionaries that has the format -> {'XS':{'FUEL1':{'FISS':{'1':{'U235':1.30}}}}, 'FUEL2':{'ABS':{'2':{'U238':4.69}}}}
      In: Dictionary deconstructedDict
      Out: Dictionary of dictionaries reconstructedDict 
    """
    #print deconstructedDict
    reconstructedDict           = {}
    perturbedPhysicalParameters = []
    perturbedMaterials          = []
    perturbedReactions          = []
    perturbedGroups             = []
    perturbedIsotopes           = []
    for i in deconstructedDict.iterkeys() :
      splittedKeywords = i.split('|')
      perturbedPhysicalParameters.append(splittedKeywords[0])
      perturbedMaterials.append(splittedKeywords[1])
      perturbedIsotopes.append(splittedKeywords[2])
      perturbedReactions.append(splittedKeywords[3])
      perturbedGroups.append(splittedKeywords[4])  
    
    for i in xrange (0,len(perturbedPhysicalParameters)):
      reconstructedDict[perturbedPhysicalParameters[i]] = {} 
      for j in xrange (0,len(perturbedMaterials)):
        reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]] = {}
        for k in xrange (0,len(perturbedIsotopes)):
          reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]][perturbedIsotopes[k]] = {} 
          for l in xrange (0,len(perturbedReactions)):
            reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]][perturbedIsotopes[k]][perturbedReactions[l]] = {}
            for m in xrange (0,len(perturbedGroups)):
              reconstructedDict[perturbedPhysicalParameters[i]][perturbedMaterials[j]][perturbedIsotopes[k]][perturbedReactions[l]][perturbedGroups[m]] = {}
    #print reconstructedDict    
    for typeKey, value in deconstructedDict.iteritems():
      keyWords = typeKey.split('|')
      reconstructedDict[keyWords[0]][keyWords[1]][keyWords[2]][keyWords[3]][keyWords[4]] = value
    #print reconstructedDict
    return reconstructedDict
   
    
  def printInput(self):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    modifiedFile = 'modif.xml'     
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
    print templatedNewXMLdict 
    
    for libraryXML in self.root.getiterator('library'): 
      #print libraryXML.attrib
      for isotopeXML in libraryXML.findall('.//isotope'):
        #print isotopeXML.attrib
        reactionList = [j.tag for j in isotopeXML]
        for k in xrange(0,len(reactionList)):
          for groupXML in isotopeXML.findall(reactionList[k]):
            if groupXML.attrib.get('gg') == None:  
              #print templatedNewXMLdict.get('XS').get(libraryXML.attrib.get('lib_name').upper()).get(isotopeXML.attrib.get('id').upper()).get(reactionList[k].upper())
              print templatedNewXMLdict.get('XS').get(libraryXML.attrib.get('lib_name').upper()).get(isotopeXML.attrib.get('id').upper()).get(reactionList[k].upper()).get(groupXML.attrib.get('g'))
              groupXML.text = templatedNewXMLdict.get('XS').get(libraryXML.attrib.get('lib_name').upper()).get(isotopeXML.attrib.get('id').upper()).get(reactionList[k].upper()).get(groupXML.attrib.get('g'))
            else: 
              groupXML.text = templatedNewXMLdict.get('XS').get(libraryXML.attrib.get('lib_name').upper()).get(isotopeXML.attrib.get('id').upper()).get(reactionList[k].upper()).get(groupXML.attrib.get(str(groupXML.attrib.get('g'))+'->'+str(groupXML.attrib.get('gg'))))
        self.tree.write(modifiedFile)
    copyfile('modif.xml', self.inputFiles)  
   

