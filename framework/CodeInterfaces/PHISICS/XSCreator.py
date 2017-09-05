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
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom

class XSCreator():

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
      for tabXML in XMLdict.get(paramXML).iterkeys():
        for matXML in XMLdict.get(paramXML).get(tabXML).iterkeys():
          for isotopeXML in XMLdict.get(paramXML).get(tabXML).get(matXML).iterkeys():
            for reactionXML in XMLdict.get(paramXML).get(tabXML).get(matXML).get(isotopeXML).iterkeys():
              for groupXML, pertValue in XMLdict.get(paramXML).get(tabXML).get(matXML).get(isotopeXML).get(reactionXML).iteritems():
                genericXMLdict[paramXML.upper()+'|'+str(tabXML).upper()+'|'+matXML.upper()+'|'+isotopeXML.upper()+'|'+reactionXML.upper()+'|'+str(groupXML).upper()] = pertValue 
    #print genericXMLdict
    return genericXMLdict

  def dictFormating_from_XML_to_perturbed(self):
    """
    Transform the dictionary of dictionaries from the XML tree to a dictionary of dictionaries
    formatted identically as the perturbed dictionary 
    the perturbed dictionary template is {'XS':{'FUEL1':{'u238':{'FISSION':{'1':1.000}}}}}
    """
    # declare the dictionaries 
    XMLdict = {}
    matList = []
    isotopeList = []
    reactionList = []
    XMLdict['XS'] = {}
    reactionList = []
    
    count = 0
    for tabulationXML in self.root.getiterator('tabulation'):
      #print count 
      count = count + 1 
      XMLdict['XS'][count] = {}
      #print tabulationXML.attrib.get('name')
      for libraryXML in tabulationXML.getiterator('library'):
        #print libraryXML.attrib.get('lib_name')
        currentMat = libraryXML.attrib.get('lib_name')
        XMLdict['XS'][count][libraryXML.attrib.get('lib_name')] = {}
        for isotopeXML in libraryXML.getiterator('isotope'):
          currentIsotope = isotopeXML.attrib.get('id')
          currentType = isotopeXML.attrib.get('type')
          #print currentType
          reactionList = [j.tag for j in isotopeXML]
          #print reactionList
          XMLdict['XS'][count][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')+isotopeXML.attrib.get('type')] = {}
          #print XMLdict
          for k in xrange (0, len(reactionList)):
            XMLdict['XS'][count][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')+isotopeXML.attrib.get('type')][reactionList[k]] = {}
            for groupXML in isotopeXML.getiterator(reactionList[k]):
              individualGroup = [x.strip() for x in groupXML.attrib.get('g').split(',')]
              individualGroupValues = [y.strip() for y in groupXML.text.split(',')]
              #print (individualGroup+individualGroupValues)
              for position in xrange(0,len(individualGroup)):
                #print (reactionList[k]+"\t\t"+individualGroup[position]+' '+individualGroupValues[position]) 
                XMLdict['XS'][count][libraryXML.attrib.get('lib_name')][isotopeXML.attrib.get('id')+isotopeXML.attrib.get('type')][reactionList[k]][individualGroup[position]] = individualGroupValues[position]
    #print XMLdict
    return XMLdict

  def tabMapping(self, tab):
    """
      link the tabulation number to the actual tabulation points 
      IN: tab: string (tabulation number)
      OUT: tabList: list (all the tabulation parameters gathered in a list)
    """
    mappingTree = ET.parse('tabMapping.xml')
    mappingRoot = mappingTree.getroot()
    for tabulationXML in mappingRoot.getiterator('tabulation'):
      if tab == tabulationXML.attrib.get('set'):
        tabList = []
        valueList = []
        for tabXML in tabulationXML.getiterator('tab'):
          tabList.append(tabXML.attrib.get('name')) 
          valueList.append(tabXML.text) 
    return tabList, valueList
    
  def prettify(self, elem):
    """
      Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
    
    
  def generateXML(self):
    """
      create an XML file from the interface  
      In: None
      Out: XS.xml (xml file) 
    """
    top = Element('scaling_library')
    
    comment = Comment('Generated by rouxpn')
    top.append(comment)
    print self.listedDict
    for XS in self.listedDict.iterkeys():
      for tabulation in self.listedDict.get('XS').iterkeys():
        topChild = SubElement(top, 'tabulation')
        tabList, valueList = self.tabMapping(tabulation)
        for i in xrange (0,len(tabList)):
          tabChild = SubElement(topChild, 'tab', {'name':tabList[i]})
          tabChild.text = valueList[i]
        for material in self.listedDict.get('XS').get(tabulation).iterkeys():
          tabulationChild = SubElement(topChild, 'library', {'lib_name':material.lower()})
          for isotope in self.listedDict.get('XS').get(tabulation).get(material).iterkeys():
            for type in self.listedDict.get('XS').get(tabulation).get(material).get(isotope).iterkeys():
              libraryChild = SubElement(tabulationChild, 'isotope', {'id':isotope.lower(), 'type':type.lower()})
              for reaction in self.listedDict.get('XS').get(tabulation).get(material).get(isotope).get(type).iterkeys():
                groupList = [] 
                valueList = []
                count = 0 
                for group,value in self.listedDict.get('XS').get(tabulation).get(material).get(isotope).get(type).get(reaction).iteritems():
                  count = count + 1 
                  numberOfGroupsPerturbed = len(self.listedDict.get('XS').get(tabulation).get(material).get(isotope).get(type).get(reaction).keys())
                  groupList.append(group)
                  valueList.append(value)
                  group.join(',')
                  if count == numberOfGroupsPerturbed:
                    groups = ','.join(str(e) for e in groupList)
                    values = ','.join(str(f) for f in valueList)
                    #print groups
                    #print values
                    reactionChild = SubElement(libraryChild, reaction.lower(), {'g':groups})
                    reactionChild.text = values
    
    file_obj = open('XS.xml', 'w')
    file_obj.write(self.prettify(top))
    #print self.prettify(top)
  
  def clean_empty(self, leanDict):
    """
      remove all the empty string in the nested dictionary reconstructedDict  
      In: reconstructedDict  (nested dictionary)
      Out: leanReconstructedDict (nested dictionary) 
    """
    if not isinstance(leanDict, (dict, list)):
        return leanDict
    if isinstance(leanDict, list):
        return [v for v in (self.clean_empty(v) for v in leanDict) if v]
    return {k: v for k, v in ((k, self.clean_empty(v)) for k, v in leanDict.items()) if v} 

   
  def __init__(self, inputFiles, **pertDict):
    """
      Parse the PHISICS XS.xml data file   
      In: XS.xml
      Out: None 
    """
    self.pertDict = pertDict
    #print self.pertDict
    #print "\n\n\n"
    #print inputFiles
    
    for key, value in self.pertDict.iteritems(): 
      self.pertDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values   
    self.inputFiles = inputFiles
    self.tree = ET.parse(self.inputFiles)
    self.root = self.tree.getroot()
    self.listedDict = self.fileReconstruction(self.pertDict)
    self.generateXML()
    self.printInput()

  def fileReconstruction(self, deconstructedDict):
    """
      Converts the formatted dictionary -> {'XS|FUEL1|U235|FISSION|1':1.30, 'XS|FUEL2|U238|ABS|2':4.69} 
      into a dictionary of dictionaries that has the format -> {'XS':{'FUEL1':{'U235':{'FISSION':{'1':1.30}}}}, 'FUEL2':{'U238':{'ABS':{'2':4.69}}}}
      In: Dictionary deconstructedDict
      Out: Dictionary of dictionaries reconstructedDict 
    """
    #print deconstructedDict
    reconstructedDict           = {}
    perturbedPhysicalParameters = []
    perturbedTabulationPoint    = []
    perturbedMaterials          = []
    perturbedIsotopes           = []
    perturbedTypes           = []
    perturbedReactions          = []
    perturbedGroups             = []
    
    
    pertDictSet = set(self.pertDict)
    deconstructedDictSet = set(deconstructedDict)
    #for variable in pertDictSet.intersection(deconstructedDictSet):
    for i in pertDictSet.intersection(deconstructedDictSet): 
      splittedKeywords = i.split('|')
      perturbedPhysicalParameters.append(splittedKeywords[0])
      perturbedTabulationPoint.append(splittedKeywords[1])
      perturbedMaterials.append(splittedKeywords[2])
      perturbedIsotopes.append(splittedKeywords[3])
      perturbedTypes.append(splittedKeywords[4])
      perturbedReactions.append(splittedKeywords[5])
      perturbedGroups.append(splittedKeywords[6])  
    
    #print perturbedReactions
    for i in xrange (0,len(perturbedPhysicalParameters)):
      reconstructedDict[perturbedPhysicalParameters[i]] = {}
      for j in xrange (0,len(perturbedTabulationPoint)):
        reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]] = {} 
        for k in xrange (0,len(perturbedMaterials)):
          reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]] = {}
          for l in xrange (0,len(perturbedIsotopes)):
            reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]][perturbedIsotopes[l]] = {} 
            for m in xrange (0,len(perturbedReactions)):
              reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]][perturbedIsotopes[l]][perturbedTypes[m]] = {} 
              for n in xrange (0,len(perturbedReactions)):
                reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]][perturbedIsotopes[l]][perturbedTypes[m]][perturbedReactions[n]] = {}
                for o in xrange (0,len(perturbedGroups)):
                  reconstructedDict[perturbedPhysicalParameters[i]][perturbedTabulationPoint[j]][perturbedMaterials[k]][perturbedIsotopes[l]][perturbedTypes[m]][perturbedReactions[n]][perturbedGroups[o]] = {}
    #print reconstructedDict
    for typeKey, value in deconstructedDict.iteritems():
      if typeKey in pertDictSet:
        keyWords = typeKey.split('|')
        #print keyWords
        reconstructedDict[keyWords[0]][keyWords[1]][keyWords[2]][keyWords[3]][keyWords[4]][keyWords[5]][keyWords[6]] = value
    #print reconstructedDict  
    leanReconstructedDict = self.clean_empty(reconstructedDict)
    #print leanReconstructedDict
    return leanReconstructedDict
   
    
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
    #print templatedNewXMLdict 
    templatedNewXMLdict = self.listedDict
    #print templatedNewXMLdict
    count = 0
    
   
    copyfile('modif.xml', self.inputFiles)  
   

