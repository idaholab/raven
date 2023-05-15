# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created May 4, 2016

@author: wangc
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os
import sys
import copy
import numpy as np

class YakMultigroupLibraryParser():
  """
    Class used to parse Yak multigroup cross section libraries, read the user provided alias files,
    perturb the libraries with variables defined in alias files and values from Raven Sampler.
    Cross sections will be reblanced based on provided information.
    In addition, this interface can be used to perturb Fission, Capture, TotalScattering, Nu, Kappa
    for given isotopes inside the libraries. In addition, the user can also perturb Transport if it
    exists in the XS library. In this case, we will treat Transport as independent variable.
    In the future, we may need to add the capability to perturb the Tablewise and Librarywise data.
  """
  #Functions Used for Reading Yak Multigroup Cross Section Library (Also including some functions for checking and recalculations)
  def __init__(self,inputFiles):
    """
      Constructor, parse the input files
      @ In, inputFiles, list(str), string list of input files that might need parsing.
      @ Out, None.
    """
    self.inputFiles     = inputFiles
    self.libs           = {} #dictionaries for libraries of tabulated xs values
    self.xmlsDict       = {} #connects libraries name and tree objects: {libraryName:objectTree}
    self.filesDict      = {} #connects files and libraries name: {file:librariesName}
    self.filesMap       = {} #connects names of files  and libraries name: {fileName:librariesName}
    self.matLibMaps     = {} #connects material id and libraries name: {matID:librariesName}
    self.matTreeMaps    = {} #connects material id and xml objects: {matID:objectTree}
    self.defaultNu      = 2.43 #number of neutrons per fission
    self.defaultKappa   = 195*1.6*10**(-13) #Energy release per fission
    self.aliases        = {} #alias to XML node dict
    self.validReactions = ['Total','Fission','Removal','Transport','Scattering','nuFission','kappaFission',
                           'FissionSpectrum','DNFraction','DNSpectrum','NeutronVelocity','DNPlambda','Absorption',
                           'Capture','Nalpha','NGamma','Flux','N2Alpha','N2N','N3N','N4N','NNProton','NProton',
                           'NDeuteron','NTriton'] #These are all valid reactions for Yak XS format
    self.perturbableReactions = ['Fission','Capture','TotalScattering','Nu','Kappa','Transport'] #These are all valid perturbable reactions for RAVEN
    self.level0Element  = 'Multigroup_Cross_Section_Libraries' #root element tag is always the same for Yak XS format
    self.level1Element  = 'Multigroup_Cross_Section_Library'   #level 1 element tag is always Multigroup_Cross_Section_Library
    self.level2Element  = ['Tabulation','AllReactions','TablewiseReactions','LibrarywiseReactions'] #These are some of the level 2 element tag with string vector xmlnode.text, without xml subnodes
    self.toBeReadXML    = [] #list of XML nodes that need to be read.
    self.libsKeys       = {} #dict to store library keys: {mglib_ID:{gridIndex:{IsotopeName:[reactions]}}}

    #read in cross-section files, unperturbed files
    for xmlFile in inputFiles:
      if not os.path.exists(xmlFile.getPath()):
        raise IOError('The following Yak multigroup cross section library file: ' + xmlFile + ' is not found')
      tree = ET.parse(xmlFile.getAbsFile())
      root = tree.getroot()
      if root.tag == self.level0Element:
        rootName = root.attrib['Name'].strip()
        self.xmlsDict[rootName] = tree
        self.filesDict[xmlFile] = rootName
        self.filesMap[xmlFile.getFilename()] = rootName
        self.libs[rootName] = {}
        self.libsKeys[rootName] = {}
        mgDict = self.libs[rootName]
        mgDictKeys =  self.libsKeys[rootName]
        self.nGroup = int(root.attrib['NGroup']) #total number of neutron energy groups
        for mgLib in root:
          mgLibID = mgLib.attrib['ID'].strip()
          self.matLibMaps[mgLibID] = rootName
          self.matTreeMaps[mgLibID] = mgLib
          mgDict[mgLibID] = {}
          mgDictKeys[mgLibID] = {}
          self._readYakXSInternal(mgLib,mgDict[mgLibID],mgDictKeys[mgLibID])
          self._readAdditionalYakXS(mgLib,mgDict[mgLibID])
          self._checkYakXS(mgDict[mgLibID],mgDictKeys[mgLibID])
      else:
        msg = 'In YakMultigroupLibraryParser, root element of XS file is always ' + self.level0Element + ';\n'
        msg = msg + 'while the given XS file has different root element: ' + root.tag + "!"
        raise IOError(msg)

  def initialize(self,aliasFiles):
    """
      Parse the input alias files
      @ In, aliasFiles, list, list of input alias files
      @ Out, None
    """
    self.aliases = {}
    self.aliasesNGroup = {}
    self.aliasesType = {}
    for xmlFile in aliasFiles:
      if not os.path.exists(xmlFile.getPath()):
        raise IOError('The following Yak cross section alias file: ' + xmlFile + ' is not found!')
      aliasTree = ET.parse(xmlFile.getAbsFile())
      root = aliasTree.getroot()
      if root.tag != self.level0Element:
        raise IOError('Invalid root tag: ' + root.tag +' is provided.' + ' The valid root tag should be: ' + self.level0Element)
      rootName = root.attrib['Name'].strip()
      if rootName in self.aliases.keys():
        raise IOError('Duplicated libraries name: ' + rootName + ' is found in provided alias files!')
      self.aliases[rootName] ={}
      self.aliasesNGroup[rootName] = int(root.attrib['NGroup'])
      aliasNGroup = int(root.attrib['NGroup'])
      self.aliasesType[rootName] = root.attrib['Type'].strip()
      subAlias = self.aliases[rootName]
      for child in root:
        if child.tag != self.level1Element:
          raise IOError('Invalid subnode tag: ' + child.tag +' is provided.' + ' The valid subnode tag should be: ' + self.level1Element)
        libID = child.attrib['ID'].strip()
        subAlias[libID] = {}
        #read the cross section alias for each library (or material)
        self._readXSAlias(child,subAlias[libID],aliasNGroup)

  def _readXSAlias(self,xmlNode,aliasXS,aliasXSGroup):
    """
      Read the cross section alias for each library
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element contains the alias information
      @ In, aliasXS, dict, dictionary used to store the cross section aliases
      @ In, aliasXSGroup, the energy group defined in provided alias file
      @ Out, None
    """
    for child in xmlNode:
      if child.tag in self.perturbableReactions:
        grid = self._stringSpacesToTuple(child.attrib['gridIndex'])
        if grid not in aliasXS.keys():
          aliasXS[grid] = {}
        mat = child.attrib['mat'].strip()
        if mat not in aliasXS[grid].keys():
          aliasXS[grid][mat] = {}
        mt = child.tag
        if mt not in aliasXS[grid][mat].keys():
          aliasXS[grid][mat][mt] = [None]*aliasXSGroup
        groupIndex = child.get('gIndex')
        if groupIndex == None:
          varsList = list(var.strip() for var in child.text.strip().split(','))
          if len(varsList) != aliasXSGroup:
            msg = str(aliasXSGroup) + ' variables should be provided for ' + child.tag + ' of material ' + child.attrib['mat']
            msg = msg + ' in grid ' + child.attrib['gridIndex'] + '! '
            msg = msg + "Only " + len(varsList) + " variables is provided!"
            raise IOError(msg)
          aliasXS[grid][mat][mt] = varsList
        else:
          pertList = list(var.strip() for var in child.text.strip().split(','))
          groups = self._stringSpacesToListInt(groupIndex)
          if len(groups) != len(pertList):
            raise IOError('The group indices is not consistent with the perturbed variables list')
          for i,g in enumerate(groups):
            aliasXS[grid][mat][mt][g-1] = pertList[i]
      else:
        raise IOError('The reaction ' + child.tag + ' can not be perturbed!')

  def _stringSpacesToTuple(self,text):
    """
      Turns a space-separated text into a tuple
      @ In, text, string, string
      @ Out, members, list(int), list of members
    """
    members = tuple(int(c.strip()) for c in text.strip().split())
    return members

  def _stringSpacesToListInt(self,text):
    """
      Turns a space-separated text into a list of int
      @ In, text, string, string
      @ Out, members, list(int), list of members
    """
    members = list(int(c.strip()) for c in text.strip().split())
    return members

  def _stringSpacesToListFloat(self,text):
    """
      Turns a space-separated text into a list of float
      @ In, text, string, string
      @ Out, members, list(float), list of members
    """
    members = list(float(c.strip()) for c in text.strip().split())
    return members

  def _stringSpacesToNumpyArray(self,text):
    """
      Turns a space-separated text into a list of float
      @ In, text, string, string
      @ Out, members, numpy.array, list of members
    """
    members = np.asarray(list(float(c.strip()) for c in text.strip().split()))
    return members

  def _stringSpacesToListString(self,text):
    """
      Turns a space-separated text into a list of constituent members
      @ In, text, string, string
      @ Out, members, list(string), list of members
    """
    members = list(c.strip() for c in text.strip().split())
    return members

  def _readYakXSInternal(self,library,pDict,keyDict):
    """
      Load the Yak multigroup library
      @ In, library, xml.etree.ElementTree.Element, xml element for cross section library defined in yak cross section files
      @ In, pDict, dict, dictionary to store the multigroup library
      @ In, keyDict, dict, dictionary to store the multigroup library node names, use to trace the cross section types for given isotope at given gridIndex
      @ Out, None
    """
    #read data for this library
    for subNode in library:
      # read tabulates
      self._readNextLevel(subNode,pDict,keyDict)

  def _parentAction(self,parentNode,libDict,keyDict):
    """
      Default action for parent nodes with children
      @ In, parentNode, xml.etree.ElementTree.Element, xml element
      @ In, libDict, dict, dictionary used to store the cross sections
      @ In, keyDict, dict, dictionary to store xml node names
      @ Out, None
    """
    for child in parentNode:
      self._readNextLevel(child,libDict,keyDict)

  def _readNextLevel(self,xmlNode,pDict,keyDict):
    """
      Uses xmlNode tag to determine next reading algorithm to perform.
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element
      @ In, pDict, dict, dictionary used to store the cross sections
      @ In, keyDict, dict, dictionary to store the xml node names
      @ Out, None
    """
    # case: child.tag
    if xmlNode.tag in self.level2Element:
      pDict[xmlNode.tag] = self._stringSpacesToListString(xmlNode.text)
    elif xmlNode.tag == 'ReferenceGridIndex':
      pDict[xmlNode.tag] = self._stringSpacesToListInt(xmlNode.text)
    elif xmlNode.tag == 'Table':
      dictKey = self._stringSpacesToTuple(xmlNode.attrib['gridIndex'])
      pDict[dictKey] = {}
      keyDict[dictKey] = {}
      self._parentAction(xmlNode,pDict[dictKey],keyDict[dictKey])
    elif xmlNode.tag == 'Tablewise':
      pDict[xmlNode.tag] = {}
      self._readTablewise(xmlNode,pDict[xmlNode.tag])
    elif xmlNode.tag == 'Isotope':
      #check if the subnode includes the XS
      pDict[xmlNode.attrib['Name'].strip()] = {}
      keyDict[xmlNode.attrib['Name'].strip()] = []
      hasSubNode = False
      for child in xmlNode:
        if child != None:
          hasSubNode = True
          break
      if hasSubNode:
        self._readIsotopeXS(xmlNode,pDict[xmlNode.attrib['Name'].strip()],keyDict[xmlNode.attrib['Name'].strip()])
    #store the xmlNode tags that have not been parsed
    else:
      self.toBeReadXML.append(xmlNode.tag)

  def _readIsotopeXS(self,node,pDict,keyList):
    """
      Reads in isotope cross section entries
      @ In, node, xml.etree.ElementTree.Element, xml element
      @ In, pDict, dict, dictionary used to store the cross section data for given isotope
      @ In, keyList, list, list of cross section reaction types
      @ Out, None
    """
    #the xs structure is same as Tablewise xs data
    self._readTablewise(node,pDict,keyList)

  def _readLibrarywise(self,node,pDict):
    """
      Reads in Librarywise entries
      @ In, node, xml.etree.ElementTree.Element, xml element
      @ In, pDict, dict, dictionary used to store the librarywise cross section data
      @ Out, None
    """
    #the xs structure is same as Tablewise xs data
    self._readTablewise(node,pDict)

  def _readTablewise(self,node,pDict,keyList=None):
    """
      Reads in Tablewise entries
      @ In, node, xml.etree.ElementTree.Element, xml element
      @ In, pDict, dict, dictionary used to store the cross section data for given tablewise entry
      @ In, keyList, list, list of cross section reaction types
      @ Out, None
    """
    orderScattering = int(node.attrib['L'])
    for child in node:
      #The following can be used to check if type of the cross sections is valid or not
      #if child.tag not in self.validReactions:
      #  raise IOError("The following reaction type " + child.tag + " is not valid!")
      if keyList != None:
        keyList.append(child.tag)
      #read all xs for all reaction types except Scattering
      if child.tag != 'Scattering':
        pDict[child.tag]= self._stringSpacesToNumpyArray(child.text)
      #read xs sections for Scattering
      else:
        #read scattering
        self._readScatteringXS(child,pDict,orderScattering)

  def _readScatteringXS(self,node,pDict,orderScattering):
    """
      Reads the Scattering block for Yak multigroup cross section library
      @ In, node, xml.etree.ElementTree.Element, xml element
      @ In, pDict, dict, dictionary used to store the scattering cross sections
      @ In, orderScattering, int, order of spherical harmonics expansioin for scattering
      @ Out, None
   """
    has_profile = False
    pDict['ScatteringOrder'] = orderScattering
    if int(node.get('profile')) == 1:
      has_profile = True
    if has_profile:
      for child in node:
        if child.tag == 'Profile':
          profileValue = self._stringSpacesToListInt(child.text)
          pDict['ScatterStart'] = profileValue[0::2]
          pDict['ScatterEnd'] = profileValue[1::2]
        elif child.tag == 'Value':
          scatteringValue = self._stringSpacesToNumpyArray(child.text) #store in 1-D array
      pDict['Scattering'] = np.zeros((self.nGroup*(orderScattering+1),self.nGroup))
      ip = 0
      for l in range(orderScattering+1):
        for g in range(self.nGroup):
          for gr in range(pDict['ScatterStart'][g+l*self.nGroup]-1,pDict['ScatterEnd'][g+l*self.nGroup]):
            pDict['Scattering'][g+l*self.nGroup][gr] = scatteringValue[ip]
            ip += 1
    else:
      scatteringValue = self._stringSpacesToNumpyArray(child.text) #store in 1-D array
      pDict[child.tag] = scatteringValue.reshape((self.nGroup*(orderScattering+1),self.nGroup))
    #calculate Total Scattering
    totScattering = np.zeros(self.nGroup)
    for g in range(self.nGroup):
      totScattering[g] = np.sum(pDict['Scattering'][0:self.nGroup,g])
    pDict['TotalScattering'] = totScattering

  def _readAdditionalYakXS(self,xmlNode,pDict):
    """
      Read addition cross sections that have not been read via method self._readYakXSInternal,
      such as Tabulation Grid, Librarywise.
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element
      @ In, pDict, dict, dictionary used to stores the cross section data
      @ Out, None
    """
    for child in xmlNode:
      #read the tabulation grid
      if child.tag in pDict['Tabulation']:
        pDict[child.tag] = self._stringSpacesToNumpyArray(child.text)
        self.toBeReadXML.remove(child.tag)
      #read the Librarywise cross section data
      elif child.tag == 'Librarywise':
        pDict[child.tag] = {}
        self._readLibrarywise(child,pDict[child.tag])
        self.toBeReadXML.remove(child.tag)
    if len(self.toBeReadXML) != 0:
      raise IOError('The following nodes xml' + str(self.toBeReadXML) + ' have not been read yet!')

  def _checkYakXS(self,pDict,keyDict):
    """
      Recalculate some undefined xs, such as 'Nu', 'Fission', 'Capture'.
      @ In, pDict, dict, dictionary stores all the cross section data for given multigroup library (or material)
      @ In, keyDict, dict, dictionary stores the multigroup library node names, use to trace the cross section types for given isotope at given gridIndex
      @ Out, None
    """
    #make sure pDict include the cross sections, if not, copy from Tablewise data
    for gridKey,isotopeDict in keyDict.items():
      for isotopeKey,reactionList in isotopeDict.items():
        if len(reactionList) == 0:
          if 'Tablewise' in pDict[gridKey].keys():
            pDict[gridKey][isotopeKey] = copy.deepcopy(pDict[gridKey]['Tablewise'])
          elif 'Librarywise' in pDict[gridKey].keys():
            pDict[gridKey][isotopeKey] = copy.deepcopy(pDict[gridKey]['Librarywise'])
          else:
            raise IOError('The isotope: ' + isotopeKey + ' is provided, but the required cross sections are not found in the library!')
        else:
          #add the missing cross sections from the Tablewise or Librarywise dictionary. Tablewise first, and then Librarywise
          if 'Tablewise' in pDict[gridKey].keys():
            for key,value in pDict[gridKey]['Tablewise'].items():
              if key not in pDict[gridKey][isotopeKey].keys():
                pDict[gridKey][isotopeKey][key] = value
          if 'Librarywise' in pDict.keys():
            for key,value in pDict['Librarywise'].items():
              if key not in pDict[gridKey][isotopeKey].keys():
                pDict[gridKey][isotopeKey][key] = value
        #calculate some independent cross sections if they are not in pDict
        #these cross sections can be: fission, total scattering, capture, nu, kappa
        self._recalculateYakXS(pDict[gridKey][isotopeKey])

  def _recalculateYakXS(self,reactionDict):
    """
      Recalculate some undefined xs, such as 'Nu', 'Fission', 'Capture'.
      @ In, reactionDict, dict, dictionary stores the cross section data for given isotope
      @ Out, None
    """
    ### fission, nu, kappa
    reactionList = reactionDict.keys()
    if 'nuFission' in reactionList:
      if 'Fission' not in reactionList:
        #calculate Fission using default Nu
        reactionDict['Fission'] = reactionDict['nuFission']/self.defaultNu
        reactionDict['Nu'] = np.ones(self.nGroup)*self.defaultNu
      else:
        nu = []
        for i in range(self.nGroup):
          if reactionDict['Fission'][i] != 0:
            nu.append(reactionDict['nuFission'][i]/reactionDict['Fission'][i])
          else:
            nu.append(self.defaultNu)
        reactionDict['Nu'] = np.asarray(nu)
      if 'kappaFission' not in reactionList:
        #calculate kappaFission using default kappa
        reactionDict['kappaFission'] = self.defaultKappa * reactionDict['Fission']
        reactionDict['Kappa'] = np.ones(self.nGroup) * self.defaultKappa
      else:
        kappa = []
        for i in range(self.nGroup):
          if reactionDict['Fission'][i] != 0:
            kappa.append(reactionDict['kappaFission'][i]/reactionDict['Fission'][i])
          else:
            kappa.append(self.defaultKappa)
        reactionDict['Kappa'] = np.asarray(kappa)
    # enable Transport perturbation if it exists.
    if 'Transport' in reactionList:
      reactionDict['perturbTransport'] = True
    else:
      reactionDict['perturbTransport'] = False
    #check and calculate total or  transport cross sections
    if 'Total' not in reactionList:
      if 'Transport' not in reactionList:
        raise IOError('Total and Transport cross sections are not found in the cross section input file, at least one of them should be provided!')
      else:
        #calculate total cross sections
        if 'Scattering' not in reactionList:
          reactionDict['Total'] = copy.copy(reactionDict['Transport'])
        elif reactionDict['ScatteringOrder'] == 0:
          reactionDict['Total'] = copy.copy(reactionDict['Transport'])
        else:
          reactionDict['Total'] = reactionDict['Transport'] + np.sum(reactionDict['Scattering'][self.nGroup:2*self.nGroup])
    else:
      if 'Transport' not in reactionList:
        #calculate transport cross sections
        if 'Scattering' not in reactionList:
          reactionDict['Transport'] = copy.copy(reactionDict['Total'])
        elif reactionDict['ScatteringOrder'] == 0:
          reactionDict['Transport'] = copy.copy(reactionDict['Total'])
        else:
          reactionDict['Transport'] = reactionDict['Total'] - np.sum(reactionDict['Scattering'][self.nGroup:2*self.nGroup])

    #Metod 1: Currently, rattlesnake will not check the consistent of provided cross sections, rattlesnake will only use Total,
    #Scattering and nuFission for the transport calculation. In this case, we will recalculate the rest cross sections
    #based on Total, Scattering and Fission.
    if 'Scattering' in reactionList:
      reactionDict['Absorption'] = reactionDict['Total'] - reactionDict['TotalScattering']
    else:
      if self.nGroup == 1 and 'Absorption' in reactionList:
        reactionDict['Scattering'] = reactionDict['Total'] - reactionDict['Absorption']
        reactionDict['TotalScattering'] = copy.copy(reactionDict['Scattering'])
      else:
        reactionDict['Absorption'] = copy.copy(reactionDict['Total'])
    #calculate capture cross sections
    if 'nuFission' in reactionList:
      reactionDict['Capture'] = reactionDict['Absorption'] - reactionDict['Fission']
    else:
      reactionDict['Capture'] = copy.copy(reactionDict['Absorption'])

    #Method 2: The following can be used to recalculate the unknown cross sections
    #either method 1 or method 2 can be used.
    '''
    #calculate scattering cross sections
    if 'Scattering' not in reactionList and self.nGroup == 1:
      reactionDict['Scattering'] = reactionDict['Total'] - reactionDict['Absorption']
      reactionDict['TotalScattering'] = copy.copy(reactionDict['Scattering'])

    #calculate absorption
    if 'Absorption' not in  reactionList:
      if 'Scattering' in reactionList:
        reactionDict['Absorption'] = reactionDict['Total'] - reactionDict['TotalScattering']
      else:
        reactionDict['Absorption'] = copy.copy(reactionDict['Total'])
      #calculate capture cross sections
      if 'Capture' not in reactionList:
        if 'nuFission' in reactionList:
          reactionDict['Capture'] = reactionDict['Absorption'] - reactionDict['Fission']
        else:
          reactionDict['Capture'] = copy.copy(reactionDict['Absorption'])
    else:
      #calculate capture cross sections
      if 'Capture' not in reactionList:
        if 'nuFission' in reactionList:
          reactionDict['Capture'] = reactionDict['Absorption'] - reactionDict['Fission']
        else:
          reactionDict['Capture'] = copy.copy(reactionDict['Absorption'])
      #calculate scattering cross sections
      if 'Scattering' not in reactionList and self.nGroup == 1:
        reactionDict['Scattering'] = reactionDict['Total'] - reactionDict['Absorption']
        reactionDict['TotalScattering'] = copy.copy(reactionDict['Scattering'])
    '''

  def perturb(self,**Kwargs):
    """
      Perturb the input cross sections based on the information provided by alias files
      @ In, Kwargs, dict, dictionary containing raven sampled var value
      @ Out, None
    """
    self.pertLib = copy.deepcopy(self.libs)
    self.modDict = Kwargs['SampledVars']
    pertFactor = copy.deepcopy(self.aliases)
    #generate the pertLib
    for libKey, libValue in pertFactor.items():
      aliasType = self.aliasesType[libKey]
      self._computePerturbations(libValue,self.pertLib[libKey],aliasType)
    for libsKey, libDict in pertFactor.items():
      aliasType = self.aliasesType[libsKey]
      for libID, gridDict in libDict.items():
        self._rebalanceXS(self.pertLib[libsKey][libID],gridDict,pertFactor[libsKey][libID],aliasType)

  def _computePerturbations(self,factors,lib,aliasType):
    """
      compute the perturbed values for input variables
      @ In, factors, dict, dictionary contains all perturbed input variables, and these variables will be
        replaced by the actual perturbed factors after this method is called.
      @ In, lib, dict, dictionary contains all the values of input variables
      @ In, aliasType, string, the type for provided alias file
      @ Out, None
    """
    for libKey, libValue in factors.items():
      if type(libValue) == dict:
        self._computePerturbations(libValue,lib[libKey],aliasType)
      elif type(libValue) == list:
        groupValues = []
        for var in libValue:
          if var in self.modDict.keys():
            groupValues.append(self.modDict[var])
          elif var == None:
            if aliasType == 'rel':
              groupValues.append(1.0)
            elif aliasType == 'abs':
              groupValues.append(0.0)
          else:
            raise IOError('The user wants to perturb ' + var + ', but this variable is not defined in the Sampler!')
        groupValues = np.asarray(groupValues)
        factors[libKey] = groupValues
        if not lib['perturbTransport'] and libKey == 'Transport':
          raise IOError('Transport can not be perturbed since it does not exist in the input XS library!')
        if aliasType == 'rel':
          lib[libKey] *= groupValues
        elif aliasType == 'abs':
          lib[libKey] += groupValues

  def _rebalanceXS(self,libDict,libKeyDict,factorDict,aliasType):
    """
      Using the perturbed cross sections to recalculate other dependent cross sections
      @ In, libDict, dict, dictionary used to store the cross section data
      @ In, libKeyDict, dict, dictionary used to store the cross section types
      @ In, factorDict, dict, dictionary used to store the perturbation factors
      @ In, aliasType, string, the type for provided alias file
      @ Out, None
    """
    for gridKey,isotopeDict in libKeyDict.items():
      for isotopeKey,reactionList in isotopeDict.items():
        #recalculate some dependent cross sections
        self._rebalanceYakXS(libDict[gridKey][isotopeKey],factorDict[gridKey][isotopeKey],aliasType)

  def _rebalanceYakXS(self,reactionDict,perturbDict,aliasType):
    """
      Recalculate some depedent xs, such as 'Total', 'Absorption', 'Scattering', 'nuFission', 'kappaFission',
      Removal, Transport.
      @ In, reactionDict, dict, dictionary used to store the cross section data
      @ In, perturbDict, dict, dictionary used to store the perturbation factors
      @ In, aliasType, string, the type for provided alias file
      @ Out, None
    """
    #fission, nu, kappa, capture, total scattering are assumed to be independent cross section types
    reactionList = perturbDict.keys()
    hasTotalScattering = False
    if 'TotalScattering' in reactionList:
      hasTotalScattering = True
    if 'Fission' in reactionDict.keys():
      reactionDict['nuFission'] = reactionDict['Fission']*reactionDict['Nu']
      reactionDict['kappaFission'] = reactionDict['Fission']*reactionDict['Kappa']
      reactionDict['Absorption'] = reactionDict['Fission'] + reactionDict['Capture']
    else:
      reactionDict['Absorption'] = copy.copy(reactionDict['Capture'])
    reactionDict['Total'] = reactionDict['Absorption'] + reactionDict['TotalScattering']
    if hasTotalScattering:
      #total scattering are perturbed
      #recalculate Scattering Cross Sections
      for g in range(self.nGroup):
        if aliasType == 'rel':
          reactionDict['Scattering'][0:self.nGroup,g] *= perturbDict['TotalScattering'][g]
        elif aliasType == 'abs':
          factor = perturbDict['TotalScattering'][g]/self.nGroup
          reactionDict['Scattering'][0:self.nGroup,g] += factor
    #recalculate Removal cross sections
    reactionDict['Removal'] = np.asarray(list(reactionDict['Total'][g] - reactionDict['Scattering'][g][g] for g in range(self.nGroup)))
    #recalculate Transport cross sections
    if not reactionDict['perturbTransport']:
      if reactionDict['Scattering'].shape[0] >= self.nGroup*2:
        reactionDict['Transport'] = reactionDict['Total'] - np.sum(reactionDict['Scattering'][self.nGroup:self.nGroup*2])
      else:
        #recalculate Transport cross sections
        reactionDict['Transport'] = copy.copy(reactionDict['Total'])

  def _addSubElementForIsotope(self,xmlNode):
    """
      Check if there is a subelement under node Isotope, if not, add the one from the Tablewise or Librarywise
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element
      @ Out, None
    """
    tableWise = xmlNode.find('Tablewise')
    if tableWise is not None:
      for child in tableWise:
        for isotope in xmlNode.findall('Isotope'):
          if isotope.find(child.tag) is not None:
            continue
          isotope.append(copy.deepcopy(child))
    libraryWise = xmlNode.find('Librarywise')
    if libraryWise is not None:
      for child in libraryWise:
        for isotope in xmlNode.findall('Isotope'):
          if isotope.find(child.tag) is not None:
            continue
          isotope.append(copy.deepcopy(child))

  def _replaceXMLNodeText(self,xmlNode,reactionDict):
    """
      Replace the xml node text (cross section values) with perturbed cross section values.
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element
      @ In, reactionDict, dict, dictionary contains the cross sections and their values
      @ Out, None
    """
    for child in xmlNode:
      if child.tag in reactionDict.keys() and child.tag != 'Scattering':
        child.text = '  '.join(['%.5e' % num for num in reactionDict[child.tag]])
      elif child.tag in reactionDict.keys() and child.tag == 'Scattering':
        for childChild in child:
          if childChild.tag == 'Value':
            msg = ''
            for g in range(reactionDict[child.tag].shape[0]):
              msg = msg + '\n' + '            '+' '.join(['%.5e' % num for num in reactionDict[child.tag][g][reactionDict['ScatterStart'][g]-1:reactionDict['ScatterEnd'][g]]])
            childChild.text = msg + '\n'

  def _prettify(self,tree):
    """
      Script for turning XML tree to be more user friendly.
      @ In, tree, xml.etree.ElementTree object, the tree form of an input file
      @ Out, pretty, string, the entire contents of the desired file to write
    """
    #make the first pass at pretty.  This will insert way too many newlines, because of how we maintain XML format.
    pretty = pxml.parseString(ET.tostring(tree.getroot())).toprettyxml(indent='  ')
    return pretty

  def writeNewInput(self,inFiles=None,**Kwargs):
    """
      Generates a new input file with the existing parsed dictionary.
      @ In, Kwargs, dict, dictionary containing raven sampled var value
      @ In, inFiles, list, list of input files
      @ Out, None.
    """
    outFiles = {}
    if inFiles == None:
      for fileInp,libKey in self.filesDict.items():
        outFiles[fileInp.getAbsFile()] = libKey
    else:
      for inFile in inFiles:
        if inFile.getFilename() in self.filesMap.keys():
          libsKey = self.filesMap[inFile.getFilename()]
          if libsKey not in self.aliases.keys():
            continue
          outFiles[inFile.getAbsFile()] = libsKey
    for outFile,libsKey in outFiles.items():
      tree = self.xmlsDict[libsKey]
      if libsKey not in self.aliases.keys():
        continue
      root = tree.getroot()
      for child in root:
        libID = child.attrib['ID'].strip()
        if libID not in self.aliases[libsKey].keys():
          continue
        for table in child.findall('Table'):
          gridIndex = self._stringSpacesToTuple(table.attrib['gridIndex'])
          if gridIndex in self.aliases[libsKey][libID].keys():
            self._addSubElementForIsotope(table)
            for subNode in table:
              if subNode.tag == 'Isotope':
                mat = subNode.attrib['Name'].strip()
                if mat not in self.aliases[libsKey][libID][gridIndex].keys():
                  continue
                self._replaceXMLNodeText(subNode,self.pertLib[libsKey][libID][gridIndex][mat])
      newFile = open(outFile,'w')
      toWrite = self._prettify(tree)
      newFile.writelines(toWrite)
      newFile.close()
