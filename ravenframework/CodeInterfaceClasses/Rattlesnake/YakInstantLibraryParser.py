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
Created May 8, 2016

@author: wangc
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import os
import sys
import copy
import numpy as np

class YakInstantLibraryParser():
  """
    Class used to parse Yak Instant cross section libraries, read the user provided alias files,
    perturb the libraries with variables defined in alias files and values from Raven Sampler.
    Cross sections will be reblanced based on provided information.
    In addition, this interface can be used to perturb Fission, Capture, TotalScattering, Nu, Kappa
    for given isotopes inside the libraries. The user can also perturb the diffusion coefficient if it
    exists in the input XS library.
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
    self.validReactions = ['TotalXS','FissionXS','RemovalXS','DiffusionCoefficient','ScatteringXS','NuFissionXS','KappaFissionXS',
                           'ChiXS','DNFraction','DNSpectrum','NeutronSpeed','DNPlambda','AbsorptionXS','CaptureXS','Nu','Kappa'] #These are all valid reactions for Yak XS format
    self.perturbableReactions = ['FissionXS','CaptureXS','TotalScatteringXS','Nu','Kappa','DiffusionCoefficient'] #These are all valid perturbable reactions for RAVEN
    self.level0Element  = 'Materials' #root element tag is always the same for Yak XS format
    self.level1Element  = 'Macros'   #level 1 element tag is always Macros
    self.level2Element  = ['material'] #These are some of the level 2 element tag with string vector xmlnode.text, without xml subnodes
    self.toBeReadXML    = [] #list of XML nodes that need to be read.
    self.libsKeys       = {} #dict to store library keys: {material_ID:{reaction:[]}}
    self.nGroup         = None # total energy groups
    self.aliasesNG      = None # total energy groups defined in alias files
    self.aliasesType    = {} # dict to store the perturbation type given in the alias files.

    #read in cross-section files, unperturbed files
    for xmlFile in inputFiles:
      if not os.path.exists(xmlFile.getPath()):
        raise IOError('The following Yak multigroup cross section library file: ' + xmlFile + ' is not found')
      tree = ET.parse(xmlFile.getAbsFile())
      root = tree.getroot()
      if root.tag == self.level0Element:
        self.xmlsDict[xmlFile.getFilename()] = tree
      else:
        msg = 'In YakInstantLibraryParser, root element of XS file is always ' + self.level0Element + ';\n'
        msg = msg + 'while the given XS file has different root element: ' + root.tag + "!"
        raise IOError(msg)
      macrosLib = root.find(self.level1Element)
      if macrosLib != None:
        if self.nGroup == None:
          self.nGroup = int(macrosLib.attrib['NG']) #total number of neutron energy groups
        elif self.nGroup != int(macrosLib.attrib['NG']):
          raise IOError('Inconsistent energy structures for give XS library ' + xmlFile.getFilename() + ' is found!')
        for matLib in macrosLib:
          matID = matLib.attrib['ID'].strip()
          scatteringOrder = int(matLib.attrib['NA'])
          self.libs[matID] = {}
          self.libsKeys[matID] = {}
          mgDict = self.libs[matID]
          mgDict['ScatteringOrder'] = scatteringOrder
          mgDictKeys =  self.libsKeys[matID]
          self._readYakXSInternal(matLib,mgDict,mgDictKeys)
          self._checkYakXS(mgDict)
      else:
        msg = 'In YakInstantLibraryParser, the node tag should be ' + self.level1Element + ';\n'
        msg = msg + 'while the given XS file has different node element: ' + macrosLib.tag + "!"
        raise IOError(msg)

  def initialize(self,aliasFiles):
    """
      Parse the input alias files
      @ In, aliasFiles, list, list of input alias files
      @ Out, None
    """
    self.aliases = {}
    for xmlFile in aliasFiles:
      if not os.path.exists(xmlFile.getPath()):
        raise IOError('The following Yak cross section alias file: ' + xmlFile + ' is not found!')
      aliasTree = ET.parse(xmlFile.getAbsFile())
      root = aliasTree.getroot()
      if root.tag != self.level0Element:
        raise IOError('Invalid root tag: ' + root.tag +' is provided.' + ' The valid root tag should be: ' + self.level0Element)
      for child in root:
        if child.tag != self.level1Element:
          raise IOError('Invalid subnode tag: ' + child.tag +' is provided.' + ' The valid subnode tag should be: ' + self.level1Element)
        if self.aliasesNG == None:
          self.aliasesNG = int(child.attrib['NG'])
        elif self.aliasesNG != int(child.attrib['NG']):
          raise IOError('Inconsistent total engergy groups were found in XS library: ' + xmlFile.getFilename())
        for matNode in child:
          matNodeID = matNode.attrib['ID'].strip()
          self.aliases[matNodeID] = {}
          aliasType = child.attrib['Type'].strip()
          self.aliasesType[matNodeID] = aliasType
          #read the cross section alias for each library (or material)
          self._readXSAlias(matNode,self.aliases[matNodeID],self.aliasesNG)

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
        mt = child.tag
        if mt not in aliasXS.keys():
          aliasXS[mt] = [None]*aliasXSGroup
        groupIndex = child.get('gIndex')
        if groupIndex == None:
          varsList = list(var.strip() for var in child.text.strip().split(','))
          if len(varsList) != aliasXSGroup:
            msg = str(aliasXSGroup) + ' variables should be provided for ' + child.tag
            msg = msg + " Only " + str(len(varsList)) + " variables is provided!"
            raise IOError(msg)
          aliasXS[mt] = varsList
        else:
          pertList = list(var.strip() for var in child.text.strip().split(','))
          groups = self._stringSpacesToListInt(groupIndex)
          if len(groups) != len(pertList):
            raise IOError('The group indices is not consistent with the perturbed variables list')
          for i,g in enumerate(groups):
            aliasXS[mt][g-1] = pertList[i]
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
      Load the Yak Instant library
      @ In, library, xml.etree.ElementTree.Element, xml element for cross section library defined in yak cross section files
      @ In, pDict, dict, dictionary to store the instant library
      @ In, keyDict, dict, dictionary to store the instatnt library node names, use to trace the cross section types for given isotope at given gridIndex
      @ Out, None
    """
    #read data for this library
    #first read scattering cross section
    profileNode = library.find('Profile')
    scatteringNode = library.find('ScatteringXS')
    self._readScatteringXS(profileNode,scatteringNode,pDict)
    for child in library:
      if child.tag == 'name':
        continue
      if child.tag == 'Profile':
        continue
      if child.tag == 'ScatteringXS':
        continue
      pDict[child.tag]= self._stringSpacesToNumpyArray(child.text)

  def _readScatteringXS(self,profile,scattering,pDict):
    """
      Reads the Scattering block for Yak multigroup cross section library
      @ In, profile, xml.etree.ElementTree.Element, xml element
      @ In, scattering, xml.etree.ElementTree.Element, xml element
      @ In, pDict, dict, dictionary used to store the scattering cross sections
      @ Out, None
    """
    has_profile = False
    if profile is not None:
      has_profile = True
    if has_profile:
      profileValue = self._stringSpacesToListInt(profile.text)
      pDict['ScatterStart'] = profileValue[0::2]
      pDict['ScatterEnd'] = profileValue[1::2]
      numRow = len(pDict['ScatterStart'])
      if scattering is not None:
        scatteringValue = self._stringSpacesToNumpyArray(scattering.text) #store in 1-D array
      else:
        raise IOError('ScatteringXS is not provided in the instant cross section library!')
      pDict[scattering.tag] = np.zeros((numRow,self.nGroup))
      ip = 0
      for g in range(numRow):
        for gr in range(pDict['ScatterStart'][g]-1,pDict['ScatterEnd'][g]):
          pDict[scattering.tag][g][gr] = scatteringValue[ip]
          ip += 1
    else:
      if scattering is not None:
        scatteringValue = self._stringSpacesToNumpyArray(scattering.text) #store in 1-D array
        pDict[scattering.tag] = scatteringValue.reshape((-1,self.nGroup))
      else:
        raise IOError('ScatteringXS is not provided in the instant cross section library')
    #calculate Total Scattering
    totScattering = np.zeros(self.nGroup)
    for g in range(self.nGroup):
      totScattering[g] = np.sum(pDict[scattering.tag][0:self.nGroup,g])
    pDict['TotalScatteringXS'] = totScattering

  def _checkYakXS(self,reactionDict):
    """
      Recalculate some undefined xs, such as 'Nu', 'Fission', 'Capture'.
      @ In, reactionDict, dict, dictionary stores the cross section data for given isotope
      @ Out, None
    """
    ### fission, nu, kappa
    reactionList = reactionDict.keys()
    if 'NuFissionXS' in reactionList:
      if 'FissionXS' not in reactionList:
        #calculate Fission using default Nu
        reactionDict['FissionXS'] = reactionDict['NuFissionXS']/self.defaultNu
        reactionDict['Nu'] = np.ones(self.nGroup)*self.defaultNu
      else:
        nu = []
        for i in range(self.nGroup):
          if reactionDict['FissionXS'][i] != 0:
            nu.append(reactionDict['NuFissionXS'][i]/reactionDict['FissionXS'][i])
          else:
            nu.append(self.defaultNu)
        reactionDict['Nu'] = np.asarray(nu)
      if 'KappaFissionXS' not in reactionList:
        #calculate kappaFission using default kappa
        reactionDict['KappaFissionXS'] = self.defaultKappa * reactionDict['FissionXS']
        reactionDict['Kappa'] = np.ones(self.nGroup) * self.defaultKappa
      else:
        kappa = []
        for i in range(self.nGroup):
          if reactionDict['FissionXS'][i] != 0:
            kappa.append(reactionDict['KappaFissionXS'][i]/reactionDict['FissionXS'][i])
          else:
            kappa.append(self.defaultKappa)
        reactionDict['Kappa'] = np.asarray(kappa)
    if 'DiffusionCoefficient' in reactionList:
      reactionDict['perturbDiffusionCoefficient'] = True
    else:
      reactionDict['perturbDiffusionCoefficient'] = False
    #check and calculate total or  transport cross sections
    if 'TotalXS' not in reactionList:
      if 'DiffusionCoefficient' not in reactionList:
        raise IOError('Total and diffusion coefficient cross sections are not found in the cross section input file, at least one of them should be provided!')
      else:
        #calculate total cross sections
        if 'ScatteringXS' not in reactionList:
          reactionDict['TotalXS'] = [1.0/(3.0*value) for value in reactionDict['DiffusionCoefficient']]
        elif reactionDict['ScatteringOrder'] == 0:
          reactionDict['TotalXS'] = [1.0/(3.0*value) for value in reactionDict['DiffusionCoefficient']]
        else:
          reactionDict['TotalXS'] =  [1.0/(3.0*value) for value in reactionDict['DiffusionCoefficient']] + np.sum(reactionDict['ScatteringXS'][self.nGroup:2*self.nGroup])
    else:
      if 'DiffusionCoefficient' not in reactionList:
        #calculate transport cross sections
        if 'ScatteringXS' not in reactionList:
          reactionDict['DiffusionCoefficient'] =  [1.0/(3.0*value) for value in reactionDict['TotalXS']]
        elif reactionDict['ScatteringOrder'] == 0:
          reactionDict['DiffusionCoefficient'] =  [1.0/(3.0*value) for value in reactionDict['TotalXS']]
        else:
          xs = reactionDict['TotalXS'] - np.sum(reactionDict['ScatteringXS'][self.nGroup:2*self.nGroup])
          reactionDict['DiffusionCoefficient'] =  [1.0/(3.0*value) for value in xs]

    #Metod 1: Currently, rattlesnake will not check the consistent of provided cross sections, rattlesnake will only use Total,
    #Scattering and nuFission for the transport calculation. In this case, we will recalculate the rest cross sections
    #based on Total, Scattering and Fission.
    if 'ScatteringXS' in reactionList:
      reactionDict['AbsorptionXS'] = reactionDict['TotalXS'] - reactionDict['TotalScatteringXS']
    else:
      if self.nGroup == 1 and 'AbsorptionXS' in reactionList:
        reactionDict['ScatteringXS'] = reactionDict['TotalXS'] - reactionDict['AbsorptionXS']
        reactionDict['TotalScatteringXS'] = copy.copy(reactionDict['ScatteringXS'])
      else:
        reactionDict['AbsorptionXS'] = copy.copy(reactionDict['TotalXS'])
    #calculate capture cross sections
    if 'NuFissionXS' in reactionList:
      reactionDict['CaptureXS'] = reactionDict['AbsorptionXS'] - reactionDict['FissionXS']
    else:
      reactionDict['CaptureXS'] = copy.copy(reactionDict['AbsorptionXS'])

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
    for matID, mtDict in pertFactor.items():
      self._computePerturbations(mtDict,self.pertLib[matID],self.aliasesType[matID])
    for matID, mtDict in pertFactor.items():
      self._rebalanceXS(self.pertLib[matID],pertFactor[matID],self.aliasesType[matID])

  def _computePerturbations(self,factors,lib,aliasType):
    """
      compute the perturbed values for input variables
      @ In, factors, dict, dictionary contains all perturbed input variables, and these variables will be
        replaced by the actual perturbed factors after this method is called.
      @ In, lib, dict, dictionary contains all the values of input variables
      @ In, aliasType, string, the type for provided alias file
      @ Out, None
    """
    for mtID, libValue in factors.items():
      groupValues = []
      for var in libValue:
        if var in self.modDict.keys():
          groupValues.append(self.modDict[var])
        elif var ==None:
          if aliasType == 'rel':
            groupValues.append(1.0)
          elif aliasType == 'abs':
            groupValues.append(0.0)
        else:
          raise IOError('The user wants to perturb ' + var + ', but this variable is not defined in the Sampler!')
      groupValues = np.asarray(groupValues)
      factors[mtID] = groupValues
      if not lib['perturbDiffusionCoefficient'] and mtID == 'DiffusionCoefficient':
        raise IOError('Diffusion Coefficient can not be perturbed since it does not exist in the XS library!')
      if aliasType == 'rel':
        lib[mtID] *= groupValues
      elif aliasType == 'abs':
        lib[mtID] += groupValues

  def _rebalanceXS(self,reactionDict,perturbDict,aliasType):
    """
      Recalculate some depedent xs, such as 'TotalXS', 'AbsorptionXS', 'ScatteringXS', 'NuFissionXS', 'KappaFissionXS',
      RemovalXS, DiffusionCoefficient.
      @ In, reactionDict, dict, dictionary used to store the cross section data
      @ In, perturbDict, dict, dictionary used to store the perturbation factors
      @ In, aliasType, string, the type for provided alias file
      @ Out, None
    """
    #fission, nu, kappa, capture, total scattering are assumed to be independent cross section types
    reactionList = perturbDict.keys()
    hasTotalScattering = False
    if 'TotalScatteringXS' in reactionList:
      hasTotalScattering = True
    if 'FissionXS' in reactionDict.keys():
      reactionDict['NuFissionXS'] = reactionDict['FissionXS']*reactionDict['Nu']
      reactionDict['KappaFissionXS'] = reactionDict['FissionXS']*reactionDict['Kappa']
      reactionDict['AbsorptionXS'] = reactionDict['FissionXS'] + reactionDict['CaptureXS']
    else:
      reactionDict['AbsorptionXS'] = copy.copy(reactionDict['CaptureXS'])
    reactionDict['TotalXS'] = reactionDict['AbsorptionXS'] + reactionDict['TotalScatteringXS']
    if hasTotalScattering:
      #total scattering are perturbed
      #recalculate Scattering Cross Sections
      for g in range(self.nGroup):
        if aliasType == 'rel':
          reactionDict['ScatteringXS'][0:self.nGroup,g] *= perturbDict['TotalScatteringXS'][g]
        elif aliasType == 'abs':
          factor = perturbDict['TotalScatteringXS'][g]/self.nGroup
          reactionDict['ScatteringXS'][0:self.nGroup,g] += factor
    #recalculate Removal cross sections
    reactionDict['RemovalXS'] = np.asarray(list(reactionDict['TotalXS'][g] - reactionDict['ScatteringXS'][g][g] for g in range(self.nGroup)))
    #recalculate diffusion coefficient cross sections
    if not reactionDict['perturbDiffusionCoefficient']:
      if reactionDict['ScatteringXS'].shape[0] >= self.nGroup*2:
        transport = reactionDict['TotalXS'] - np.sum(reactionDict['ScatteringXS'][self.nGroup:self.nGroup*2])
        reactionDict['DiffusionCoefficient'] = [1.0/(3.0*value) for value in transport]
      else:
        reactionDict['DiffusionCoefficient'] = [1.0/(3.0*value) for value in reactionDict['TotalXS']]


  def _replaceXMLNodeText(self,xmlNode,reactionDict):
    """
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element
      @ In, reactionDict, dict, dictionary contains the cross sections and their values
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'name':
        continue
      if child.tag == 'Profile':
        continue
      if child.tag in reactionDict.keys() and child.tag != 'ScatteringXS':
        child.text = '  '.join(['%.5e' % num for num in reactionDict[child.tag]])
      elif child.tag in reactionDict.keys() and child.tag == 'ScatteringXS':
        msg = ''
        for g in range(reactionDict[child.tag].shape[0]):
          msg = msg + '\n' + '            '+' '.join(['%.5e' % num for num in reactionDict[child.tag][g][reactionDict['ScatterStart'][g]-1:reactionDict['ScatterEnd'][g]]])
        child.text = msg + '\n'

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
    for outFile in inFiles:
      with open(outFile.getAbsFile(),'w') as newFile:
        tree = self.xmlsDict[outFile.getFilename()]
        root = tree.getroot()
        for child in root:
          for mat in child:
            matID = mat.attrib['ID'].strip()
            if matID not in self.aliases.keys():
              continue
            self._replaceXMLNodeText(mat,self.pertLib[matID])

        toWrite = self._prettify(tree)
        newFile.writelines(toWrite)
