from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import os
import sys
import copy
import numpy as np

class YakMultigroupLibraryParser():
  """
    import the user-edited input file, build list of strings with replaceable parts
  """
  def __init__(self,inputFiles):
    """
      Accept the input file and store XS data
      @ In, inputFiles, list(str), string list of input filenames that might need parsing.
      @ Out, None.
    """
    self.inputFiles     = inputFiles
    self.tabs           = {} #dict of tab points, keyed on tabNames
    self.reactionTypes  = [] #list of reaction types to be included
    self.tableReacts    = [] #list of tablewise reactions
    self.libs           = {} #dictionaries for libraries of tabulated xs values
    self.xmlLibs        = {} #unperturbed version of libs
    self.filesDict      = {} #connects files and trees
    self.defaultNu      = 2.43 #number of neutrons per fission
    self.defaultKappa   = 195*1.6*10**(-13) #Energy release per fission
    self.aliases        = {} #alias to XML node dict
    self.validReactions = ['Total','Fission','Removal','Transport','Scattering','nuFission','kappaFission',
                           'FissionSpectrum','DNFraction','DNSpectrum','NeutronVelocity','DNPlambda'] #These are all valid reactions for Yak XS format
    self.perturbableReactions = ['Fission','Capture','TotalScattering','Nu','Kappa'] #These are all valid perturbable reactions for RAVEN
    self.level0Element  = 'Multigroup_Cross_Section_Libraries' #root element tag is always the same for Yak XS format
    self.level1Element  = 'Multigroup_Cross_Section_Library'   #level 1 element tag is always Multigroup_Cross_Section_Library
    self.level2Element  = ['Tabulation','AllReactions','TablewiseReactions','LibrarywiseReactions'] #These are some of the level 2 element tag with string vector xmlnode.text, without xml subnodes
    self.toBeReadXML    = [] #list of XML nodes that need to be read.
    self.libsKeys       = {} #dict to store library keys: {mglib_ID:{gridIndex:{IsotopeName:[reactions]}}}

    #read in cross-section files, unperturbed files
    libFiles = inputFiles # FIXME self._findXSFiles(inputFiles)
    for xmlFile in libFiles:
      tree = ET.parse(xmlFile)
      root = tree.getroot()
      self.filesDict[xmlFile] = tree
      if root.tag == self.level0Element:
        self.xmlLibs[root.attrib['Name']] = tree
        self.nGroup = int(root.attrib['NGroup']) #total number of neutron energy groups
        for mgLib in root:
          self.libs[mgLib.attrib['ID']] = {}
          self.libsKeys[mgLib.attrib['ID']] = {}
          self._readYakXSInternal(mgLib,self.libs[mgLib.attrib['ID']],self.libsKeys[mgLib.attrib['ID']])
          self._readAdditionalYakXS(mgLib,self.libs[mgLib.attrib['ID']])
          self._checkYakXS(self.libs[mgLib.attrib['ID']],self.libsKeys[mgLib.attrib['ID']])
      else:
        msg = 'In YakMultigroupLibraryParser, root element of XS file is always ' + self.rootElement + ';\n'
        msg = msg + 'while the given XS file has different root element: ' + root.tag + "!"
        raise IOError(msg)

  def initialize(self,aliasTree):
    """
      Initialize aliases
      @ In, aliasTree, xml.etree.ElementTree.ElementTree, alias tree
      @ Out, None
    """
    perturbable = ['Fission','Capture','Scattering','Nu','Kappa']
    root = aliasTree.getroot()
    self.aliases={}
    for child in root:
      if child.text in self.aliases.keys():
        raise IOError('YakMaterialsParser: there were duplicate names in the aliases file for '+child.text)
      if child.attrib['Reaction'] not in perturbable:
        raise IOError('YakMaterialsParser: %s is not perturbable!  Options are %s' %(child.attrib['Reaction'],str(perturbable)))
      self.aliases[child.text] = child

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
      @ In, library, xml.etree.ElementTree.Element, element
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
      @ In, parentNode, xml.etree.ElementTree.Element, element
      @ In, libDict, dict, dictionary of multigroup library
      @ In, keyDict, dict, dictionary to store the multigroup library node names, use to trace the cross section types for given isotope at given gridIndex
      @ Out, None
    """
    for child in parentNode:
      self._readNextLevel(child,libDict,keyDict)

  def _readNextLevel(self,xmlNode,pDict,keyDict):
    """
      Uses xmlNode tag to determine next reading algorithm to perform.
      @ In, xmlNode, xml.etree.ElementTree.Element, element
      @ In, pDict, dict, dictionary for child's parent
      @ In, keyDict, dict, dictionary to store the multigroup library node names, use to trace the cross section types for given isotope at given gridIndex
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
      pDict[xmlNode.attrib['Name']] = {}
      keyDict[xmlNode.attrib['Name']] = []
      hasSubNode = False
      for child in xmlNode:
        if child != None:
          hasSubNode = True
          break
      if hasSubNode:
        self._readIsotopXS(xmlNode,pDict[xmlNode.attrib['Name']],keyDict[xmlNode.attrib['Name']])
    #store the xmlNode tags that have not been parsed
    else:
      self.toBeReadXML.append(xmlNode.tag)

  def _readIsotopeXS(self,node,pDict,keyList):
    """
      Reads in Tablewise entry for rattlesnake and stores values
      @ In, node, xml.etree.ElementTree.Element, node
      @ In, pDict, dict, xml dictionary
      @ In, keyList, list, dictionary to store the multigroup library node names, use to trace the cross section types for given isotope at given gridIndex
      @ Out, None
    """
    #the xs structure is same as Tablewise xs data
    self._readTablewise(node,pDict,keyList)

  def _readLibrarywise(self,node,pDict):
    """
      Reads in Librarywise entry for rattlesnake and stores values
      @ In, node, xml.etree.ElementTree.Element, node
      @ In, pDict, dict, xml dictionary
      @ Out, None
    """
    #the xs structure is same as Tablewise xs data
    self._readTablewise(node,pDict)

  def _readTablewise(self,node,pDict,keyList=None):
    """
      Reads in Tablewise entry for rattlesnake and stores values
      @ In, node, xml.etree.ElementTree.Element, node
      @ In, pDict, dict, xml dictionary
      @ In, keyList, list, list to store the multigroup library node names, use to trace the cross section types for given isotope at given gridIndex
      @ Out, None
    """
    orderScattering = int(node.attrib['L'])
    for child in node:
      #FIXME the validReactions is documented in Yak, but it seems this is not sure. Other cross sections can also be included, such as Capture, Nalpha, ...
      #if child.tag not in self.validReactions:
      #  raise IOError("The following reaction type " + child.tag + " is not valid!")
      if keyList != None:
        keyList.append(child.tag)
      #read all xs for all reaction types except Scattering
      if child.tag != 'Scattering':
        pDict[child.tag]= self._stringSpacesToNumpyArray(child.text)
      #read xs sections for Scattering
      else:
        #TODO scattering is hard to read in.
        self._readScatteringXS(child,pDict,orderScattering)

  def _readScatteringXS(self,node,pDict,orderScattering):
    """
      Reads the Scattering block for Yak multigroup cross section library
      @ In, node, xml.etree.ElementTree.Element, xml node
      @ In, pDict, dict, xml dictionary
      @ In, orderScattering, int, order of spherical harmonics expansioin for scattering
      @ Out, None
   """
    has_profile = False
    pDict['ScatteringOrder'] = orderScattering
    if int(node.get('profile')) == 1: has_profile = True
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
      totScattering[g] = np.sum(pDict['Scattering'][g])
    pDict['TotalScattering'] = totScattering

  def _readAdditionalYakXS(self,xmlNode,pDict):
    """
      Read addition cross sections that have not been read via method self._readYakXSInternal,
      such as Tabulation Grid, Librarywise.
      @ In, pDict, dict, dictionary stores all the cross section data for given multigroup library (or material)
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
      @ In, keyDict, dict, dictionary to store the multigroup library node names, use to trace the cross section types for given isotope at given gridIndex
      @ Out, None
    """
    #make sure pDict include the cross sections, if not, copy from Tablewise data
    for gridKey,isotopeDict in keyDict.items():
      for isotopeKey,reactionList in isotopeDict.items():
        if len(reactionList) == 0:
          pDict[gridKey][isotopeKey] = copy.deepcopy(pDict[gridKey]['Tablewise'])
        #calculate some independent cross sections if they are not in pDict
        #these cross sections can be: fission, total scattering, capture, nu, kappa
        self._recalculateYakXS(pDict[gridKey][isotopeKey])

  def _recalculateYakXS(self,reactionDict):
    """
      Recalculate some undefined xs, such as 'Nu', 'Fission', 'Capture'.
      @ In, reactionDict, dict, dictionary stores all the cross section data for given multigroup library (or material) at given gridIndex and given Isotope
      @ Out, None
    """
    ### fission, nu, kappa
    reactionList = reactionDict.keys()
    if 'nuFission' in reactionList:
      if 'Fission' not in reactionList:
        #recalculate Fission using default Nu
        reactionDict['Fission'] = reactionDict['nuFission']/self.defaultNu
        reactionDict['nu'] = np.ones(self.nGroup)*self.defaultNu
      else:
        nu = []
        for i in range(self.nGroup):
          if reactionDict['Fission'][i] != 0:
            nu.append(reactionDict['nuFission'][i]/reactionDict['Fission'][i])
          else:
            nu.append(self.defaultNu)
        reactionDict['nu'] = np.asarray(nu)
      if 'kappaFission' not in reactionList:
        #recalculate kappaFission using default kappa
        reactionDict['kappaFission'] = self.defaultKappa * reactionDict['Fission']
        reactionDict['kappa'] = np.ones(self.nGroup) * self.defaultKappa
      else:
        kappa = []
        for i in range(self.nGroup):
          if reactionDict['Fission'][i] != 0:
            kappa.append(reactionDict['kappaFission'][i]/reactionDict['Fission'][i])
          else:
            kappa.append(self.defaultKappa)
        reactionDict['kappa'] = np.asarray(kappa)
    # calculate absoption
    hasTotal = False
    if 'Absorption' not in  reactionList:
      if 'Total' in reactionList:
        reactionDict['Absorption'] = reactionDict['Total'] - reactionDict['TotalScattering']
      else:
        raise IOError('Total cross section is required for this interface, but not provide!')
    else:
      #recalculate capture cross sections
      if 'Capture' not in reactionList:
        if 'nuFission' in reactionList:
          reactionDict['Capture'] = reactionDict['Absorption'] - reactionDict['Fission']
        else:
          reactionDict['Capture'] = reactionDict['Absorption']
    #we may also need to consider to recalculate Removal and DiffusionCoefficient XS.
    #we will implement in the future.

