from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import xml.etree.ElementTree as ET
import os
import sys
import copy

class YakMaterialsParser():
  """
    import the user-edited input file, build list of strings with replacable parts
  """
  def __init__(self,inputFiles):
    """
      Accept the input file and store XS data
      @ In, inputFiles, list(str), string list of input filenames that might need parsing.
      @Out, None.
    """
    self.inputFiles    = inputFiles
    self.tabs          = {} #dict of tab points, keyed on tabNames
    self.reactionTypes = [] #list of reaction types to be included
    self.tableReacts   = [] #list of tablewise reactions
    self.libs          = {} #dictionaries for libraries of tabulated xs values
    self.xmlLibs       = {} #unperturbed version of libs
    self.filesDict     = {} #connects files and trees
    self.defaultNu     = 2.43 #n per fission
    self.defaultKappa  = 195*1.6*10**(-13) #E per fission
    self.aliases       = {} #alias to XML node dict

    #read in cross-section files
    libfiles = inputFiles# FIXME self._findXSFiles(inputFiles)
    for xmlfile in libfiles:
      tree = ET.parse(xmlfile)
      root = tree.getroot()
      self.filesDict[tree] = xmlfile
      if root.tag == 'Multigroup_Cross_Section_Libraries':
        self.xmlLibs[root.attrib['Name']] = tree
        self._readNextLevel(root,self.libs)
      else: raise IOError('In YakMaterialsParser, root of XS file is not implemented: '+root.tag)

  def initialize(self,aliasTree):
    """
      Writes in aliases
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

  ##################################################
  #               READING METHODS                  #
  ##################################################
  def _findXSFiles(self,files):
    """
      Finds the xml input files labeled as cross section files
      @ In, files, list, File objects
      @ Out, fs, list(File object), file objects
    """
    fs=[]
    for f in files:
      if f.subtype == 'XS': fs.append(f)
    if len(fs)>0: return fs
    raise IOError('In YakMaterialsParser, no cross section file found!')

  def _spacesToList(self,text):
    """
      Turns a space-separated list into a list of constituent members
      @ In, text, string, string
      @ Out, members, list, list of members
    """
    members = []
    for t in list(c.strip() for c in text.strip().split(' ')):
      if t=='': continue
      members.append(t)
    return members

  def _readXMLInfo(node,nDict):
    """
      Reads in basic XML information (tag, attributes, text)
      @ In, node, xml.etree.ElementTree.Element, node to read
      @ In, nDict, dict, dict to add info to
      @ Out, _readXMLInfo, dict, new dict
    """
    nDict[node.tag] = {'attrib':dict(node.attrib), 'text':str(node.text)}
    return nDict[node.tag]

  def _readNextLevel(child,pDict,path=''):
    """
      Uses child tag to determine next reading algorithm to perform.
      @ In, child, xml.etree.ElementTree.Element, element
      @ In, pDict, dict, dictionary for child's parent
      @ In, path, string, identifiers to cross sections
      @ Out, None
    """
    def parentAction(path):
      """
        Default action for parent nodes with children
        @ In, path, string, unique identifier to cross section
        @ Out, None
      """
      #curDict = self._readXMLInfo(child,pDict)
      for cchild in child:
        self._readNextLevel(cchild,pDict,path)
      #end parentAction method
    def addToPath(pDict,key,value):
      """
        Formats path for consistency
        @ In, pDict, dict, data storage dictionary
        @ In, key, string, label
        @ In, value, string, value
        @ Out, nstr, string, formatted string
      """
      if 'dictLabel' not in pDict.keys():
        pDict['dictLabel']=key
      if key != pDict['dictLabel']:
        raise RuntimeError('Yak Materials Parser has doubly-used labels: "%s" and "%s"!' %(key,pDict['dictLabel']))
      if value not in pDict.keys():
        pDict[value]={}
      else:
        raise RuntimeError('Yak Materials Parser has duplicate ID entry: "%s"!' %value)
      return key+': '+str(value)+'|'
      #end addToPath method
    # case: child.tag
    if   child.tag == 'Multigroup_Cross_Section_Libraries':
      path+=addToPath(pDict,'Library Group',cchild.attrib['Name'])
      self.nG = int(child.attrib['NGroup'])
      parentAction(path)
    elif child.tag == 'Multigroup_Cross_Section_Library':
      path+=addToPath(pDict,'Library ID',cchild.attrib['ID'])
      parentAction(path)
    elif child.tag == 'Table':
      path+=addToPath(pDict,'Grid Index',cchild.attrib['gridIndex'])
      parentAction(path)
    elif child.tag == 'Isotope':
      path+=addToPath(pDict,'Isotope',cchild.attrib['Name'])
      parentAction(path)
    elif child.tag == 'Tabulation':
      for t in self._spacesToList(child.text):
        self.tabs[t]=[]
    elif child.tag in self.tabs.keys(): #FIXME depends on order of xml file, and assumes floats!
      self.tabs[child.tag] = list(float(s) for s in self._spacesToList(child.text))
    elif child.tag == 'AllReactions':
      self.reactionTypes = self._spacesToList(child.text)
    elif child.tag == 'TablewiseReactions':
      self.tableReacts = self._spacesToList(child.text)
    elif child.tag == 'Tablewise':
      self._readTablewise(child,pDict,path)
    else:
      raise IOError('YakMaterialParser tried to parse node <'+child.tag+'> but it was not recognized!')

  def _readTablewise(self,node,pDict,path):
    """
      Reads in Tablewise entry for rattlesnake and stores values
      @ In, node, xml.etree.ElementTree.Element, node
      @ In, pDict, dict, xml dictionary
      @ In, path, string, |-separated path of identifiers
    """
    have = {}
    for child in node:
      have[child.tag] = {}
      xsNode = have[child.tag]
      if child.tag != 'Scattering':
        groupwise = list(float(s) for s in self._spacesToList(child.text))
        for g,value in enumerate(groupwise):
          xsNode[g] = value
      else:
        #TODO scattering is hard to read in.
    #now calculate basic values: fission, scattering, capture, nu, kappa(?)
    baseXS = pDict
    baseXS['Fission'   ] = {}
    baseXS['Nu'        ] = {}
    baseXS['Kappa'     ] = {}
    baseXS['Scattering'] = {}
    baseXS['Capture'   ] = {}
    baseXS['Chi'       ] = {}
    ### fission, nu, kappa
    if 'Fission' not in have.keys():
      for g in range(self.nG):
        baseXS['Kappa'][g] = self.defaultKappa
      if 'kappaFission' in have.keys():
        for g in range(self.nG):
          baseXS['Fission'][g] = have['kappaFission'][g]/self.defaultKappa
          baseXS['Nu'][g] =  have['nuFission'][g]/baseXS['Fission'][g]
      else:
        for g in range(self.nG):
          baseXS['Nu'][g] = self.defaultNu
          baseXS['Fission'][g] = have['nuFission'][g]/self.defaultNu
    else:
      baseXS['Fission'][g] = have['Fission'][g]
      if 'kappaFission' in have.keys():
        for g in range(self.nG):
          baseXS['Kappa'][g] = have['kappaFission'][g]/baseXS['Fission'][g]
      for g in range(self.nG):
        baseXS['Nu'][g] =  have['nuFission'][g]/baseXS['Fission'][g]
    ### scattering
    for gFrom in range(self.nG):
      baseXS['Scattering'][gFrom] = {}
      for gTo in range(self.nG):
        baseXS['Scattering'][gFrom][gTo] = have['Scattering'][gFrom].get(gTo,0.0)
    ### capture
    if 'Capture' in have.keys():
      for g in range(self.nG):
        baseXS['Capture'][g] = have['Capture'][g]
    else:
      if 'Absorption' in have.keys():
        for g in range(self.nG):
          baseXS['Capture'][g] = have['Absorption'][g] - baseXS['Fission'][g]
      else:
        for g in range(self.nG):
          absorb = have['Total'][g] - sum(have['Scattering'][g].values())
          baseXS['Capture'][g] = absorb - baseXS['Fission'][g]
    ### Chi
    for g in range(self.nG):
      baseXS['Chi'][g] = have['FissionSpectrum'][g]
    #TODO WORKING
    # TODO set baseXS somewhere useful

  ##################################################
  #              MODIFYING METHODS                 #
  ##################################################
  def modifyInternalDictionary(self,**Kwargs):
    """
      Edits the parsed data
      @ In, **Kwargs, dict, including moddit (the dictionary of variable:value to replace) and additionalEdits.
      @ Out, None.
    """
    moddict = Kwargs['SampledVars']
    for alias,value in moddict.keys():





### FROM GENERIC ###
    moddict = Kwargs['SampledVars']
    self.adldict = Kwargs['additionalEdits']
    iovars = []
    for key,value in self.adldict.items():
      if type(value)==dict:
        for k in value.keys():
          iovars.append(k)
      elif type(value)==list:
        for v in value:
          iovars.append(v)
      else:
        iovars.append(value)
    newFileStrings={}
    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile] if inputFile in self.varPlaces[var].keys() else []:
          if var in moddict.keys():
            if var in self.formats.keys():
              if inputFile in self.formats[var].keys():
                if any(formVal in self.formats[var][inputFile][0] for formVal in self.acceptFormats.keys()):
                  formatstringc = "{:"+self.formats[var][inputFile][0].strip()+"}"
                  self.segments[inputFile][place] = formatstringc.format(self.formats[var][inputFile][1](moddict[var]))
                else: self.segments[inputFile][place] = str(moddict[var]).strip().rjust(self.formats[var][inputFile][1](self.formats[var][inputFile][0]))
            else: self.segments[inputFile][place] = str(moddict[var])
          elif var in self.defaults.keys():
            if var in self.formats.keys():
              if inputFile in self.formats[var].keys():
                if any(formVal in self.formats[var][inputFile][0] for formVal in self.acceptFormats.keys()):
                  formatstringc = "{:"+self.formats[var][inputFile][0].strip()+"}"
                  self.segments[inputFile][place] = formatstringc.format(self.formats[var][inputFile][1](self.defaults[var][inputFile]))
                else: self.segments[inputFile][place] = str(self.defaults[var][inputFile]).strip().rjust(self.formats[var][inputFile][1](self.formats[var][inputFile][0]))
            else: self.segments[inputFile][place] = self.defaults[var][inputFile]
          elif var in iovars: continue #this gets handled in writeNewInput
          else: raise IOError('Generic Parser: Variable '+var+' was not sampled and no default given!')

  def writeNewInput(self,inFiles,origFiles):
    '''
    Generates a new input file with the existing parsed dictionary.
    @ In, inFiles, Files list of new input files to return
    @ In, origFiles, the original list of Files, used for key names
    @Out, None.
    '''
    #get the right IO names put in
    case = 'out~'+inFiles[0].getBase() #FIXME the first entry? This is bad! Forces order somewhere in input file
    # however, I can't seem to generate an error with this, so maybe it's okay
    def getFileWithExtension(fileList,ext):
      '''
      Just a script to get the file with extension ext from the fileList.
      @ In, fileList, the Files list of files to pick from.
      @Out, ext, the string extension that the desired filename ends with.
      '''
      found=False
      for index,inputFile in enumerate(fileList):
        if inputFile.getExt() == ext:
          found=True
          break
      if not found: raise IOError('No InputFile with extension '+ext+' found!')
      return index,inputFile

    for var in self.varPlaces.keys():
      for inputFile in self.segments.keys():
        for place in self.varPlaces[var][inputFile] if inputFile in self.varPlaces[var].keys() else []:
          for iotype,adlvar in self.adldict.items():
            if iotype=='output':
              if var==self.adldict[iotype]:
                self.segments[inputFile][place] = case
                break
            elif iotype=='input':
              if var in self.adldict[iotype].keys():
                self.segments[inputFile][place] = getFileWithExtension(inFiles,self.adldict[iotype][var][0].strip('.'))[1].getAbsFile()
                break
    #now just write the files.
    for f,inFile in enumerate(origFiles):
      outfile = inFiles[f]
      if os.path.isfile(outfile.getAbsFile()): os.remove(outfile.getAbsFile())
      outfile.writelines(''.join(self.segments[inFile.getFilename()]))
      outfile.close()
