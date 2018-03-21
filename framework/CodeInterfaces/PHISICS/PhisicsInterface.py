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
Created on July 5th, 2017 
@author: rouxpn 
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import shutil
import re
from CodeInterfaceBaseClass import CodeInterfaceBase
import phisicsdata
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom
import fileinput 
import sys

class Phisics(CodeInterfaceBase):
  """
    Code interface for PHISICS
  """ 
  def getFilename(self):
    """
      Retriever for full filename.
      @ In, None
      @ Out, __base, string, filename
    """
    if self.__ext is not None:
      return '.'.join([self.__base,self.__ext])
    else:
      return self.__base
    
  def getBase(self):
    """
      Retriever for file base.
      @ In, None
      @ Out, __base, string path
    """
    return self.__base 
  
  def getNumberOfMPI(self,string):
    """
      Gets the number of MPI requested by the user in the RAVEN input
      @ In, string, string, string from the Kwargs containing the number of MPI
      @ Out, MPInumber, integer, number of MPI used in the calculation
    """
    return int(string.split(" ")[-2])
    
  def outputFileNames(self,pathFile):
    """
      Collects the output file names from lib_inp_path xml file
      @ In, pathFile, string, lib_path_input file 
      @ Out, None
    """
    pathTree = ET.parse(pathFile)
    pathRoot = pathTree.getroot()
    self.outputFileNameDict = {}
    xmlNodes = ['reactions','atoms_plot','atoms_csv','decay_heat','bu_power','flux','repository']
    for xmlNodeNumber in xrange (0,len(xmlNodes)):
      for xmlNode in pathRoot.getiterator(xmlNodes[xmlNodeNumber]):
        self.outputFileNameDict[xmlNodes[xmlNodeNumber]] = xmlNode.text
  
  def syncLibPathFileWithRavenInp(self,pathFile,currentInputFiles,keyWordDict):
    """
      Parses the lib_file input and writes the correct library path in the lib_path.xml, based in the raven input 
      @ In, pathFile, string, lib_path_input file
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, keyWordDict, dictionary, keys: type associated to an input file. Values: integer, unique associated to each input file
      @ Out, pathFile, string, lib_path_input file, updated with user-defined path
    """
    pathTree = ET.parse(pathFile)
    pathRoot = pathTree.getroot()
    typeList = ['IsotopeList','mass','decay','budep','FissionYield','FissQValue','CRAM_coeff_PF','N,G','N,Gx','N,2N','N,P','N,ALPHA','AlphaDecay','BetaDecay','BetaxDecay','Beta+Decay','Beta+xDecay','IntTraDecay']
    libPathList = ['iso_list_inp','mass_a_weight_inp','decay_lib','xs_sep_lib','fiss_yields_lib','fiss_q_values_lib','cram_lib','n_gamma','n_gamma_ex','n_2n','n_p','n_alpha','alpha','beta','beta_ex','beta_plus','beta_plus_ex','int_tra']
    
    for typeNumber in xrange(0,len(typeList)):
      for libPathText in pathRoot.getiterator(libPathList[typeNumber]):
        libPathText.text = currentInputFiles[keyWordDict[typeList[typeNumber].lower()]].getAbsFile() 
    pathTree.write(pathFile)
    
  def syncPathToLibFile(self,depletionRoot,depletionFile,depletionTree,libPathFile):
    """
      prints the name of the file that contains the path to the libraries (lib_path.xml) in the depletion input (dep.xml). 
      @ In, depletionRoot, xml.etree.ElementTree.Element, depletion input XML node
      @ In, depletionFile, string, path to depletion_input.xml
      @ In, depletionTree, xml.etree.ElementTree.Element, XML tree from the depletion_input.xml
      @ In, libPathFile, string, lib_path.xml file
      @ Out, None 
    """
    if depletionTree.find('.//input_files') is None:  
      for line in fileinput.FileInput(depletionFile, inplace = 1):
        if '<DEPLETION_INPUT>' in line:
          line = line.replace('<DEPLETION_INPUT>','<DEPLETION_INPUT>'+'\n\t'+'<input_files>'+libPathFile+'</input_files>')
        sys.stdout.write(line)
    else:
      depletionTree.find('.//input_files').text = libPathFile
      depletionTree.write(depletionFile)

  def forcePrintLibraries(self,libraryInput):
    """
      Check if the flag 'print libraries' is on. if not, it is automatically added. 
      @ In, library input, string, Xs-library input 
      @ Out, None 
    """
    libraryTree = ET.parse(libraryInput)
    libraryRoot = libraryTree.getroot()
    check = False 
    for child in libraryRoot.getiterator("XS-library"):
      for attrib,value in child.attrib.iteritems():
        if attrib == 'print_libraries' and value == '1':
          check = True
          break 
    if check is False: 
      raise ValueError('The attribute print_libraries=1 is missing in the node '+str(child)+' from the input '+str(libraryInput))
    if check is True:
      return 
    
  def getTitle(self,depletionRoot):
    """
      Gets the job title. It will become later the instant output file name. If the title flag is not in the 
      instant input, the job title is defaulted to 'defaultInstant'
      @ In, depletionRoot, xml.etree.ElementTree.Element, depletion input XML node
      @ Out, None  
    """
    self.jobTitle =  'defaultInstant'
    for child in depletionRoot.findall(".//title"):
      self.jobTitle = child.text
      break 
    return 
    
  def verifyMrtauFlagsAgree(self,depletionRoot):
    """
      Verifies the node "standalone"'s text in the depletion_input xml. if the standalone flag 
      in the depletion_input disagrees with the mrtau standalone flag in the raven input, 
      the codes errors out. 
      @ In, depletionRoot, xml.etree.ElementTree.Element, depletion input XML node
      @ Out, None
    """
    for child in depletionRoot.findall(".//standalone"):
      isMrtauStandAlone = child.text.lower()
      tag = child.tag
      break 
    if self.mrtauStandAlone == False and isMrtauStandAlone == 'yes':
      raise  ValueError("\n\n Error. The flags controlling the Mrtau standalone mode are incorrect. The node <standalone> in depletion_input file disagrees with the node <mrtauStandAlone> in the raven input. \n the matching solutions are: <mrtauStandAlone>yes</mrtauStandAlone> and <"+tag+">True<"+tag+">\n <mrtauStandAlone>no</mrtauStandAlone> and <"+tag+">False<"+tag+">")
    if self.mrtauStandAlone == True and isMrtauStandAlone == 'no':
      raise  ValueError("\n\n Error. The flags controlling the Mrtau standalone mode are incorrect. The node <standalone> in depletion_input file disagrees with the node <mrtauStandAlone> in the raven input. \n the matching solutions are: <mrtauStandAlone>yes</mrtauStandAlone> and <"+tag+">True<"+tag+">\n <mrtauStandAlone>no</mrtauStandAlone> and <"+tag+">False<"+tag+">")
  
  def parseControlOptions(self,depletionFile,libPathFile):
    """
      Parses the Material.xml data file and put the isotopes name as key and 
      the decay constant relative to the isotopes as values 
      @ In, depletionFile, string, depletion_input file
      @ In, libpathFile, string, lib_inp_path file 
      @ In, inpFile, string, Instant input file 
      @ Out, None 
    """  
    depletionTree = ET.parse(depletionFile)
    depletionRoot = depletionTree.getroot()
    self.verifyMrtauFlagsAgree(depletionRoot)
    self.getTitle(depletionRoot)
    self.syncPathToLibFile(depletionRoot,depletionFile,depletionTree,libPathFile)
    return 
  
  def distributeVariablesToParsers(self,perturbedVars):
    """
      Transforms the dictionary into dictionary of dictionaries. This dictionary renders easy the distribution 
      of the variable to their corresponding parser. For example, if the two variables are the following: 
      {'FY|FAST|PU241|SE78':1.0, 'DECAY|BETA|U235':2.0}, the output dict will be: 
      {'FY':{'FY|FAST|PU241|SE78':1.0}, 'DECAY':{'DECAY|BETA|U235':2.0}}
      @ In, perturbedVars, dictionary, dictionary of the perturbed variables
      @ Out, distributedPerturbedVars, dictionary of dictionaries containing the perturbed variables
    """
    distributedPerturbedVars = {}
    pertType = []
    for i in perturbedVars.iterkeys():        # teach what are the type of perturbation (decay FY etc...) 
      splittedKeywords = i.split('|')
      pertType.append(splittedKeywords[0])
    for i in xrange (0,len(pertType)):        # declare all the dictionaries according the different type of pert
      distributedPerturbedVars[pertType[i]] = {}
    for key, value in perturbedVars.items():  # populate the dictionaries 
      splittedKeywords = key.split('|')
      for j in xrange (0,len(pertType)):
        if splittedKeywords[0] == pertType[j] :
          distributedPerturbedVars[pertType[j]][key] = value
    return distributedPerturbedVars
  
  def addDefaultExtension(self):
    """
      Possible input extensions found in the input files. 
      @ In, None
      @ Out, None 
    """
    self.addInputExtension(['xml','dat','path'])
  
  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    validPerturbation = ['additive', 'multiplicative', 'absolute']
    self.perturbXS = validPerturbation[1] # default is cross section perturbation multiplicative mode
    setOfPerturbations = set(validPerturbation)
    #default values if the flag is not in the raven input  
    self.tabulation       = True
    self.mrtauStandAlone  = False
    self.mrtauExecutable  = None 
    self.phisicsRelap     = False 
    self.printSpatialRR   = False 
    self.printSpatialFlux = False
    
    for child in xmlNode:
      if child.tag == 'PerturbXS':
        if child.text.lower() in set(validPerturbation): 
          self.perturbXS = child.text.lower()
        else: 
          raise ValueError("\n\nThe type of perturbation --"+child.text.lower()+"-- is not valid. You can choose one of the following \n"+"\n".join(set(validPerturbation)))
      
      if child.tag == 'tabulation':
        self.tabulation = None
        if (child.text.lower() == 't' or child.text.lower() == 'true'):  
          self.tabulation = True
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'): 
          self.tabulation = False 
        else: 
          raise ValueError("\n\n The tabulation node -- <"+child.tag+"> -- only supports the following text (case insensitive): \n True \n T \n False \n F" )
      
      if child.tag == 'mrtauStandAlone':
        self.mrtauStandAlone = None 
        if (child.text.lower() == 't' or child.text.lower() == 'true'):  
          self.mrtauStandAlone = True 
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'):  
          self.mrtauStandAlone = False 
        else: 
          raise ValueError("\n\n The flag activating MRTAU standalone mode -- <"+child.tag+"> -- only supports the following text (case insensitive): \n True \n T \n False \n F. \n Default Value is False" )
      if child.tag == 'mrtauStandAloneExecutable' and self.mrtauStandAlone is True:
        self.mrtauExecutable = child.text
      
      if child.tag == 'printSpatialRR':
        if (child.text.lower() == 't' or child.text.lower() == 'true'):  
          self.printSpatialRR = True
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'): 
          self.printSpatialRR = False
        else: 
          raise ValueError("\n the node "+child.tag+"has to be a boolean entry")
      
      if child.tag == 'printSpatialFlux':
        if (child.text.lower() == 't' or child.text.lower() == 'true'):  
          self.printSpatialFlux = True
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'): 
          self.printSpatialFlux = False
        else: 
          raise ValueError("\n the node "+child.tag+"has to be a boolean entry")
   
  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    dict = self.mapInputFileType(inputFiles)
    if self.mrtauStandAlone == True: 
      executable = self.mrtauExecutable
      commandToRun = executable
    outputfile = 'out~'+inputFiles[dict['inp'.lower()]].getBase()
    if self.mrtauStandAlone == False: 
      commandToRun = executable + ' ' +inputFiles[dict['inp'.lower()]].getFilename() + ' ' + inputFiles[dict['Xs-library'.lower()]].getFilename() + ' ' + inputFiles[dict['Material'.lower()]].getFilename() + ' ' + inputFiles[dict['Depletion_input'.lower()]].getFilename() + ' ' + self.instantOutput
      commandToRun = commandToRun.replace("\n"," ")
      commandToRun  = re.sub("\s\s+" , " ", commandToRun )
    returnCommand = [('parallel',commandToRun)], outputfile
    return returnCommand
    
  def finalizeCodeOutput(self,command,output,workingDir,**phiRel):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      This methods also calls the method 'mergeOutput' if MPI mode is used, in order to merge all the output files into one 
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ In, phiRel, dictionary, contains a key 'phiRel': True if phisics Relap is in coupled mode, empty otherwise
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    if "phiRel" in phiRel:
      pass 
    else: 
      phiRel['phiRel'] = False 
    splitWorkDir = workingDir.split('/')
    self.pertNumber = splitWorkDir[-1]
    phisicsdata.phisicsdata(self.instantOutput,workingDir,self.mrtauStandAlone,self.jobTitle,self.outputFileNameDict,self.numberOfMPI,phiRel['phiRel'],self.printSpatialRR,self.printSpatialFlux)
    if self.mrtauStandAlone == False:
      return self.jobTitle+'-'+str(self.pertNumber).strip()
    if self.mrtauStandAlone == True:
      return 'mrtau'+'-'+str(self.pertNumber).strip()

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      The line Task ended is searched in the phisics output as successful job message. 
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = True
    with open(os.path.join(workingDir,output), 'r') as f:
      for line in f:
        if re.search(r'task\s+ended',line,re.IGNORECASE):
          failure = False 
    return failure 
    
  def mapInputFileType(self,currentInputFiles):
    """
      Assigns a unique integer to the input file Types 
      @ In, currentInputFiles,  list,  list of current input files (input files from last this method call)
      @ Out, keyWordDict, dictionary, dictionary have input file types as keyword, and its related order of appearance (interger) as value
    """
    keyWordDict = {} 
    count = 0
    for inFile in currentInputFiles:
      keyWordDict[inFile.getType().lower()] = count 
      count = count + 1
    return keyWordDict
    
  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      this generate a new input file depending on which sampler is chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ @ currentInputFiles, list,  list of current input files (input files from last this method call) (perturbed)
    """ 
    import DecayParser
    import FissionYieldParser
    import QValuesParser
    import MaterialParser
    import PathParser
    import XSCreator    
    self.typeDict = {}   
    
    self.typeDict = self.mapInputFileType(currentInputFiles)
    self.distributedPerturbedVars = self.distributeVariablesToParsers(Kwargs['SampledVars'] )
    self.parseControlOptions(currentInputFiles[self.typeDict['depletion_input']].getAbsFile(), currentInputFiles[self.typeDict['path']].getAbsFile()) 
    self.syncLibPathFileWithRavenInp(currentInputFiles[self.typeDict['path']].getAbsFile(),currentInputFiles,self.typeDict)
    self.outputFileNames(currentInputFiles[self.typeDict['path']].getAbsFile())
    self.instantOutput = self.jobTitle+'.o'
    self.forcePrintLibraries(currentInputFiles[self.typeDict['xs-library']].getAbsFile())
    self.depInp = currentInputFiles[self.typeDict['depletion_input']].getAbsFile()
    self.phisicsInp = currentInputFiles[self.typeDict['inp']].getAbsFile()
    if Kwargs['precommand'] == '': self.numberOfMPI = 1 
    else                         : self.numberOfMPI = self.getNumberOfMPI(Kwargs['precommand'])   
    
    for perturbedParam in self.distributedPerturbedVars.iterkeys():
      if perturbedParam == 'DECAY'      : 
        DecayParser.DecayParser(currentInputFiles[self.typeDict['decay']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'DENSITY'    : 
        MaterialParser.MaterialParser(currentInputFiles[self.typeDict['material']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'FY'         : 
        FissionYieldParser.FissionYieldParser(currentInputFiles[self.typeDict['fissionyield']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'QVALUES'    : 
        QValuesParser.QValuesParser(currentInputFiles[self.typeDict['fissqvalue']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'ALPHADECAY' : 
        PathParser.PathParser(currentInputFiles[self.typeDict['alphadecay']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETA+DECAY' : 
        PathParser.PathParser(currentInputFiles[self.typeDict['beta+decay']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETA+XDECAY': 
        PathParser.PathParser(currentInputFiles[self.typeDict['beta+xdecay']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETADECAY'  : 
        PathParser.PathParser(currentInputFiles[self.typeDict['betadecay']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETAXDECAY' : 
        PathParser.PathParser(currentInputFiles[self.typeDict['betaxdecay']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'INTTRADECAY': 
        PathParser.PathParser(currentInputFiles[self.typeDict['inttradecay']].getAbsFile(),**self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'XS'         : 
        XSCreator.XSCreator(currentInputFiles[self.typeDict['xs']].getAbsFile(), self.tabulation,**self.distributedPerturbedVars[perturbedParam])
    return currentInputFiles
