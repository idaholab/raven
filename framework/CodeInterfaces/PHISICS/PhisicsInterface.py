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
from  __builtin__ import any as bAny
from CodeInterfaceBaseClass import CodeInterfaceBase
import phisicsdata
import xml.etree.ElementTree as ET


class Phisics(CodeInterfaceBase):
  """
    this class is used a part of a code dictionary to specialize Model.Code for RELAP5-3D Version 4.0.3
  """ 
  def getPath(self):
    """
      Retriever for path.
      @ In, None
      @ Out, __path, string, path
    """
    return self.__path
    
  def getBase(self):
    """
      Retriever for file base.
      @ In, None
      @ Out, __base, string path
    """
    return self.__base 

  def getTitle(self, depletionRoot):
    """
      get the job title. It will become later the instant output file name. 
      @ In, self.mrtauStandAlone, True = mrtau is ran standalone, False = mrtau in not ran standalone
      @ In, depletionRoot, XML tree from the depletion_input.xml
      @ In, inpRoot, XML tree from the inp.xml
      @ Out, fileTitle, string 
    """
    for child in depletionRoot.findall(".//title"):
      jobTitle = str(child.text)
      break 
    return jobTitle
    
  def verifyMrtauFlagsAgree(self, depletionRoot):
    """
      Verifies the node "standalone"'s text in the depletion_input xml. if the standalone flag 
      in the depletion_input disagrees with the mrtau standalone flag in the raven input, 
      the codes errors out
      @ In, mrtauStandAlone, True = mrtau is ran standalone, False = mrtau in not ran standalone
      @ In, depletionRoot, XML tree from the depletion_input.xml
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
  
  def parseControlOptions(self, depletionFile):
    """
      Parse the Material.xml data file and put the isotopes name as key and 
      the decay constant relative to the isotopes as values 
      @ In, depletionFile, string, depletion_input file
      @ In, inpFile, string, Instant input file 
    """   
    depletionTree = ET.parse(depletionFile)
    depletionRoot = depletionTree.getroot()
    self.verifyMrtauFlagsAgree(depletionRoot)
    jobTitle = self.getTitle(depletionRoot)
    return jobTitle 
  
  def distributeVariablesToParsers(self, perturbedVars):
    """
      This module takes the perturbedVars dictionary. perturbedVars contains all the variables to be perturbed. 
      The module transform the dictionary into dictionary of dictionary. This dictionary renders easy the distribution 
      of the variable to their corresponding parser. For example, if the two variables are the following: 
      {'FY|FAST|PU241|SE78':1.0, 'DECAY|BETA|U235':2.0}, the output dict will be: 
      {'FY':{'FY|FAST|PU241|SE78':1.0}, 'DECAY':{'DECAY|BETA|U235':2.0}}
      In: perturbVars, dictionary 
      out: distributedVars, dictionary of dictionary 
    """
    distributedPerturbedVars = {}
    pertType = []
    #print (perturbedVars)
    # teach what are the type of perturbation (decay FY etc...)
    for i in perturbedVars.iterkeys():
      splittedKeywords = i.split('|')
      pertType.append(splittedKeywords[0])
    # declare all the dictionaries according the different type of pert
    for i in xrange (0,len(pertType)):
      distributedPerturbedVars[pertType[i]] = {}
    # populate the dictionaries 
    for key, value in perturbedVars.items():
      splittedKeywords = key.split('|')
      for j in xrange (0,len(pertType)):
        if splittedKeywords[0] == pertType[j] :
          distributedPerturbedVars[pertType[j]][key] = value
    #print (distributedPerturbedVars)
    return distributedPerturbedVars
  
  def addDefaultExtension(self):
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
    self.tabulation = True
    self.mrtauStandAlone = False
    self.mrtauExecutable = None 
    for child in xmlNode:
      if child.tag == 'PerturbXS':
        if child.text.lower() in set(validPerturbation): self.perturbXS = child.text.lower()
        else: raise ValueError("\n\nThe type of perturbation --"+child.text.lower()+"-- is not valid. You can choose one of the following \n"+"\n".join(set(validPerturbation)))
      if child.tag == 'tabulation':
        self.tabulation = None
        if (child.text.lower() == 't' or child.text.lower() == 'true'):  self.tabulation = True
        if (child.text.lower() == 'f' or child.text.lower() == 'false'): self.tabulation = False 
        if (self.tabulation is None): raise ValueError("\n\n The tabulation node -- <"+child.tag+"> -- only supports the following text (case insensitive): \n True \n T \n False \n F" )
      if child.tag == 'mrtauStandAlone':
        self.mrtauStandAlone = None 
        if (child.text.lower() == 't' or child.text.lower() == 'true'):  self.mrtauStandAlone = True 
        if (child.text.lower() == 'f' or child.text.lower() == 'false'):  self.mrtauStandAlone = False 
        if (self.mrtauStandAlone is None): raise ValueError("\n\n The flag activating MRTAU standalone mode -- <"+child.tag+"> -- only supports the following text (case insensitive): \n True \n T \n False \n F. \n Default Value is False" )
      if child.tag == 'mrtauStandAloneExecutable' and self.mrtauStandAlone == True:
        self.mrtauExecutable = child.text
        
  def switchExecutable(self):
    """
      This module replaces the executable if the user chooses to use MRTAU in standalone mode. 
      @ In, None 
      @ Out, mrtauExecutable, string, absolute path to mrtau executable
    """
    return self.mrtauExecutable
   
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
    if self.mrtauStandAlone == True: 
      executable = self.switchExecutable()
    found = False
    index = 0
    outputfile = 'out~'+inputFiles[index].getBase()
    #printprint (outputfile)
    if clargs: addflags = clargs['text']
    else     : addflags = ''
    #commandToRun = executable + ' -i ' + inputFiles[index].getFilename() + ' -o ' + outputfile  + '.o' + ' -r ' + outputfile  + '.r' + addflags
    #commandToRun = executable + ' -i ' + inputFiles[index].getFilename() + ' -o ' + outputfile  + '.o' +  addflags
    commandToRun = executable
    commandToRun = commandToRun.replace("\n"," ")
    commandToRun  = re.sub("\s\s+" , " ", commandToRun )
    #print (commandToRun)
    returnCommand = [('parallel',commandToRun)], outputfile
    #print (commandToRun)
    #print (returnCommand)
    return returnCommand
  
  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
    """
    output = 'Dpl_INSTANT.outp-0'
    splitWorkDir = workingDir.split('/')
    pertNumber = splitWorkDir[-1]
    outputobj=phisicsdata.phisicsdata(output, workingDir, self.mrtauStandAlone, self.jobTitle)
    return "keff"+str(pertNumber).strip()
    #if outputobj.hasAtLeastMinorData(): outputobj.writeCSV(os.path.join(workingDir,output+'.csv'))
    #else: raise IOError('Relap5 output file '+ command.split('-o')[0].split('-i')[-1].strip()+'.o' + ' does not contain any minor edits. It might be crashed!')

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      (e.g. in RELAP5 would be the keyword "********")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    #from  __builtin__ import any as bAny
    #failure = True
    #errorWord = ["ERROR the number of materials in mat_map_to_instant block"]
    #try   : outputToRead = open(os.path.join(workingDir,output+'.o'),"r")
    #except: return failure
    #readLines = outputToRead.readlines()
    #for goodMsg in errorWord:
    #  if bAny(goodMsg in x for x in readLines):
    #    failure = False
    #    break
    
    
    failure = False
    return failure

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      this generate a new input file depending on which sampler is chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """ 
    import DecayParser
    import FissionYieldParser
    import QValuesParser
    import MaterialParser
    import PathParser
    import XSCreator
    
    keyWordDict = {} 
    count = 0
    #print (currentInputFiles)
    #print (Kwargs)
    #print (Kwargs['SampledVars'])
    perturbedVars = Kwargs['SampledVars']
    
    for inFile in currentInputFiles:
      keyWordDict[inFile.getType().lower()] = count 
      count = count + 1

    distributedPerturbedVars = self.distributeVariablesToParsers(perturbedVars)
    #print (distributedPerturbedVars)
    #print (currentInputFiles)
    #print (self.tabulation)
    booleanTab = self.tabulation 
    self.jobTitle = self.parseControlOptions(currentInputFiles[keyWordDict['depletion_input']].getAbsFile())
    
    for i in distributedPerturbedVars.iterkeys():
      if i == 'DECAY'         : decayParser        = DecayParser.DecayParser(currentInputFiles[keyWordDict['decay']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'DENSITY'       : materialParser     = MaterialParser.MaterialParser(currentInputFiles[keyWordDict['material']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'FY'            : FissionYieldParser = FissionYieldParser.FissionYieldParser(currentInputFiles[keyWordDict['fissionyield']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'QVALUES'       : QValuesParser      = QValuesParser.QValuesParser(currentInputFiles[keyWordDict['fissqvalue']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'ALPHADECAY'    : BetaDecayParser    = PathParser.PathParser(currentInputFiles[keyWordDict['alphadecay']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'BETA+DECAY'    : BetaDecayParser    = PathParser.PathParser(currentInputFiles[keyWordDict['beta+decay']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'BETA+XDECAY'   : BetaDecayParser    = PathParser.PathParser(currentInputFiles[keyWordDict['beta+xdecay']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'BETADECAY'     : BetaDecayParser    = PathParser.PathParser(currentInputFiles[keyWordDict['betadecay']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'BETAXDECAY'    : BetaDecayParser    = PathParser.PathParser(currentInputFiles[keyWordDict['betaxdecay']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'INTTRADECAY'   : BetaDecayParser    = PathParser.PathParser(currentInputFiles[keyWordDict['inttradecay']].getAbsFile(), **distributedPerturbedVars[i])
      if i == 'XS'            : XSParser           = XSCreator.XSCreator(currentInputFiles[keyWordDict['xs']].getAbsFile(), booleanTab, **distributedPerturbedVars[i])
    return currentInputFiles
    

