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
Created on February 1st 2018
@author: rouxpn
"""
import os
import re
from . import combine
import xml.etree.ElementTree as ET
import copy
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from . import phisicsdata
from ..RELAP5 import relapdata
from .PhisicsInterface import Phisics
from ..RELAP5.Relap5Interface import Relap5


class PhisicsRelap5(CodeInterfaceBase):
  """
    Class that links the PHISICS and RELAP interfaces.
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.PhisicsInterface = Phisics()
    self.PhisicsInterface.addDefaultExtension()
    self.Relap5Interface = Relap5()
    self.Relap5Interface.addDefaultExtension()

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    relapInputs, phisicsInputs= self.tailorRelap5InputFiles(oriInputFiles)
    self.PhisicsInterface.initialize(runInfo, phisicsInputs)
    self.Relap5Interface.initialize(runInfo, relapInputs)

  def definePhisicsVariables(self):
    """
      Lists the variables perturbable within PHISICS. The other variables will be related to RELAP.
      @ In, None
      @ Out, phisicsVariables, list
    """
    self.phisicsVariables = ['DENSITY','XS','DECAY','FY','QVALUES','ALPHADECAY','BETA+DECAY','BETA+XDECAY','BETADECAY','BETAXDECAY','INTTRADECAY']

  def defineRelapInput(self,currentInputFiles):
    """
      Lists the input file types relative to RELAP5. The other input types will be related to PHISICS. The RELAP input
      files have a type attribute strating with the string 'relap'
      @ In, currentInputFiles,  list,  list of current input files (input files from last this method call)
      @ Out, , list
    """
    relapInputTypes = []
    for inFile in currentInputFiles:
      if re.match(r'relap',inFile.getType().lower()):
        relapInputTypes.append(inFile.getType().lower())
    return relapInputTypes

  def addDefaultExtension(self):
    """
      Possible input extensions found in the input files.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['xml','dat','path','bin','i','inp','x','in'])

  def depTime(self,inputXML,searchDict,dictKeys):
    """
      Synchronizes the PHISICS time and RELAP time based on the input.
      The input read are the xml depletion file (PHISICS) and xml input file (PHISICS).
      The resulting ditionary is use in the "combine" class to print the PHISICS and RELAP csv output in the proper time lines.
      @ In, inputXML, depletion input name
      @ In, searchDict, dictionary, dictionary containing dummy keys, and the searched keywords as values
      @ In, dictKeys, dictionary, dictionary containing dummy keys, the values are the keys of the output dict timeDict
      @ Out, timeDict, dictionary
    """
    timeDict = {}
    libraryTree = ET.parse(inputXML)
    libraryRoot = libraryTree.getroot()
    for key in searchDict:
      for child in libraryRoot.iter(searchDict[key]):
        timeDict[dictKeys[key]] = child.text
    return timeDict

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    self.PhisicsInterface._readMoreXML(xmlNode)
    self.Relap5Interface._readMoreXML(xmlNode)

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual
                                       command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    mapDict = self.mapInputFileType(inputFiles)
    outputfile = 'out~'+inputFiles[mapDict['relapInp'.lower()]].getBase()
    self.outFileName = inputFiles[mapDict['relapInp'.lower()]].getBase() # used in finalizeCodeOutput
    self.outputExt = '.o'
    commandToRun = executable + ' -i ' +inputFiles[mapDict['relapInp'.lower()]].getFilename() + ' -o  ' + self.outFileName + self.outputExt
    commandToRun = commandToRun.replace("\n"," ")
    commandToRun  = re.sub(r"\s\s+", " ",commandToRun)
    returnCommand = [('parallel',commandToRun)], outputfile
    return returnCommand

  def mapInputFileType(self,currentInputFiles):
    """
      Assigns a number to the input file Types.
      @ In, currentInputFiles,  list,  list of current input files (input files from last this method call)
      @ Out, keyWordDict, dictionary, dictionary have input file types as keyword, and its related order of appearance (interger) as value
    """
    keyWordDict = {}
    count = 0
    for inFile in currentInputFiles:
      keyWordDict[inFile.getType().lower()] = count
      count = count + 1
    return keyWordDict

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
      This methods also calls the method 'mergeOutput' if MPI mode is used, in order to merge all the output files into one.
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, None
    """
    # RELAP post processing
    dataRelap = self.Relap5Interface.finalizeCodeOutput(command,self.outFileName,workingDir)
    # PHISICS post processing
    dataPhisics = self.PhisicsInterface.finalizeCodeOutput(command,output,workingDir,phiRel=True,relapOut=self.outFileName)
    cmb = combine.combine(workingDir,dataRelap, dataPhisics,self.depTimeDict,self.inpTimeDict)
    response = cmb.returnData()
    return response

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
    failure = True
    failure = self.Relap5Interface.checkForOutputFailure(self.outFileName,workingDir)
    if failure:
      raise IOError('The check Output Failure returns a job failed')
    return failure

  def tailorRelap5InputFiles(self,currentInputFiles):
    """
      Generates a list of relap5 files only, in order to pass it to the RELAP parsers.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ Out, relap5CurrentInputFiles, list, list of RELAP current input files (input files from last this method call)
      @ Out, phisicsCurrentInputFiles, list, list of PHISICS current input files (input files from last this method call)
    """
    relap5CurrentInputFiles = []
    phisicsCurrentInputFiles = []
    relapInputTypes = self.defineRelapInput(currentInputFiles)
    for inFile in currentInputFiles:
      if inFile.getType().lower() in set(relapInputTypes):
        relap5CurrentInputFiles.append(inFile)
      else:
        phisicsCurrentInputFiles.append(inFile)
    return relap5CurrentInputFiles, phisicsCurrentInputFiles

  def tailorSampledVariables(self,perturbedVars):
    """
      Shapes the key 'SampledVars' from the Kwargs that suits properly the code of interest.
      Hence, the variables relative to PHISICS and RELAP5 are separated to feed to the two codes only the variables they recognize.
      @ In, perturbedVars, dictionary, dictionary of the PHISICS and RELAP variables
      @ Out, None
    """
    passToDesignatedCode = {}
    passToDesignatedCode['phisics'] = {}
    passToDesignatedCode['phisics']['SampledVars'] = {}
    passToDesignatedCode['relap5']  = {}
    passToDesignatedCode['relap5']['SampledVars']  = {}
    for var,value in perturbedVars.items():
      if var.split('|')[0] in set(self.phisicsVariables):
        passToDesignatedCode['phisics']['SampledVars'][var] = value
      else:
        passToDesignatedCode['relap5']['SampledVars'][var] = value
    return passToDesignatedCode

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      Generates a new input file depending on which sampler is chosen.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, currentInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    if self.PhisicsInterface.mrtauStandAlone:  # errors out if MRTAU standalone and PHISICS/RELAP5 are activated simultanously
      raise ValueError('MRTAU cannot be used in standalone mode in PHISICS/RELAP5 calculations.')
    self.definePhisicsVariables()
    self.outputDeck = self.Relap5Interface.outputDeck
    perturbedVars = Kwargs['SampledVars']
    localKwargs = copy.deepcopy(Kwargs)
    perturbedVars = localKwargs.pop('SampledVars')
    passToDesignatedCode = self.tailorSampledVariables(perturbedVars)
    passToDesignatedCode['phisics'].update(localKwargs)
    passToDesignatedCode['relap5'].update(localKwargs)
    relap5CurrentInputFiles, phisicsCurrentInputFiles = self.tailorRelap5InputFiles(currentInputFiles)
    # PHISICS
    self.PhisicsInterface.createNewInput(phisicsCurrentInputFiles,oriInputFiles,samplerType,**passToDesignatedCode['phisics'])
    # RELAP
    self.Relap5Interface.createNewInput(relap5CurrentInputFiles,oriInputFiles,samplerType,**passToDesignatedCode['relap5'])
    self.depTimeDict = self.depTime(self.PhisicsInterface.depInp,{'search1':'n_tab_interval','search2':'tab_time_step'},{'search1':'nBUsteps','search2':'timeSteps'})
    self.inpTimeDict = self.depTime(self.PhisicsInterface.phisicsInp,{'search1':'TH_between_BURN'},{'search1':'TH_between_BURN'})
    return currentInputFiles
