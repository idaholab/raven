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
import os
import re
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..Generic.GenericParser import GenericParser
from . import DecayParser
from . import FissionYieldParser
from . import QValuesParser
from . import MaterialParser
from . import PathParser
from . import XSCreator
from . import MassParser  # for MRTAU standalone
from . import phisicsdata
import xml.etree.ElementTree as ET


class Phisics(CodeInterfaceBase):
  """
    Code interface for PHISICS
  """

  def getNumberOfMpi(self, string):
    """
      Gets the number of MPI requested by the user in the RAVEN input.
      @ In, string, string, string from the Kwargs containing the number of MPI
      @ Out, getNumberOfMpi, integer, number of MPI used in the calculation
    """
    return int(string.split(" ")[-2])

  def outputFileNames(self, pathFile):
    """
      Collects the output file names from xml path file.
      @ In, pathFile, string, lib_path_input file
      @ Out, None
    """
    pathTree = ET.parse(pathFile)
    pathRoot = pathTree.getroot()
    self.outputFileNameDict = {}
    xmlNodes = [
        'reactions', 'atoms_plot', 'atoms_csv', 'decay_heat', 'bu_power',
        'flux', 'repository'
    ]
    for xmlNodeNumber in range(0, len(xmlNodes)):
      for xmlNode in pathRoot.iter(xmlNodes[xmlNodeNumber]):
        self.outputFileNameDict[xmlNodes[xmlNodeNumber]] = xmlNode.text

  def syncLibPathFileWithRavenInp(self, pathFile, currentInputFiles,
                                  keyWordDict):
    """
      Parses the xml path file and writes the correct library path in the xml path file, based in the raven input.
      @ In, pathFile, string, xml path file
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, keyWordDict, dictionary, keys: type associated to an input file. Values: integer, unique associated to each input file
      @ Out, pathFile, string, lib_path_input file, updated with user-defined path
    """
    pathTree = ET.parse(pathFile)
    pathRoot = pathTree.getroot()
    typeList = [
        'IsotopeList', 'mass', 'decay', 'budep', 'FissionYield', 'FissQValue',
        'CRAM_coeff_PF', 'N,G', 'N,Gx', 'N,2N', 'N,P', 'N,ALPHA', 'AlphaDecay',
        'BetaDecay', 'BetaxDecay', 'Beta+Decay', 'Beta+xDecay', 'IntTraDecay'
    ]
    libPathList = [
        'iso_list_inp', 'mass_a_weight_inp', 'decay_lib', 'xs_sep_lib',
        'fiss_yields_lib', 'fiss_q_values_lib', 'cram_lib', 'n_gamma',
        'n_gamma_ex', 'n_2n', 'n_p', 'n_alpha', 'alpha', 'beta', 'beta_ex',
        'beta_plus', 'beta_plus_ex', 'int_tra'
    ]

    for typeNumber in range(len(typeList)):
      for libPathText in pathRoot.iter(libPathList[typeNumber]):
        libPathText.text = os.path.join(
            currentInputFiles[keyWordDict[typeList[typeNumber].lower()]]
            .subDirectory, currentInputFiles[keyWordDict[typeList[typeNumber]
                                                         .lower()]].getBase() +
            '.' + currentInputFiles[keyWordDict[typeList[typeNumber]
                                                .lower()]].getExt())
    pathTree.write(pathFile)

  def syncPathToLibFile(self, depletionRoot, depletionFile, depletionTree,
                        libPathFile):
    """
      Prints the name of the files that contains the path to the libraries in the xml depletion input.
      @ In, depletionRoot, xml.etree.ElementTree.Element, depletion input xml node
      @ In, depletionFile, string, path to xml depletion file
      @ In, depletionTree, xml.etree.ElementTree.Element, xml tree from the depletion_input.xml
      @ In, libPathFile, string, lib_path.xml file
      @ Out, None
    """
    if depletionTree.find('.//input_files') is None:
      inputFilesNode = ET.Element("input_files")
      inputFilesNode.text = libPathFile
      depletionTree.getroot().insert(0,inputFilesNode)
    else:
      depletionTree.find('.//input_files').text = libPathFile
    depletionTree.write(depletionFile)

  def getTitle(self, depletionRoot):
    """
      Gets the job title. It will become later the INSTANT output file name. If the title flag is not in the
      INSTANT input, the job title is defaulted to 'defaultInstant'.
      @ In, depletionRoot, xml.etree.ElementTree.Element, depletion input xml node
      @ Out, None
    """
    self.jobTitle = 'defaultInstant'
    for child in depletionRoot.findall(".//title"):
      self.jobTitle = child.text.strip()
      if " " in self.jobTitle:
        raise IOError("Job title can not have spaces in the title but must be a single string. E.g. from "+self.jobTitle+ " to "+ self.jobTitle.replace(" ",""))
      break
    return

  def timeUnit(self, depletionRoot):
    """
      Parses the xml depletion file to find the time unit. Default: seconds (string).
      @ In, depletionRoot, xml.etree.ElementTree.Element, depletion input xml node
      @ Out, None
    """
    self.timeControl = 'seconds'
    for child in depletionRoot.findall(".//time_control"):
      self.timeControl = child.attrib.get("type")
      break

  def parseControlOptions(self, depletionFile, libPathFile):
    """
      Parses the xml depletion file and library path name file to obtain the control options.
      Verifies if the mrtau flag agree between the RAVEN input and depletion file, Gets the decay heat options.
      Gets the job name and synchronizes the files names from RAVEN in the library path file. Get the time units.
      @ In, depletionFile, string, xml depletion input file name
      @ In, libPathFile, string, xml library path file name
      @ Out, None
    """
    depletionTree = ET.parse(depletionFile)
    depletionRoot = depletionTree.getroot()
    self.findDecayHeatFlag(depletionRoot)
    self.timeUnit(depletionRoot)
    self.getTitle(depletionRoot)
    self.syncPathToLibFile(depletionRoot, depletionFile, depletionTree,
                           libPathFile)

  def distributeVariablesToParsers(self, perturbedVars):
    """
      Transforms a dictionary into dictionary of dictionaries. This dictionary renders easy the distribution
      of the variables to their corresponding parser. For example, if the two variables are the following:
      {'FY|FAST|PU241|SE78':1.0, 'DECAY|BETA|U235':2.0}, the output dict will become:
      {'FY':{'FY|FAST|PU241|SE78':1.0}, 'DECAY':{'DECAY|BETA|U235':2.0}}
      @ In, perturbedVars, dictionary, dictionary of the perturbed variables
      @ Out, distributedPerturbedVars, dictionary of dictionaries containing the perturbed variables
    """
    distributedPerturbedVars = {}
    pertType = []
    for i in perturbedVars.keys():
      # teach what are the type of perturbation (decay FY etc...)
      if "|" in i:
        pertType.append(i.split('|')[0])
      else:
        pertType.append("generic")
    for i in range(0, len(pertType)):
      # declare all the dictionaries according the different type of pert
      distributedPerturbedVars[pertType[i]] = {}
    for key, value in perturbedVars.items():
      # populate the dictionaries
      splittedKeywords = key.split('|') if '|' in key else ["generic"]
      for j in range(0, len(pertType)):
        if splittedKeywords[0] == pertType[j]:
          distributedPerturbedVars[pertType[j]][key] = value
    return distributedPerturbedVars

  def addDefaultExtension(self):
    """
      Possible input extensions found in the input files.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['xml', 'dat', 'path'])

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize some
      members based on inputs. This can be overloaded in specialize code interface in order to read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node
      @ Out, None.
    """
    #default values if the flag is not in the raven input
    self.mrtauStandAlone = False
    self.phisicsRelap = False
    self.printSpatialRR = False
    self.printSpatialFlux = False
    self.executableMrtau = None
    for child in xmlNode:
      if child.tag == 'mrtauStandAlone':
        self.mrtauStandAlone = None
        if (child.text.lower() == 't' or child.text.lower() == 'true'):
          self.mrtauStandAlone = True
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'):
          self.mrtauStandAlone = False
        else:
          raise ValueError(
              "\n\n The flag activating MRTAU standalone mode -- <" +
              child.tag +
              "> -- only supports the following text (case insensitive): \n True \n T \n False \n F. \n Default Value is False"
          )
      if child.tag == 'printSpatialRR':
        if (child.text.lower() == 't' or child.text.lower() == 'true'):
          self.printSpatialRR = True
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'):
          self.printSpatialRR = False
        else:
          raise ValueError(
              "\n the node " + child.tag + "has to be a boolean entry")

      if child.tag == 'printSpatialFlux':
        if (child.text.lower() == 't' or child.text.lower() == 'true'):
          self.printSpatialFlux = True
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'):
          self.printSpatialFlux = False
        else:
          raise ValueError(
              "\n the node " + child.tag + "has to be a boolean entry")

  def generateCommand(self, inputFiles, executable, clargs=None, fargs=None, preExec=None):
    """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have
                              been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in
                                    the input (e.g. under the node < Code >< clargstype =0 input0arg =0
                                    i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can
                                   specify in the input (e.g. under the node < Code >< clargstype =0 input0arg
                                   =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual
                                       command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to
                                   run the code (string), returnCommand[1] is the name of the output root
    """
    mapDict = self.mapInputFileType(inputFiles)
    if self.mrtauStandAlone:
      commandToRun = executable
      outputfile = 'out~'
    else:
      commandToRun = executable + ' -i ' + inputFiles[mapDict['inp'.lower(
      )]].getFilename() + ' -xs ' + inputFiles[mapDict['Xs-library'.lower(
      )]].getFilename() + ' -mat ' + inputFiles[mapDict['Material'.lower(
      )]].getFilename() + ' -dep ' + inputFiles[mapDict['Depletion_input'.lower(
      )]].getFilename() + ' -o ' + self.instantOutput
      commandToRun = commandToRun.replace("\n", " ")
      commandToRun = re.sub(r"\s\s+", " ", commandToRun)
      outputfile = 'out~' + inputFiles[mapDict['inp'.lower()]].getBase()
    returnCommand = [('parallel', commandToRun)], outputfile
    return returnCommand

  def finalizeCodeOutput(self, command, output, workingDir, **phiRel):
    """
      This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv.
      This methods also calls the method 'mergeOutput' if MPI mode is used, in order to merge all the output files into one
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ In, phiRel, dictionary, contains a key 'phiRel', value is True if PHISICS/RELAP is in coupled mode, empty otherwise
                                     and a key 'relapOut', value is the RELAP main output name
      @ Out, response, dict, the dictionary containing the output data
    """
    phisicsDataDict = {}
    if "phiRel" not in phiRel:
      phiRel['phiRel'] = False
    if "relapOut" not in phiRel:
      phiRel['relapOut'] = None
    phisicsDataDict['relapOut'] = phiRel[
        'relapOut']  # RELAP output name, needed if MPI = 1
    phisicsDataDict['output'] = output
    phisicsDataDict['timeControl'] = self.timeControl
    phisicsDataDict['decayHeatFlag'] = self.decayHeatFlag
    phisicsDataDict['instantOutput'] = self.instantOutput
    phisicsDataDict['workingDir'] = workingDir
    phisicsDataDict['mrtauStandAlone'] = self.mrtauStandAlone
    phisicsDataDict['jobTitle'] = self.jobTitle
    phisicsDataDict['mrtauFileNameDict'] = self.outputFileNameDict
    phisicsDataDict['numberOfMPI'] = self.numberOfMPI
    phisicsDataDict['phiRel'] = phiRel['phiRel']
    phisicsDataDict['printSpatialRR'] = self.printSpatialRR
    phisicsDataDict['printSpatialFlux'] = self.printSpatialFlux
    phisicsDataDict['pertVariablesDict'] = self.distributedPerturbedVars
    # read outputs
    outputParser = phisicsdata.phisicsdata(phisicsDataDict)
    response = outputParser.returnData()
    return response

  def checkForOutputFailure(self, output, workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a
      return code that is 0.
      This can happen in those codes that record the failure of the job (e.g.
      not converged, etc.) as normal termination (returncode == 0).
      This method can be used, for example, to parse the outputfile looking for a
      special keyword that testifies that a particular job got failed.
      The line Task ended is searched in the PHISICS output as successful job message.
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = True
    if not self.mrtauStandAlone:
      outputFile = None
      if os.path.exists(os.path.join(workingDir, self.instantOutput)):
        outputFile = os.path.join(workingDir, self.instantOutput)
      elif os.path.exists(os.path.join(workingDir, self.instantOutput+"0")):
        outputFile = os.path.join(workingDir, self.instantOutput+"0")
      elif os.path.exists(os.path.join(workingDir, self.instantOutput+"-0")):
        outputFile = os.path.join(workingDir, self.instantOutput+"-0")
      if outputFile is not None:
        with open(outputFile, 'r') as f:
          for line in f:
            if re.search(r'task\s+ended', line, re.IGNORECASE):
              failure = False
    else:
      if os.path.exists(os.path.join(workingDir, 'Errors_CPUt.txt')):
        with open(os.path.join(workingDir, 'Errors_CPUt.txt'), 'r') as f:
          for line in f:
            if re.search(r'NO\s+ERRORS\s+IN\s+THE\s+CALCULATION', line, re.IGNORECASE):
              failure = False
    return failure

  def mapInputFileType(self, currentInputFiles):
    """
      Assigns a unique integer to the input file Types.
      @ In, currentInputFiles,  list,  list of current input files (input files
                                       from last this method call)
      @ Out, keyWordDict, dict, dictionary have input file types as keys,
                                and its related order of appearance (interger) as values
    """
    keyWordDict = {}
    count = 0
    for inFile in currentInputFiles:
      keyWordDict[inFile.getType().lower()] = count
      count = count + 1
    return keyWordDict

  def checkInput(self):
    """
      Check that the input files required by the interface are given by the user in the Input block.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ Out, None
    """
    mandatoryType = ["depletion_input", "path"]
    if not self.mrtauStandAlone:
      mandatoryType.extend(["inp", "material", "xs-library"])
    else:
      mandatoryType.append("mass")
    for key in mandatoryType:
      if key not in self.typeDict.keys():
        raise IOError(
            "Error in Input block. The input file with the type attribute " +
            key + " is missing \n")

  def isThereTabMappinp(self, currentInputFiles):
    """
      If an Input file has a type attribute 'tabMapping', tabulation mapping is considered to be True.
      No tabulation mapping otherwise.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ Out, isThereTabMappinp, boolean, True if a tabulation mapping file exist, flase otherwise
      @ Out, isThereTabMappinp, string, path to tabulation mapping file name
    """
    try:
      return True, currentInputFiles[self.typeDict['tabmap']].getAbsFile()
    except KeyError:  # no tabMap type attribute, hence, no tab Mapping desired
      return False, None

  def findDecayHeatFlag(self, depletionRoot):
    """
      Parses the xml depletion input, and return the decay heat flag in the input.
      1 (default) no decay heat printed, 2 means decay heat in KW, 3 means decay heat in MeV/s
      @ In, depletionRoot, xml.etree.ElementTree.Element, depletion input xml node
      @ Out, None
    """
    self.decayHeatFlag = 1
    for child in depletionRoot.findall(".//decay_heat_type"):
      self.decayHeatFlag = int(child.text)
      break

  def verifyPhiFlags(self):
    """
      Verifies if the flag <blengen> is in the phisics input file. The flags activates the
      reaction rate printing (solve report, fission matrices). Also verifies if the flag <echo> is
      set to 2. echo 2 prints the k-eff in the output.
      @ In, None
      @ Out, None
    """
    blengenFlag = False
    echoFlag = False
    inpTree = ET.parse(self.phisicsInp)
    inpRoot = inpTree.getroot()
    for child in inpRoot.findall(".//blengen"):
      if child.text.lower() == 't' or child.text == 'true':
        blengenFlag = True
    for child in inpRoot.findall(".//echo"):
      if child.text == '2':
        echoFlag = True
    if not blengenFlag or not echoFlag:
      raise ValueError(
          "\n Flag error in " + self.phisicsInp +
          ". The flag blengen has to be True, and the flag echo has to be set to 2."
      )

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType,
                     **Kwargs):
    """
      This generate a new input file depending on which sampler is chosen.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, currentInputFiles, list, list of current input files (input files from last this method call) (perturbed)
    """

    self.typeDict = {}
    self.typeDict = self.mapInputFileType(currentInputFiles)
    self.checkInput()
    self.distributedPerturbedVars = self.distributeVariablesToParsers(
        Kwargs['SampledVars'])
    self.parseControlOptions(
        currentInputFiles[self.typeDict['depletion_input']].getAbsFile(),
        currentInputFiles[self.typeDict['path']].getAbsFile())
    self.syncLibPathFileWithRavenInp(
        currentInputFiles[self.typeDict['path']].getAbsFile(),
        currentInputFiles, self.typeDict)
    self.outputFileNames(currentInputFiles[self.typeDict['path']].getAbsFile())
    self.instantOutput = self.jobTitle.replace(" ", "") + '.o'
    self.depInp = currentInputFiles[self.typeDict[
        'depletion_input']].getAbsFile()  # for PHISICS/RELAP interface
    if not self.mrtauStandAlone:
      self.phisicsInp = currentInputFiles[self.typeDict[
          'inp']].getAbsFile()  # for PHISICS/RELAP interface
      self.verifyPhiFlags()
    booleanTabMap, tabMapFileName = self.isThereTabMappinp(currentInputFiles)
    if Kwargs['precommand'] == '':
      self.numberOfMPI = 1
    else:
      self.numberOfMPI = self.getNumberOfMpi(Kwargs['precommand'])
    for perturbedParam in self.distributedPerturbedVars.keys():
      if perturbedParam == 'DECAY':
        DecayParser.DecayParser(
            currentInputFiles[self.typeDict['decay']].getAbsFile(),
            currentInputFiles[self.typeDict['decay']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'DENSITY':
        MaterialParser.MaterialParser(
            currentInputFiles[self.typeDict['material']].getAbsFile(),
            currentInputFiles[self.typeDict['material']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'FY':
        FissionYieldParser.FissionYieldParser(
            currentInputFiles[self.typeDict['fissionyield']].getAbsFile(),
            currentInputFiles[self.typeDict['fissionyield']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'QVALUES':
        QValuesParser.QValuesParser(
            currentInputFiles[self.typeDict['fissqvalue']].getAbsFile(),
            currentInputFiles[self.typeDict['fissqvalue']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'ALPHADECAY':
        PathParser.PathParser(
            currentInputFiles[self.typeDict['alphadecay']].getAbsFile(),
            currentInputFiles[self.typeDict['alphadecay']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETA+DECAY':
        PathParser.PathParser(
            currentInputFiles[self.typeDict['beta+decay']].getAbsFile(),
            currentInputFiles[self.typeDict['beta+decay']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETA+XDECAY':
        PathParser.PathParser(
            currentInputFiles[self.typeDict['beta+xdecay']].getAbsFile(),
            currentInputFiles[self.typeDict['beta+xdecay']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETADECAY':
        PathParser.PathParser(
            currentInputFiles[self.typeDict['betadecay']].getAbsFile(),
            currentInputFiles[self.typeDict['betadecay']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'BETAXDECAY':
        PathParser.PathParser(
            currentInputFiles[self.typeDict['betaxdecay']].getAbsFile(),
            currentInputFiles[self.typeDict['betaxdecay']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'INTTRADECAY':
        PathParser.PathParser(
            currentInputFiles[self.typeDict['inttradecay']].getAbsFile(),
            currentInputFiles[self.typeDict['inttradecay']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'XS':
        XSCreator.XSCreator(
            currentInputFiles[self.typeDict['xs']].getAbsFile(), booleanTabMap,
            currentInputFiles[self.typeDict['xs']].getPath(), tabMapFileName,
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'MASS':
        MassParser.MassParser(
            currentInputFiles[self.typeDict['mass']].getAbsFile(),
            currentInputFiles[self.typeDict['mass']].getPath(),
            **self.distributedPerturbedVars[perturbedParam])
      if perturbedParam == 'generic':
        # check and modify modelpar.inp file
        modelParParser = GenericParser(currentInputFiles)
        modelParParser.modifyInternalDictionary(**Kwargs)
        modelParParser.writeNewInput(currentInputFiles,oriInputFiles)


      # add CSV output from depletion
      tree = ET.parse(currentInputFiles[self.typeDict['depletion_input']].getAbsFile())
      depletionRoot = tree.getroot()
      outc = depletionRoot.find(".//output_control")
      plotType = outc.find(".//plot_type")
      if plotType is None or plotType.text.strip() != '3':
        plotType = ET.Element("plot_type") if plotType is None else plotType
      plotType.text = '3'
      try:
        outc.remove(plotType)
      except ValueError:
        print('Phisics INTERFACE: added plot_type node and set to 3!')
      outc.append(plotType)
      tree.write(currentInputFiles[self.typeDict['depletion_input']].getAbsFile())
    return currentInputFiles
