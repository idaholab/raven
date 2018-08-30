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
Created on Jan 08th, 2018 
@author: rouxpn 
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import re
from CodeInterfaceBaseClass import CodeInterfaceBase
from PhisicsInterface import Phisics
import phisicsdata
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom

class Newt(CodeInterfaceBase):
  """
    this class is used a part of a code dictionary to specialize Model. 
  """ 
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    CodeInterfaceBase.__init__(self)
    self.PhisicsInterface = Phisics()
    self.PhisicsInterface.addDefaultExtension()
  
  def definePhisicsVariables(self):
    """
      lists the variables perturbable within PHISICS. The other variables will be related to relap. 
      @ In, None, 
      @ Out, phisicsVariables, list 
    """
    self.phisicsVariables = ['DENSITY','XS','DECAY','FY','QVALUES','ALPHADECAY','BETA+DECAY','BETA+XDECAY','BETADECAY','BETAXDECAY','INTTRADECAY']
  
  def getPhisicsRefValues(self,rootDir):
    """
      Parses the 252-group phisics reference output. The NEWT 252-group and PHISICS is ran offline. SO the PHISICS
      output has to be completed before the the RAVEN was run. The reference output (numbers and instant output, and instant flux csv file)
      have to be placed in the root directory, where the RAVEN input is located. The name HTGR-test.o-0 and numbers-0.csv, Dpl_INSTANT_HTGR_test_flux_mat.csv
      are hardcoded so far. 
      @ In, rootDir, string, root directory
      @ Out, None 
    """
    phisicsDataDict = {}
    phisicsDataDict['timeControl'] = self.PhisicsInterface.timeControl
    phisicsDataDict['decayHeatFlag'] = self.PhisicsInterface.decayHeatFlag
    phisicsDataDict['instantOutput'] = 'HTGR-test.o'
    phisicsDataDict['workingDir'] = rootDir
    phisicsDataDict['mrtauStandAlone'] = False
    phisicsDataDict['jobTitle'] = 'HTGR_test'
    phisicsDataDict['mrtauFileNameDict'] = self.PhisicsInterface.outputFileNameDict
    phisicsDataDict['numberOfMPI'] = 1
    phisicsDataDict['phiRel'] = False
    phisicsDataDict['printSpatialRR'] = True 
    phisicsDataDict['printSpatialFlux'] = False 
    phisicsDataDict['pertVariablesDict'] = self.PhisicsInterface.distributedPerturbedVars
    phisicsDataDict['parseFlux'] = None # The flux in fine group structure is collapsed only AFTER the broad group simulations are done. At this stage, only the reference fine group is parsed, but the broad calc. have not been performed yet, so impossible to use the parseFlux method in phisicsdata 
    phisicsdata.phisicsdata(phisicsDataDict)
    
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
  
  def getNumberOfMPI(self,string):
    """
      gets the number of MPI requested by the user in the RAVEN input
      @ In, string, string, string from the Kwargs containing the number of MPI
      @ Out, MPInumber, integer, number of MPI used in the calculation
    """
    return int(string.split(" ")[-2])
  
  def addDefaultExtension(self):
    """
      Possible input extensions found in the input files. 
      @ In, None
      @ Out, None 
    """
    self.addInputExtension(['xml','dat','path','inp','pbs'])
  
  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      Only one option is possible. You can choose here, if multi-deck mode is activated, from which deck you want to load the results
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    self.xmlNode = xmlNode
    self.readPhisicsRef = False 
    self.numberOfLattice = 1 # number of lattice cell defined, for phisics use. default 1
    for child in xmlNode:
      if child.tag == 'AMPXconverterExecutable':
        self.ampxConverterExecutable = child.text
      if child.tag == 'phisicsExecutable':
        self.phisicsExecutable = child.text
      if child.tag == 'disadFactorScript':
        self.disadFactorScript = child.text
      if child.tag == 'readPhisicsRef':
        if (child.text.lower() == 't' or child.text.lower() == 'true'):  
          self.readPhisicsRef = True
        elif (child.text.lower() == 'f' or child.text.lower() == 'false'): 
          self.readPhisicsRef = False
        else: 
          raise ValueError("\n the node "+child.tag+"has to be a boolean entry")
      if child.tag == 'numberOfLattice':
        self.numberOfLattice = child.text
  
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
    
    newtFiles = []
    dict = self.PhisicsInterface.mapInputFileType(inputFiles)
    outputfile = 'out~'+inputFiles[dict['inp']].getBase()
    for i in range(int(self.numberOfLattice)):
      commandToRunNewt = executable + ' -N 1 -m -r -T TEMPnewt'+str(i)+'.inp newt'+str(i)+'.inp  &>log_run_$PBS_JOBNAME.out'
      commandToRunNewt = commandToRunNewt.replace("\n"," ")
      commandToRunNewt = re.sub("\s\s+" , " ", commandToRunNewt )
      if i == 0:
        allLatticeNewt = [('serial',commandToRunNewt)]
      else :
        allLatticeNewt = allLatticeNewt + [('serial',commandToRunNewt)]
    
    for i in range(int(self.numberOfLattice)): 
      commandToRunAMPXconverter = self.ampxConverterExecutable + ' -i TEMPnewt'+str(i)+'.inp/ft30f001 -o ft30f001_'+str(i)+' -f legacy'
      commandToRunAMPXconverter = commandToRunAMPXconverter.replace("\n"," ")
      commandToRunAMPXconverter = re.sub("\s\s+" , " ", commandToRunAMPXconverter )
      if i == 0:
        allConverters = [('serial',commandToRunAMPXconverter)]
      else :
        allConverters = allConverters + [('serial',commandToRunAMPXconverter)]
    
    commandToRunDisadFactorScript = 'python ' + self.disadFactorScript + 'scaleParser.py'
    commandToRunDisadFactorScript = commandToRunDisadFactorScript.replace("\n"," ")
    commandToRunDisadFactorScript = re.sub("\s\s+" , " ", commandToRunDisadFactorScript )

    self.instantOutput = self.PhisicsInterface.instantOutput
    commandToRunPhisics = self.phisicsExecutable + ' ' +inputFiles[dict['inp'.lower()]].getFilename() + ' ' + inputFiles[dict['Xs-library'.lower()]].getFilename() + ' ' + inputFiles[dict['Material'.lower()]].getFilename() + ' ' + inputFiles[dict['Depletion_input'.lower()]].getFilename() + ' ' + self.instantOutput
    commandToRunPhisics = commandToRunPhisics.replace("\n"," ")
    commandToRunPhisics  = re.sub("\s\s+" , " ", commandToRunPhisics )

    returnCommand = allLatticeNewt + allConverters + [('serial', commandToRunDisadFactorScript)] +[('parallel',commandToRunPhisics)], outputfile
    #returnCommand = [('parallel',commandToRunPhisics)], outputfile
    
    #returnCommand = [('serial',commandToRunPhisics)], outputfile
    #returnCommand = [('serial',commandToRunAMPXconverter)], outputfile
    #returnCommand = [('serial',commandToRunNewt)], outputfile 
    #returnCommand = [('serial','python '+executable+ ' -i ' +inputFiles[index].getFilename()+' -o '+outputfile+'.e')], outputfile
    #returnCommand = cubitCommand + mooseCommand, mooseOut #can only send one...#(cubitOut,mooseOut)
    #print (commandToRun)
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
    phisicsRavenCsv = self.PhisicsInterface.finalizeCodeOutput(command,output,workingDir,parseFlux=True)
    return phisicsRavenCsv
    
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
    self.PhisicsInterface.checkForOutputFailure(output,workingDir)
  
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
    import inputParser

    latticeNames = []
    self.definePhisicsVariables()
    perturbedVars = Kwargs['SampledVars'] 
    keyWordDict = self.PhisicsInterface.mapInputFileType(currentInputFiles)
    # parse Phisics
    self.PhisicsInterface._readMoreXML(self.xmlNode)
    self.PhisicsInterface.createNewInput(currentInputFiles,oriInputFiles,samplerType,**Kwargs)
    self.distributedPerturbedVars = self.PhisicsInterface.distributedPerturbedVars
    # parse phisics ref 
    if self.readPhisicsRef is True and Kwargs['prefix'] == '1':
      self.getPhisicsRefValues(os.getcwd())
    
    newtInput = currentInputFiles[keyWordDict['newt']]
    # defines the names of each lattice --- newt1.inp newt2.inp etc. ---
    for i in range (int(self.numberOfLattice)):
      latticeNames.append(newtInput.getBase()+str(i)+'.'+newtInput.getAbsFile().split(".")[-1])
    for paramVariable in self.distributedPerturbedVars.iterkeys():
      if paramVariable == 'group' : 
        inputParser.inputParser(newtInput.getPath(), latticeNames, self.numberOfLattice, **self.distributedPerturbedVars[paramVariable])
    return currentInputFiles
    

