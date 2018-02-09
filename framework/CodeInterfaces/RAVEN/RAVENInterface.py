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
Created on Sept 10, 2017

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
from  __builtin__ import any
import copy
import numpy as np
from utils import utils
from CodeInterfaceBaseClass import CodeInterfaceBase
import DataObjects
import csvUtilities

class RAVEN(CodeInterfaceBase):
  """
    this class is used as part of a code dictionary to specialize Model.Code for RAVEN
  """
  def __init__(self):
    CodeInterfaceBase.__init__(self)
    self.printTag  = 'RAVEN INTERFACE'
    self.outputPrefix = 'out~'
    self.outStreamsNamesAndType = {} # Outstreams names and type {'outStreamName':[DataObjectName,DataObjectType]}
    # path to the module that contains the function to modify and convert the sampled vars (optional)
    # 2 methods are going to be inquired (if present and needed):
    # - convertNotScalarSampledVariables
    # - manipulateScalarSampledVariables
    self.extModForVarsManipulationPath = None
    # 'noscalar' = True if convertNotScalarSampledVariables exists in extModForVarsManipulation module
    # 'scalar'   = True if manipulateScalarSampledVariables exists in extModForVarsManipulation module
    self.hasMethods                = {'noscalar':False, 'scalar':False}
    # inner workind directory
    self.innerWorkingDir = ''
    # linked DataObjects
    self.linkedDataObjectOutStreamsNames = None

  def addDefaultExtension(self):
    """
      This method sets a list of default extensions a specific code interface accepts for the input files.
      This method should be overwritten if these are not acceptable defaults.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['xml'])

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      In this case, this is used for locating an external python module where the variables can be modified and the "vector" variables
      can be splitted in multiple single variables
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    if os.path.basename(xmlNode.find("executable").text) != 'raven_framework':
      raise IOError(self.printTag+' ERROR: executable must be "raven_framework" (in whatever location)!')

    linkedDataObjects = xmlNode.find("outputExportOutStreams")
    if linkedDataObjects is None:
      raise IOError(self.printTag+' ERROR: outputExportOutStreams node not present. You must input at least one OutStream (max 2)!')
    self.linkedDataObjectOutStreamsNames = linkedDataObjects.text.split(",")
    if len(self.linkedDataObjectOutStreamsNames) > 2:
      raise IOError(self.printTag+' ERROR: outputExportOutStreams node. The maximum number of linked OutStreams are 2 (1 for PointSet and 1 for HistorySet)!')

    child = xmlNode.find("conversionModule")
    if child is not None:
      self.extModForVarsManipulationPath = os.path.expanduser(child.text.strip())
      if not os.path.isabs(self.extModForVarsManipulationPath):
        self.extModForVarsManipulationPath = os.path.abspath(self.extModForVarsManipulationPath)
      # check if it exist
      if not os.path.exists(self.extModForVarsManipulationPath):
        raise IOError(self.printTag+' ERROR: the conversionModule "'+self.extModForVarsManipulationPath+'" has not been found!')
      extModForVarsManipulation = utils.importFromPath(self.extModForVarsManipulationPath)
      if extModForVarsManipulation is None:
        raise IOError(self.printTag+' ERROR: the conversionModule "'+self.extModForVarsManipulationPath+'" failed to be imported!')
      # check if the methods are there
      if 'convertNotScalarSampledVariables' in extModForVarsManipulation.__dict__.keys():
        self.hasMethods['noscalar'] = True
      if 'manipulateScalarSampledVariables' in extModForVarsManipulation.__dict__.keys():
        self.hasMethods['scalar'  ] = True
      if not self.hasMethods['scalar'] and not self.hasMethods['noscalar']:
        raise IOError(self.printTag +' ERROR: the conversionModule "'+self.extModForVarsManipulationPath
                                    +'" does not contain any of the usable methods! Expected at least '
                                    +'one of: "manipulateScalarSampledVariables" and/or "manipulateScalarSampledVariables"!')

  def __findInputFile(self,inputFiles):
    """
      Method to return the index of the RAVEN input file (error out in case it is not found)
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ Out, inputFileIndex, int, index of the RAVEN input file
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getType().lower() == 'raven':
        inputFileIndex = index
        if found:
          raise IOError(self.printTag+" ERROR: Currently the RAVEN interface allows only one input file (xml). ExternalXML and Merging Files will be added in the future!")
        found = True
    if not found:
      raise IOError(self.printTag+' ERROR: None of the input files are tagged with the "type" "raven" (e.g. <Input name="aName" type="raven">inputFileName.xml</Input>)')
    return inputFileIndex

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    index = self.__findInputFile(inputFiles)
    outputfile = self.outputPrefix+inputFiles[index].getBase()
    # we set the command type to serial since the SLAVE RAVEN handles the parallel on its own
    executeCommand = [('serial',executable+ ' '+inputFiles[index].getFilename())]
    returnCommand = executeCommand, outputfile
    return returnCommand

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      this generates a new input file depending on which sampler has been chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    import RAVENparser
    if 'dynamiceventtree' in str(samplerType).strip().lower():
      raise IOError(self.printTag+' ERROR: DynamicEventTree-based sampling not supported!')
    index = self.__findInputFile(currentInputFiles)
    parser = RAVENparser.RAVENparser(currentInputFiles[index].getAbsFile())
    # get the OutStreams names
    self.outStreamsNamesAndType = parser.returnOutstreamsNamesAnType()
    # check if the linked DataObjects are among the Outstreams
    pointSetNumber, historySetNumber = 0, 0
    for outstream, dataObj in self.outStreamsNamesAndType.items():
      if outstream in self.linkedDataObjectOutStreamsNames:
        if dataObj[1].strip() == 'PointSet':
          pointSetNumber+=1
        else:
          historySetNumber+=1
        if pointSetNumber > 1 or historySetNumber > 1:
          raise IOError(self.printTag+' ERROR: Only one OutStream for PointSet and/or one for HistorySet can be linked as output export!')
    if pointSetNumber == 0 and historySetNumber == 0:
      raise IOError(self.printTag+' ERROR: No one of the OutStreams linked to this interface have been found in the SLAVE RAVEN!'
                                 +' Expected: "'+' '.join(self.linkedDataObjectOutStreamsNames)+'" but found "'
                                 +' '.join(self.outStreamsNamesAndType.keys())+'"!')
    # get variable groups
    varGroupNames = parser.returnVarGroups()
    if len(varGroupNames) > 0:
      # check if they are not present in the linked outstreams
      for outstream in self.linkedDataObjectOutStreamsNames:
        inputNode = self.outStreamsNamesAndType[outstream][2].find("Input")
        outputNode = self.outStreamsNamesAndType[outstream][2].find("Output")
        inputVariables = inputNode.text.split(",") if inputNode is not None else []
        outputVariables =  outputNode.text.split(",") if outputNode is not None else []
        if any (varGroupName in inputVariables+outputVariables for varGroupName in varGroupNames):
          raise IOError(self.printTag+' ERROR: The VariableGroup system is not supported in the current ' +
                                      'implementation of the interface for the DataObjects specified in the '+
                                      '<outputExportOutStreams> XML node!')
    # get inner working dir
    self.innerWorkingDir = parser.workingDir
    # get sampled variables
    modifDict = Kwargs['SampledVars']
    # check if there are noscalar variables
    vectorVars = {}
    totSizeExpected = 0
    for var, value in modifDict.items():
      if np.asarray(value).size > 1:
        vectorVars[var] = np.asarray(value)
        totSizeExpected += vectorVars[var].size
    if len(vectorVars) > 0 and not self.hasMethods['noscalar']:
      raise IOError(self.printTag+' ERROR: No scalar variables ('+','.join(vectorVars.keys())
                                  + ') have been detected but no convertNotScalarSampledVariables has been inputted!')
    # check if ext module has been inputted
    if self.hasMethods['noscalar'] or self.hasMethods['scalar']:
      extModForVarsManipulation = utils.importFromPath(self.extModForVarsManipulationPath)
    if self.hasMethods['noscalar']:
      if len(vectorVars) > 0:
        toPopOut = vectorVars.keys()
        try:
          newVars = extModForVarsManipulation.convertNotScalarSampledVariables(vectorVars)
          if type(newVars).__name__ != 'dict':
            raise IOError(self.printTag+' ERROR: convertNotScalarSampledVariables must return a dictionary!')
          # DEBUGG this is failing b/c Index and Variable both being counted!
          #if len(newVars) != totSizeExpected:
          #  raise IOError(self.printTag+' ERROR: The total number of variables expected from method convertNotScalarSampledVariables is "'+str(totSizeExpected)+'". Got:"'+str(len(newVars))+'"!')
          modifDict.update(newVars)
          for noscalarVar in toPopOut:
            modifDict.pop(noscalarVar)
        except TypeError:
          raise IOError(self.printTag+' ERROR: convertNotScalarSampledVariables accept only one argument convertNotScalarSampledVariables(variableDict)')
      else:
        print(self.printTag+' Warning: method "convertNotScalarSampledVariables" has been inputted but no "no scalar" variables have been found!')
    # check if ext module has the method to manipulate the variables
    if self.hasMethods['scalar']:
      try:
        extModForVarsManipulation.manipulateScalarSampledVariables(modifDict)
      except TypeError:
        raise IOError(self.printTag+' ERROR: manipulateScalarSampledVariables accept only one argument manipulateScalarSampledVariables(variableDict)')

    # we work on batchSizes here
    newBatchSize = Kwargs['NumMPI']
    internalParallel = Kwargs.get('internalParallel',False)
    if int(Kwargs['numberNodes']) > 0:
      # we are in a distributed memory machine => we allocate a node file
      nodeFileToUse = os.path.join(Kwargs['BASE_WORKING_DIR'],"node_" +str(Kwargs['INDEX']))
      if os.path.exists(nodeFileToUse):
        modifDict['RunInfo|mode'           ] = 'mpi'
        modifDict['RunInfo|mode|nodefile'  ] = nodeFileToUse
      else:
        raise IOError(self.printTag+' ERROR: The nodefile "'+str(nodeFileToUse)+'" does not exist!')
    if internalParallel or newBatchSize > 1:
      # either we have an internal parallel or NumMPI > 1
      modifDict['RunInfo|batchSize'       ] = newBatchSize
    #modifDict['RunInfo|internalParallel'] = internalParallel
    #make tree
    modifiedRoot = parser.modifyOrAdd(modifDict,save=True,allowAdd = True)
    #make input
    parser.printInput(modifiedRoot,currentInputFiles[index].getAbsFile())
    # copy slave files
    parser.copySlaveFiles(currentInputFiles[index].getPath())
    return currentInputFiles

  def checkForOutputFailure(self,output,workingDir):
    """
      This method is called by the RAVEN code at the end of each run  if the return code is == 0.
      This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
      This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
      This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
      (e.g. in RAVEN would be the keyword "raise")
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, failure, bool, True if the job is failed, False otherwise
    """
    failure = False
    try:
      outputToRead = open(os.path.join(workingDir,output),"r")
    except IOError:
      failure = True
      print(self.printTag+' ERROR: The RAVEN SLAVE log file  "'+str(os.path.join(workingDir,output))+'" does not exist!')
    if not failure:
      readLines = outputToRead.readlines()
      if not any("Run complete" in x for x in readLines[-20:]):
        failure = True
      del readLines
    if not failure:
      for filename in self.linkedDataObjectOutStreamsNames:
        outStreamFile = os.path.join(workingDir,self.innerWorkingDir,filename+".csv")
        try:
          fileObj = open(outStreamFile,"r")
        except IOError:
          print(self.printTag+' ERROR: The RAVEN SLAVE output file "'+str(outStreamFile)+'" does not exist!')
          failure = True
        if not failure:
          readLines = fileObj.readlines()
          if any("nan" in x.lower() for x in readLines):
            failure = True
            print(self.printTag+' ERROR: Found nan in RAVEN SLAVE output "'+str(outStreamFile)+'!')
            break
          del readLines
    return failure

  def finalizeCodeOutput(self,command,output,workingDir):
    """
      this method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
      It can be used for those codes, that do not create CSV files to convert the whatever output formats into a csv
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, dataObjectsToReturn, dict, optional, this is a special case for RAVEN only. It returns the constructed dataobjects
                                                 (internally we check if the return variable is a dict and if it is returned by RAVEN (if not, we error out))
    """

    ##### TODO This is an exception to the way CodeInterfaces usually run.
    # The return dict for this CodeInterface is a dictionary of data objects (either one or two of them, up to one each point set and history set).
    # Normally, the end result of this method is producing a CSV file with the data to load.
    # When data objects are returned, this triggers an "if" path in Models/Code.py in evaluateSample().  There's an "else"
    # that can be found by searching for the comment "we have the DataObjects -> raven-runs-raven case only so far".
    # See more details there.
    #####
    dataObjectsToReturn = {}
    numRlz = None
    for filename in self.linkedDataObjectOutStreamsNames:
      # load the output CSV into a data object, so we can return that
      ## load the XML initialization information and type
      dataObjectInfo = self.outStreamsNamesAndType[filename]
      # create an instance of the correct data object type
      data = DataObjects.returnInstance(dataObjectInfo[1],None)
      # dummy message handler to handle message parsing, TODO this stinks and should be fixed.
      data.messageHandler = DataObjects.XDataObject.MessageCourier()
      # initialize the data object by reading the XML
      data._readMoreXML(dataObjectInfo[2])
      # set the name, then load the data
      data.name = filename
      data.load(os.path.join(workingDir,self.innerWorkingDir,filename),style='csv')
      # check consistency of data object number of realizations
      if numRlz is None:
        # set the standard if you're the first data object
        numRlz = len(data)
      else:
        # otherwise, check that the number of realizations is appropriate
        if len(data) != numRlz:
          raise IOError('The number of realizations in output CSVs from the inner RAVEN run are not consistent!  In "{}" received "{}" realization(s), but other data objects had "{}" realization(s)!'.format(data.name,len(data),numRlz))
      # store the object to return
      dataObjectsToReturn[dataObjectInfo[0]] = data
    return dataObjectsToReturn


