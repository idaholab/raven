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
import os
import numpy as np
from sys import platform
from ravenframework.utils import utils
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ravenframework import DataObjects
from ravenframework import Databases

class RAVEN(CodeInterfaceBase):
  """
    this class is used as part of a code dictionary to specialize Model.Code for RAVEN
  """
  def __init__(self):
    CodeInterfaceBase.__init__(self)
    self.preCommand = "python " # this is the precommand
    self.printTag  = 'RAVEN INTERFACE'
    self.outputPrefix = 'out~'
    self.outStreamsNamesAndType = {} # Outstreams names and type {'outStreamName':[DataObjectName,DataObjectType]}
    self.outDatabases = {} # as outStreams, but {name: path/and/file} for databases
    # path to the module that contains the function to modify and convert the sampled vars (optional)
    # 2 methods are going to be inquired (if present and needed):
    # - convertNotScalarSampledVariables
    # - manipulateScalarSampledVariables
    self.extModForVarsManipulationPath = None
    # 'noscalar' = True if convertNotScalarSampledVariables exists in extModForVarsManipulation module
    # 'scalar'   = True if manipulateScalarSampledVariables exists in extModForVarsManipulation module
    self.hasMethods                = {'noscalar':False, 'scalar':False}
    # inner working directory
    self.innerWorkingDir = ''
    # linked DataObjects and Databases
    self.linkedDataObjectOutStreamsNames = None
    self.linkedDatabaseName = None
    # input manipulation module
    self.inputManipulationModule = None
    self.printFailedRuns = False  # whether to print failed runs to the screen

  def addDefaultExtension(self):
    """
      This method sets a list of default extensions a specific code interface accepts for the input files.
      This method should be overwritten if these are not acceptable defaults.
      @ In, None
      @ Out, None
    """
    self.addInputExtension(['xml'])

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class and initialize
      some members based on inputs. This can be overloaded in specialize code interface in order to
      read specific flags.
      In this case, this is used for locating an external python module where the variables can be modified and the "vector" variables
      can be splitted in multiple single variables
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None.
    """
    baseName = os.path.basename(xmlNode.find("executable").text)
    if baseName not in ['raven_framework','raven_framework.py']:
      raise IOError(self.printTag+' ERROR: executable must be "raven_framework" (in whatever location)! Got "'+baseName+'"!')

    linkedDataObjects = xmlNode.find("outputExportOutStreams")
    linkedDatabases = xmlNode.find('outputDatabase')
    if linkedDataObjects is None and linkedDatabases is None:
      raise IOError(self.printTag+' ERROR: Neither <outputExportOutStreams> nor <outputDatabase> node is present. '+
                    'You must indicate an output from the inner run!')
    if linkedDataObjects is not None and linkedDatabases is not None:
      raise IOError(self.printTag+' ERROR: Only one of <outputExportOutStreams> or <outputDatabase> can be present!')
    if linkedDataObjects is not None:
      self.linkedDataObjectOutStreamsNames = linkedDataObjects.text.split(",")
    elif linkedDatabases is not None:
      self.linkedDatabaseName = linkedDatabases.text.strip()

    if self.linkedDataObjectOutStreamsNames is not None and len(self.linkedDataObjectOutStreamsNames) > 2:
      raise IOError(self.printTag+' ERROR: outputExportOutStreams node. The maximum number of linked OutStreams are 2 (1 for PointSet and 1 for HistorySet)!')
    if self.linkedDatabaseName is not None and len(self.linkedDatabaseName.split(',')) > 1:
      raise IOError(self.printTag+' ERROR: outputExportOutStreams node. The maximum number of linked OutStreams are 2 (1 for PointSet and 1 for HistorySet)!')

    # load conversion modules
    self.conversionDict = {} # {modulePath : {'variables': [], 'noScalar': 0, 'scalar': 0}, etc }
    child = xmlNode.find("conversion")
    if child is not None:
      for moduleNode in child:
        # get the module to be used for conversion
        source = moduleNode.attrib.get('source', None)
        if source is None:
          raise IOError(self.printTag+' ERROR: no module "source" listed in "conversion" subnode attributes!')
        # fix up the path
        ## should be relative to the working dir!
        source = os.path.expanduser(source)
        if not os.path.isabs(source):
          source = os.path.abspath(os.path.join(self._ravenWorkingDir, source))
        # check for existence
        if not os.path.exists(source):
          raise IOError(self.printTag+' ERROR: the conversionModule "{}" was not found!'
                        .format(source))
        # check module is imported
        checkImport = utils.importFromPath(source)
        if checkImport is None:
          raise IOError(self.printTag+' ERROR: the conversionModule "{}" failed on import!'
                        .format(source))
        # variable conversion modules
        if moduleNode.tag == 'module':
          # check methods are in place
          noScalar = 'convertNotScalarSampledVariables' in checkImport.__dict__
          scalar = 'manipulateScalarSampledVariables' in checkImport.__dict__
          if not (noScalar or scalar):
            raise IOError(self.printTag +' ERROR: the conversionModule "'+source
                          +'" does not contain any of the usable methods! Expected at least '
                          +'one of: "manipulateScalarSampledVariables" and/or "manipulateScalarSampledVariables"!')
          # acquire the variables to be modified
          varNode = moduleNode.find('variables')
          if varNode is None:
            raise IOError(self.printTag+' ERROR: no node "variables" listed in "conversion|module" subnode!')
          variables = [x.strip() for x in varNode.text.split(',')]
          self.conversionDict[source] = {'variables':variables, 'noScalar':noScalar, 'scalar':scalar}

        # custom input file manipulation
        elif moduleNode.tag == 'input':
          self.inputManipulationModule = source

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

  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None, preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs that have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """

    index = self.__findInputFile(inputFiles)
    outputfile = self.outputPrefix+inputFiles[index].getBase()
    # we set the command type to serial since the SLAVE RAVEN handles the parallel on its own
    # executable command will either be the direct raven_framework, or
    # executable command will be: "python <path>/raven_framework.py"
    # in which case make sure executable ends with .py
    # Note that for raven_framework to work, it needs to be in the path.
    if executable == 'raven_framework':
      self.preCommand = ''
    elif not executable.endswith(".py"):
      executable += ".py"
    executeCommand = [('serial', self.preCommand + executable+ ' ' + inputFiles[index].getFilename())]
    returnCommand = executeCommand, outputfile

    return returnCommand

  def initialize(self, runInfo, oriInputFiles):
    """
      Method to initialize the run of a new step
      @ In, runInfo, dict,  dictionary of the info in the <RunInfo> XML block
      @ In, oriInputFiles, list, list of the original input files
      @ Out, None
    """
    from . import RAVENparser
    index = self.__findInputFile(oriInputFiles)
    parser = RAVENparser.RAVENparser(oriInputFiles[index].getAbsFile())
    # get the OutStreams names
    self.outStreamsNamesAndType, self.outDatabases = parser.returnOutputs()
    # check if the linked DataObjects are among the Outstreams
    if self.linkedDataObjectOutStreamsNames:
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
    else: # self.linkedDatabaseName
      for dbName, dbXml in self.outDatabases.items():
        if dbName == self.linkedDatabaseName:
          break
      else:
        # the one we want wasn't found!
        raise IOError(f'{self.printTag} ERROR: The Database named "{self.linkedDatabaseName}" listed '+
                      'in <outputDatabase> was not found among the written Databases in active Steps in the inner RAVEN! '+
                      f'Found: {list(self.outDatabases.keys())}')

    # get variable groups
    varGroupNames = parser.returnVarGroups()
    ## store globally
    self.variableGroups = varGroupNames
    # get inner working dir
    self.innerWorkingDir = parser.workingDir

  def createNewInput(self, currentInputFiles, oriInputFiles, samplerType, **Kwargs):
    """
      this generates a new input file depending on which sampler has been chosen
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RAVEN stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    from . import RAVENparser
    if 'dynamiceventtree' in str(samplerType).strip().lower():
      raise IOError(self.printTag+' ERROR: DynamicEventTree-based sampling not supported!')
    index = self.__findInputFile(currentInputFiles)
    parser = RAVENparser.RAVENparser(currentInputFiles[index].getAbsFile())
    # get sampled variables
    modifDict = Kwargs['SampledVars']

    # apply conversion scripts
    for source, convDict in self.conversionDict.items():
      module = utils.importFromPath(source)
      varVals = dict((var,np.asarray(modifDict[var])) for var in convDict['variables'])
      # modify vector+ variables that need to be flattened
      if convDict['noScalar']:
        # call conversion
        newVars = module.convertNotScalarSampledVariables(varVals)
        # check type
        if type(newVars).__name__ != 'dict':
          raise IOError(self.printTag+' ERROR: convertNotScalarSampledVariables in "{}" must return a dictionary!'.format(source))
        # apply new and/or updated values
        modifDict.update(newVars)
      # modify scalar variables
      if convDict['scalar']:
        # call conversion, value changes happen in-place
        module.manipulateScalarSampledVariables(modifDict)
    # we work on batchSizes here
    newBatchSize = Kwargs['NumMPI']
    internalParallel = Kwargs.get('internalParallel',False)
    if int(Kwargs['numberNodes']) > 0:
      # we are in a distributed memory machine => we allocate a node file
      nodeFileToUse = os.path.join(Kwargs['BASE_WORKING_DIR'],"node_" +str(Kwargs['INDEX']))
      if not os.path.exists(nodeFileToUse):
        if "PBS_NODEFILE" not in os.environ:
          raise IOError(self.printTag+' ERROR: The nodefile "'+str(nodeFileToUse)+'" and PBS_NODEFILE enviroment var do not exist!')
        else:
          nodeFileToUse = os.environ["PBS_NODEFILE"]
      if len(parser.tree.findall('./RunInfo/mode')) == 1:
        #If there is a mode node, give it a nodefile.
        modifDict['RunInfo|mode|nodefile'  ] = nodeFileToUse
    if internalParallel or newBatchSize > 1:
      # either we have an internal parallel or NumMPI > 1
      modifDict['RunInfo|batchSize'] = newBatchSize

    if 'headNode' in Kwargs:
      modifDict['RunInfo|headNode'] = Kwargs['headNode']
    if 'remoteNodes' in Kwargs:
      if Kwargs['remoteNodes'] is not None and len(Kwargs['remoteNodes']):
        modifDict['RunInfo|remoteNodes'] = ','.join(Kwargs['remoteNodes'])
    #modifDict['RunInfo|internalParallel'] = internalParallel
    # make tree
    modifiedRoot = parser.modifyOrAdd(modifDict, save=True, allowAdd=True)
    # modify tree
    if self.inputManipulationModule is not None:
      module = utils.importFromPath(self.inputManipulationModule)
      modifiedRoot = module.modifyInput(modifiedRoot,modifDict)
    # write input file
    parser.printInput(modifiedRoot, currentInputFiles[index].getAbsFile())
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
    # check for log file
    ## NOTE this can be falsely accepted if the run dir isn't cleared before running,
    ##      which it automatically is but can be disabled
    toCheck = os.path.join(workingDir, self.innerWorkingDir, '.ravenStatus')
    if not os.path.isfile(toCheck):
      print(f'RAVENInterface WARNING: Could not find {toCheck}, assuming failed RAVEN run.')
      return True
    # check for output CSV (and data)
    if not failure:
      if self.linkedDataObjectOutStreamsNames:
        for filename in self.linkedDataObjectOutStreamsNames:
          outStreamFile = os.path.join(workingDir,self.innerWorkingDir,filename+".csv")
          try:
            fileObj = open(outStreamFile,"r")
          except IOError:
            print(self.printTag+' ERROR: The RAVEN INNER output file "'+str(outStreamFile)+'" does not exist!')
            failure = True
          if not failure:
            readLines = fileObj.readlines()
            if any("nan" in x.lower() for x in readLines):
              failure = True
              print(self.printTag+' ERROR: Found nan in RAVEN INNER output "'+str(outStreamFile)+'!')
              break
            del readLines
      else:
        dbName = self.linkedDatabaseName
        path = self.outDatabases[dbName]
        fullPath = os.path.join(workingDir, self.innerWorkingDir, path)
        if not os.path.isfile(fullPath):
          print(f'{self.printTag} ERROR: The RAVEN INNER output file "{os.path.abspath(fullPath)}" was not found!')
          failure = True
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
    if self.linkedDataObjectOutStreamsNames:
      for filename in self.linkedDataObjectOutStreamsNames:
        # load the output CSV into a data object, so we can return that
        ## load the XML initialization information and type
        dataObjectInfo = self.outStreamsNamesAndType[filename]
        # create an instance of the correct data object type
        data = DataObjects.factory.returnInstance(dataObjectInfo[1])
        # initialize the data object by reading the XML
        data.readXML(dataObjectInfo[2], variableGroups=self.variableGroups)
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
    else: # self.linkedDatabaseName
      dbName = self.linkedDatabaseName
      path = self.outDatabases[dbName]
      fullPath = os.path.join(workingDir, self.innerWorkingDir, path)
      data = DataObjects.factory.returnInstance('DataSet')
      info = {'WorkingDir': self._ravenWorkingDir}
      db = Databases.factory.returnInstance('NetCDF')
      db.applyRunInfo(info)
      db.databaseDir, db.filename = os.path.split(fullPath)
      db.loadIntoData(data)
      dataObjectsToReturn[dbName] = data
    return dataObjectsToReturn
