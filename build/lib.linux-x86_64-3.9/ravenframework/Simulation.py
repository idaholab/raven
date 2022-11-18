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
Module that contains the driver for the whole simulation flow (Simulation Class)
"""

import os
import subprocess
import sys
import io
import string
import datetime
import threading
import time
import numpy as np

from .BaseClasses import MessageUser
from . import Steps
from . import DataObjects
from . import Files
from . import Samplers
from . import Optimizers
from . import Models
from . import Metrics
from . import Distributions
from . import Databases
from . import Functions
from . import OutStreams
from .JobHandler import JobHandler
from .utils import utils, TreeStructure, xmlUtils, mathUtils
from . import Decorators
from .Application import __QtAvailable
from .Interaction import Interaction
if __QtAvailable:
  from .Application import InteractiveApplication

# Load up plugins!
# -> only available on specially-marked base types
Models.Model.loadFromPlugins()

class SimulationMode(MessageUser):
  """
    SimulationMode allows changes to the how the simulation
    runs are done.  modifySimulation lets the mode change runInfoDict
    and other parameters.  remoteRunCommand lets a command to run RAVEN
    remotely be specified.
  """
  def __init__(self, *args):
    """
      Constructor
      @ In, args, list, unused positional arguments
      @ Out, None
    """
    super().__init__()
    self.printTag = 'SIMULATION MODE'

  def remoteRunCommand(self, runInfoDict):
    """
      If this returns None, do nothing. If it returns a dictionary,
      use the dictionary to run raven remotely.
      @ In, runInfoDict, dict, the run info
      @ Out, remoteRunCommand, dict, the information for the remote command.
      The dictionary should have a "args" key that is used as a command to
      a subprocess.call.  It optionally can have a "cwd" for the current
      working directory and a "env" for the environment to use for the command.
    """

    return None

  def modifyInfo(self, runInfoDict):
    """
      modifySimulation is called after the runInfoDict has been setup.
      This allows the mode to change any parameters that need changing.
      This typically modifies the precommand and the postcommand that
      are put in front of the command and after the command.
      @ In, runInfoDict, dict, the run info
      @ Out, dictionary to use for modifications.  If empty, no changes
    """
    import multiprocessing
    newRunInfo = {}
    try:
      if multiprocessing.cpu_count() < runInfoDict['batchSize']:
        self.raiseAWarning(f"cpu_count {multiprocessing.cpu_count()} < batchSize {runInfoDict['batchSize']}")
    except NotImplementedError:
      pass
    if runInfoDict['NumThreads'] > 1:
      newRunInfo['threadParameter'] = runInfoDict['threadParameter']
      # add number of threads to the post command.
      newRunInfo['postcommand'] = f" {newRunInfo['threadParameter']} {runInfoDict['postcommand']}"

    return newRunInfo

  def XMLread(self,xmlNode):
    """
      XMLread is called with the mode node, and can be used to
      get extra parameters needed for the simulation mode.
      @ In, xmlNode, xml.etree.ElementTree.Element, the xml node that belongs to this class instance
      @ Out, None
    """
    pass

# Note that this has to be after SimulationMode is defined or the CustomModes
# don't see SimulationMode when they import Simulation. Still bad practice though
from . import CustomModes

def splitCommand(s):
  """
    Splits the string s into a list that can be used for the command
    So for example splitCommand("ab bc c 'el f' \"bar foo\" ") ->
    ['ab', 'bc', 'c', 'el f', 'bar foo']
    Bugs: Does not handle quoted strings with different kinds of quotes
    @ In, s, string, the command to split
    @ Out, retList, list, the list of splitted command
  """
  n = 0
  retList = []
  inQuote = False
  buffer = ""
  while n < len(s):
    current = s[n]
    if current in string.whitespace and not inQuote:
      if len(buffer) > 0:
        # found end of command
        retList.append(buffer)
        buffer = ""
    elif current in "\"'":
      if inQuote:
        inQuote = False
      else:
        inQuote = True
    else:
      buffer = buffer + current
    n += 1
  if len(buffer) > 0:
    retList.append(buffer)

  return retList

class Simulation(MessageUser):
  """
    This is a class that contain all the objects needed to run the simulation
    Usage:
    myInstance = Simulation()                          Generate the instance
    myInstance.XMLread(xml.etree.ElementTree.Element)  This method generates all the objects living in the simulation
    myInstance.initialize()                            This method takes care of setting up the directory/file environment with proper checks
    myInstance.run()                                   This method runs the simulation
    Utility methods:
     myInstance.printDicts                             prints the dictionaries representing the whole simulation
     myInstance.setInputFiles                          re-associate the set of files owned by the simulation
     myInstance.getDefaultInputFile                    return the default name of the input file read by the simulation
    Inherited from the BaseType class:
     myInstance.whoAreYou()                            inherited from BaseType class-
     myInstance.myClassmyCurrentSetting()              see BaseType class-

    --how to add a new entity <myClass> to the simulation--
    Add an import for the module where it is defined. Convention is that the module is named with the plural
     of the base class of the module: <MyModule>=<myClass>+'s'.
     The base class of the module is by convention named as the new type of simulation component <myClass>.
     The module should contain a set of classes named <myType> that are child of the base class <myClass>.
     The module should possess a function <MyModule>.factory.returnInstance('<myType>') that returns a pointer to the class <myType>.
    Add in Simulation.__init__ the following
     self.<myClass>Dict = {}
     self.entityModules['<myClass>'] = <MyModule>
     self.entities['<myClass>'  ] = self.<myClass>+'Dict'
    The XML describing the new entity should be organized as it follows:
     <MyModule (camelback with first letter capital)>
       <MyType (camelback with first letter capital) name='here a user given name' subType='here additional specialization'>
         <if needed more xml nodes>
       </MyType>
     </MyModule>

    --Comments on the simulation environment--
    every type of element living in the simulation should be uniquely identified by type and name not by sub-type
    !!!!Wrong!!!!!!!!!!!!!!!!:
    Class: distribution, type: normal,     name: myDistribution
    Class: distribution, type: triangular, name: myDistribution
    Correct:
    type: distribution, type: normal,      name: myNormalDist
    type: distribution, type: triangular,  name: myTriDist

    Using the attribute in the xml node <MyType> type discouraged to avoid confusion
  """

  def __init__(self, frameworkDir, verbosity='all', interactive=Interaction.No):
    """
      Constructor
      @ In, frameworkDir, string, absolute path to framework directory
      @ In, verbosity, string, optional, general verbosity level
      @ In, interactive, Interaction, optional, toggles the ability to provide
        an interactive UI or to run to completion without human interaction
      @ Out, None
    """
    super().__init__()
    self.FIXME = False
    # set the numpy print threshold to avoid ellipses in array truncation
    np.set_printoptions(threshold=np.inf)
    self.verbosity = verbosity
    callerLength = 25
    tagLength = 15
    suppressErrs = False
    self.messageHandler.initialize({'verbosity': self.verbosity,
                                    'callerLength': callerLength,
                                    'tagLength': tagLength,
                                    'suppressErrs': suppressErrs})
    # ensure messageHandler time has been reset (important if re-running simulation)
    self.messageHandler.starttime = time.time()
    sys.path.append(os.getcwd())
    # flag for checking if simulation has been run before
    self.ranPreviously = False
    # this dictionary contains the general info to run the simulation
    self.runInfoDict = {}
    self.runInfoDict['DefaultInputFile'  ] = 'test.xml' # Default input file to use
    self.runInfoDict['SimulationFiles'   ] = []     # the xml input file
    self.runInfoDict['ScriptDir'         ] = os.path.join(os.path.dirname(frameworkDir), "scripts") # the location of the pbs script interfaces
    self.runInfoDict['FrameworkDir'      ] = frameworkDir  # the directory where the framework is located
    self.runInfoDict['RemoteRunCommand'  ] = os.path.join(frameworkDir,'raven_ec_qsub_command.sh')
    self.runInfoDict['NodeParameter'     ] = '--hostfile' # the parameter used to specify the files where the nodes are listed
    self.runInfoDict['MPIExec'           ] = 'mpiexec'  # the command used to run mpi commands
    self.runInfoDict['threadParameter'   ] = '--n-threads=%NUM_CPUS%'  # the command used to run multi-threading commands.
                                                                       # The "%NUM_CPUS%" is a wildcard to replace. In this way for commands
                                                                       # that require the num of threads to be inputted without a
                                                                       # blank space we can have something like --my-nthreads=%NUM_CPUS%
                                                                       # (e.g. --my-nthreads=10), otherwise we can have something like
                                                                       # -omp %NUM_CPUS% (e.g. -omp 10). If not present, a blank
                                                                       # space is always added (e.g. --mycommand => --mycommand 10)
    self.runInfoDict['includeDashboard'  ] = False  # in case of internalParallel True, instantiate the RAY dashboard (https://docs.ray.io/en/master/ray-dashboard.html)? Default: False
    self.runInfoDict['WorkingDir'        ] = ''     # the directory where the framework should be running
    self.runInfoDict['TempWorkingDir'    ] = ''     # the temporary directory where a simulation step is run
    self.runInfoDict['NumMPI'            ] = 1      # the number of mpi process by run
    self.runInfoDict['NumThreads'        ] = 1      # Number of Threads by run
    self.runInfoDict['numProcByRun'      ] = 1      # Total number of core used by one run (number of threads by number of mpi)
    self.runInfoDict['batchSize'         ] = 1      # number of contemporaneous runs
    self.runInfoDict['internalParallel'  ] = False  # activate internal parallel (parallel python). If True parallel python is used, otherwise multi-threading is used
    self.runInfoDict['ParallelCommand'   ] = ''     # the command that should be used to submit jobs in parallel (mpi)
    self.runInfoDict['ThreadingCommand'  ] = ''     # the command that should be used to submit multi-threaded
    self.runInfoDict['totalNumCoresUsed' ] = 1      # total number of cores used by driver
    self.runInfoDict['queueingSoftware'  ] = ''     # queueing software name
    self.runInfoDict['stepName'          ] = ''     # the name of the step currently running
    self.runInfoDict['precommand'        ] = ''     # Add to the front of the command that is run
    self.runInfoDict['postcommand'       ] = ''     # Added after the command that is run.
    self.runInfoDict['delSucLogFiles'    ] = False  # If a simulation (code run) has not failed, delete the relative log file (if True)
    self.runInfoDict['deleteOutExtension'] = []     # If a simulation (code run) has not failed, delete the relative output files with the listed extension (comma separated list, for example: 'e,r,txt')
    self.runInfoDict['mode'              ] = ''     # Running mode.  Currently the only mode supported is mpi but others can be added with custom modes.
    self.runInfoDict['Nodes'             ] = []     # List of  node IDs. Filled only in case RAVEN is run in a DMP machine
    self.runInfoDict['expectedTime'      ] = '10:00:00' # How long the complete input is expected to run.
    self.runInfoDict['logfileBuffer'     ] = int(io.DEFAULT_BUFFER_SIZE)*50 # logfile buffer size in bytes
    self.runInfoDict['clusterParameters' ] = []     # Extra parameters to use with the qsub command.
    self.runInfoDict['maxQueueSize'      ] = None

    # A set of dictionaries that collect the instances of all objects needed in the simulation.
    # The keys are the user given names of data, sampler, etc.
    # The value is the instance of the corresponding class
    self.stepsDict            = {}
    self.dataDict             = {}
    self.samplersDict         = {}
    self.modelsDict           = {}
    self.distributionsDict    = {}
    self.dataBasesDict        = {}
    self.functionsDict        = {}
    self.filesDict            = {} #  for each file returns an instance of a Files class
    self.metricsDict          = {}
    self.outStreamsDict       = {}
    self.__stepSequenceList     = [] # the list of step of the simulation

    # list of supported queue-ing software:
    self.knownQueueingSoftware = []
    self.knownQueueingSoftware.append('None')
    self.knownQueueingSoftware.append('PBS Professional')

    # Dictionary of mode handlers
    self.__modeHandlerDict = CustomModes.modeHandlers

    # this dictionary contains the static factory that returns the instance of one of the allowed entities in the simulation
    # the keys are the name of the module that contains the instance of that specific entity
    self.entityModules  = {}
    self.entityModules['Steps'        ] = Steps
    self.entityModules['DataObjects'  ] = DataObjects
    self.entityModules['Samplers'     ] = Samplers
    self.entityModules['Optimizers'   ] = Optimizers
    self.entityModules['Models'       ] = Models
    self.entityModules['Distributions'] = Distributions
    self.entityModules['Databases'    ] = Databases
    self.entityModules['Functions'    ] = Functions
    self.entityModules['Files'        ] = Files
    self.entityModules['Metrics'      ] = Metrics
    self.entityModules['OutStreams'   ] = OutStreams

    # Mapping between an entity type and the dictionary containing the instances for the simulation
    self.entities = {}
    self.entities['Steps'        ] = self.stepsDict
    self.entities['DataObjects'  ] = self.dataDict
    self.entities['Samplers'     ] = self.samplersDict
    self.entities['Optimizers'   ] = self.samplersDict
    self.entities['Models'       ] = self.modelsDict
    self.entities['RunInfo'      ] = self.runInfoDict
    self.entities['Files'        ] = self.filesDict
    self.entities['Distributions'] = self.distributionsDict
    self.entities['Databases'    ] = self.dataBasesDict
    self.entities['Functions'    ] = self.functionsDict
    self.entities['Metrics'      ] = self.metricsDict
    self.entities['OutStreams'   ] = self.outStreamsDict

    # The QApplication
    # The benefit of this enumerated type is that anything other than
    # Interaction.No will evaluate to true here and correctly make the
    # interactive app.
    if interactive:
      self.app = InteractiveApplication([], interactive)
    else:
      self.app = None

    # the handler of the runs within each step
    self.jobHandler = JobHandler()
    # handle the setting of how the jobHandler act
    self.__modeHandler = SimulationMode(self)
    self.printTag = 'SIMULATION'
    self.pollingThread = None # set up when simulation is run to allow subsequent runs without reinstantiating everything

  @Decorators.timingProfile
  def setInputFiles(self, inputFiles):
    """
      Method that can be used to set the input files that the program received.
      These are currently used for cluster running where the program
      needs to be restarted on a different node.
      @ In, inputFiles, list, input files list
      @ Out, None
    """
    self.runInfoDict['SimulationFiles'] = inputFiles

  def getDefaultInputFile(self):
    """
      Returns the default input file to read
      @ In, None
      @ Out, defaultInputFile, string, default input file
    """
    defaultInputFile = self.runInfoDict['DefaultInputFile']
    return defaultInputFile

  def __createAbsPath(self, fileIn):
    """
      Assuming that the file in is already in the self.filesDict it places, as value, the absolute path
      @ In, fileIn, string, the file name that needs to be made "absolute"
      @ Out, None
    """
    curfile = self.filesDict[fileIn]
    path = os.path.normpath(self.runInfoDict['WorkingDir'])
    curfile.prependPath(path) # this respects existing path from the user input, if any

  def XMLpreprocess(self, node, cwd):
    """
      Preprocess the input file, load external xml files into the main ET
      @ In, node, TreeStructure.InputNode, element of RAVEN input file
      @ In, cwd, string, current working directory (for relative path searches)
      @ Out, None
    """
    xmlUtils.expandExternalXML(node,cwd)

  def XMLread(self, xmlNode, runInfoSkip=None, xmlFilename=None):
    """
      instantiates the classes from the input XML file needed to represent all entities in the simulation
      @ In, xmlNode, ElementTree.Element, xml node to read in
      @ In, runInfoSkip, set, optional, nodes to skip
      @ In, xmlFilename, string, optional, xml filename for relative directory
      @ Out, None
    """
    self.raiseADebug("Reading XML", xmlFilename)
    self.setOptionalAttributes(xmlNode)
    self.instantiateEntities(xmlNode, runInfoSkip, xmlFilename)

    # If requested, duplicate input
    # NOTE: All substitutions to the XML input tree should be done BEFORE this point!!
    if self.runInfoDict.get('printInput',False):
      fileName = os.path.join(self.runInfoDict['WorkingDir'],self.runInfoDict['printInput'])
      self.raiseAMessage('Writing duplicate input file:',fileName)
      with open(fileName, 'w') as outFile:
        outFile.writelines(utils.toString(TreeStructure.tostring(xmlNode))+'\n') #\n for no-end-of-line issue
    if not set(self.__stepSequenceList).issubset(set(self.stepsDict.keys())):
      self.raiseAnError(IOError, f'The step list: {self.__stepSequenceList} contains steps that have not been declared: {list(self.stepsDict.keys())}')

  def setOptionalAttributes(self, xmlNode):
    """
      Sets optional attributes for the simulation
      @ In, xmlNode, ElementTree.Element, XML node to read
      @ Out, None
    """
    unknownAttribs = utils.checkIfUnknowElementsinList(['printTimeStamps', 'verbosity', 'color', 'profile'], list(xmlNode.attrib.keys()))
    if len(unknownAttribs) > 0:
      errorMsg = 'The following attributes are unknown:'
      for element in unknownAttribs:
        errorMsg += ' ' + element
      self.raiseAnError(IOError, errorMsg)
    self.verbosity = xmlNode.attrib.get('verbosity', 'all').lower()
    if 'printTimeStamps' in xmlNode.attrib.keys():
      self.raiseADebug(f'Setting "printTimeStamps" to {xmlNode.attrib["printTimeStamps"]}')
      self.messageHandler.setTimePrint(xmlNode.attrib['printTimeStamps'])
    if 'color' in xmlNode.attrib.keys():
      self.raiseADebug(f'Setting color output mode to {xmlNode.attrib["color"]}')
      self.messageHandler.setColor(xmlNode.attrib['color'])
    if 'profile' in xmlNode.attrib.keys():
      thingsToProfile = list(p.strip().lower() for p in xmlNode.attrib['profile'].split(','))
      if 'jobs' in thingsToProfile:
        self.jobHandler.setProfileJobs(True)
    self.messageHandler.verbosity = self.verbosity

  def instantiateRunInfo(self, xmlNode, runInfoSkip=None, xmlFilename=None):
    """
      Instantiates RunInfo entity
      @ In, xmlNode, ElementTree.Element, XML node to read
      @ In, runInfoSkip, set, optional, nodes to skip
      @ In, xmlFilename, string, optional, XML filename for relative directory
    """
    runInfoNode = xmlNode.find('RunInfo')
    if runInfoNode is None:
      self.raiseAnError(IOError, 'The RunInfo node is missing!')
    self.__readRunInfo(runInfoNode, runInfoSkip, xmlFilename)

  def buildVariableGroups(self, xmlNode):
    """
      Gets variable groups from XML
      @ In, xmlNode, ElementTree.Element, XML node to read
      @ Out, varGroups, dict, variable groups
    """
    varGroupNode = xmlNode.find('VariableGroups')
    # get variable groups from XML
    if varGroupNode is not None:
      varGroups = mathUtils.readVariableGroups(varGroupNode)
    else:
      varGroups={}

    return varGroups

  def instantiateEntities(self, xmlNode, runInfoSkip=None, xmlFilename=None):
    """
      Instantiates all entities for simulation from XML
      @ In, xmlNode, ElementTree.Element, XML node to read
      @ In, runInfoSkip, set, optional, nodes to skip
      @ In, xmlFilename, string, optional, XML filename for relative directory
      @ Out, None
    """
    self.instantiateRunInfo(xmlNode, runInfoSkip, xmlFilename)
    # build variable groups
    varGroups = self.buildVariableGroups(xmlNode)
    # read other nodes
    for inputBlock in xmlNode:
      # inputBlock is one of RunInfo, Files, VariableGroups, Distributions, Samplers, Optimizers
      # DataObjects, Databases, OutStreams, Models, Functions, Metrics, or Steps
      if inputBlock.tag == 'VariableGroups':
        continue # we did these already
      xmlUtils.replaceVariableGroups(inputBlock, varGroups)
      if inputBlock.tag in self.entities:
        className = inputBlock.tag
        # we already took care of RunInfo block
        if className in ['RunInfo']:
          continue
        self.raiseADebug(f'-- Reading the block: {inputBlock.tag} --')
        if len(inputBlock.attrib) == 0:
          globalAttributes = {}
        else:
          globalAttributes = inputBlock.attrib
        module = self.entityModules[className]
        if module.factory.returnInputParameter:
          paramInput = module.returnInputParameter()
          paramInput.parseNode(inputBlock)
          # block is specific input block: MonteCarlo, Uniform, PointSet, etc.
          for block in paramInput.subparts:
            blockName = block.getName()
            entity = module.factory.returnInstance(blockName)
            entity.applyRunInfo(self.runInfoDict)
            entity.handleInput(block, globalAttributes=globalAttributes)
            name = entity.name
            self.entities[className][name] = entity
        else:
          for block in inputBlock:
            kind, name, entity = module.factory.instanceFromXML(block)
            self.raiseADebug(f'Reading class "{kind}" named "{name}" ...')
            # place the instance in the proper dictionary (self.entities[Type]) under class name as key
            if name in self.entities[className]:
              self.raiseAnError(IOError, f'Two objects of class "{className}" have the same name "{name}"!')
            self.entities[className][name] = entity
            entity.applyRunInfo(self.runInfoDict)
            entity.readXML(block, varGroups, globalAttributes=globalAttributes)
      else:
        # tag not in entities, check if it's a documentation tag
        if inputBlock.tag not in ['TestInfo']:
          self.raiseAnError(IOError, f'<{inputBlock.tag}> is not among the known simulation components {repr(inputBlock)}')
    # If requested, duplicate input
    # ###NOTE: All substitutions to the XML input tree should be done BEFORE this point!!
    if self.runInfoDict.get('printInput', False):
      fileName = os.path.join(self.runInfoDict['WorkingDir'],self.runInfoDict['printInput'])
      self.raiseAMessage('Writing duplicate input file:', fileName)
      outFile = open(fileName, 'w')
      outFile.writelines(utils.toString(TreeStructure.tostring(xmlNode))+'\n') #\n for no-end-of-line issue
      outFile.close()

  def initialize(self):
    """
      Method to initialize the simulation.
      Check/create working directory, check/set up the parallel environment, call step consistency checker
      @ In, None
      @ Out, None
    """
    # move the full simulation environment in the working directory
    self.raiseADebug(f'Moving to working directory: {self.runInfoDict["WorkingDir"]}')
    os.chdir(self.runInfoDict['WorkingDir'])
    # add the new working dir to the path
    sys.path.append(os.getcwd())
    # clear the raven status file, if any
    self.clearStatusFile()
    # check consistency and fill the missing info for the // runs (threading, mpi, batches)
    self.runInfoDict['numProcByRun'] = self.runInfoDict['NumMPI']*self.runInfoDict['NumThreads']
    oldTotalNumCoresUsed = self.runInfoDict['totalNumCoresUsed']
    self.runInfoDict['totalNumCoresUsed'] = self.runInfoDict['numProcByRun']*self.runInfoDict['batchSize']
    if self.runInfoDict['totalNumCoresUsed'] < oldTotalNumCoresUsed:
      # This is used to reserve some cores
      self.runInfoDict['totalNumCoresUsed'] = oldTotalNumCoresUsed
    elif oldTotalNumCoresUsed > 1:
      # If 1, probably just default
      self.raiseAWarning(f"overriding totalNumCoresUsed {oldTotalNumCoresUsed} to {self.runInfoDict['totalNumCoresUsed']}")
    # transform all files in absolute path
    for key in self.filesDict:
      self.__createAbsPath(key)
    # Let the mode handler do any modification here
    newRunInfo = self.__modeHandler.modifyInfo(dict(self.runInfoDict))
    for key in newRunInfo:
      # Copy in all the new keys
      self.runInfoDict[key] = newRunInfo[key]
    self.jobHandler.applyRunInfo(self.runInfoDict)
    self.__remoteRunCommand = self.__modeHandler.remoteRunCommand(dict(self.runInfoDict))
    if self.__remoteRunCommand is None:
      # If __remoteRunCommand is None, then we are *not* going to run remotely,
      # so need to start jobhandler stuff
      self.jobHandler.initialize()

    for stepName, stepInstance in self.stepsDict.items():
      self.checkStep(stepInstance,stepName)

  def checkStep(self, stepInstance, stepName):
    """
      This method checks the coherence of the simulation step by step
      @ In, stepInstance, instance, instance of the step
      @ In, stepName, string, the name of the step to check
      @ Out, None
    """
    for [role, myClass, _, name] in stepInstance.parList:
      if myClass != 'Step' and myClass not in list(self.entities):
        self.raiseAnError(IOError, f'For step named "{stepName}" the role "{role}" has been assigned to an unknown class type "{myClass}"!')
      if name not in self.entities[myClass]:
        self.raiseADebug(f'name: {name}')
        self.raiseADebug(f'myClass: {myClass}')
        self.raiseADebug(f'list: {list(self.entities[myClass].keys())}')
        self.raiseADebug(f'entities[myClass] {self.entities[myClass]}')
        self.raiseAnError(IOError, f'In step "{stepName}" the class "{myClass}" named "{name}" supposed to be used for the role "{role}" has not been found!')

  def __readRunInfo(self, xmlNode, runInfoSkip, xmlFilename):
    """
      Method that reads the xml input file for the RunInfo block
      @ In, xmlNode, xml.etree.Element, the xml node that belongs to Simulation
      @ In, runInfoSkip, string, the runInfo step to skip
      @ In, xmlFilename, string, xml input file name
      @ Out, None
    """
    if 'verbosity' in xmlNode.attrib.keys():
      self.verbosity = xmlNode.attrib['verbosity']
    self.raiseAMessage(f'Global verbosity level is "{self.verbosity}"', verbosity='quiet')
    if runInfoSkip is None:
      runInfoSkipIter = set()
    else:
      runInfoSkipIter = runInfoSkip
    for element in xmlNode:
      if element.tag in runInfoSkipIter:
        self.raiseAWarning(f"Skipped element {element.tag}")
      elif element.tag == 'printInput':
        text = element.text.strip() if element.text is not None else ''
        # extension fixing
        if len(text) >= 4 and text[-4:].lower() == '.xml':
          text = text[:-4]
        # if the user asked to not print input instead of leaving off tag, respect it
        if utils.stringIsFalse(text):
          self.runInfoDict['printInput'] = False
        # if the user didn't provide a name, provide a default
        elif len(text) < 1:
          self.runInfoDict['printInput'] = 'duplicated_input.xml'
        # otherwise, use the user-provided name
        else:
          self.runInfoDict['printInput'] = text+'.xml'
      elif element.tag == 'WorkingDir':
        # first store the cwd, the "CallDir"
        self.runInfoDict['CallDir'] = os.getcwd()
        # then get the requested "WorkingDir"
        tempName = element.text
        if element.text is None:
          self.raiseAnError(IOError, 'RunInfo.WorkingDir is empty! Use "." to signify "work here" or specify a directory.')
        if '~' in tempName:
          tempName = os.path.expanduser(tempName)
        xmlDirectory = os.path.dirname(os.path.abspath(xmlFilename))
        self.runInfoDict['InputDir'] = xmlDirectory
        if os.path.isabs(tempName):
          self.runInfoDict['WorkingDir'] = tempName
        elif "runRelative" in element.attrib:
          self.runInfoDict['WorkingDir'] = os.path.abspath(tempName)
        else:
          if xmlFilename is None:
            self.raiseAnError(IOError, 'Relative working directory requested but xmlFilename is None.')
          # store location of the input
          xmlDirectory = os.path.dirname(os.path.abspath(xmlFilename))
          self.runInfoDict['InputDir'] = xmlDirectory
          rawRelativeWorkingDir = element.text.strip()
          # working dir is file location + relative working dir
          self.runInfoDict['WorkingDir'] = os.path.join(xmlDirectory,rawRelativeWorkingDir)
        utils.makeDir(self.runInfoDict['WorkingDir'])
      elif element.tag == 'maxQueueSize':
        try:
          self.runInfoDict['maxQueueSize'] = int(element.text)
        except ValueError:
          self.raiseAnError(f'Value given for RunInfo.maxQueueSize could not be converted to integer: {element.text}')
      elif element.tag == 'RemoteRunCommand':
        tempName = element.text
        if '~' in tempName:
          tempName = os.path.expanduser(tempName)
        if os.path.isabs(tempName):
          self.runInfoDict['RemoteRunCommand'] = tempName
        else:
          self.runInfoDict['RemoteRunCommand'] = os.path.abspath(os.path.join(self.runInfoDict['FrameworkDir'], tempName))
      elif element.tag == 'NodeParameter':
        self.runInfoDict['NodeParameter'] = element.text.strip()
      elif element.tag == 'MPIExec':
        self.runInfoDict['MPIExec'] = element.text.strip()
      elif element.tag == 'threadParameter':
        self.runInfoDict['threadParameter'] = element.text.strip()
      elif element.tag == 'JobName':
        self.runInfoDict['JobName'] = element.text.strip()
      elif element.tag == 'ParallelCommand':
        self.runInfoDict['ParallelCommand'] = element.text.strip()
      elif element.tag == 'queueingSoftware':
        self.runInfoDict['queueingSoftware'] = element.text.strip()
      elif element.tag == 'ThreadingCommand':
        self.runInfoDict['ThreadingCommand'] = element.text.strip()
      elif element.tag == 'NumThreads':
        self.runInfoDict['NumThreads'] = int(element.text)
      elif element.tag == 'totalNumCoresUsed':
        self.runInfoDict['totalNumCoresUsed'] = int(element.text)
      elif element.tag == 'NumMPI':
        self.runInfoDict['NumMPI'] = int(element.text)
      elif element.tag == 'internalParallel':
        self.runInfoDict['internalParallel'] = utils.interpretBoolean(element.text)
        dashboard = element.attrib.get("dashboard",'False')
        self.runInfoDict['includeDashboard'  ] = utils.interpretBoolean(dashboard)
      elif element.tag == 'batchSize':
        self.runInfoDict['batchSize'] = int(element.text)
      elif element.tag.lower() == 'maxqueuesize':
        self.runInfoDict['maxQueueSize'] = int(element.text)
      elif element.tag == 'MaxLogFileSize':
        self.runInfoDict['MaxLogFileSize'] = int(element.text)
      elif element.tag == 'precommand':
        self.runInfoDict['precommand'] = element.text
      elif element.tag == 'postcommand':
        self.runInfoDict['postcommand'] = element.text
      elif element.tag == 'deleteOutExtension':
        self.runInfoDict['deleteOutExtension'] = element.text.strip().split(',')
      elif element.tag == 'headNode':
        self.runInfoDict['headNode'] = element.text.strip()
      elif element.tag == 'remoteNodes':
        self.runInfoDict['remoteNodes'] = [el.strip() for el in element.text.strip().split(',')]
      elif element.tag == 'PYTHONPATH':
        self.runInfoDict['UPDATE_PYTHONPATH'] = element.text.strip()
      elif element.tag == 'delSucLogFiles'    :
        if utils.stringIsTrue(element.text):
          self.runInfoDict['delSucLogFiles'] = True
        else:
          self.runInfoDict['delSucLogFiles'] = False
      elif element.tag == 'logfileBuffer':
        self.runInfoDict['logfileBuffer'] = utils.convertMultipleToBytes(element.text.lower())
      elif element.tag == 'clusterParameters':
        self.runInfoDict['clusterParameters'].extend(splitCommand(element.text)) # extend to allow adding parameters at different points.
      elif element.tag == 'mode'               :
        self.runInfoDict['mode'] = element.text.strip().lower()
        # parallel environment
        if self.runInfoDict['mode'] in self.__modeHandlerDict:
          self.__modeHandler = self.__modeHandlerDict[self.runInfoDict['mode']](self)
          self.__modeHandler.XMLread(element)
        else:
          self.raiseAnError(IOError, f"Unknown mode {self.runInfoDict['mode']}")
      elif element.tag == 'expectedTime':
        self.runInfoDict['expectedTime'] = element.text.strip()
      elif element.tag == 'Sequence':
        for stepName in element.text.split(','):
          self.__stepSequenceList.append(stepName.strip())
      elif element.tag == 'DefaultInputFile':
        self.runInfoDict['DefaultInputFile'] = element.text.strip()
      elif element.tag == 'CustomMode' :
        modeName = element.text.strip()
        modeClass = element.attrib["class"]
        modeFile = element.attrib["file"]
        # XXX This depends on if the working directory has been set yet.
        # So switching the order of WorkingDir and CustomMode can
        # cause different results.
        modeFile = modeFile.replace("%BASE_WORKING_DIR%", self.runInfoDict['WorkingDir'])
        modeFile = modeFile.replace("%FRAMEWORK_DIR%", self.runInfoDict['FrameworkDir'])
        modeDir, modeFilename = os.path.split(modeFile)
        if modeFilename.endswith(".py"):
          modeModulename = modeFilename[:-3]
        else:
          modeModulename = modeFilename
        os.sys.path.append(modeDir)
        module = __import__(modeModulename)
        if modeName in self.__modeHandlerDict:
          self.raiseAWarning(f"duplicate mode definition {modeName}")
        self.__modeHandlerDict[modeName] = module.__dict__[modeClass]
      else:
        self.raiseAnError(IOError, f'RunInfo element "{element.tag}" unknown!')

  def printDicts(self):
    """
      utility function to print a summary of the dictionaries
      @ In, None
      @ Out, None
    """
    def __prntDict(Dict, msg):
      """utility function to print a single dictionary"""
      for key in Dict:
        msg += key + '= ' + str(Dict[key]) + '\n'
      return msg
    msg=''
    msg=__prntDict(self.runInfoDict, msg)
    msg=__prntDict(self.stepsDict, msg)
    msg=__prntDict(self.dataDict, msg)
    msg=__prntDict(self.samplersDict, msg)
    msg=__prntDict(self.modelsDict, msg)
    msg=__prntDict(self.metricsDict, msg)
    msg=__prntDict(self.filesDict, msg)
    msg=__prntDict(self.dataBasesDict, msg)
    msg=__prntDict(self.outStreamsDict, msg)
    msg=__prntDict(self.entityModules, msg)
    msg=__prntDict(self.entities, msg)
    self.raiseADebug(msg)

  def getEntity(self, kind, name):
    """
      Return an entity from RAVEN simulation
      @ In, kind, str, type of entity (e.g. DataObject, Sampler)
      @ In, name, str, identifier for entity (i.e. name of the entity)
      @ Out, entity, instance, RAVEN instance (None if not found)
    """
    # TODO is this the fastest way to get-and-check objects?
    kindGroup = self.entities.get(kind, None)
    if kindGroup is None:
      self.raiseAnError(KeyError,f'Entity kind "{kind}" not recognized! Found: {list(self.entities.keys())}')
    entity = kindGroup.get(name, None)
    if entity is None:
      self.raiseAnError(KeyError,f'No entity named "{name}" found among "{kind}" entities! Found: {list(self.entities[kind].keys())}')
    return entity

  def initiateStep(self, stepName):
    """
      This method assembles and initializes the step to make it ready to be executed
      @ In, stepName, str, the step to initialize
      @ Out, (stepInputDict, stepInstance), tuple, tuple of step input dictionary and step instance
    """
    stepInstance = self.stepsDict[stepName]   # retrieve the instance of the step
    if self.ranPreviously:
      stepInstance.flushStep()
    self.raiseAMessage(f'-- Beginning {stepInstance.type} step "{stepName}" ... --')
    self.runInfoDict['stepName'] = stepName   # provide the name of the step to runInfoDict
    stepInputDict = {}                        # initialize the input dictionary for a step. Never use an old one!!!!!
    stepInputDict['Input' ] = []              # set the Input to an empty list
    stepInputDict['Output'] = []              # set the Output to an empty list
    # fill the take a step input dictionary just to recall: key= role played in the step b= Class, c= Type, d= user given name
    for role, entity, _, name in stepInstance.parList:
      # Only for input and output we allow more than one object passed to the step, so for those we build a list
      if role == 'Input':
        stepInputDict[role].append(self.getEntity(entity, name))
      elif role == 'Output':
        if self.ranPreviously and entity == 'DataObjects':
          # if simulation was run previously, output DataObjects need to be flushed
          flushDataObject = self.getEntity(entity, name)
          flushDataObject.flush()
          # now add to stepInputDict
          stepInputDict[role].append(flushDataObject)
        else:
          stepInputDict[role].append(self.getEntity(entity, name))
      elif role in ['Optimizer', 'Sampler', 'SolutionExport'] and self.ranPreviously:
        # if simulation was run previously, flush Optimizers, Samplers, and SolutionExport DataObjects
        flusher = self.getEntity(entity, name)
        if 'flush' in dir(flusher):
          flusher.flush()
        # now add to stepInputDict
        stepInputDict[role] = flusher
      else:
        stepInputDict[role] = self.getEntity(entity, name)

    # add the global objects
    stepInputDict['jobHandler'] = self.jobHandler
    # generate the needed assembler to send to the step
    for key in stepInputDict:
      if isinstance(stepInputDict[key], list):
        stepindict = stepInputDict[key]
      else:
        stepindict = [stepInputDict[key]]
      # check assembler. NB. If the assembler refers to an internal object the relative dictionary
      # needs to have the format {'internal':[(None,'variableName'),(None,'variable name')]}
      for stp in stepindict:
        self.generateAllAssemblers(stp)
    return stepInputDict, stepInstance

  def executeStep(self, stepInputDict, stepInstance):
    """
      This method executes and finalizes the step
      @ In, stepInputDict, dict, the step dictionary
      @ In, stepInstance, instance, step instance
      @ Out, None
    """
    stepInstance.takeAstep(stepInputDict)
    # FilePrint has a finalize method that should be run at the end of the step,
    # run this method for each instance
    for output in stepInputDict['Output']:
      if "_printer" in output.__dict__:
        output._printer.finalize()

    self.raiseAMessage(f'-- End step {stepInstance.name} of type: {stepInstance.type} --\n')

  def finalizeSimulation(self):
    """
      This method is called at end of the simulation to finalize it
      It is in charge of shutting down the job handler and cleaning up the execution
      @ In, None
      @ Out, None
    """
    self.jobHandler.shutdown()
    self.messageHandler.printWarnings()
    # implicitly, the job finished successfully if we got here -- self.pollingThread.is_Alive()
    # returns False and no new jobs can be queued.
    self.writeStatusFile()
    self.ranPreviously = True

  def stepSequence(self):
    """
      Return step sequence
      @ In, None
      @ Out, stepSequence, list(str), list of step in sequence
    """
    return self.__stepSequenceList

  def run(self):
    """
      Run the simulation
      @ In, None
      @ Out, None
    """
    # set the Simulation time
    self.messageHandler.starttime = time.time()
    readtime = datetime.datetime.fromtimestamp(self.messageHandler.starttime).strftime('%Y-%m-%d %H:%M:%S')
    self.raiseAMessage('Simulation started at', readtime, verbosity='silent')

    self.pollingThread = threading.Thread(target=self.jobHandler.startLoop)
    # This allows RAVEN to exit when the only thing left is the JobHandler
    # This should no longer be necessary since the jobHandler now has an off
    # switch that this object can flip when it is complete, however, if
    # simulation fails before it is finished, we should probably still ensure
    # that this thread is killed as well, so maybe it is best to keep it for
    # now.
    self.pollingThread.daemon = True
    self.pollingThread.start()
    # to do list
    # can we remove the check on the existence of the file, it might make more sense just to check in case they are input and before the step they are used
    self.raiseADebug('entering the run')

    # controlling the PBS environment
    if self.__remoteRunCommand is not None:
      subprocess.call(args=self.__remoteRunCommand["args"],
                      cwd=self.__remoteRunCommand.get("cwd", None),
                      env=self.__remoteRunCommand.get("env", None))
      self.raiseADebug('Submitted in queue! Shutting down Jobhandler!')
      #Note that jobhandler has not been initialized,
      # so no need to call self.jobHandler.shutdown()
      return
    # initialize, then execute, steps
    for stepName in self.__stepSequenceList:
      stepInputDict, stepInstance = self.initiateStep(stepName)
      self.executeStep(stepInputDict, stepInstance)
    # finalize the simulation
    self.finalizeSimulation()
    self.raiseADebug(time.ctime())
    self.raiseAMessage('Run complete!\n', forcePrint=True)

    return 0

  def generateAllAssemblers(self, objectInstance):
    """
      This method is used to generate all assembler objects at the Step construction stage
      @ In, objectInstance, Instance, Instance of RAVEN entity, i.e. Input, Sampler, Model
      @ Out, None
    """
    if "whatDoINeed" in dir(objectInstance):
      neededobjs    = {}
      neededObjects = objectInstance.whatDoINeed()
      for mainClassStr in neededObjects.keys():
        if mainClassStr not in self.entities and mainClassStr != 'internal':
          self.raiseAnError(IOError,'Main Class '+mainClassStr+' unknown!')
        neededobjs[mainClassStr] = {}
        for obj in neededObjects[mainClassStr]:
          if obj[1] in vars(self):
            neededobjs[mainClassStr][obj[1]] = vars(self)[obj[1]]
          elif obj[1] in self.entities[mainClassStr].keys():
            if obj[0]:
              if obj[0] not in self.entities[mainClassStr][obj[1]].type:
                self.raiseAnError(IOError, f'Type of requested object {obj[1]} does not match the actual type! {obj[0]} != {self.entities[mainClassStr][obj[1]].type}')
            neededobjs[mainClassStr][obj[1]] = self.entities[mainClassStr][obj[1]]
            self.generateAllAssemblers(neededobjs[mainClassStr][obj[1]])
          elif obj[1] in 'all':
            # if 'all' we get all the objects of a certain 'mainClassStr'
            for allObject in self.entities[mainClassStr]:
              neededobjs[mainClassStr][allObject] = self.entities[mainClassStr][allObject]
          else:
            self.raiseAnError(IOError, f'Requested object <{obj[1]}> is not part of the Main Class <{mainClassStr}>! \nOptions are: {list(self.entities[mainClassStr].keys())}')
      objectInstance.generateAssembler(neededobjs)

  def clearStatusFile(self):
    """
      Remove the status file from disk so we can really tell when RAVEN has successfully finished.
      This doesn't seem to be a very robust strategy, but it is working for now.
      @ In, None
      @ Out, None
    """
    try:
      os.remove('.ravenStatus')
    except OSError as e:
      if os.path.isfile('.ravenStatus'):
        self.raiseAWarning(f'RAVEN status file detected but not removable! Got: "{e}"')

  def writeStatusFile(self):
    """
      Write a status file to disk so we can really tell when RAVEN has successfully finished.
      This doesn't seem to be a very robust strategy, but it is working for now.
      @ In, None
      @ Out, None
    """
    with open('.ravenStatus', 'w') as f:
      f.writelines('Success')
      #force it to disk
      f.flush()
      os.fsync(f.fileno())

  def resetSimulation(self):
    """
      Resets and re-initializes for re-running the Simulation
      @ In, None
      @ Out, None
    """
    if self.jobHandler.completed:
      # this must be False in order to set up the queue
      self.jobHandler.completed = False

    # reset warning counts and messages for new simulation if simulation ran previously
    if self.ranPreviously:
      self.messageHandler.warningCount = []
      self.messageHandler.warnings = []

    # re-initialize all databases if overwrite is desired
    if self.ranPreviously:
      dbs = self.entities['Databases']
      for db in dbs:
        tmpDatabase = dbs[db]
        tmpDatabase.initializeDatabase()
