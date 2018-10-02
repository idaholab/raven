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
Module that contains the driver for the whole the simulation flow (Simulation Class)
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import os,subprocess
import math
import sys
import io
import string
import datetime
import numpy as np
import threading
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import Steps
import DataObjects
import Files
import Samplers
import Optimizers
import Models
import Metrics
import Distributions
import Databases
import Functions
import OutStreams
from JobHandler import JobHandler
import MessageHandler
import VariableGroups
from utils import utils,TreeStructure,xmlUtils
from Application import __QtAvailable
from Interaction import Interaction
if __QtAvailable:
  from Application import InteractiveApplication
#Internal Modules End--------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
class SimulationMode(MessageHandler.MessageUser):
  """
    SimulationMode allows changes to the how the simulation
    runs are done.  modifySimulation lets the mode change runInfoDict
    and other parameters.  remoteRunCommand lets a command to run RAVEN
    remotely be specified.
  """
  def __init__(self,messageHandler):
    """
      Constructor
      @ In, messageHandler, instance, instance of the MessageHandler class
      @ Out, None
    """
    self.messageHandler = messageHandler
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
      are put infront of the command and after the command.
      @ In, runInfoDict, dict, the run info
      @ Out, dictionary to use for modifications.  If empty, no changes
    """
    import multiprocessing
    try:
      if multiprocessing.cpu_count() < runInfoDict['batchSize']:
        self.raiseAWarning("cpu_count",multiprocessing.cpu_count(),"< batchSize",runInfoDict['batchSize'])
    except NotImplementedError:
      pass
    return {}

  def XMLread(self,xmlNode):
    """
      XMLread is called with the mode node, and can be used to
      get extra parameters needed for the simulation mode.
      @ In, xmlNode, xml.etree.ElementTree.Element, the xml node that belongs to this class instance
      @ Out, None
    """
    pass

#Note that this has to be after SimulationMode is defined or the CustomModes
#don't see SimulationMode when they import Simulation
import CustomModes

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
        #found end of command
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

#----------------------------------------------------------------------

#
#
#
#-----------------------------------------------------------------------------------------------------
class Simulation(MessageHandler.MessageUser):
  """
    This is a class that contain all the object needed to run the simulation
    Usage:
    myInstance = Simulation()                          !Generate the instance
    myInstance.XMLread(xml.etree.ElementTree.Element)  !This method generate all the objects living in the simulation
    myInstance.initialize()                            !This method takes care of setting up the directory/file environment with proper checks
    myInstance.run()                                   !This method run the simulation
    Utility methods:
     myInstance.printDicts                              !prints the dictionaries representing the whole simulation
     myInstance.setInputFiles                           !re-associate the set of files owned by the simulation
     myInstance.getDefaultInputFile                     !return the default name of the input file read by the simulation
    Inherited from the BaseType class:
     myInstance.whoAreYou()                             !inherited from BaseType class-
     myInstance.myClassmyCurrentSetting()               !see BaseType class-

    --how to add a new entity <myClass> to the simulation--
    Add an import for the module where it is defined. Convention is that the module is named with the plural
     of the base class of the module: <MyModule>=<myClass>+'s'.
     The base class of the module is by convention named as the new type of simulation component <myClass>.
     The module should contain a set of classes named <myType> that are child of the base class <myClass>.
     The module should possess a function <MyModule>.returnInstance('<myType>',caller) that returns a pointer to the class <myType>.
    Add in Simulation.__init__ the following
     self.<myClass>Dict = {}
     self.addWhatDict['<myClass>'] = <MyModule>
     self.whichDict['<myClass>'  ] = self.<myClass>+'Dict'
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

  def __init__(self,frameworkDir,verbosity='all',interactive=Interaction.No):
    """
      Constructor
      @ In, frameworkDir, string, absolute path to framework directory
      @ In, verbosity, string, optional, general verbosity level
      @ In, interactive, Interaction, optional, toggles the ability to provide
        an interactive UI or to run to completion without human interaction
      @ Out, None
    """
    self.FIXME          = False
    #set the numpy print threshold to avoid ellipses in array truncation
    np.set_printoptions(threshold=np.inf)
    #establish message handling: the error, warning, message, and debug print handler
    self.messageHandler = MessageHandler.MessageHandler()
    self.verbosity      = verbosity
    callerLength        = 25
    tagLength           = 15
    suppressErrs        = False
    self.messageHandler.initialize({'verbosity':self.verbosity,
                                    'callerLength':callerLength,
                                    'tagLength':tagLength,
                                    'suppressErrs':suppressErrs})
    readtime = datetime.datetime.fromtimestamp(self.messageHandler.starttime).strftime('%Y-%m-%d %H:%M:%S')
    sys.path.append(os.getcwd())
    #this dictionary contains the general info to run the simulation
    self.runInfoDict = {}
    self.runInfoDict['DefaultInputFile'  ] = 'test.xml'   #Default input file to use
    self.runInfoDict['SimulationFiles'   ] = []           #the xml input file
    self.runInfoDict['ScriptDir'         ] = os.path.join(os.path.dirname(frameworkDir),"scripts") # the location of the pbs script interfaces
    self.runInfoDict['FrameworkDir'      ] = frameworkDir # the directory where the framework is located
    self.runInfoDict['RemoteRunCommand'  ] = os.path.join(frameworkDir,'raven_qsub_command.sh')
    self.runInfoDict['NodeParameter'     ] = '-f'         # the parameter used to specify the files where the nodes are listed
    self.runInfoDict['MPIExec'           ] = 'mpiexec'    # the command used to run mpi commands
    self.runInfoDict['WorkingDir'        ] = ''           # the directory where the framework should be running
    self.runInfoDict['TempWorkingDir'    ] = ''           # the temporary directory where a simulation step is run
    self.runInfoDict['NumMPI'            ] = 1            # the number of mpi process by run
    self.runInfoDict['NumThreads'        ] = 1            # Number of Threads by run
    self.runInfoDict['numProcByRun'      ] = 1            # Total number of core used by one run (number of threads by number of mpi)
    self.runInfoDict['batchSize'         ] = 1            # number of contemporaneous runs
    self.runInfoDict['internalParallel'  ] = False        # activate internal parallel (parallel python). If True parallel python is used, otherwise multi-threading is used
    self.runInfoDict['ParallelCommand'   ] = ''           # the command that should be used to submit jobs in parallel (mpi)
    self.runInfoDict['ThreadingCommand'  ] = ''           # the command should be used to submit multi-threaded
    self.runInfoDict['totalNumCoresUsed' ] = 1            # total number of cores used by driver
    self.runInfoDict['queueingSoftware'  ] = ''           # queueing software name
    self.runInfoDict['stepName'          ] = ''           # the name of the step currently running
    self.runInfoDict['precommand'        ] = ''           # Add to the front of the command that is run
    self.runInfoDict['postcommand'       ] = ''           # Added after the command that is run.
    self.runInfoDict['delSucLogFiles'    ] = False        # If a simulation (code run) has not failed, delete the relative log file (if True)
    self.runInfoDict['deleteOutExtension'] = []           # If a simulation (code run) has not failed, delete the relative output files with the listed extension (comma separated list, for example: 'e,r,txt')
    self.runInfoDict['mode'              ] = ''           # Running mode.  Curently the only mode supported is mpi but others can be added with custom modes.
    self.runInfoDict['Nodes'             ] = []           # List of  node IDs. Filled only in case RAVEN is run in a DMP machine
    self.runInfoDict['expectedTime'      ] = '10:00:00'   # How long the complete input is expected to run.
    self.runInfoDict['logfileBuffer'     ] = int(io.DEFAULT_BUFFER_SIZE)*50 # logfile buffer size in bytes
    self.runInfoDict['clusterParameters' ] = []           # Extra parameters to use with the qsub command.
    self.runInfoDict['maxQueueSize'      ] = None

    #Following a set of dictionaries that, in a manner consistent with their names, collect the instance of all objects needed in the simulation
    #Theirs keywords in the dictionaries are the the user given names of data, sampler, etc.
    #The value corresponding to a keyword is the instance of the corresponding class
    self.stepsDict            = {}
    self.dataDict             = {}
    self.samplersDict         = {}
    self.modelsDict           = {}
    self.distributionsDict    = {}
    self.dataBasesDict        = {}
    self.functionsDict        = {}
    self.filesDict            = {} #  for each file returns an instance of a Files class
    self.metricsDict          = {}
    self.OutStreamManagerPlotDict  = {}
    self.OutStreamManagerPrintDict = {}
    self.stepSequenceList     = [] #the list of step of the simulation

    #list of supported queue-ing software:
    self.knownQueueingSoftware = []
    self.knownQueueingSoftware.append('None')
    self.knownQueueingSoftware.append('PBS Professional')

    #Dictionary of mode handlers for the
    self.__modeHandlerDict           = CustomModes.modeHandlers
    #self.__modeHandlerDict['mpi']    = CustomModes.MPISimulationMode
    #self.__modeHandlerDict['mpilegacy'] = CustomModes.MPILegacySimulationMode

    #this dictionary contain the static factory that return the instance of one of the allowed entities in the simulation
    #the keywords are the name of the module that contains the specialization of that specific entity
    self.addWhatDict  = {}
    self.addWhatDict['Steps'            ] = Steps
    self.addWhatDict['DataObjects'      ] = DataObjects
    self.addWhatDict['Samplers'         ] = Samplers
    self.addWhatDict['Optimizers'       ] = Optimizers
    self.addWhatDict['Models'           ] = Models
    self.addWhatDict['Distributions'    ] = Distributions
    self.addWhatDict['Databases'        ] = Databases
    self.addWhatDict['Functions'        ] = Functions
    self.addWhatDict['Files'            ] = Files
    self.addWhatDict['Metrics'          ] = Metrics
    self.addWhatDict['OutStreams' ] = {}
    self.addWhatDict['OutStreams' ]['Plot' ] = OutStreams
    self.addWhatDict['OutStreams' ]['Print'] = OutStreams


    #Mapping between an entity type and the dictionary containing the instances for the simulation
    self.whichDict = {}
    self.whichDict['Steps'           ] = self.stepsDict
    self.whichDict['DataObjects'     ] = self.dataDict
    self.whichDict['Samplers'        ] = self.samplersDict
    self.whichDict['Optimizers'      ] = self.samplersDict
    self.whichDict['Models'          ] = self.modelsDict
    self.whichDict['RunInfo'         ] = self.runInfoDict
    self.whichDict['Files'           ] = self.filesDict
    self.whichDict['Distributions'   ] = self.distributionsDict
    self.whichDict['Databases'       ] = self.dataBasesDict
    self.whichDict['Functions'       ] = self.functionsDict
    self.whichDict['Metrics'         ] = self.metricsDict
    self.whichDict['OutStreams'] = {}
    self.whichDict['OutStreams']['Plot' ] = self.OutStreamManagerPlotDict
    self.whichDict['OutStreams']['Print'] = self.OutStreamManagerPrintDict

    # The QApplication
    ## The benefit of this enumerated type is that anything other than
    ## Interaction.No will evaluate to true here and correctly make the
    ## interactive app.
    if interactive:
      self.app = InteractiveApplication([],self.messageHandler, interactive)
    else:
      self.app = None

    #the handler of the runs within each step
    self.jobHandler    = JobHandler()
    #handle the setting of how the jobHandler act
    self.__modeHandler = SimulationMode(self.messageHandler)
    self.printTag = 'SIMULATION'
    self.raiseAMessage('Simulation started at',readtime,verbosity='silent')


    self.pollingThread = threading.Thread(target=self.jobHandler.startLoop)
    ## This allows RAVEN to exit when the only thing left is the JobHandler
    ## This should no longer be necessary since the jobHandler now has an off
    ## switch that this object can flip when it is complete, however, if
    ## simulation fails before it is finished, we should probably still ensure
    ## that this thread is killed as well, so maybe it is best to keep it for
    ## now.
    self.pollingThread.daemon = True
    self.pollingThread.start()

  def setInputFiles(self,inputFiles):
    """
      Method that can be used to set the input files that the program received.
      These are currently used for cluster running where the program
      needs to be restarted on a different node.
      @ In, inputFiles, list, input files list
      @ Out, None
    """
    self.runInfoDict['SimulationFiles'   ] = inputFiles

  def getDefaultInputFile(self):
    """
      Returns the default input file to read
      @ In, None
      @ Out, defaultInputFile, string, default input file
    """
    defaultInputFile = self.runInfoDict['DefaultInputFile']
    return defaultInputFile

  def __createAbsPath(self,fileIn):
    """
      Assuming that the file in is already in the self.filesDict it places, as value, the absolute path
      @ In, fileIn, string, the file name that needs to be made "absolute"
      @ Out, None
    """
    curfile = self.filesDict[fileIn]
    path = os.path.normpath(self.runInfoDict['WorkingDir'])
    curfile.prependPath(path) #this respects existing path from the user input, if any

  def XMLpreprocess(self,node,cwd):
    """
      Preprocess the input file, load external xml files into the main ET
      @ In, node, TreeStructure.InputNode, element of RAVEN input file
      @ In, cwd, string, current working directory (for relative path searches)
      @ Out, None
    """
    xmlUtils.expandExternalXML(node,cwd)

  def XMLread(self,xmlNode,runInfoSkip = set(),xmlFilename=None):
    """
      parses the xml input file, instances the classes need to represent all objects in the simulation
      @ In, xmlNode, ElementTree.Element, xml node to read in
      @ In, runInfoSkip, set, optional, nodes to skip
      @ In, xmlFilename, string, optional, xml filename for relative directory
      @ Out, None
    """
    #TODO update syntax to note that we read InputTrees not XmlTrees
    unknownAttribs = utils.checkIfUnknowElementsinList(['printTimeStamps','verbosity','color','profile'],list(xmlNode.attrib.keys()))
    if len(unknownAttribs) > 0:
      errorMsg = 'The following attributes are unknown:'
      for element in unknownAttribs:
        errorMsg += ' ' + element
      self.raiseAnError(IOError,errorMsg)
    self.verbosity = xmlNode.attrib.get('verbosity','all').lower()
    if 'printTimeStamps' in xmlNode.attrib.keys():
      self.raiseADebug('Setting "printTimeStamps" to',xmlNode.attrib['printTimeStamps'])
      self.messageHandler.setTimePrint(xmlNode.attrib['printTimeStamps'])
    if 'color' in xmlNode.attrib.keys():
      self.raiseADebug('Setting color output mode to',xmlNode.attrib['color'])
      self.messageHandler.setColor(xmlNode.attrib['color'])
    if 'profile' in xmlNode.attrib.keys():
      thingsToProfile = list(p.strip().lower() for p in xmlNode.attrib['profile'].split(','))
      if 'jobs' in thingsToProfile:
        self.jobHandler.setProfileJobs(True)
    self.messageHandler.verbosity = self.verbosity
    runInfoNode = xmlNode.find('RunInfo')
    if runInfoNode is None:
      self.raiseAnError(IOError,'The RunInfo node is missing!')
    self.__readRunInfo(runInfoNode,runInfoSkip,xmlFilename)
    ### expand variable groups before continuing ###
    ## build variable groups ##
    varGroupNode = xmlNode.find('VariableGroups')
    # init, read XML for variable groups
    if varGroupNode is not None:
      varGroups = xmlUtils.readVariableGroups(varGroupNode,self.messageHandler,self)
    else:
      varGroups={}
    # read other nodes
    for child in xmlNode:
      if child.tag=='VariableGroups':
        continue #we did these before the for loop
      if child.tag in list(self.whichDict.keys()):
        self.raiseADebug('-'*2+' Reading the block: {0:15}'.format(str(child.tag))+2*'-')
        Class = child.tag
        if len(child.attrib.keys()) == 0:
          globalAttributes = {}
        else:
          globalAttributes = child.attrib
          #if 'verbosity' in globalAttributes.keys(): self.verbosity = globalAttributes['verbosity']
        if Class not in ['RunInfo','OutStreams'] and "returnInputParameter" in self.addWhatDict[Class].__dict__:
          paramInput = self.addWhatDict[Class].returnInputParameter()
          paramInput.parseNode(child)
          for childChild in paramInput.subparts:
            childName = childChild.getName()
            if "name" not in childChild.parameterValues:
              self.raiseAnError(IOError,'not found name attribute for '+childName +' in '+Class)
            name = childChild.parameterValues["name"]
            if "needsRunInfo" in self.addWhatDict[Class].__dict__:
              self.whichDict[Class][name] = self.addWhatDict[Class].returnInstance(childName,self.runInfoDict,self)
            else:
              self.whichDict[Class][name] = self.addWhatDict[Class].returnInstance(childName,self)
            self.whichDict[Class][name].handleInput(childChild, self.messageHandler, varGroups, globalAttributes=globalAttributes)
        elif Class != 'RunInfo':
          for childChild in child:
            subType = childChild.tag
            if 'name' in childChild.attrib.keys():
              name = childChild.attrib['name']
              self.raiseADebug('Reading type '+str(childChild.tag)+' with name '+name)
              #place the instance in the proper dictionary (self.whichDict[Type]) under his name as key,
              #the type is the general class (sampler, data, etc) while childChild.tag is the sub type
              #if name not in self.whichDict[Class].keys():  self.whichDict[Class][name] = self.addWhatDict[Class].returnInstance(childChild.tag,self)
              if Class != 'OutStreams':
                if name not in self.whichDict[Class].keys():
                  if "needsRunInfo" in self.addWhatDict[Class].__dict__:
                    self.whichDict[Class][name] = self.addWhatDict[Class].returnInstance(childChild.tag,self.runInfoDict,self)
                  else:
                    self.whichDict[Class][name] = self.addWhatDict[Class].returnInstance(childChild.tag,self)
                else:
                  self.raiseAnError(IOError,'Redundant naming in the input for class '+Class+' and name '+name)
              else:
                if name not in self.whichDict[Class][subType].keys():
                  self.whichDict[Class][subType][name] = self.addWhatDict[Class][subType].returnInstance(childChild.tag,self)
                else:
                  self.raiseAnError(IOError,'Redundant  naming in the input for class '+Class+' and sub Type'+subType+' and name '+name)
              #now we can read the info for this object
              #if globalAttributes and 'verbosity' in globalAttributes.keys(): localVerbosity = globalAttributes['verbosity']
              #else                                                      : localVerbosity = self.verbosity
              if Class != 'OutStreams':
                self.whichDict[Class][name].readXML(childChild, self.messageHandler, varGroups, globalAttributes=globalAttributes)
              else:
                self.whichDict[Class][subType][name].readXML(childChild, self.messageHandler, globalAttributes=globalAttributes)
            else:
              self.raiseAnError(IOError,'not found name attribute for one '+Class)
      else:
        #tag not in whichDict, check if it's a documentation tag
        if child.tag not in ['TestInfo']:
          self.raiseAnError(IOError,'<'+child.tag+'> is not among the known simulation components '+repr(child))
    # If requested, duplicate input
    # ###NOTE: All substitutions to the XML input tree should be done BEFORE this point!!
    if self.runInfoDict.get('printInput',False):
      fileName = os.path.join(self.runInfoDict['WorkingDir'],self.runInfoDict['printInput'])
      self.raiseAMessage('Writing duplicate input file:',fileName)
      outFile = open(fileName,'w')
      outFile.writelines(TreeStructure.tostring(xmlNode)+'\n') #\n for no-end-of-line issue
      outFile.close()
    if not set(self.stepSequenceList).issubset(set(self.stepsDict.keys())):
      self.raiseAnError(IOError,'The step list: '+str(self.stepSequenceList)+' contains steps that have not been declared: '+str(list(self.stepsDict.keys())))

  def initialize(self):
    """
      Method to intialize the simulation.
      Check/created working directory, check/set up the parallel environment, call step consistency checker
      @ In, None
      @ Out, None
    """
    #move the full simulation environment in the working directory
    self.raiseADebug('Moving to working directory:',self.runInfoDict['WorkingDir'])
    os.chdir(self.runInfoDict['WorkingDir'])
    #add also the new working dir to the path
    sys.path.append(os.getcwd())
    #check consistency and fill the missing info for the // runs (threading, mpi, batches)
    self.runInfoDict['numProcByRun'] = self.runInfoDict['NumMPI']*self.runInfoDict['NumThreads']
    oldTotalNumCoresUsed = self.runInfoDict['totalNumCoresUsed']
    self.runInfoDict['totalNumCoresUsed'] = self.runInfoDict['numProcByRun']*self.runInfoDict['batchSize']
    if self.runInfoDict['totalNumCoresUsed'] < oldTotalNumCoresUsed:
      #This is used to reserve some cores
      self.runInfoDict['totalNumCoresUsed'] = oldTotalNumCoresUsed
    elif oldTotalNumCoresUsed > 1:
      #If 1, probably just default
      self.raiseAWarning("overriding totalNumCoresUsed",oldTotalNumCoresUsed,"to", self.runInfoDict['totalNumCoresUsed'])
    #transform all files in absolute path
    for key in self.filesDict.keys():
      self.__createAbsPath(key)
    #Let the mode handler do any modification here
    newRunInfo = self.__modeHandler.modifyInfo(dict(self.runInfoDict))
    for key in newRunInfo:
      #Copy in all the new keys
      self.runInfoDict[key] = newRunInfo[key]
    self.jobHandler.initialize(self.runInfoDict,self.messageHandler)
    # only print the dictionaries when the verbosity is set to debug
    #if self.verbosity == 'debug': self.printDicts()
    for stepName, stepInstance in self.stepsDict.items():
      self.checkStep(stepInstance,stepName)

  def checkStep(self,stepInstance,stepName):
    """
      This method checks the coherence of the simulation step by step
      @ In, stepInstance, instance, instance of the step
      @ In, stepName, string, the name of the step to check
      @ Out, None
    """
    for [role,myClass,objectType,name] in stepInstance.parList:
      if myClass!= 'Step' and myClass not in list(self.whichDict.keys()):
        self.raiseAnError(IOError,'For step named '+stepName+' the role '+role+' has been assigned to an unknown class type '+myClass)
      if myClass != 'OutStreams':
        if name not in list(self.whichDict[myClass].keys()):
          self.raiseADebug('name:',name)
          self.raiseADebug('myClass:',myClass)
          self.raiseADebug('list:',list(self.whichDict[myClass].keys()))
          self.raiseADebug('whichDict[myClass]',self.whichDict[myClass])
          self.raiseAnError(IOError,'In step '+stepName+' the class '+myClass+' named '+name+' supposed to be used for the role '+role+' has not been found')
      else:
        if objectType not in self.whichDict[myClass].keys():
          self.raiseAnError(IOError,'In step "{}" class "{}" the type "{}" is not recognized!'.format(stepName,myClass,objectType))
        if name not in self.whichDict[myClass][objectType].keys():
          self.raiseADebug('name: '+name)
          self.raiseADebug('list: '+str(list(self.whichDict[myClass][objectType].keys())))
          self.raiseADebug(str(self.whichDict[myClass][objectType]))
          self.raiseAnError(IOError,'In step '+stepName+' the class '+myClass+' named '+name+' supposed to be used for the role '+role+' has not been found')

      if myClass != 'Files':
        # check if object type is consistent
        if myClass != 'OutStreams':
          objtype = self.whichDict[myClass][name].type
        else:
          objtype = self.whichDict[myClass][objectType][name].type
        if objectType != objtype.replace("OutStream",""):
          objtype = self.whichDict[myClass][name].type
          #self.raiseAnError(IOError,'In step '+stepName+' the class '+myClass+' named '+name+' used for role '+role+' has mismatching type. Type is "'+objtype.replace("OutStream","")+'" != inputted one "'+objectType+'"!')

  def __readRunInfo(self,xmlNode,runInfoSkip,xmlFilename):
    """
      Method that reads the xml input file for the RunInfo block
      @ In, xmlNode, xml.etree.Element, the xml node that belongs to Simulation
      @ In, runInfoSkip, string, the runInfo step to skip
      @ In, xmlFilename, string, xml input file name
      @ Out, None
    """
    if 'verbosity' in xmlNode.attrib.keys():
      self.verbosity = xmlNode.attrib['verbosity']
    self.raiseAMessage('Global verbosity level is "',self.verbosity,'"',verbosity='quiet')
    for element in xmlNode:
      if element.tag in runInfoSkip:
        self.raiseAWarning("Skipped element ",element.tag)
      elif element.tag == 'printInput':
        text = element.text.strip() if element.text is not None else ''
        #extension fixing
        if len(text) >= 4 and text[-4:].lower() == '.xml':
          text = text[:-4]
        # if the user asked to not print input instead of leaving off tag, respect it
        if text.lower() in utils.stringsThatMeanFalse():
          self.runInfoDict['printInput'] = False
        # if the user didn't provide a name, provide a default
        elif len(text)<1:
          self.runInfoDict['printInput'] = 'duplicated_input.xml'
        # otherwise, use the user-provided name
        else:
          self.runInfoDict['printInput'] = text+'.xml'
      elif element.tag == 'WorkingDir':
        # first store the cwd, the "CallDir"
        self.runInfoDict['CallDir'] = os.getcwd()
        # then get the requested "WorkingDir"
        tempName = element.text
        if '~' in tempName:
          tempName = os.path.expanduser(tempName)
        if os.path.isabs(tempName):
          self.runInfoDict['WorkingDir'] = tempName
        elif "runRelative" in element.attrib:
          self.runInfoDict['WorkingDir'] = os.path.abspath(tempName)
        else:
          if xmlFilename == None:
            self.raiseAnError(IOError,'Relative working directory requested but xmlFilename is None.')
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
          self.raiseAnError('Value give for RunInfo.maxQueueSize could not be converted to integer: {}'.format(element.text))
      elif element.tag == 'RemoteRunCommand':
        tempName = element.text
        if '~' in tempName:
          tempName = os.path.expanduser(tempName)
        if os.path.isabs(tempName):
          self.runInfoDict['RemoteRunCommand'] = tempName
        else:
          self.runInfoDict['RemoteRunCommand'] = os.path.abspath(os.path.join(self.runInfoDict['FrameworkDir'],tempName))
      elif element.tag == 'NodeParameter':
        self.runInfoDict['NodeParameter'] = element.text.strip()
      elif element.tag == 'MPIExec':
        self.runInfoDict['MPIExec'] = element.text.strip()
      elif element.tag == 'JobName':
        self.runInfoDict['JobName'           ] = element.text.strip()
      elif element.tag == 'ParallelCommand':
        self.runInfoDict['ParallelCommand'   ] = element.text.strip()
      elif element.tag == 'queueingSoftware':
        self.runInfoDict['queueingSoftware'  ] = element.text.strip()
      elif element.tag == 'ThreadingCommand':
        self.runInfoDict['ThreadingCommand'  ] = element.text.strip()
      elif element.tag == 'NumThreads':
        self.runInfoDict['NumThreads'        ] = int(element.text)
      elif element.tag == 'totalNumCoresUsed':
        self.runInfoDict['totalNumCoresUsed' ] = int(element.text)
      elif element.tag == 'NumMPI':
        self.runInfoDict['NumMPI'            ] = int(element.text)
      elif element.tag == 'internalParallel':
        self.runInfoDict['internalParallel'  ] = utils.interpretBoolean(element.text)
      elif element.tag == 'batchSize':
        self.runInfoDict['batchSize'         ] = int(element.text)
      elif element.tag.lower() == 'maxqueuesize':
        self.runInfoDict['maxQueueSize'      ] = int(element.text)
      elif element.tag == 'MaxLogFileSize':
        self.runInfoDict['MaxLogFileSize'    ] = int(element.text)
      elif element.tag == 'precommand':
        self.runInfoDict['precommand'        ] = element.text
      elif element.tag == 'postcommand':
        self.runInfoDict['postcommand'       ] = element.text
      elif element.tag == 'deleteOutExtension':
        self.runInfoDict['deleteOutExtension'] = element.text.strip().split(',')
      elif element.tag == 'delSucLogFiles'    :
        if element.text.lower() in utils.stringsThatMeanTrue():
          self.runInfoDict['delSucLogFiles'    ] = True
        else:
          self.runInfoDict['delSucLogFiles'    ] = False
      elif element.tag == 'logfileBuffer':
        self.runInfoDict['logfileBuffer'] = utils.convertMultipleToBytes(element.text.lower())
      elif element.tag == 'clusterParameters':
        self.runInfoDict['clusterParameters'].extend(splitCommand(element.text)) #extend to allow adding parameters at different points.
      elif element.tag == 'mode'               :
        self.runInfoDict['mode'] = element.text.strip().lower()
        #parallel environment
        if self.runInfoDict['mode'] in self.__modeHandlerDict:
          self.__modeHandler = self.__modeHandlerDict[self.runInfoDict['mode']](self.messageHandler)
          self.__modeHandler.XMLread(element)
        else:
          self.raiseAnError(IOError,"Unknown mode "+self.runInfoDict['mode'])
      elif element.tag == 'expectedTime':
        self.runInfoDict['expectedTime'      ] = element.text.strip()
      elif element.tag == 'Sequence':
        for stepName in element.text.split(','):
          self.stepSequenceList.append(stepName.strip())
      elif element.tag == 'DefaultInputFile':
        self.runInfoDict['DefaultInputFile'] = element.text.strip()
      elif element.tag == 'CustomMode' :
        modeName = element.text.strip()
        modeClass = element.attrib["class"]
        modeFile = element.attrib["file"]
        #XXX This depends on if the working directory has been set yet.
        # So switching the order of WorkingDir and CustomMode can
        # cause different results.
        modeFile = modeFile.replace("%BASE_WORKING_DIR%",self.runInfoDict['WorkingDir'])
        modeFile = modeFile.replace("%FRAMEWORK_DIR%",self.runInfoDict['FrameworkDir'])
        modeDir, modeFilename = os.path.split(modeFile)
        if modeFilename.endswith(".py"):
          modeModulename = modeFilename[:-3]
        else:
          modeModulename = modeFilename
        os.sys.path.append(modeDir)
        module = __import__(modeModulename)
        if modeName in self.__modeHandlerDict:
          self.raiseAWarning("duplicate mode definition " + modeName)
        self.__modeHandlerDict[modeName] = module.__dict__[modeClass]
      else:
        self.raiseAnError(IOError,'RunInfo element "'+element.tag +'" unknown!')

  def printDicts(self):
    """
      utility function capable to print a summary of the dictionaries
      @ In, None
      @ Out, None
    """
    def __prntDict(Dict,msg):
      """utility function capable to print a dictionary"""
      for key in Dict:
        msg+=key+'= '+str(Dict[key])+'\n'
      return msg
    msg=''
    msg=__prntDict(self.runInfoDict,msg)
    msg=__prntDict(self.stepsDict,msg)
    msg=__prntDict(self.dataDict,msg)
    msg=__prntDict(self.samplersDict,msg)
    msg=__prntDict(self.modelsDict,msg)
    msg=__prntDict(self.metricsDict,msg)
    #msg=__prntDict(self.testsDict,msg)
    msg=__prntDict(self.filesDict,msg)
    msg=__prntDict(self.dataBasesDict,msg)
    msg=__prntDict(self.OutStreamManagerPlotDict,msg)
    msg=__prntDict(self.OutStreamManagerPrintDict,msg)
    msg=__prntDict(self.addWhatDict,msg)
    msg=__prntDict(self.whichDict,msg)
    self.raiseADebug(msg)

  def run(self):
    """
      Run the simulation
      @ In, None
      @ Out, None
    """
    #to do list
    #can we remove the check on the esistence of the file, it might make more sense just to check in case they are input and before the step they are used
    self.raiseADebug('entering the run')
    #controlling the PBS environment
    remoteRunCommand = self.__modeHandler.remoteRunCommand(dict(self.runInfoDict))
    if remoteRunCommand is not None:
      subprocess.call(args=remoteRunCommand["args"],
                      cwd=remoteRunCommand.get("cwd", None),
                      env=remoteRunCommand.get("env", None))
      return
    #loop over the steps of the simulation
    for stepName in self.stepSequenceList:
      stepInstance                     = self.stepsDict[stepName]   #retrieve the instance of the step
      self.raiseAMessage('-'*2+' Beginning step {0:50}'.format(stepName+' of type: '+stepInstance.type)+2*'-')#,color='green')
      self.runInfoDict['stepName']     = stepName                   #provide the name of the step to runInfoDict
      stepInputDict                    = {}                         #initialize the input dictionary for a step. Never use an old one!!!!!
      stepInputDict['Input' ]          = []                         #set the Input to an empty list
      stepInputDict['Output']          = []                         #set the Output to an empty list
      #fill the take a a step input dictionary just to recall: key= role played in the step b= Class, c= Type, d= user given name
      for [key,b,c,d] in stepInstance.parList:
        #Only for input and output we allow more than one object passed to the step, so for those we build a list
        if key == 'Input' or key == 'Output':
          if b == 'OutStreams':
            stepInputDict[key].append(self.whichDict[b][c][d])
          else:
            stepInputDict[key].append(self.whichDict[b][d])
        else:
          stepInputDict[key] = self.whichDict[b][d]
      #add the global objects
      stepInputDict['jobHandler'] = self.jobHandler
      #generate the needed assembler to send to the step
      for key in stepInputDict.keys():
        if type(stepInputDict[key]) == list:
          stepindict = stepInputDict[key]
        else:
          stepindict = [stepInputDict[key]]
        # check assembler. NB. If the assembler refers to an internal object the relative dictionary
        # needs to have the format {'internal':[(None,'variableName'),(None,'variable name')]}
        for stp in stepindict:
          self.generateAllAssemblers(stp)
      #if 'Sampler' in stepInputDict.keys(): stepInputDict['Sampler'].generateDistributions(self.distributionsDict)
      #running a step
      stepInstance.takeAstep(stepInputDict)
      #---------------here what is going on? Please add comments-----------------
      for output in stepInputDict['Output']:
        if self.FIXME:
          self.raiseAMessage('This is for the filter, it needs to go when the filtering strategy is done')
        if "finalize" in dir(output):
          output.finalize()
      self.raiseAMessage('-'*2+' End step {0:50} '.format(stepName+' of type: '+stepInstance.type)+2*'-'+'\n')#,color='green')
    self.jobHandler.shutdown()
    self.messageHandler.printWarnings()
    self.raiseAMessage('Run complete!',forcePrint=True)

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
        if mainClassStr not in self.whichDict.keys() and mainClassStr != 'internal':
          self.raiseAnError(IOError,'Main Class '+mainClassStr+' needed by '+stp.name + ' unknown!')
        neededobjs[mainClassStr] = {}
        for obj in neededObjects[mainClassStr]:
          if obj[1] in vars(self):
            neededobjs[mainClassStr][obj[1]] = vars(self)[obj[1]]
          elif obj[1] in self.whichDict[mainClassStr].keys():
            if obj[0]:
              if obj[0] not in self.whichDict[mainClassStr][obj[1]].type:
                self.raiseAnError(IOError,'Type of requested object '+obj[1]+' does not match the actual type!'+ obj[0] + ' != ' + self.whichDict[mainClassStr][obj[1]].type)
            neededobjs[mainClassStr][obj[1]] = self.whichDict[mainClassStr][obj[1]]
            self.generateAllAssemblers(neededobjs[mainClassStr][obj[1]])
          elif obj[1] in 'all':
            # if 'all' we get all the objects of a certain 'mainClassStr'
            for allObject in self.whichDict[mainClassStr]:
              neededobjs[mainClassStr][allObject] = self.whichDict[mainClassStr][allObject]
          else:
            self.raiseAnError(IOError,'Requested object '+obj[1]+' is not part of the Main Class '+mainClassStr + '!')
      objectInstance.generateAssembler(neededobjs)
