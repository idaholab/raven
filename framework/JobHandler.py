"""
Created on Mar 5, 2013

@author: alfoa, cogljj, crisr
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3-------------------------------------------

#External Modules---------------------------------------------------------------
import time
import collections
import subprocess
try               : import Queue as queue
except ImportError: import queue
import os
import signal
import copy
import sys
import abc
#import logging, logging.handlers
import threading
import random
import socket
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
import utils
from BaseClasses import BaseType
# for internal parallel
if sys.version_info.major == 2:
  import pp
  import ppserver
else:
  print("pp does not support python3")
# end internal parallel module
import MessageHandler
import Runners
#Internal Modules End-----------------------------------------------------------

class JobHandler(MessageHandler.MessageUser):
  """
    JobHandler class. This handles the execution of any job in the RAVEN
    framework
  """
  def __init__(self):
    """
      Init method
      @ In, None
      @ Out, None
    """
    self.printTag                     = 'Job Handler'
    self.runInfoDict                  = {}
    self.mpiCommand                   = ''
    self.threadingCommand             = ''
    self.initParallelPython           = False

    ## The following variables are protected by the __queueLock
    self.__running                    = []
    self.__queue                      = collections.deque()
    self.__clientQueue                = collections.deque()
    self.__clientRunning              = collections.deque()
    self.__nextId                     = 0
    ## End block of __queueLock protected variables

    self.__queueLock                  = threading.RLock()

    ## List of submitted job identifiers
    self.__submittedJobs              = []
    ## Dict of failed jobs of the form { identifer: metadata }
    self.__failedJobs                 = {}

    #self.__noResourcesJobs            = []

  def initialize(self,runInfoDict,messageHandler):
    """
      Method to initialize the JobHandler
      @ In, runInfoDict, dict, dictionary of run info settings
      @ In, messageHandler, MessageHandler object, instance of the global RAVEN
        message handler
      @ Out, None
    """
    self.runInfoDict = runInfoDict
    self.messageHandler = messageHandler
    if self.runInfoDict['NumMPI'] != 1 and len(self.runInfoDict['ParallelCommand']) > 0:
      self.mpiCommand = self.runInfoDict['ParallelCommand'] + ' ' + str(self.runInfoDict['NumMPI'])

    if self.runInfoDict['NumThreads'] != 1 and len(self.runInfoDict['ThreadingCommand']) > 0:
      self.threadingCommand = self.runInfoDict['ThreadingCommand'] + ' ' + str(self.runInfoDict['NumThreads'])

    #initialize PBS
    with self.__queueLock:
      self.__running       = [None]*self.runInfoDict['batchSize']
      self.__clientRunning = [None]*self.runInfoDict['batchSize']

  def __initializeParallelPython(self):
    """
      Internal method that is aimed to initialize the internal parallel system.
      It initilizes the paralle python implementation (with socketing system) in
      case RAVEN is run in a cluster with multiple nodes or the NumMPI > 1,
      otherwise multi-threading is used.
      @ In, None
      @ Out, None
    """
    ## Check if the list of unique nodes is present and, in case, initialize the
    ## socket
    if self.runInfoDict['internalParallel']:
      if len(self.runInfoDict['Nodes']) > 0:
        availableNodes = [nodeId.strip() for nodeId in self.runInfoDict['Nodes']]

        ## Set the initial port randomly among the user accessible ones
        ## Is there any problem if we select the same port as something else?
        randomPort = random.randint(1024,65535)

        ## Get localHost and servers
        localHostName, ppservers = self.__runRemoteListeningSockets(randomPort)
        self.raiseADebug("Local host is "+ localHostName)

        if len(ppservers) == 0:
          ## We are on a single node
          self.ppserver = pp.Server(ncpus=len(availableNodes))
        else:
          ## We are using multiple nodes
          self.raiseADebug("Servers found are " + ','.join(ppservers))
          self.raiseADebug("Server port in use is " + str(randomPort))
          self.ppserver = pp.Server(ncpus=0, ppservers=tuple(ppservers))
      else:
         ## We are using the parallel python system
        self.ppserver = pp.Server(ncpus=int(self.runInfoDict['totalNumCoresUsed']))
    else:
      ## We are just using threading
      self.ppserver = None

    self.initParallelPython = True

  def __getLocalAndRemoteMachineNames(self):
    """
      Method to get the qualified host and remote nodes' names
      @ In, None
      @ Out, hostNameMapping, dict, dictionary containing the qualified names
        {'local':hostName,'remote':{nodeName1:IP1,nodeName2:IP2,etc}}
    """
    hostNameMapping = {'local':"",'remote':{}}

    ## Store the local machine name as its fully-qualified domain name (FQDN)
    hostNameMapping['local'] = str(socket.getfqdn()).strip()
    self.raiseADebug("Local Host is " + hostNameMapping['local'])

    ## collect the qualified hostnames for each remote node
    for nodeId in list(set(self.runInfoDict['Nodes'])):
      hostNameMapping['remote'][nodeId.strip()] = socket.gethostbyname(nodeId.strip())
      self.raiseADebug("Remote Host identified " + hostNameMapping['remote'][nodeId.strip()])

    return hostNameMapping

  def __runRemoteListeningSockets(self,newPort):
    """
      Method to activate the remote sockets for parallel python
      @ In, newPort, integer, the comunication port to use
      @ Out, (qualifiedHostName, ppservers), tuple, tuple containining:
             - in position 0 the host name and
             - in position 1 the list containing the nodes in which the remote
               sockets have been activated
    """
    ## Get the local machine name and the remote nodes one
    hostNameMapping = self.__getLocalAndRemoteMachineNames()
    qualifiedHostName =  hostNameMapping['local']
    remoteNodesIP = hostNameMapping['remote']

    ## Strip out the nodes' names
    availableNodes = [node.strip() for node in self.runInfoDict['Nodes']]

    ## Get unique nodes
    uniqueNodes    = list(set(availableNodes))
    ppservers      = []

    if len(uniqueNodes) > 1:
      ## There are remote nodes that need to be activated

      ## Locate the ppserver script to be executed
      ppserverScript = os.path.join(self.runInfoDict['FrameworkDir'],"contrib","pp","ppserver.py")

      ## Modify the python path used by the local environment
      localenv = os.environ.copy()
      pathSeparator = os.pathsep
      localenv["PYTHONPATH"] = pathSeparator.join(sys.path)

      for nodeId in uniqueNodes:
        ## Build the filename
        outFileName = nodeId.strip()+"_port:"+str(newPort)+"_server_out.log"
        outFileName = os.path.join(self.runInfoDict['WorkingDir'], outFileName)

        outFile = open(outFileName, 'w')

        ## Check how many processors are available in the node
        ntasks = availableNodes.count(nodeId)
        remoteHostName =  remoteNodesIP[nodeId]

        ## Activate the remote socketing system

        ## Next line is a direct execute of a ppserver:
        #subprocess.Popen(['ssh', nodeId, "python2.7", ppserverScript,"-w",str(ntasks),"-i",remoteHostName,"-p",str(newPort),"-t","1000","-g",localenv["PYTHONPATH"],"-d"],shell=False,stdout=outFile,stderr=outFile,env=localenv)

        ## Instead, let's build the command and then call the os-agnostic version
        command=" ".join(["python",ppserverScript,"-w",str(ntasks),"-i",remoteHostName,"-p",str(newPort),"-t","1000","-g",localenv["PYTHONPATH"],"-d"])
        utils.pickleSafeSubprocessPopen(['ssh',nodeId,"COMMAND='"+command+"'",self.runInfoDict['RemoteRunCommand']],shell=False,stdout=outFile,stderr=outFile,env=localenv)
        ## e.g., ssh nodeId COMMAND='python ppserverScript -w stuff'

        ## update list of servers
        ppservers.append(nodeId+":"+str(newPort))

    return qualifiedHostName, ppservers

  def addExternal(self,executeCommands,outputFile,workingDir,identifier = None, metadata=None,codePointer=None,uniqueHandler="any"):
    """
      Method to add an external runner (an external code) in the handler list
      @ In, executeCommands, list of tuple(string), ('parallel'/'serial',
        <execution command>)
      @ In, outputFile, string, output file name
      @ In, workingDir, string, working directory
      @ In, identifier, string, optional, the job identifier
      @ In, metadata, dict, optional, dictionary of metadata
      @ In, codePointer, derived CodeInterfaceBaseClass object, optional,
        pointer to code interface
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ Out, None
    """
    ##FIXME what uses this?  Still precommand for whole line if multiapp case?
    precommand = self.runInfoDict['precommand']
    ## It appears precommand is usually used for mpiexec - however, there could
    ## be other uses....

    commands=[]
    for runtype,cmd in executeCommands:
      newCommand=''
      if runtype.lower() == 'parallel':
        newCommand += precommand
        if self.mpiCommand !='':
          newCommand += ' '+self.mpiCommand+' '
        ##FIXME are these two exclusive?
        if self.threadingCommand !='':
          newCommand += ' '+ self.threadingCommand +' '
        newCommand += cmd+' '
        newCommand += self.runInfoDict['postcommand']
        commands.append(newCommand)
      elif runtype.lower() == 'serial':
        commands.append(cmd)
      else:
        self.raiseAnError(IOError,'For execution command <'+cmd+'> the run type was neither "serial" nor "parallel"!  Instead received: ',runtype,'\nPlease check the code interface.')

    command = ' && '.join(commands)+' '

    with self.__queueLock:
      runner = Runners.ExternalRunner(self.messageHandler, command,workingDir,
                                      self.runInfoDict['logfileBuffer'],
                                      identifier, outputFile, metadata,
                                      codePointer, uniqueHandler)
      self.__queue.append(runner)

    self.raiseAMessage('Execution command submitted:',command)
    self.__submittedJobs.append(identifier)
    self.addRuns()
    #if self.howManyFreeSpots()>0: self.addRuns()

  def addInternal(self,Input,functionToRun,identifier,metadata=None, modulesToImport = [], forceUseThreads = False, uniqueHandler="any",clientQueue = False):
    """
      Method to add an internal run (function execution)
      @ In, Input, list, list of Inputs that are going to be passed to the
        function to be executed as *args
      @ In, functionToRun,function or method, the function that needs to be
        executed
      @ In, identifier, string, the job identifier
      @ In, metadata, dict, optional, dictionary of metadata associated to this
        run
      @ In, modulesToImport, list, optional, list of modules that need to be
        imported for internal parallelization (parallel python). This list
        should be generated with the method returnImportModuleString in utils.py
      @ In, forceUseThreads, bool, optional, flag that, if True, is going to
        force the usage of multi-threading even if parallel python is activated
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ In, clientQueue, boolean, optional, if this run needs to be added in the
        clientQueue
      @ Out, None
    """
    ## internal server is initialized only in case an internal calc is requested
    if not self.initParallelPython:
      self.__initializeParallelPython()

    skipFunctions = [utils.metaclass_insert(abc.ABCMeta,BaseType)]
    if self.ppserver is None or forceUseThreads:
      internalJob = Runners.SharedMemoryRunner(self.messageHandler, Input,
                                               functionToRun, modulesToImport,
                                               identifier, metadata,
                                               skipFunctions, uniqueHandler)
    else:
      internalJob = Runners.DistributedMemoryRunner(self.messageHandler,
                                                    self.ppserver, Input,
                                                    functionToRun,
                                                    modulesToImport, identifier,
                                                    metadata, skipFunctions,
                                                    uniqueHandler)
    with self.__queueLock:
      if not clientQueue:
        self.__queue.append(internalJob)
      else:
        self.__clientQueue.append(internalJob)

    self.__submittedJobs.append(identifier)
    self.addRuns()
    #if self.howManyFreeSpots()>0: self.addRuns()

  def addInternalClient(self,Input,functionToRun,identifier,metadata=None, uniqueHandler="any"):
    """
      Method to add an internal run (function execution), without consuming
      resources (free spots). This can be used for client handling (see
      metamodel)
      @ In, Input, list, list of Inputs that are going to be passed to the
        function to be executed as *args
      @ In, functionToRun,function or method, the function that needs to be
        executed
      @ In, identifier, string, the job identifier
      @ In, metadata, dict, optional, dictionary of metadata associated to this
        run
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner.
      @ Out, None
    """
    #self.__clientQueue.append()
    #self.__running.append(None)
    self.addInternal(Input, functionToRun, identifier, metadata,
                     forceUseThreads = True, uniqueHandler = uniqueHandler,
                     clientQueue = True)
    #self.__noResourcesJobs.append(identifier)

  def isFinished(self):
    """
      Method to check if all the runs in the queue are finished
      @ In, None
      @ Out, isFinished, bool, True all the runs in the queue are finished
    """
    with self.__queueLock:
      if not len(self.__queue) == 0:
        return False

      for i in range(len(self.__running)):
        if self.__running[i] and not self.__running[i].isDone():
          return False

      for elem in self.__clientRunning:
        if elem and not elem.isDone():
          return False

    return True

  def isThisJobFinished(self, identifier):
    """
      Method to check if the run identified by "identifier" is finished
      @ In, identifier, string, identifier
      @ Out, isFinished, bool, True if the job identified by "identifier" is
        finished
    """
    isFinished = None
    with self.__queueLock:
      for i in range(len(self.__running)):
        if self.__running[i] is not None and self.__running[i].identifier.strip() == identifier.strip():
          isFinished = self.__running[i].isDone()
          break

      if isFinished is None:
        for elem in self.__clientRunning:
          if elem is not None:
            if elem.identifier.strip() == identifier.strip():
              isFinished = elem.isDone()
              break

    if isFinished is None:
      self.raiseAnError(RuntimeError,"Job "+identifier+" is unknown!")

    return isFinished

  def getNumberOfFailures(self):
    """
      Method to get the number of executions that failed
      @ In, None
      @ Out, len(self.__failedJobs), int, number of failures
    """
    return len(self.__failedJobs)

  def getListOfFailedJobs(self):
    """
      Method to get list of failed jobs
      @ In, None
      @ Out, __failedJobs, list, list of the identifiers (jobs) that failed
    """
    return self.__failedJobs

  def howManyFreeSpots(self):
    """
      Method to get the number of free spots in the running queue
      @ In, None
      @ Out, cntFreeSpots, int, number of free spots
    """
    cntFreeSpots = 0
    #if len(self.__queue) == 0:
    with self.__queueLock:
      for i in range(len(self.__running)):
        ## Why this if conditional?
        if self.__running[i] is not None and self.__running[i].isDone():
          cntFreeSpots += 1
        else:
          cntFreeSpots += 1
    #cntFreeSpots-=len(self.__queue)
    return cntFreeSpots

  def howManyFreeSpotsForClients(self):
    """
      Method to get the number of free spots in the client queue (same size of
      __running queue)
      @ In, None
      @ Out, cntFreeSpots, int, number of free spots
    """
    cntFreeSpots = 0
    with self.__queueLock:
      for i in range(len(self.__clientRunning)):
        if self.__clientRunning[i] and self.__clientRunning[i].isDone():
          cntFreeSpots += 1
    return cntFreeSpots

  def getFinished(self, removeFinished=True, jobIdentifier = '', uniqueHandler = "any"):
    """
      Method to get the list of jobs that ended (list of objects)
      @ In, removeFinished, bool, optional, flag to control if the finished jobs
        need to be removed from the queue
      @ In, jobIdentifier, string, optional, if specified, only collects
        finished runs with a particular jobIdentifier.
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        each runner. If provided, just the jobs that have the uniqueIdentifier
        will be retrieved. By default uniqueHandler = 'any' => all the jobs for
        which no uniqueIdentifier has been set up are going to be retrieved
      @ Out, finished, list, list of finished jobs (InternalRunner or
        ExternalRunner objects) (if jobIdentifier is None), else the finished
        job identified by jobIdentifier
    """
    finished             = []

    with self.__queueLock:
      ## The code handling these two lists was the exact same, I have taken the
      ## liberty of condensing these loops into one and removing some of the
      ## redundant checks to make this code a bit simpler.
      for runList in [self.__running, self.__clientRunning]:
        runsToBeRemoved = []
        for i,run in enumerate(runList):
          if run is not None and run.isDone():
            ## If the jobIdentifier does not match or the uniqueHandler does not
            ## match, then don't bother trying to do anything with it
            if not run.identifier.startswith(jobIdentifier) \
            or uniqueHandler != run.uniqueHandler:
              continue

            finished.append(run)
            if removeFinished:
              self.__checkAndRemoveFinished(run)
              runsToBeRemoved.append(i)

        for i in runsToBeRemoved:
          runList[i] = None

      self.addRuns()
    #end with self.__queueLock
    return finished

  def __checkAndRemoveFinished(self, running):
    """
      Method to check if a run is finished and remove it from the queque
      @ In, running, instance, the job instance (InternalRunner or ExternalRunner)
      @ Out, None
    """
    with self.__queueLock:
      returnCode = running.getReturnCode()
      if returnCode != 0:
        self.raiseAMessage(" Process Failed "+str(running)+' '+str(running.command)+" returnCode "+str(returnCode))
        self.__failedJobs[running.identifier]=(returnCode,copy.deepcopy(running.getMetadata()))

        if isinstance(running, Runners.ExternalRunner):
          outputFilename = running.getOutputFilename()
          if os.path.exists(outputFilename):
            self.raiseAMessage(open(outputFilename,"r").read())
          else:
            self.raiseAMessage(" No output "+outputFilename)
      else:
        ## The following code performs any user-specified file cleanup and only
        ## applies to ExternalRunners, since InternalRunners do not generate
        ## any of their own output currently (Subject to change).
        if isinstance(running, Runners.ExternalRunner):
          ## Check if the user specified to delete the log files of successful
          ## runs
          if self.runInfoDict['delSucLogFiles']:
            self.raiseAMessage(' Run "' +running.identifier+'" ended smoothly, removing log file!')
            if os.path.exists(running.getOutputFilename()):
              os.remove(running.getOutputFilename())

          ## Check if the user specified any file extensions for clean up
          for fileExt in self.runInfoDict['deleteOutExtension']:
            if not fileExt.startswith("."):
              fileExt = "." + fileExt

            fileList = [ f for f in os.listdir(running.getWorkingDir()) if f.endswith(fileExt) ]

            for f in fileList:
              os.remove(f)

  def addRuns(self):
    """
      Method to start running the jobs in queue.  If there are empty slots
      takes jobs out of the queue and starts running them.
      @ In, None
      @ Out, None
    """
    with self.__queueLock:
      if self.howManyFreeSpots() > 0:
        for i in range(len(self.__running)):
          if self.__running[i] == None and len(self.__queue) > 0:
            item = self.__queue.popleft()
            if isinstance(running, Runners.ExternalRunner):
              command = item.command
              command = command.replace("%INDEX%",str(i))
              command = command.replace("%INDEX1%",str(i+1))
              command = command.replace("%CURRENT_ID%",str(self.__nextId))
              command = command.replace("%CURRENT_ID1%",str(self.__nextId+1))
              command = command.replace("%SCRIPT_DIR%",self.runInfoDict['ScriptDir'])
              command = command.replace("%FRAMEWORK_DIR%",self.runInfoDict['FrameworkDir'])
              command = command.replace("%WORKING_DIR%",item.getWorkingDir())
              command = command.replace("%BASE_WORKING_DIR%",self.runInfoDict['WorkingDir'])
              command = command.replace("%METHOD%",os.environ.get("METHOD","opt"))
              command = command.replace("%NUM_CPUS%",str(self.runInfoDict['NumThreads']))
              item.command = command
            self.__running[i] = item
            self.__running[i].start() #FIXME this call is really expensive; can it be reduced?
            self.__nextId += 1
    with self.__queueLock:
      if self.__clientRunning.count(None) > 0 and len(self.__clientQueue) !=0:
        for i in range(len(self.__clientRunning)):
          if len(self.__clientQueue) == 0:
            break
          if self.__clientRunning[i] is None:
            self.__clientRunning[i] = self.__clientQueue.popleft()
            self.__clientRunning[i].start()
            self.__nextId += 1


  def getFinishedNoPop(self):
    """
      Method to get the list of jobs that ended (list of objects) without removing them from the queue
      @ In, None
      @ Out, finished, list, list of finished jobs (InternalRunner or ExternalRunner objects)
    """
    finished = self.getFinished(False)
    return finished

  def getNumSubmitted(self):
    """
      Method to get the number of submitted jobs
      @ In, None
      @ Out, len(self.__submittedJobs), int, number of submitted jobs
    """
    return len(self.__submittedJobs)

  def startingNewStep(self):
    """
      Method to reset the __submittedJobs to an empty list.
      @ In, None
      @ Out, None
    """
    self.__submittedJobs = []

  def terminateAll(self):
    """
      Method to clear out the queue by killing all running processes.
      @ In, None
      @ Out, None
    """
    with self.__queueLock:
      while not len(self.__queue) == 0        : self.__queue.popleft()
      while not len(self.__clientQueue) == 0  : self.__clientQueue.popleft()
      for i in range(len(self.__running))     :
        if self.__running[i] is not None: self.__running[i].kill()
      if self.__clientRunning.count(None) != len(self.__clientRunning):
        for i in range(len(self.__clientRunning))     :
          if self.__running[i] is not None: self.__clientRunning[i].kill()

  def numRunning(self):
    """
      Returns the number of runs currently running.
      @ In, None
      @ Out, activeRuns, int, number of active runs
    """
    activeRuns = sum(run is not None for run in self.__running)
    return activeRuns
