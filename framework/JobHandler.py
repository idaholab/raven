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

import os
import copy
import sys
import abc
#import logging, logging.handlers
import threading
import random
import socket
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from utils import utils
from BaseClasses import BaseType
import MessageHandler
import Runners
# for internal parallel
if sys.version_info.major == 2:
  import pp
  import ppserver
else:
  print("pp does not support python3")
# end internal parallel module
#Internal Modules End-----------------------------------------------------------


## FIXME: Finished jobs can bog down the queue waiting for other objects to take
## them away. Can we shove them onto a different list and free up the job queue?

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
    self.printTag         = 'Job Handler'
    self.runInfoDict      = {}

    ## Is it one or the other?
    self.mpiCommand       = ''
    self.threadingCommand = ''

    self.isParallelPythonInitialized = False

    self.sleepTime  = 0.005
    self.completed = False

    ## Stops the pending queue from getting too big. TODO: expose this to the
    ## user
    self.maxQueueSize = 1000

    ############################################################################
    ## The following variables are protected by the __queueLock

    ## Placeholders for each actively running job. When a job finishes, its
    ## spot in one of these lists will be reset to None and the next Runner will
    ## be placed in a free None spot, and set to start
    self.__running       = []
    self.__clientRunning = []

    ## Queue of jobs to be run, when something on the list above opens up, the
    ## corresponding queue will pop a job (Runner) and put it into that location
    ## and set it to start
    self.__queue       = collections.deque()
    self.__clientQueue = collections.deque()

    ## A counter used for uniquely identifying the next id for an ExternalRunner
    ## InternalRunners will increment this counter, but do not use it currently
    self.__nextId = 0

    ## List of finished jobs. When a job finishes, it is placed here until
    ## something from the main thread can remove them.
    self.__finished = []

    ## End block of __queueLock protected variables
    ############################################################################

    self.__queueLock = threading.RLock()

    ## List of submitted job identifiers, includes jobs that have completed as
    ## this list is not cleared until a new step is entered
    self.__submittedJobs = []
    ## Dict of failed jobs of the form { identifer: metadata }
    self.__failedJobs = {}

    #self.__noResourcesJobs = []

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
            self.raiseAMessage(repr(open(outputFilename,"r").read()).replace("\\n","\n"))
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

    self.isParallelPythonInitialized = True

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

  def startLoop(self):
    """
    This function begins the polling loop for the JobHandler where it will
    constantly fill up its running queue with jobs in its pending queue and
    unload finished jobs into its finished queue to be extracted by
    """
    while not self.completed:
      self.addRuns()
      self.cleanRuns()
      ## TODO May want to revisit this:
      ## http://stackoverflow.com/questions/29082268/python-time-sleep-vs-event-wait
      time.sleep(self.sleepTime)

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
      runner = Runners.ExternalRunner(self.messageHandler, command, workingDir,
                                      self.runInfoDict['logfileBuffer'],
                                      identifier, outputFile, metadata,
                                      codePointer, uniqueHandler)
      self.__queue.append(runner)
      self.__submittedJobs.append(identifier)

    self.raiseAMessage('Execution command submitted:',command)

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
    if not self.isParallelPythonInitialized:
      self.__initializeParallelPython()

    if self.ppserver is None or forceUseThreads:
      internalJob = Runners.SharedMemoryRunner(self.messageHandler, Input,
                                               functionToRun, identifier,
                                               metadata, uniqueHandler)
    else:
      skipFunctions = [utils.metaclass_insert(abc.ABCMeta,BaseType)]
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
    self.addInternal(Input, functionToRun, identifier, metadata,
                     forceUseThreads = True, uniqueHandler = uniqueHandler,
                     clientQueue = True)

  def isFinished(self):
    """
      Method to check if all the runs in the queue are finished
      @ In, None
      @ Out, isFinished, bool, True all the runs in the queue are finished
    """
    with self.__queueLock:
      ## If there is still something left in the queue, we are not done yet.
      if len(self.__queue) > 0 or len(self.__clientQueue) > 0:
        return False

      ## Otherwise, let's look at our running lists and see if there is a job
      ## that is not done.
      for run in self.__running+self.__clientRunning:
        if run:
          return False

    ## Are there runs that need to be claimed? If so, then I cannot say I am
    ## done.
    if len(self.getFinishedNoPop()) > 0:
      return False

    return True

  def availability(self, client=False):
    """
    Returns the number of runs that can be added until we consider our queue
    saturated
    @ In, client, bool, if true, then return the values for the
    __clientQueue, otherwise use __queue
    @ Out, availability, int the number of runs that can be added until we
      reach saturation
    """
    if client:
      availability = self.maxQueueSize - len(self.__clientQueue)
    else:
      availability = self.maxQueueSize - len(self.__queue)
    return availability

  def isThisJobFinished(self, identifier):
    """
      Method to check if the run identified by "identifier" is finished
      @ In, identifier, string, identifier
      @ Out, isFinished, bool, True if the job identified by "identifier" is
        finished
    """
    identifier = identifier.strip()
    with self.__queueLock:
      ## Look through the finished jobs and attempt to find a matching
      ## identifier. If the job exists here, it is finished
      for run in self.__finished:
        if run.identifier == identifier:
          return True

      ## Look through the pending jobs and attempt to find a matching identifier
      ## If the job exists here, it is not finished
      for queue in [self.__queue, self.__clientQueue]:
        for run in queue:
          if run.identifier == identifier:
            return False

      ## Look through the running jobs and attempt to find a matching identifier
      ## If the job exists here, it is not finished
      for run in self.__running+self.__clientRunning:
        if run is not None and run.identifier == identifier:
          return False

    ##  If you made it here and we still have not found anything, we have got
    ## problems.
    self.raiseAnError(RuntimeError,"Job "+identifier+" is unknown!")

  def getFailedJobs(self):
    """
      Method to get list of failed jobs
      @ In, None
      @ Out, __failedJobs, list, list of the identifiers (jobs) that failed
    """
    return self.__failedJobs

  def getFinished(self, removeFinished=True, jobIdentifier = '', uniqueHandler = "any"):
    """
      Method to get the list of jobs that ended (list of objects)
      @ In, removeFinished, bool, optional, flag to control if the finished jobs
        need to be removed from the queue
      @ In, jobIdentifier, string, optional, if specified, only collects
        finished runs that start with this text. If not specified collect all.
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        each runner. If provided, just the jobs that have the uniqueIdentifier
        will be retrieved. By default uniqueHandler = 'any' => all the jobs for
        which no uniqueIdentifier has been set up are going to be retrieved
      @ Out, finished, list, list of finished jobs (InternalRunner or
        ExternalRunner objects) (if jobIdentifier is None), else the finished
        jobs matching the base case jobIdentifier
    """
    finished = []

    ## If the user does not specify a jobIdentifier, then set it to the empty
    ## string because every job will match this starting string.
    if jobIdentifier is None:
      jobIdentifier = ''

    with self.__queueLock:
      runsToBeRemoved = []
      for i,run in enumerate(self.__finished):
        ## If the jobIdentifier does not match or the uniqueHandler does not
        ## match, then don't bother trying to do anything with it
        if not run.identifier.startswith(jobIdentifier) \
        or uniqueHandler != run.uniqueHandler:
          continue

        finished.append(run)
        if removeFinished:
          runsToBeRemoved.append(i)
          self.__checkAndRemoveFinished(run)

      ##Since these indices are sorted, reverse them to ensure that when we
      ## delete something it will not shift anything to the left (lower index)
      ## than it.
      for i in reversed(runsToBeRemoved):
        del self.__finished[i]
    ## end with self.__queueLock

    return finished

  def getFinishedNoPop(self):
    """
      Method to get the list of jobs that ended (list of objects) without
      removing them from the queue
      @ In, None
      @ Out, finished, list, list of finished jobs (InternalRunner or
        ExternalRunner objects)
    """
    finished = self.getFinished(False)
    return finished

  def numFreeSpots(self, client=False):
    """
      Method to get the number of free spots in one of the running queues
      @ In, client, bool, if true, then return the values for the
        __clientRunning list, otherwise use __running
      @ Out, cntFreeSpots, int, number of free spots
    """
    cntFreeSpots = 0
    with self.__queueLock:
      ## The process is the same for both lists, so let's establish which one
      ## we are working on and then get to business.
      if client:
        runList = self.__clientRunning
        # queue = self.__clientQueue
      else:
        runList = self.__running
        # queue = self.__queue

      for run in runList:
        if run is None:
          cntFreeSpots += 1

    return cntFreeSpots

  def numRunning(self):
    """
      Returns the number of runs currently running.
      @ In, None
      @ Out, activeRuns, int, number of active runs
    """
    with self.__queueLock:
      activeRuns = sum(run is not None for run in self.__running)
    return activeRuns

  def numSubmitted(self):
    """
      Method to get the number of submitted jobs
      @ In, None
      @ Out, len(self.__submittedJobs), int, number of submitted jobs
    """
    return len(self.__submittedJobs)

  def addRuns(self):
    """
      Method to start running the jobs in queue.  If there are empty slots
      takes jobs out of the queue and starts running them.
      @ In, None
      @ Out, None
    """
    with self.__queueLock:
      emptySlots = [i for i,run in enumerate(self.__running) if run is None]
      for i in emptySlots:
        if len(self.__queue) > 0:
          item = self.__queue.popleft()
          if isinstance(item, Runners.ExternalRunner):
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
          ##FIXME this call is really expensive; can it be reduced?
          self.__running[i].start()
          self.__nextId += 1
        else:
          break

    with self.__queueLock:
      emptySlots = [i for i,run in enumerate(self.__clientRunning) if run is None]
      for i in emptySlots:
        if len(self.__clientQueue) > 0:
          self.__clientRunning[i] = self.__clientQueue.popleft()
          self.__clientRunning[i].start()
          self.__nextId += 1
        else:
          break

  def cleanRuns(self):
    """
    Method that will remove finished jobs from the queue and place them into the
    finished queue to be read by some other thread.
    @ In, None
    @ Out, None
    """
    with self.__queueLock:
      ## The code handling these two lists was the exact same, I have taken the
      ## liberty of condensing these loops into one and removing some of the
      ## redundant checks to make this code a bit simpler.
      for runList in [self.__running, self.__clientRunning]:
        for i,run in enumerate(runList):
          if run is not None and run.isDone():
            self.__finished.append(run)
            runList[i] = None

  def startingNewStep(self):
    """
      Method to reset the __submittedJobs to an empty list.
      @ In, None
      @ Out, None
    """
    with self.__queueLock:
      self.__submittedJobs = []

  def shutdown(self):
    """
    This function will mark the job handler as done, so it can shutdown its
    polling thread.
    @ In, None
    @ Out, None
    """
    self.completed = True

  def terminateAll(self):
    """
      Method to clear out the queue by killing all running processes.
      @ In, None
      @ Out, None
    """
    with self.__queueLock:
      for queue in [self.__queue, self.__clientQueue]:
        queue.clear()

      for runList in [self.__running, self.__clientRunning]:
        unfinishedRuns = [run for run in runList if run is not None]
        for run in unfinishedRuns:
          run.kill()
