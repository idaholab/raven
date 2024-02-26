
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
import time
import collections
import os
import copy
import sys
import threading
from random import randint
import socket
import re

from .utils import importerUtils as im
from .utils import utils
from .utils.utils import ParallelLibEnum
from .BaseClasses import BaseType
from . import Runners
from . import Models
# for internal parallel
_rayAvail = im.isLibAvail("ray")
_daskAvail = im.isLibAvail("dask")
if _daskAvail:
  import dask
  import dask.distributed
if _rayAvail:
  import ray

# end internal parallel module
# Internal Modules End-----------------------------------------------------------

# FIXME: Finished jobs can bog down the queue waiting for other objects to take
# them away. Can we shove them onto a different list and free up the job queue?

class JobHandler(BaseType):
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
    super().__init__()
    self.printTag = 'Job Handler' # Print tag of this object
    self.runInfoDict = {}         # Container of the running info (RunInfo block in the input file)
    self.__isDistributedInitialized = False # Is Ray or Dask Initialized?
    self._server = None         # Variable containing the info about the RAY or DASK parallel server.
                                  # If None, multi-threading is used
    self.sleepTime = 1e-4         # Sleep time for collecting/inquiring/submitting new jobs
    self.completed = False        # Is the execution completed? When True, the JobHandler is shut down
    self.__profileJobs = False    # Determines whether to collect and print job timing summaries at the end of job runs.
    self.maxQueueSize = None      # Prevents the pending queue from growing indefinitely, but also
                                  # allowing extra jobs to be queued to prevent starving
                                  # parallelized environments of jobs.

    ############################################################################
    # The following variables are protected by the __queueLock

    # Placeholders for each actively running job. When a job finishes, its
    # spot in one of these lists will be reset to None and the next Runner will
    # be placed in a free None spot, and set to start
    self.__running       = []
    self.__clientRunning = []

    # Queue of jobs to be run, when something on the list above opens up, the
    # corresponding queue will pop a job (Runner) and put it into that location
    # and set it to start
    self.__queue       = collections.deque()
    self.__clientQueue = collections.deque()

    # A counter used for uniquely identifying the next id for an ExternalRunner
    # InternalRunners will increment this counter, but do not use it currently
    self.__nextId = 0

    # List of finished jobs. When a job finishes, it is placed here until
    # something from the main thread can remove them.
    self.__finished = []

    # End block of __queueLock protected variables
    ############################################################################

    self.__queueLock = threading.RLock()
    # List of submitted job identifiers, includes jobs that have completed as
    # this list is not cleared until a new step is entered
    self.__submittedJobs = []
    # Dict of failed jobs of the form { identifier: metadata }
    self.__failedJobs = {}
    # Dict containing info about batching
    self.__batching = collections.defaultdict()
    self.rayInstanciatedOutside = None
    self.daskInstanciatedOutside = None
    self.remoteServers = None
    self.daskSchedulerFile = None
    self._daskScheduler = None

  def __getstate__(self):
    """
      This function return the state of the JobHandler
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    state = copy.copy(self.__dict__)
    state.pop('_JobHandler__queueLock')
    #XXX we probably need to record how this was init, and store that
    # such as the scheduler file
    if self._parallelLib == ParallelLibEnum.dask and '_server' in state:
      state.pop('_server')
    return state

  def __setstate__(self, d):
    """
      Initialize the JobHandler with the data contained in newstate
      @ In, d, dict, it contains all the information needed by the JobHandler to be initialized
      @ Out, None
    """
    self.__dict__.update(d)
    self.__queueLock = threading.RLock()

  def applyRunInfo(self, runInfo):
    """
      Allows access to the RunInfo data
      @ In, runInfo, dict, info from RunInfo
      @ Out, None
    """
    self.runInfoDict = runInfo

  def initialize(self):
    """
      Method to initialize the JobHandler
      @ In, None
      @ Out, None
    """
    # set the maximum queue size (number of jobs to queue past the running number)
    self.maxQueueSize = self.runInfoDict['maxQueueSize']
    # defaults to None; if None, then use batchSize instead
    if self.maxQueueSize is None:
      self.maxQueueSize = self.runInfoDict['batchSize']
    # if requested max size less than 1, we can't do that, so take 1 instead
    if self.maxQueueSize < 1:
      self.raiseAWarning('maxQueueSize was set to be less than 1!  Setting to 1...')
      self.maxQueueSize = 1
    self.raiseADebug('Setting maxQueueSize to', self.maxQueueSize)

    # initialize PBS
    with self.__queueLock:
      self.__running       = [None]*self.runInfoDict['batchSize']
      self.__clientRunning = [None]*self.runInfoDict['batchSize']
    self._parallelLib = ParallelLibEnum.shared
    if self.runInfoDict['parallelMethod'] is not None and self.runInfoDict['parallelMethod'] != ParallelLibEnum.distributed:
      self._parallelLib = self.runInfoDict['parallelMethod']
    elif self.runInfoDict['internalParallel'] or \
         self.runInfoDict['parallelMethod'] is not None and self.runInfoDict['parallelMethod'] == ParallelLibEnum.distributed:
      #If ParallelLibEnum.distributed or internalParallel True
      # than choose a library automatically.
      if _daskAvail:
        self._parallelLib = ParallelLibEnum.dask
      elif _rayAvail:
        self._parallelLib = ParallelLibEnum.ray
      else:
        self.raiseAWarning("Distributed Running requested but no parallel method found")
        self._parallelLib = ParallelLibEnum.shared
    desiredParallelMethod = f"parallelMethod: {self.runInfoDict['parallelMethod']} internalParallel: {self.runInfoDict['internalParallel']}"
    self.raiseADebug(f"Using parallelMethod: {self._parallelLib} because Input: {desiredParallelMethod} and Ray Availablility: {_rayAvail} and Dask Availabilitiy: {_daskAvail}")
    if self._parallelLib == ParallelLibEnum.dask and not _daskAvail:
      self.raiseAnError(RuntimeError, f"dask requested but not available. {desiredParallelMethod}")
    if self._parallelLib == ParallelLibEnum.ray and not _rayAvail:
      self.raiseAnError(RuntimeError, f"ray requested but not available. {desiredParallelMethod}")
    # internal server is initialized only in case an internal calc is requested
    if not self.__isDistributedInitialized:
      self.__initializeDistributed()

  def __checkAndRemoveFinished(self, running):
    """
      Method to check if a run is finished and remove it from the queque
      @ In, running, instance, the job instance (InternalRunner or ExternalRunner)
      @ Out, None
    """
    with self.__queueLock:
      returnCode = running.getReturnCode()
      if returnCode != 0:
        metadataFailedRun = running.getMetadata()
        metadataToKeep = metadataFailedRun
        if metadataFailedRun is not None:
          metadataKeys      = list(metadataFailedRun.keys())
          if 'jobHandler' in metadataKeys:
            metadataKeys.pop(metadataKeys.index("jobHandler"))
            metadataToKeep = { keepKey: metadataFailedRun[keepKey] for keepKey in metadataKeys }
        # FIXME: The running.command was always internal now, so I removed it.
        # We should probably find a way to give more pertinent information.
        self.raiseAMessage(f" Process Failed {running.identifier}:{running} internal returnCode {returnCode}")
        self.__failedJobs[running.identifier]=(returnCode,copy.deepcopy(metadataToKeep))

  def __initializeDistributed(self):
    """
      Internal method that is aimed to initialize the internal parallel system.
      It initializes the RAY or DASK implementation (with socketing system) in
      case RAVEN is run in a cluster with multiple nodes or the NumMPI > 1,
      otherwise multi-threading is used.
      @ In, None
      @ Out, None
    """
    self.raiseADebug("Initializing parallel InternalParallel: {0} Nodes: {1}".format(self.runInfoDict['internalParallel'],len(self.runInfoDict['Nodes'])))
    if self._parallelLib != ParallelLibEnum.shared:
      # dashboard?
      db = self.runInfoDict['includeDashboard']
      # Check if the list of unique nodes is present and, in case, initialize the
      servers = None
      sys.path.append(self.runInfoDict['WorkingDir'])
      if 'UPDATE_PYTHONPATH' in self.runInfoDict:
        sys.path.extend([p.strip() for p in self.runInfoDict['UPDATE_PYTHONPATH'].split(":")])

      if _rayAvail:
        # update the python path and working dir
        olderPath = os.environ["PYTHONPATH"].split(os.pathsep) if "PYTHONPATH" in os.environ else []
        os.environ["PYTHONPATH"] = os.pathsep.join(set(olderPath+sys.path))

      # is ray instanciated outside?
      self.rayInstanciatedOutside = 'headNode' in self.runInfoDict
      self.daskInstanciatedOutside = 'schedulerFile' in self.runInfoDict
      if len(self.runInfoDict['Nodes']) > 0 or self.rayInstanciatedOutside or self.daskInstanciatedOutside:
        availableNodes = [nodeId.strip() for nodeId in self.runInfoDict['Nodes']]
        uniqueN = list(set(availableNodes))
        # identify the local host name and get the number of local processors
        localHostName = self.__getLocalHost()
        self.raiseADebug("Head host name is   : ", localHostName)
        # number of processors
        nProcsHead = availableNodes.count(localHostName)
        if not nProcsHead:
          self.raiseAWarning("# of local procs are 0. Only remote procs are avalable")
          self.raiseAWarning(f'Head host name "{localHostName}" /= Avail Nodes "'+', '.join(uniqueN)+'"!')
        self.raiseADebug("# of local procs    : ", str(nProcsHead))
        self.raiseADebug("# of total procs    : ", str(len(availableNodes)))
        if nProcsHead != len(availableNodes) or self.rayInstanciatedOutside or self.daskInstanciatedOutside:
          if self.rayInstanciatedOutside:
            address = self.runInfoDict['headNode']
          elif self.daskInstanciatedOutside:
            self.daskSchedulerFile = self.runInfoDict['schedulerFile']
          else:
            # create head node cluster
            # port 0 lets ray choose an available port
            address = self.__runHeadNode(nProcsHead, 0)
          if self._parallelLib == ParallelLibEnum.ray:
            # add names in runInfo
            self.runInfoDict['headNode'] = address
            self.raiseADebug("Head host IP      :", address)
          if self._parallelLib == ParallelLibEnum.dask:
            # add file in runInfo
            self.runInfoDict['schedulerFile'] = self.daskSchedulerFile
            self.raiseADebug('scheduler file     :', self.daskSchedulerFile)
          ## Get servers and run ray or dask remote listener
          if self.rayInstanciatedOutside or self.daskInstanciatedOutside:
            servers = self.runInfoDict['remoteNodes']
          else:
            servers = self.__runRemoteListeningSockets(address, localHostName)
          # add names in runInfo
          self.runInfoDict['remoteNodes'] = servers
          if self._parallelLib == ParallelLibEnum.ray:
            ## initialize ray server with nProcs
            self._server = ray.init(address=address,log_to_driver=False,include_dashboard=db)
          elif self._parallelLib == ParallelLibEnum.dask:
            if self.daskSchedulerFile is not None:
              #handle multinode and prestarted configurations
              self._server = dask.distributed.Client(scheduler_file=self.daskSchedulerFile)
            else:
              #Start locally
              cluster = dask.distributed.LocalCluster()
              self._server = dask.distributed.Client(cluster)
          else:
            self.raiseAWarning("No supported server")
          if self._parallelLib == ParallelLibEnum.ray:
            self.raiseADebug("NODES IN THE CLUSTER : ", str(ray.nodes()))
        else:
          if self._parallelLib == ParallelLibEnum.ray:
            self.raiseADebug("Executing RAY in the cluster but with a single node configuration")
            self._server = ray.init(num_cpus=nProcsHead,log_to_driver=False,include_dashboard=db)
          elif self._parallelLib == ParallelLibEnum.dask:
            self.raiseADebug("Executing DASK in the cluster but with a single node configuration")
            #Start locally
            cluster = dask.distributed.LocalCluster()
            self._server = dask.distributed.Client(cluster)
      else:
        self.raiseADebug("Initializing", str(self._parallelLib), "locally with num_cpus: ", self.runInfoDict['totalNumCoresUsed'])
        if self._parallelLib == ParallelLibEnum.ray:
          self._server = ray.init(num_cpus=int(self.runInfoDict['totalNumCoresUsed']),include_dashboard=db)
        elif self._parallelLib == ParallelLibEnum.dask:
          #handle local method
          cluster = dask.distributed.LocalCluster(n_workers=int(self.runInfoDict['totalNumCoresUsed']))
          self._server = dask.distributed.Client(cluster)
        else:
          self.raiseAWarning("parallellib creation not handled")
      if self._parallelLib == ParallelLibEnum.ray:
        self.raiseADebug("Head node IP address: ", self._server.address_info['node_ip_address'])
        self.raiseADebug("Redis address       : ", self._server.address_info['redis_address'])
        self.raiseADebug("Object store address: ", self._server.address_info['object_store_address'])
        self.raiseADebug("Raylet socket name  : ", self._server.address_info['raylet_socket_name'])
        self.raiseADebug("Session directory   : ", self._server.address_info['session_dir'])
        self.raiseADebug("GCS Address         : ", self._server.address_info['gcs_address'])
        if servers:
          self.raiseADebug("# of remote servers : ", str(len(servers)))
          self.raiseADebug("Remote servers      : ", " , ".join(servers))
      else:
        self.raiseADebug("JobHandler initialized without ray")
    else:
      ## We are just using threading
      self._server = None
      self.raiseADebug("JobHandler initialized with threading")
    # ray or dask is initialized
    self.__isDistributedInitialized = True

  def __getLocalAndRemoteMachineNames(self):
    """
      Method to get the qualified host and remote nodes' names
      @ In, None
      @ Out, hostNameMapping, dict, dictionary containing the qualified names of the remote nodes
    """
    hostNameMapping = {}
    ## collect the qualified hostnames for each remote node
    for nodeId in list(set(self.runInfoDict['Nodes'])):
      hostNameMapping[nodeId.strip()] = socket.gethostbyname(nodeId.strip())
      self.raiseADebug('Host "'+nodeId.strip()+'" identified with IP: ', hostNameMapping[nodeId.strip()])

    return hostNameMapping

  def __getLocalHost(self):
    """
      Method to get the name of the local host
      @ In, None
      @ Out, __getLocalHost, string, the local host name
    """
    return str(socket.getfqdn()).strip()

  def __shutdownParallel(self):
    """
      shutdown the parallel protocol
      @ In, None
      @ Out, None
    """
    if self._parallelLib == ParallelLibEnum.ray and self._server is not None and not self.rayInstanciatedOutside:
      # we need to ssh and stop each remote node cluster (ray)
      servers = []
      if 'remoteNodes' in self.runInfoDict:
        servers += self.runInfoDict['remoteNodes']
      if 'headNode' in self.runInfoDict:
        servers += [self.runInfoDict['headNode']]
      # get local enviroment
      localEnv = os.environ.copy()
      localEnv["PYTHONPATH"] = os.pathsep.join(sys.path)
      for nodeAddress in servers:
        self.raiseAMessage("Shutting down ray at address: "+ nodeAddress)
        command="ray stop"
        rayTerminate = utils.pickleSafeSubprocessPopen(['ssh',nodeAddress.split(":")[0],"COMMAND='"+command+"'","RAVEN_FRAMEWORK_DIR='"+self.runInfoDict["FrameworkDir"]+"'",self.runInfoDict['RemoteRunCommand']],shell=False,env=localEnv)
        rayTerminate.wait()
        if rayTerminate.returncode != 0:
          self.raiseAWarning("RAY FAILED TO TERMINATE ON NODE: "+nodeAddress)
      # shutdown ray API (object storage, plasma, etc.)
      ray.shutdown()
    elif self._parallelLib == ParallelLibEnum.dask and self._server is not None and not self.rayInstanciatedOutside:
      self._server.close()
      if self._daskScheduler is not None:
        self._daskScheduler.terminate()

  def __runHeadNode(self, nProcs, port=None):
    """
      Method to activate the head ray server
      @ In, nProcs, int, the number of processors
      @ In, port, int, desired port (None: ray default, 0: ray finds available)
      @ Out, address, str, the retrieved address (ip:port)
    """
    address = None
    # get local enviroment
    localEnv = os.environ.copy()
    localEnv["PYTHONPATH"] = os.pathsep.join(sys.path)
    if self._parallelLib == ParallelLibEnum.ray:
      command = ["ray", "start", "--head"]
      if nProcs is not None:
        command.append("--num-cpus="+str(nProcs))
      if port is not None:
        command.append("--port="+str(port))
      outFile = open("ray_head.ip", 'w')
      rayStart = utils.pickleSafeSubprocessPopen(command,shell=False,stdout=outFile, stderr=outFile, env=localEnv)
      rayStart.wait()
      outFile.close()
      if rayStart.returncode != 0:
        self.raiseAnError(RuntimeError, f"RAY failed to start on the --head node! Return code is {rayStart.returncode}")
      else:
        address = self.__getRayInfoFromStart("ray_head.ip")
    elif self._parallelLib == ParallelLibEnum.dask:
      self.daskSchedulerFile = os.path.join(self.runInfoDict['WorkingDir'],"scheduler.json")
      if os.path.exists(self.daskSchedulerFile):
        self.raiseADebug("Removing "+str(self.daskSchedulerFile))
        os.remove(self.daskSchedulerFile)

      tries = 0
      succeeded = False
      while not succeeded:
        #If there is a way to tell dask scheduler to automatically choose a
        # port, please change this to that.
        scheduler = utils.pickleSafeSubprocessPopen(["dask","scheduler",
                                                     "--scheduler-file",
                                                     self.daskSchedulerFile,
                                                     "--port",str(8786+tries)])

        waitCount = 0.0
        while not (os.path.exists(self.daskSchedulerFile) or scheduler.poll() is not None or waitCount > 20.0):
          time.sleep(0.1)
          waitCount += 0.1
        if os.path.exists(self.daskSchedulerFile) and scheduler.poll() is None:
          succeeded = True
          self._daskScheduler = scheduler
          self.raiseADebug("dask scheduler started with "+str(self.daskSchedulerFile))
          break
        if scheduler.poll() is None:
          self.raiseAWarning("killing dask scheduler")
          scheduler.terminate()
        tries += 1
        if tries > 20:
          succeeded = False
          self.raiseAWarning("failed to start dask scheduler")
          self.daskSchedulerFile = None
          break
      if succeeded:
        #do equivelent of dask worker start in start_dask.sh:
        # dask worker --nworkers $NUM_CPUS --scheduler-file $SCHEDULER_FILE  >> $OUTFILE
        outFile = open(os.path.join(self.runInfoDict['WorkingDir'],
                                    "server_debug_"+self.__getLocalHost()),'w')
        command = ["dask","worker","--scheduler-file",self.daskSchedulerFile]
        if nProcs is not None:
          command.extend(("--nworkers",str(nProcs)))
        headDaskWorker = utils.pickleSafeSubprocessPopen(command,shell=False,
                                stdout=outFile, stderr=outFile, env=localEnv)
    return address

  def __getRayInfoFromStart(self, rayLog):
    """
      Read Ray info from shell return script for ray
      @ In, rayLog, str, the ray output log
      @ Out, address, str, the retrieved address (ip:port)
    """
    with open(rayLog, 'r') as rayLogObj:
      for line in rayLogObj.readlines():
        match = re.search("ray start --address='([^']*)'", line)
        if match:
          address = match.groups()[0]
          return address
    self.raiseAWarning("ray start address not found in "+str(rayLog))
    return None

  def __updateListeningSockets(self, localHostName):
    """
      Update the path in the remote nodes
      @ In, localHostName, string, the head node name
      @ Out, None
    """
    ## Get the local machine name and the remote nodes one
    remoteNodesIP = self.__getLocalAndRemoteMachineNames()
    ## Strip out the nodes' names
    availableNodes = [node.strip() for node in self.runInfoDict['Nodes']]
    ## Get unique nodes
    uniqueNodes  = list(set(list(set(availableNodes))) - set([localHostName]))
    self.remoteServers = {}
    if len(uniqueNodes) > 0:
      ## There are remote nodes that need to be activated
      ## Modify the python path used by the local environment
      localEnv = os.environ.copy()
      pathSeparator = os.pathsep
      if "PYTHONPATH" in localEnv and len(localEnv["PYTHONPATH"].strip()) > 0:
        previousPath = localEnv["PYTHONPATH"].strip()+pathSeparator
      else:
        previousPath = ""
      localEnv["PYTHONPATH"] = previousPath+pathSeparator.join(sys.path)
      ## Start
      for nodeId in uniqueNodes:
        remoteHostName =  remoteNodesIP[nodeId]
        ## Activate the remote socketing system
        ## let's build the command and then call the os-agnostic version
        if _rayAvail:
          self.raiseADebug("Updating RAY server in node:", nodeId.strip())
          runScript = os.path.join(self.runInfoDict['FrameworkDir'],"RemoteNodeScripts","update_path_in_remote_servers.sh")
          command=" ".join([runScript,"--remote-node-address",nodeId," --working-dir ",self.runInfoDict['WorkingDir']])
          self.raiseADebug("command is:", command)
          command += " --python-path "+localEnv["PYTHONPATH"]
          self.remoteServers[nodeId] = utils.pickleSafeSubprocessPopen([command],shell=True,env=localEnv)

  def __removeLibPythonFromPath(self, pythonPath):
    """
      Method to remove the python library from the path (which can cause
      problems with different python version)
      @ In, pythonPath, string, the original python path
      @ Out, pythonPath, string, the python path with lib.python removed.
    """
    if re.search("lib.python", pythonPath):
      #strip out python libraries from path
      #XXX ideally, this would have a way to tell if
      # the paths we are stripping are the real builtin python paths
      # instead of just using a regular expression
      splitted=pythonPath.split(os.pathsep)
      newpath = []
      for part in splitted:
        if not re.search("lib.python", part):
          newpath.append(part)
        else:
          self.raiseADebug(f"removepath: {part}")
      return os.pathsep.join(newpath)
    return pythonPath

  def __runRemoteListeningSockets(self, address, localHostName):
    """
      Method to activate the remote sockets for parallel python
      @ In, address, string, the head node redis address
      @ In, localHostName, string, the local host name
      @ Out, servers, list, list containing the nodes in which the remote sockets have been activated
    """
    ## Get the local machine name and the remote nodes one
    remoteNodesIP = self.__getLocalAndRemoteMachineNames()

    ## Strip out the nodes' names
    availableNodes = [node.strip() for node in self.runInfoDict['Nodes']]

    ## Get unique nodes
    uniqueNodes  = list(set(availableNodes) - set([localHostName]))
    servers      = []
    self.remoteServers = {}
    if len(uniqueNodes) > 0:
      ## There are remote nodes that need to be activated
      ## Modify the python path used by the local environment
      localEnv = os.environ.copy()
      pathSeparator = os.pathsep
      if "PYTHONPATH" in localEnv and len(localEnv["PYTHONPATH"].strip()) > 0:
        previousPath = localEnv["PYTHONPATH"].strip()+pathSeparator
      else:
        previousPath = ""
      localEnv["PYTHONPATH"] = previousPath+pathSeparator.join(sys.path)
      ## Start
      for nodeId in uniqueNodes:
        ## Check how many processors are available in the node
        ntasks = availableNodes.count(nodeId)
        remoteHostName =  remoteNodesIP[nodeId]

        ## Activate the remote socketing system
        ## let's build the command and then call the os-agnostic version
        if self._parallelLib == ParallelLibEnum.ray:
          self.raiseADebug("Setting up RAY server in node: "+nodeId.strip())
          runScript = os.path.join(self.runInfoDict['FrameworkDir'],"RemoteNodeScripts","start_remote_servers.sh")
          command=" ".join([runScript,"--remote-node-address",nodeId, "--address",address, "--num-cpus",str(ntasks)," --working-dir ",self.runInfoDict['WorkingDir']," --raven-framework-dir",self.runInfoDict["FrameworkDir"],"--remote-bash-profile",self.runInfoDict['RemoteRunCommand']])
          self.raiseADebug("command is: "+command)
          command += " --python-path "+localEnv["PYTHONPATH"]
          self.remoteServers[nodeId] = utils.pickleSafeSubprocessPopen([command],shell=True,env=localEnv)
        elif self._parallelLib == ParallelLibEnum.dask:
          remoteServerScript = os.path.join(self.runInfoDict['FrameworkDir'],
                                            "RemoteNodeScripts","start_dask.sh")
          outputFile = os.path.join(self.runInfoDict['WorkingDir'],"server_debug_"+nodeId)
          command = ['ssh',nodeId,remoteServerScript,outputFile,
                     self.daskSchedulerFile,str(ntasks),
                     self.runInfoDict["FrameworkDir"],
                     self.runInfoDict['RemoteRunCommand'],
                     self.runInfoDict['WorkingDir']]
          self.raiseADebug("command is: "+" ".join(command))
          command.append(self.__removeLibPythonFromPath(localEnv["PYTHONPATH"]))
          self.remoteServers[nodeId] = utils.pickleSafeSubprocessPopen(command, env=localEnv)
        ## update list of servers
        servers.append(nodeId)
      if self._parallelLib == ParallelLibEnum.ray or self._parallelLib == ParallelLibEnum.dask:
        #wait for the servers to finish starting (prevents zombies)
        for nodeId in uniqueNodes:
          self.remoteServers[nodeId].wait()
          self.raiseADebug("server "+str(nodeId)+" result: "+str(self.remoteServers[nodeId]))

    return servers

  def sendDataToWorkers(self, data):
    """
      Method to send data to workers (if ray activated) and return a reference
      If ray is not used, the data is simply returned, otherwise an object reference id is returned
      @ In, data, object, any data to send to workers
      @ Out, ref, ray.ObjectRef or object, the reference or the object itself
    """
    if self._server is not None and self._parallelLib == ParallelLibEnum.ray:
      ref = ray.put(copy.deepcopy(data))
    else:
      ref = copy.deepcopy(data)
    return ref

  def startLoop(self):
    """
    This function begins the polling loop for the JobHandler where it will
    constantly fill up its running queue with jobs in its pending queue and
    unload finished jobs into its finished queue to be extracted by
    """
    while not self.completed:
      self.fillJobQueue()
      self.cleanJobQueue()
      # TODO May want to revisit this:
      # http://stackoverflow.com/questions/29082268/python-time-sleep-vs-event-wait
      # probably when we move to Python 3.
      time.sleep(self.sleepTime)

  def addJob(self, args, functionToRun, identifier, metadata=None, forceUseThreads = False, uniqueHandler="any", clientQueue = False, groupInfo = None):
    """
      Method to add an internal run (function execution)
      @ In, args, dict, this is a list of arguments that will be passed as
        function parameters into whatever method is stored in functionToRun.
        e.g., functionToRun(*args)
      @ In, functionToRun,function or method, the function that needs to be
        executed
      @ In, identifier, string, the job identifier
      @ In, metadata, dict, optional, dictionary of metadata associated to this
        run
      @ In, forceUseThreads, bool, optional, flag that, if True, is going to
        force the usage of multi-threading even if parallel python is activated
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ In, groupInfo, dict, optional, {id:string, size:int}.
        - "id": it is a special keyword attached to
          this runner to identify that this runner belongs to a special set of runs that need to be
          grouped together (all will be retrievable only when all the runs ended).
        - "size", number of runs in this group self.__batching
        NOTE: If the "size" of the group is only set the first time a job of this group is added.
              Consequentially the size is immutable
      @ In, clientQueue, boolean, optional, if this run needs to be added in the
        clientQueue
      @ Out, None
    """
    assert "original_function" in dir(functionToRun), "to parallelize a function, it must be" \
           " decorated with RAVEN Parallel decorator"
    if self._server is None or forceUseThreads:
      internalJob = Runners.factory.returnInstance('SharedMemoryRunner', args,
                                                   functionToRun.original_function,
                                                   identifier=identifier,
                                                   metadata=metadata,
                                                   uniqueHandler=uniqueHandler,
                                                   profile=self.__profileJobs)
    else:
      if self._parallelLib == ParallelLibEnum.dask:
        arguments =  tuple([self._server] + list(args))
      else:
        arguments = args
      if self._parallelLib == ParallelLibEnum.dask:
        internalJob = Runners.factory.returnInstance('DaskRunner', arguments,
                                                     functionToRun.original_function,
                                                     identifier=identifier,
                                                     metadata=metadata,
                                                     uniqueHandler=uniqueHandler,
                                                     profile=self.__profileJobs)

      elif self._parallelLib == ParallelLibEnum.ray:
        internalJob = Runners.factory.returnInstance('RayRunner', arguments,
                                                     functionToRun.remote,
                                                     identifier=identifier,
                                                     metadata=metadata,
                                                     uniqueHandler=uniqueHandler,
                                                     profile=self.__profileJobs)
    # set the client info
    internalJob.clientRunner = clientQueue
    #  set the groupping id if present
    if groupInfo is not None:
      groupId =  groupInfo['id']
      # TODO: create method in Runner to set flags,ids,etc in the instanciated runner
      internalJob.groupId = groupId
      if groupId not in self.__batching:
        # NOTE: The size of the group is only set once the first job beloning to a group is added
        #       ***** THE size of a group is IMMUTABLE *****
        self.__batching[groupId] = {"counter": 0, "ids": [], "size": groupInfo['size'], 'finished': []}
      self.__batching[groupId]["counter"] += 1
      if self.__batching[groupId]["counter"] > self.__batching[groupId]["size"]:
        self.raiseAnError(RuntimeError, f"group id {groupId} is full. Size reached:")
      self.__batching[groupId]["ids"].append(identifier)
    # add the runner in the Queue
    self.reAddJob(internalJob)

  def reAddJob(self, runner):
    """
      Method to add a runner object in the queue
      @ In, runner, Runner Instance, this is the instance of the runner that we want to readd in the queque
      @ Out, None
    """
    with self.__queueLock:
      if not runner.clientRunner:
        self.__queue.append(runner)
      else:
        self.__clientQueue.append(runner)
      if self.__profileJobs:
        runner.trackTime('queue')
      self.__submittedJobs.append(runner.identifier)

  def addClientJob(self, args, functionToRun, identifier, metadata=None, uniqueHandler="any"):
    """
      Method to add an internal run (function execution), without consuming
      resources (free spots). This can be used for client handling (see
      metamodel)
      @ In, args, dict, this is a list of arguments that will be passed as
        function parameters into whatever method is stored in functionToRun.
        e.g., functionToRun(*args)
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
    self.addJob(args, functionToRun, identifier, metadata,
                forceUseThreads = True, uniqueHandler = uniqueHandler,
                clientQueue = True)

  def addFinishedJob(self, data, metadata=None, uniqueHandler="any", profile=False):
    """
      Takes an already-finished job (for example, a restart realization) and adds it to the finished queue.
      @ In, data, dict, completed realization
      @ In, data, dict, fully-evaluated realization
      @ In, metadata, dict, optional, dictionary of metadata associated with
        this run
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ In, profile, bool, optional, if True then at de-construction timing statements will be printed
      @ Out, None
    """
    # create a placeholder runner
    run = Runners.factory.returnInstance('PassthroughRunner', data, None,
                                         metadata=metadata,
                                         uniqueHandler=uniqueHandler,
                                         profile=profile)
    # place it on the finished queue
    with self.__queueLock:
      self.__finished.append(run)

  def isFinished(self, uniqueHandler=None):
    """
      Method to check if all the runs in the queue are finished, or if a specific job(s) is done (jobIdentifier or uniqueHandler)
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        each runner. If provided, just the jobs that have the uniqueIdentifier
        will be checked. By default uniqueHandler = None => all the jobs for
        which no uniqueIdentifier has been set up are going to be checked
      @ Out, isFinished, bool, True all the runs in the queue are finished
    """

    #    FIXME: The following two lines of codes have been a temporary fix for timing issues
    #       on the collections of jobs in the jobHandler. This issue has emerged when
    #       performing batching. It is needed to review the relations between jobHandler
    #       and the Step when retrieving multiple jobs.
    #       An issue has been opened: 'JobHandler and Batching #1402'

    with self.__queueLock:
      # If there is still something left in the queue, we are not done yet.
      if len(self.__queue)>0 or len(self.__clientQueue)>0:
        return False

      # Otherwise, let's look at our running lists and see if there is a job
      # that is not done.
      for run in self.__running+self.__clientRunning:
        if run:
          if uniqueHandler is None or uniqueHandler == run.uniqueHandler:
            return False
    # Are there runs that need to be claimed? If so, then I cannot say I am done.
    numFinished = len(self.getFinishedNoPop())
    if numFinished != 0:
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
    # Due to possibility of memory explosion, we should include the finished
    # queue when considering whether we should add a new job. There was an
    # issue when running on a distributed system where we saw that this list
    # seemed to be growing indefinitely as the main thread was unable to clear
    # that list within a reasonable amount of time. The issue on the main thread
    # should also be addressed, but at least we can prevent it on this end since
    # the main thread's issue may be legitimate.

    maxCount = self.maxQueueSize
    finishedCount = len(self.__finished)

    if client:
      if maxCount is None:
        maxCount = self.__clientRunning.count(None)
      queueCount = len(self.__clientQueue)
    else:
      if maxCount is None:
        maxCount = self.__running.count(None)
      queueCount = len(self.__queue)

    availability = maxCount - queueCount - finishedCount
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
      # Look through the finished jobs and attempt to find a matching
      # identifier. If the job exists here, it is finished
      for run in self.__finished:
        if run.identifier == identifier:
          return True

      # Look through the pending jobs and attempt to find a matching identifier
      # If the job exists here, it is not finished
      for queue in [self.__queue, self.__clientQueue]:
        for run in queue:
          if run.identifier == identifier:
            return False

      # Look through the running jobs and attempt to find a matching identifier
      # If the job exists here, it is not finished
      for run in self.__running+self.__clientRunning:
        if run is not None and run.identifier == identifier:
          return False

    #  If you made it here and we still have not found anything, we have got
    # problems.
    self.raiseAnError(RuntimeError,"Job "+identifier+" is unknown!")

  def areTheseJobsFinished(self, uniqueHandler="any"):
    """
      Method to check if all the runs in the queue are finished
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        each runner. If provided, just the jobs that have the uniqueIdentifier
        will be retrieved. By default uniqueHandler = 'any' => all the jobs for
        which no uniqueIdentifier has been set up are going to be retrieved
      @ Out, isFinished, bool, True all the runs in the queue are finished
    """
    uniqueHandler = uniqueHandler.strip()
    with self.__queueLock:
      for run in self.__finished:
        if run.uniqueHandler == uniqueHandler:
          return False

      for queue in [self.__queue, self.__clientQueue]:
        for run in queue:
          if run.uniqueHandler == uniqueHandler:
            return False

      for run in self.__running + self.__clientRunning:
        if run is not None and run.uniqueHandler == uniqueHandler:
          return False

    self.raiseADebug("The jobs with uniqueHandler ", uniqueHandler, "are finished")

    return True

  def getFailedJobs(self):
    """
      Method to get list of failed jobs
      @ In, None
      @ Out, __failedJobs, list, list of the identifiers (jobs) that failed
    """
    return self.__failedJobs

  def getFinished(self, removeFinished=True, jobIdentifier='', uniqueHandler="any"):
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
      @ Out, finished, list, list of list containing finished jobs (InternalRunner or
        ExternalRunner objects) (if jobIdentifier is None), else the finished
        jobs matching the base case jobIdentifier
        NOTE:
        - in case the runs belong to a groupID (batching), each element of the list
         contains a list of the finished runs belonging to that group (Batch)
        - otherwise a flat list of jobs are returned.
        For example:
        finished =    [job1, job2, [job3.1, job3.2], job4 ] (job3.1/3.2 belong to the same groupID)
                   or [job1, job2, job3, job4]
    """
    # If the user does not specify a jobIdentifier, then set it to the empty
    # string because every job will match this starting string.
    if jobIdentifier is None:
      jobIdentifier = ''

    with self.__queueLock:
      finished = []
      runsToBeRemoved = []
      for i,run in enumerate(self.__finished):
        # If the jobIdentifier does not match or the uniqueHandler does not
        # match, then don't bother trying to do anything with it
        if not run.identifier.startswith(jobIdentifier) \
           or uniqueHandler != run.uniqueHandler:
          continue
        # check if the run belongs to a subgroup and in case
        if run.groupId in self.__batching:
          if not run in self.__batching[run.groupId]['finished']:
            self.__batching[run.groupId]['finished'].append(run)
        else:
          finished.append(run)

        if removeFinished:
          runsToBeRemoved.append(i)
          self.__checkAndRemoveFinished(run)
          #FIXME: IF THE RUN IS PART OF A BATCH AND IT FAILS, WHAT DO WE DO? alfoa
      # check if batches are ready to be returned
      for groupId in list(self.__batching.keys()):
        if len(self.__batching[groupId]['finished']) >  self.__batching[groupId]['size']:
          self.raiseAnError(RuntimeError,'The batching system got corrupted. Open an issue in RAVEN github!')
        if removeFinished:
          if len(self.__batching[groupId]['finished']) ==  self.__batching[groupId]['size']:
            doneBatch = self.__batching.pop(groupId)
            finished.append(doneBatch['finished'])
        else:
          doneBatch = self.__batching[groupId]
          finished.append(doneBatch['finished'])

        # Since these indices are sorted, reverse them to ensure that when we
        # delete something it will not shift anything to the left (lower index)
        # than it.
      if removeFinished:
        for i in reversed(runsToBeRemoved):
          self.__finished[i].trackTime('collected')
          del self.__finished[i]

      # end with self.__queueLock
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

  # Deprecating this function because I don't think it is doing the right thing
  # People using the job handler should be asking for what is available not the
  # number of free spots in the running block. Only the job handler should be
  # able to internally alter or query the running and clientRunning queues.
  # The outside environment can only access the queue and clientQueue variables.
  # def numFreeSpots(self, client=False):

  def numRunning(self):
    """
      Returns the number of runs currently running.
      @ In, None
      @ Out, activeRuns, int, number of active runs
    """
    #with self.__queueLock:
    # The size of the list does not change, only its contents, so I don't
    # think there should be any conflict if we are reading a variable from
    # one thread and updating it on the other thread.
    activeRuns = sum(run is not None for run in self.__running)

    return activeRuns

  def numRunningTotal(self):
    """
      Returns the number of runs currently running in both lists.
      @ In, None
      @ Out, activeRuns, int, number of active runs
    """
    activeRuns = sum(run is not None for run in self.__running + self.__clientRunning)
    return activeRuns

  def _numQueuedTotal(self):
    """
      Returns the number of runs currently waiting in both queues.
      @ In, None
      @ Out, queueSize, int, number of runs in queue
    """
    queueSize = len(self.__queue) + len(self.__clientQueue)
    return queueSize

  def numSubmitted(self):
    """
      Method to get the number of submitted jobs
      @ In, None
      @ Out, len(self.__submittedJobs), int, number of submitted jobs
    """
    return len(self.__submittedJobs)

  def fillJobQueue(self):
    """
      Method to start running the jobs in queue.  If there are empty slots
      takes jobs out of the queue and starts running them.
      @ In, None
      @ Out, None
    """
    # Only the jobHandler's startLoop thread should have write access to the
    # self.__running variable, so we should be able to safely query this outside
    # of the lock given that this function is called only on that thread as well.
    emptySlots = [i for i,run in enumerate(self.__running) if run is None]

    # Don't bother acquiring the lock if there are no empty spots or nothing
    # in the queue (this could be simultaneously added to by the main thread,
    # but I will be back here after a short wait on this thread so I am not
    # concerned about this potential inconsistency)
    if len(emptySlots) > 0 and len(self.__queue) > 0:
      with self.__queueLock:
        for i in emptySlots:
          # The queue could be emptied during this loop, so we will to break
          # out as soon as that happens so we don't hog the lock.
          if len(self.__queue) > 0:
            item = self.__queue.popleft()

            # Okay, this is a little tricky, but hang with me here. Whenever
            # a code model is run, we need to replace some of its command
            # parameters. The way we do this is by looking at the job instance
            # and checking if the first argument (the self in
            # self.evaluateSample) is an instance of Code, if so, then we need
            # to replace the execution command. Is this fragile? Possibly. We may
            # want to revisit this on the next iteration of this code.
            if len(item.args) > 0 and isinstance(item.args[0], Models.Code):
              kwargs = {}
              if self._server is not None:
                for infoKey in ['headNode','remoteNodes','schedulerFile']:
                  if infoKey in self.runInfoDict:
                    kwargs[infoKey] = self.runInfoDict[infoKey]
              kwargs['INDEX'] = str(i)
              kwargs['INDEX1'] = str(i+i)
              kwargs['CURRENT_ID'] = str(self.__nextId)
              kwargs['CURRENT_ID1'] = str(self.__nextId+1)
              kwargs['SCRIPT_DIR'] = self.runInfoDict['ScriptDir']
              kwargs['FRAMEWORK_DIR'] = self.runInfoDict['FrameworkDir']


              # This will not be used since the Code will create a new
              # directory for its specific files and will spawn a process there
              # so we will let the Code fill that in. Note, the line below
              # represents the WRONG directory for an instance of a code!
              # It is however the correct directory for a MultiRun step
              # -- DPM 5/4/17
              kwargs['WORKING_DIR'] = item.args[0].workingDir
              kwargs['BASE_WORKING_DIR'] = self.runInfoDict['WorkingDir']
              kwargs['METHOD'] = os.environ.get("METHOD","opt")
              kwargs['NUM_CPUS'] = str(self.runInfoDict['NumThreads'])
              item.args[3].update(kwargs)

            self.__running[i] = item
            self.__running[i].start()
            self.__running[i].trackTime('started')
            self.__nextId += 1
          else:
            break

    # Repeat the same process above, only for the clientQueue
    emptySlots = [i for i,run in enumerate(self.__clientRunning) if run is None]
    if len(emptySlots) > 0 and len(self.__clientQueue) > 0:
      with self.__queueLock:
        for i in emptySlots:
          if len(self.__clientQueue) > 0:
            self.__clientRunning[i] = self.__clientQueue.popleft()
            self.__clientRunning[i].start()
            self.__clientRunning[i].trackTime('jobHandler_started')
            self.__nextId += 1
          else:
            break

  def cleanJobQueue(self):
    """
    Method that will remove finished jobs from the queue and place them into the
    finished queue to be read by some other thread.
    @ In, None
    @ Out, None
    """
    # The code handling these two lists was the exact same, I have taken the
    # liberty of condensing these loops into one and removing some of the
    # redundant checks to make this code a bit simpler.
    for runList in [self.__running, self.__clientRunning]:
      with self.__queueLock:
        # We need the queueLock, because if terminateJobs runs kill on it,
        #  kill changes variables that can cause run.isDone to error out.
        for i,run in enumerate(runList):
          if run is not None and run.isDone():
            self.__finished.append(run)
            self.__finished[-1].trackTime('jobHandler_finished')
            runList[i] = None

  def setProfileJobs(self,profile=False):
    """
      Sets whether profiles for jobs are printed or not.
      @ In, profile, bool, optional, if True then print timings for jobs when they are garbage collected
      @ Out, None
    """
    self.__profileJobs = profile

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
    self.__shutdownParallel()

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

  def terminateJobs(self, ids):
    """
      Kills running jobs that match the given ids.
      @ In, ids, list(str), job prefixes to terminate
      @ Out, None
    """
    #WARNING: terminateJobs modifies the running queue, which
    # fillJobQueue assumes can't happen
    queues = [self.__queue, self.__clientQueue, self.__running, self.__clientRunning]
    with self.__queueLock:
      for _, queue in enumerate(queues):
        toRemove = []
        for job in queue:
          if job is not None and job.identifier in ids:
            # this assumes that each uniqueHandle only exists once in any queue anywhere
            ids.remove(job.identifier)
            toRemove.append(job)
        for job in toRemove:
          # for fixed-spot queues, need to replace job with None not remove
          if isinstance(queue,list):
            job.kill()
            queue[queue.index(job)] = None
          # for variable queues, can just remove the job
          else:
            queue.remove(job)
          self.raiseADebug(f'Terminated job "{job.identifier}" by request.')
    if len(ids):
      self.raiseADebug('Tried to remove some jobs but not found in any queues:',', '.join(ids))
