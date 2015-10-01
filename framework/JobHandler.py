"""
Created on Mar 5, 2013

@author: crisr
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
try               : import Queue as queue
except ImportError: import queue
import subprocess
import os
import signal
import copy
import sys
import abc
#import logging, logging.handlers
import threading
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
from BaseClasses import BaseType
# for internal parallel
import pp
import ppserver
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class ExternalRunner(MessageHandler.MessageUser):
  """
  Class for running external codes
  """
  def __init__(self,messageHandler,command,workingDir,bufsize,output=None,metadata=None,codePointer=None):
    """ Initialize command variable"""
    self.codePointerFailed = None
    self.messageHandler = messageHandler
    self.command    = command
    self.bufsize    = bufsize
    workingDirI     = None
    if    output!=None:
      self.output   = output
      if os.path.split(output)[0] != workingDir: workingDirI = os.path.split(output)[0]
      if len(str(output).split("~")) > 1:
        self.identifier =  str(output).split("~")[1]
      else:
        # try to find the identifier in the folder name
        # to eliminate when the identifier is passed from outside
        def splitall(path):
          allparts = []
          while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
              allparts.insert(0, parts[0])
              break
            elif parts[1] == path: # sentinel for relative paths
              allparts.insert(0, parts[1])
              break
            else:
              path = parts[0]
              allparts.insert(0, parts[1])
          return allparts
        splitted = splitall(str(output))
        if len(splitted) >= 2: self.identifier= splitted[-2]
        else: self.identifier= 'generalOut'
    else:
      self.output   = os.path.join(workingDir,'generalOut')
      self.identifier = 'generalOut'
    if workingDirI: self.__workingDir = workingDirI
    else          : self.__workingDir = workingDir
    ####### WARNING: THIS DEEPCOPY MUST STAY!!!! DO NOT REMOVE IT ANYMORE. ANDREA #######
    self.__metadata   = copy.deepcopy(metadata)
    ####### WARNING: THIS DEEPCOPY MUST STAY!!!! DO NOT REMOVE IT ANYMORE. ANDREA #######
    self.codePointer  = codePointer

# BEGIN: KEEP THIS COMMENTED PORTION HERE, I NEED IT FOR LATER USE. ANDREA
    # Initialize logger
    #self.logger     = self.createLogger(self.identifier)
    #self.addLoggerHandler(self.identifier, self.output, 100000, 1)
#   def createLogger(self,name):
#     """
#     Function to create a logging object
#     @ In, name: name of the logging object
#     @ Out, logging object
#     """
#     return logging.getLogger(name)
#
#   def addLoggerHandler(self,logger_name,filename,max_size,max_number_files):
#     """
#     Function to create a logging object
#     @ In, logger_name     : name of the logging object
#     @ In, filename        : log file name (with path)
#     @ In, max_size        : maximum file size (bytes)
#     @ In, max_number_files: maximum number of files to be created
#     @ Out, None
#     """
#     hadler = logging.handlers.RotatingFileHandler(filename,'a',max_size,max_number_files)
#     logging.getLogger(logger_name).addHandler(hadler)
#     logging.getLogger(logger_name).setLevel(logging.INFO)
#     return
#
#   def outStreamReader(self, out_stream):
#     """
#     Function that logs every line received from the out stream
#     @ In, out_stream: output stream
#     @ In, logger    : the instance of the logger object
#     @ Out, logger   : the logger itself
#     """
#     while True:
#       line = out_stream.readline()
#       if len(line) == 0 or not line:
#         break
#       self.logger.info('%s', line)
#       #self.logger.debug('%s', line.srip())
#   END: KEEP THIS COMMENTED PORTION HERE, I NEED IT FOR LATER USE. ANDREA

  def isDone(self):
    """
    Function to inquire the process to check if the calculation is finished
    """
    self.__process.poll()
    return self.__process.returncode != None

  def getReturnCode(self):
    """
    Function to inquire the process to get the return code
    If the self.codePointer is available (!= None), this method
    inquires it to check if the process return code is a false negative (or positive).
    The first time the codePointer is inquired, it calls the function and store the result
    => sub-sequential calls to getReturnCode will not inquire the codePointer anymore but
    just return the stored value
    """
    returnCode = self.__process.returncode
    if self.codePointer != None and returnCode == 0:
      if  self.codePointerFailed == None:  self.codePointerFailed = self.codePointer.checkForOutputFailure(self.output,self.getWorkingDir())
      if  self.codePointerFailed: returnCode = 1
    return returnCode

  def returnEvaluation(self):
    """
    Function to return the External runner evaluation (outcome/s). Since in process, return None
    """
    return None

  def returnMetadata(self):
    """
    Function to return the External runner metadata
    """
    return self.__metadata

  def start(self):
    """
    Function to run the driven code
    """
    oldDir = os.getcwd()
    os.chdir(self.__workingDir)
    localenv = dict(os.environ)
    outFile = open(self.output,'w', self.bufsize)
    self.__process = subprocess.Popen(self.command,shell=True,stdout=outFile,stderr=outFile,cwd=self.__workingDir,env=localenv)
    os.chdir(oldDir)
    #self.__process = subprocess.Popen(self.command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=self.__workingDir,env=localenv)
    #self.thread = threading.Thread(target=self.outStreamReader, args=(self.__process.stdout,))
    #self.thread.daemon = True
    #self.thread.start()

  def kill(self):
    """
    Function to kill the subprocess of the driven code
    """
    #In python 2.6 this could be self.process.terminate()
    self.raiseAMessage("Terminating "+self.__process.pid+' '+self.command)
    os.kill(self.__process.pid,signal.SIGTERM)

  def getWorkingDir(self):
    """
    Function to get the working directory path
    """
    return self.__workingDir

  def getOutputFilename(self):
    """
    Function to get the output filenames
    """
    return os.path.join(self.__workingDir,self.output)
#
#
#
#
class InternalRunner(MessageHandler.MessageUser):
  #import multiprocessing as multip
  def __init__(self,messageHandler,ppserver, Input,functionToRun, frameworkModules = [], identifier=None,metadata=None, globs = None, functionToSkip = None):
    # we keep the command here, in order to have the hook for running exec code into internal models
    self.command  = "internal"
    self.messageHandler = messageHandler
    self.ppserver = ppserver
    self.__thread = None
    if    identifier!=None:
      if "~" in identifier: self.identifier =  str(identifier).split("~")[1]
      else                : self.identifier =  str(identifier)
    else: self.identifier = 'generalOut'
    if type(Input) != tuple: self.raiseAnError(IOError,"The input for InternalRunner needs to be a tuple!!!!")
    #the Input needs to be a tuple. The first entry is the actual input (what is going to be stored here), the others are other arg the function needs
    if self.ppserver == None: self.subque = queue.Queue()
    self.functionToRun   = functionToRun
    self.__runReturn     = None
    self.__hasBeenAdded  = False
    self.__input         = copy.copy(Input)
    self.__metadata      = copy.copy(metadata)
    self.__globals       = copy.copy(globs)
    self.__frameworkMods = copy.copy(frameworkModules)
    self._functionToSkip = functionToSkip
    self.retcode         = 0

  def __deepcopy__(self,memo):
    """This is the method called with copy.deepcopy.  Overwritten to remove some keys.
    @ In, memo, dictionary required by deepcopy method
    @Out, deep copy of this object
    """
    cls = self.__class__
    newobj = cls.__new__(cls)
    memo[id(self)] = newobj
    copydict = self.__dict__
    ### these things can't be deepcopied ###
    toRemove = ['functionToRun','subque','_InternalRunner__thread']
    for k,v in copydict.items():
      if k in toRemove: continue
      setattr(newobj,k,copy.deepcopy(v,memo))
    return newobj

  def start_pp(self):
    if self.ppserver != None:
      if len(self.__input) == 1: self.__thread = self.ppserver.submit(self.functionToRun, args= (self.__input[0],), depfuncs=(), modules = tuple(list(set(self.__frameworkMods))),functionToSkip=self._functionToSkip)
      else                     : self.__thread = self.ppserver.submit(self.functionToRun, args= self.__input, depfuncs=(), modules = tuple(list(set(self.__frameworkMods))),functionToSkip=self._functionToSkip)
    else:
      if len(self.__input) == 1: self.__thread = threading.Thread(target = lambda q,  arg : q.put(self.functionToRun(arg)), name = self.identifier, args=(self.subque,self.__input[0]))
      else                     : self.__thread = threading.Thread(target = lambda q, *arg : q.put(self.functionToRun(*arg)), name = self.identifier, args=(self.subque,)+tuple(self.__input))
      self.__thread.daemon = True
      self.__thread.start()

  def isDone(self):
    if self.__thread == None: return True
    else:
      if self.ppserver != None: return self.__thread.finished
      else:                     return not self.__thread.is_alive()

  def getReturnCode(self):
    """Returns the return code from running the code.  If return code not yet set, set it.
    @ In, None
    @Out, return code, usually integer
    """
    if self.ppserver is None and hasattr(self,'subque'):
      if self.subque.empty(): #is this necessary and sufficient for all failed runs?
        self.__runReturn = -1
        self.retcode = -1
    else:
      self.raiseAWarning('FIXME Not yet implemented: handle this!')
    return self.retcode

  def returnEvaluation(self):
    if self.isDone():
      if not self.__hasBeenAdded:
        if self.ppserver is not None:
          self.__runReturn = self.__thread()
        else:
          if self.subque.empty(): self.__runReturn = None #queue is empty!
          else: self.__runReturn = self.subque.get(timeout=1)
        self.__hasBeenAdded = True
        if self.__runReturn is None:
          self.retcode = -1
          return self.retcode
      return (self.__input[0],self.__runReturn)
    else: return -1 #control return code

  def returnMetadata(self): return self.__metadata

  def start(self):
    try: self.start_pp()
    except Exception as ae:
      self.raiseAMessage("InternalRunner job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
      self.retcode = -1

  def kill(self):
    self.raiseAMessage("Terminating "+self.__thread.pid+ " Identifier " + self.identifier)
    if self.ppserver != None: os.kill(self.__thread.tid,signal.SIGTERM)
    else: os.kill(self.__thread.pid,signal.SIGTERM)

class JobHandler(MessageHandler.MessageUser):
  def __init__(self):
    self.printTag               = 'Job Handler'
    self.runInfoDict            = {}
    self.mpiCommand             = ''
    self.threadingCommand       = ''
    self.initParallelPython     = False
    self.submitDict             = {}
    self.submitDict['External'] = self.addExternal
    self.submitDict['Internal'] = self.addInternal
    self.externalRunning        = []
    self.internalRunning        = []
    self.__running              = []
    self.__queue                = queue.Queue()
    self.__nextId               = 0
    self.__numSubmitted         = 0
    self.__numFailed            = 0
    self.__failedJobs           = {} #dict of failed jobs, keyed on identifer, valued on metadata

  def initialize(self,runInfoDict,messageHandler):
    self.runInfoDict = runInfoDict
    self.messageHandler = messageHandler
    if self.runInfoDict['NumMPI'] !=1 and len(self.runInfoDict['ParallelCommand']) > 0:
      self.mpiCommand = self.runInfoDict['ParallelCommand']+' '+str(self.runInfoDict['NumMPI'])
    if self.runInfoDict['NumThreads'] !=1 and len(self.runInfoDict['ThreadingCommand']) > 0:
      self.threadingCommand = self.runInfoDict['ThreadingCommand'] +' '+str(self.runInfoDict['NumThreads'])
    #initialize PBS
    self.__running = [None]*self.runInfoDict['batchSize']

  def __initializeParallelPython(self):
    """
      Internal method that is aimed to initialize the internal parallel system.
      It initilizes the paralle python implementation (with socketing system) in case
      RAVEN is run in a cluster with multiple nodes or the NumMPI > 1,
      otherwise multi-threading is used.
      @ In, None
      @ Out, None
    """
    # check if the list of unique nodes is present and, in case, initialize the socket
    if len(self.runInfoDict['Nodes']) > 0:
      # initialize the socketing system
      #ppserverScript = os.path.join(self.runInfoDict['FrameworkDir'],"contrib","pp","ppserver.py -a")
      ppserverScript = os.path.join(self.runInfoDict['FrameworkDir'],"contrib","pp","ppserver.py")
      # create the servers in the reserved nodes
      localenv = dict(os.environ)
      #localenv['PYTHONPATH'] = ''
      ppservers = []
      for nodeid in [node.strip() for node in set(self.runInfoDict['Nodes'])]:
        outFile = open(nodeid.strip()+"_server_out.log",'w')
        # check how many processors are available in the node
        ntasks = self.runInfoDict['Nodes'].count(nodeid)
        process = subprocess.Popen(['ssh', nodeid, ppserverScript,"-w",str(ntasks),"-d"],shell=False,stdout=outFile,stderr=outFile,env=localenv)
        ppservers.append(nodeid)
      #for nodeid in self.runInfoDict['Nodes']: subprocess.call(['ssh ', nodeid, ppserverScript])
      #for nodeid in self.runInfoDict['Nodes']: subprocess.Popen('ssh '+nodeid+' '+ ppserverScript , shell=True) #,env=localenv)
      # create the server handler
      #ppservers=("*",)
      #ppservers = tuple(nodeid.split(".")[0])
      self.ppserver     = pp.Server(ppservers=tuple(ppservers)) #,ncpus=int(self.runInfoDict['totalNumCoresUsed']))
      #self.ppserver     = pp.Server(ncpus=int(self.runInfoDict['totalNumCoresUsed']), ppservers=tuple(self.runInfoDict['Nodes']))
    else:
      if self.runInfoDict['NumMPI'] > 1: self.ppserver = pp.Server(ncpus=int(self.runInfoDict['totalNumCoresUsed'])) # we use the parallel python
      else                             : self.ppserver = None        # we just use threading!
    self.initParallelPython = True

  def addExternal(self,executeCommands,outputFile,workingDir,metadata=None,codePointer=None):
    """
      Method to add an external runner (an external code) in the handler list
      @ In, executeCommands, tuple(string), ('parallel'/'serial', <execution command>)
      @ In, outputFile, string, output file name
      @ In, workingDir, string, working directory
      @ In, metadata, dict, optional, dictionary of metadata
      @ In, codePointer, derived CodeInterfaceBaseClass object, optional, pointer to code interface
      @ Out, None
    """
    precommand = self.runInfoDict['precommand'] #FIXME what uses this?  Still precommand for whole line if multiapp case?
    #it appears precommand is usually used for mpiexec - however, there could be other uses....
    commands=[]
    for runtype,cmd in executeCommands:
      newcom=''
      if runtype.lower() == 'parallel':
        newcom += precommand
        if self.mpiCommand !='':
          newcom += ' '+self.mpiCommand+' '
        if self.threadingCommand !='': #FIXME are these two exclusive?
          newcom += ' '+ self.threadingCommand +' '
        newcom += cmd+' '
        newcom+= self.runInfoDict['postcommand']
        commands.append(newcom)
      elif runtype.lower() == 'serial':
        commands.append(cmd)
      else:
        self.raiseAnError(IOError,'For execution command <'+cmd+'> the run type was neither "serial" nor "parallel"!  Instead got:',runtype,'\nCheck the code interface.')
    command= ' && '.join(commands)+' '
    self.__queue.put(ExternalRunner(self.messageHandler,command,workingDir,self.runInfoDict['logfileBuffer'],outputFile,metadata,codePointer))
    self.raiseAMessage('Execution command submitted:',command)
    self.__numSubmitted += 1
    if self.howManyFreeSpots()>0: self.addRuns()

  def addInternal(self,Input,functionToRun,identifier,metadata=None, modulesToImport = [], globs = None):
    #internal serve is initialized only in case an internal calc is requested
    if not self.initParallelPython: self.__initializeParallelPython()
    self.__queue.put(InternalRunner(self.messageHandler,self.ppserver, Input, functionToRun, modulesToImport, identifier, metadata, globs, functionToSkip=[utils.metaclass_insert(abc.ABCMeta,BaseType)]))
    self.__numSubmitted += 1
    if self.howManyFreeSpots()>0: self.addRuns()

  def isFinished(self):
    if not self.__queue.empty():
      return False
    for i in range(len(self.__running)):
      if self.__running[i] and not self.__running[i].isDone():
        return False
    return True

  def getNumberOfFailures(self):
    return self.__numFailed

  def getListOfFailedJobs(self):
    return self.__failedJobs

  def howManyFreeSpots(self):
    cnt_free_spots = 0
    if self.__queue.empty():
      for i in range(len(self.__running)):
        if self.__running[i]:
          if self.__running[i].isDone():
            cnt_free_spots += 1
        else:
          cnt_free_spots += 1
    return cnt_free_spots

  def getFinished(self, removeFinished=True, prefix=None):
    """collects the finished jobs and returns them.
    @ In, optional, removeFinished, if true removes jobs from handler storage.
    @ In, prefix, if specified only collects finished runs with a particular prefix.
    @Out, list of finished jobs.
    """
    finished = []
    for i in range(len(self.__running)):
      if self.__running[i] and self.__running[i].isDone():
        if prefix is not None:
          if self.__running[i].identifier.startswith(prefix):
            finished.append(self.__running[i])
          else:
            continue
        else:
          finished.append(self.__running[i])
        if removeFinished:
          running = self.__running[i]
          returncode = running.getReturnCode()
          if returncode != 0:
            self.raiseAMessage(" Process Failed "+str(running)+' '+str(running.command)+" returncode "+str(returncode))
            self.__numFailed += 1
            self.__failedJobs[running.identifier]=(returncode,copy.deepcopy(running.returnMetadata()))
            if type(running).__name__ == "External":
              outputFilename = running.getOutputFilename()
              if os.path.exists(outputFilename): self.raiseAMessage(open(outputFilename,"r").read())
              else: self.raiseAMessage(" No output "+outputFilename)
          else:
            if self.runInfoDict['delSucLogFiles'] and running.__class__.__name__ != 'InternalRunner':
              self.raiseAMessage(' Run "' +running.identifier+'" ended smoothly, removing log file!')
              if os.path.exists(running.getOutputFilename()): os.remove(running.getOutputFilename())
            if len(self.runInfoDict['deleteOutExtension']) >= 1 and running.__class__.__name__ != 'InternalRunner':
              for fileExt in self.runInfoDict['deleteOutExtension']:
                if not fileExt.startswith("."): fileExt = "." + fileExt
                filelist = [ f for f in os.listdir(running.getWorkingDir()) if f.endswith(fileExt) ]
                for f in filelist: os.remove(f)
          self.__running[i] = None
    if not self.__queue.empty(): self.addRuns()
    return finished

  def addRuns(self):
    for i in range(len(self.__running)):
      if self.__running[i] == None and not self.__queue.empty():
        item = self.__queue.get()
        if "External" in item.__class__.__name__ :
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

  def getFinishedNoPop(self):
    return self.getFinished(False)

  def getNumSubmitted(self):
    return self.__numSubmitted

  def startingNewStep(self):
    self.__numSubmitted = 0

  def terminateAll(self):
    #clear out the queue
    while not self.__queue.empty(): self.__queue.get()
    for i in range(len(self.__running)):
      if self.__running[i] is not None: self.__running[i].kill()

  def numRunning(self):
    return sum(run is not None for run in self.__running)
