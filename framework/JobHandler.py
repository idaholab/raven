'''
Created on Mar 5, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

try:
  import Queue as queue
except ImportError:
  import queue
import subprocess
import os
import signal
import copy
from utils import returnPrintTag
#import logging, logging.handlers
import threading 

class ExternalRunner:
  def __init__(self,command,workingDir,output=None,metadata=None):
    ''' Initialize command variable'''
    self.command    = command
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
    self.__metadata   = metadata
    # Initialize logger
    #self.logger     = self.createLogger(self.identifier)
    #self.addLoggerHandler(self.identifier, self.output, 100000, 1)
#   def createLogger(self,name):
#     '''
#     Function to create a logging object
#     @ In, name: name of the logging object
#     @ Out, logging object 
#     '''
#     return logging.getLogger(name)
#     
#   def addLoggerHandler(self,logger_name,filename,max_size,max_number_files):
#     '''
#     Function to create a logging object
#     @ In, logger_name     : name of the logging object
#     @ In, filename        : log file name (with path)
#     @ In, max_size        : maximum file size (bytes)
#     @ In, max_number_files: maximum number of files to be created
#     @ Out, None 
#     '''
#     hadler = logging.handlers.RotatingFileHandler(filename,'a',max_size,max_number_files)
#     logging.getLogger(logger_name).addHandler(hadler)
#     logging.getLogger(logger_name).setLevel(logging.INFO)
#     return 
# 
#   def outStreamReader(self, out_stream):
#     '''
#     Function that logs every line received from the out stream
#     @ In, out_stream: output stream
#     @ In, logger    : the instance of the logger object
#     @ Out, logger   : the logger itself 
#     '''
#     while True:
#       line = out_stream.readline()
#       if len(line) == 0 or not line:
#         break
#       self.logger.info('%s', line)
#       #self.logger.debug('%s', line.srip())

  def isDone(self):
    self.__process.poll()
    return self.__process.returncode != None

  def getReturnCode(self): return self.__process.returncode

  def returnEvaluation(self): return None
  
  def returnMetadata(self): return self.__metadata
  
  def start(self):
    oldDir = os.getcwd()
    os.chdir(self.__workingDir)
    localenv = dict(os.environ)
    localenv['PYTHONPATH'] = ''
    outFile = open(self.output,'w')
    self.__process = subprocess.Popen(self.command,shell=True,stdout=outFile,stderr=outFile,cwd=self.__workingDir,env=localenv)
    os.chdir(oldDir)
    #self.__process = subprocess.Popen(self.command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=self.__workingDir,env=localenv)
    #self.thread = threading.Thread(target=self.outStreamReader, args=(self.__process.stdout,)) 
    #self.thread.daemon = True
    #self.thread.start()
  
  def kill(self):
    #In python 2.6 this could be self.process.terminate()
    print(returnPrintTag('JOB HANDLER')+ ": Terminating ",self.__process.pid,self.command)
    os.kill(self.__process.pid,signal.SIGTERM)    

  def getWorkingDir(self): return self.__workingDir

  def getOutputFilename(self): return os.path.join(self.__workingDir,self.output)
#
#
#
#
class InternalRunner:
  #import multiprocessing as multip
  def __init__(self,Input,functionToRun,identifier=None,metadata=None):
    # we keep the command here, in order to have the hook for running exec code into internal models
    self.command = "internal"
    if    identifier!=None: 
      if "~" in identifier: self.identifier =  str(identifier).split("~")[1]
      else                : self.identifier =  str(identifier)
    else: self.identifier = 'generalOut'
    if type(Input) != tuple: raise IOError(returnPrintTag('JOB HADLER') + ": ERROR -> The input for InternalRunner needs to be a tuple!!!!")
    #the Input needs to be a tuple. The first entry is the actual input (what is going to be stored here), the others are other arg the function needs
    self.subque          = queue.Queue()
    self.functionToRun   = functionToRun
    if len(Input) == 1: self.__thread = threading.Thread(target = lambda q,  arg : q.put(self.functionToRun(arg)), name = self.identifier, args=(self.subque,)+Input)
    else              : self.__thread = threading.Thread(target = lambda q, *arg : q.put(self.functionToRun(arg)), name = self.identifier, args=(self.subque,)+Input)
    self.__thread.daemon = True 
    self.__runReturn     = None
    self.__hasBeenAdded  = False
    try:   self.__input         = copy.deepcopy(Input[0])
    except:self.__input         = copy.copy(Input[0])
    self.__metadata      = copy.deepcopy(metadata)
    self.retcode         = 0

  def isDone(self):
    return not self.__thread.is_alive()

  def getReturnCode(self): return self.retcode
  
  def returnEvaluation(self):
    if self.isDone(): 
      if not self.__hasBeenAdded:
        self.__runReturn = copy.deepcopy(self.subque.get(timeout=1))   
        self.__hasBeenAdded = True
      return (self.__input,self.__runReturn)
    else: return -1 #control return code   
  
  def returnMetadata(self): return self.__metadata
  
  def start(self): 
    try: self.__thread.start()
    except Exception as ae:
      print(returnPrintTag('JOB HADLER')+"ERROR -> InternalRunner job "+self.identifier+" failed with error:"+ str(ae) +" !")
      self.retcode = -1
  
  def kill(self): 
    print(returnPrintTag('JOB HADLER')+": Terminating ",self.__thread.ident(), " Identifier " + self.identifier)
    os.kill(self.__thread.ident(),signal.SIGTERM)    

class JobHandler:
  def __init__(self):
    self.runInfoDict       = {}
    self.mpiCommand        = ''
    self.threadingCommand  = ''
    self.submitDict = {}
    self.submitDict['External'] = self.addExternal
    self.submitDict['Internal'] = self.addInternal
    self.externalRunning        = []
    self.internalRunning        = []
    self.__running = []
    self.__queue = queue.Queue()
    self.__nextId = 0
    self.__numSubmitted = 0
    self.__numFailed = 0
    self.__failedJobs = []
    
  def initialize(self,runInfoDict):
    self.runInfoDict = runInfoDict
    if self.runInfoDict['NumMPI'] !=1 and len(self.runInfoDict['ParallelCommand']) > 0:
      self.mpiCommand = self.runInfoDict['ParallelCommand']+' '+str(self.runInfoDict['NumMPI'])
    if self.runInfoDict['NumThreads'] !=1 and len(self.runInfoDict['ThreadingCommand']) > 0:
      self.threadingCommand = self.runInfoDict['ThreadingCommand'] +' '+str(self.runInfoDict['NumThreads'])
    #initialize PBS
    self.__running = [None]*self.runInfoDict['batchSize']

  def addExternal(self,executeCommand,outputFile,workingDir,metadata=None):
    #probably something more for the PBS
    command = self.runInfoDict['precommand']
    if self.mpiCommand !='':
      command += self.mpiCommand+' '
    if self.threadingCommand !='':
      command +=self.threadingCommand+' '
    command += executeCommand
    command += self.runInfoDict['postcommand']
    self.__queue.put(ExternalRunner(command,workingDir,outputFile,metadata))
    self.__numSubmitted += 1
    if self.howManyFreeSpots()>0: self.addRuns()
    
  def addInternal(self,Input,functionToRun,identifier,metadata=None):
    self.__queue.put(InternalRunner(Input,functionToRun,identifier,metadata))
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

  def getFinished(self, removeFinished=True):
    #print("getFinished "+str(self.__running)+" "+str(self.__queue.qsize()))
    finished = []
    for i in range(len(self.__running)):
      if self.__running[i] and self.__running[i].isDone():
        finished.append(self.__running[i])
        if removeFinished:
          running = self.__running[i]
          returncode = running.getReturnCode()
          if returncode != 0:
            print(returnPrintTag('JOB HADLER')+": Process Failed ",running,running.command," returncode",returncode)
            self.__numFailed += 1
            self.__failedJobs.append(running.identifier)
            outputFilename = running.getOutputFilename()
            if os.path.exists(outputFilename):
              print(open(outputFilename,"r").read())
            else:
              print(returnPrintTag('JOB HADLER')+" No output ",outputFilename)
          else:
            if self.runInfoDict['delSucLogFiles'] and running.__class__.__name__ != 'InternalRunner':
              print('JOB HANDLER'.ljust(25) + ': Run "' +running.identifier+'" ended smoothly, removing log file!')
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
        self.__running[i].start()
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
    for i in range(len(self.__running)): self.__running[i].kill()


