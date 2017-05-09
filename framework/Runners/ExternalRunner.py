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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import abc
import platform
import shlex
#import logging, logging.handlers

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from BaseClasses import BaseType
import MessageHandler
import Runners
from .Runner import Runner
#Internal Modules End--------------------------------------------------------------------------------

class ExternalRunner(Runner):
  """
    Class for running external codes
  """
  def __init__(self,messageHandler, command, workingDir, bufferSize, identifier = None, output = None ,metadata = None, codePointer = None, uniqueHandler = "any"):
    """
      Initialize command variable
      @ In, messageHandler, MessageHandler instance, the global RAVEN message handler instance
      @ In, command, list, list of commands that needs to be executed
      @ In, workingDir, string, absolute path of the working directory
      @ In, bufferSize, int, buffer size for logger
      @ In, identifier, string, optional, id of this job
      @ In, output, string, optional, output filename root
      @ In, metadata, dict, optional, dictionary of metadata associated with this ExternalRunner
      @ In, codePointer, CodeInterface instance, optional, instance of the code interface associated with this ExternalRunner
      @ In, uniqueHandler, string, optional, it is a special keyword attached to this runner. For example, if present, to retrieve this runner using the method jobHandler.getFinished, the uniqueHandler needs to be provided.
                                             if uniqueHandler == 'any', every "client" can get this runner
      @ Out, None
    """
    ## First, allow the base class to handle the commonalities
    ##   Note, we are manually deep copying the metadata below, so here we will
    ##   feed the base class nothing for it.
    super(ExternalRunner, self).__init__(messageHandler, command, identifier, None, uniqueHandler)

    ## Other parameters passed at initialization
    self.workingDir  = workingDir
    self.bufferSize  = bufferSize
    #### WARNING: THIS DEEPCOPY MUST STAY! DO NOT REMOVE IT ANYMORE. ANDREA ####
    self.metadata    = copy.deepcopy(metadata)
    #### WARNING: THIS DEEPCOPY MUST STAY! DO NOT REMOVE IT ANYMORE. ANDREA ####
    self.codePointer = codePointer

    ## Other parameters manipulated internally
    self.codePointerFailed = None

    ## Set the output according to the user parameters
    if output is not None:
      self.output   = output
      if identifier is None:
        ## If an identifier has not been established, try to grab it from the
        ## preifx in the metadata
        if metadata is not None and 'prefix' in metadata.keys():
          self.identifier = metadata['prefix']
        else:
          ## Lastly, try to find the identifier in the folder name
          ## if the identifier was passed from outside
          splitPaths = os.path.split(str(output))
          if len(splitPaths) >= 2:
            self.identifier = splitPaths[-2]
    else:
      self.output = os.path.join(workingDir,'generalOut')

  # BEGIN: KEEP THIS COMMENTED PORTION HERE, I NEED IT FOR LATER USE. ANDREA
  #   Initialize logger
  #   self.logger     = self.createLogger(self.identifier)
  #   self.addLoggerHandler(self.identifier, self.output, 100000, 1)

  # def createLogger(self,name):
  #   """
  #   Function to create a logging object
  #   @ In, name: name of the logging object
  #   @ Out, logging object
  #   """
  #   return logging.getLogger(name)

  # def addLoggerHandler(self,logger_name,filename,max_size,max_number_files):
  #   """
  #   Function to create a logging object
  #   @ In, logger_name     : name of the logging object
  #   @ In, filename        : log file name (with path)
  #   @ In, max_size        : maximum file size (bytes)
  #   @ In, max_number_files: maximum number of files to be created
  #   @ Out, None
  #   """
  #   hadler = logging.handlers.RotatingFileHandler(filename,'a',max_size,max_number_files)
  #   logging.getLogger(logger_name).addHandler(hadler)
  #   logging.getLogger(logger_name).setLevel(logging.INFO)
  #   return

  # def outStreamReader(self, out_stream):
  #   """
  #   Function that logs every line received from the out stream
  #   @ In, out_stream: output stream
  #   @ In, logger    : the instance of the logger object
  #   @ Out, logger   : the logger itself
  #   """
  #   while True:
  #     line = out_stream.readline()
  #     if len(line) == 0 or not line:
  #       break
  #     self.logger.info('%s', line)
  #     #self.logger.debug('%s', line.srip())
  #   END: KEEP THIS COMMENTED PORTION HERE, I NEED IT FOR LATER USE. ANDREA

  def isDone(self):
    """
      Function to inquire the process to check if the calculation is finished
      @ In, None
      @ Out, finished, bool, is this run finished?
    """
    ## If the process has not been started yet, then return False
    if not self.started:
      return False

    self.__process.poll()
    finished = self.__process.returncode != None
    return finished

  def getReturnCode(self):
    """
      Function to inquire the process to get the return code
      If the self.codePointer is available (!= None), this method
      inquires it to check if the process return code is a false negative (or positive).
      The first time the codePointer is inquired, it calls the function and store the result
      => sub-sequential calls to getReturnCode will not inquire the codePointer anymore but
      just return the stored value
      @ In, None
      @ Out, returnCode, int, return code.  1 if the checkForOutputFailure is true, otherwise the process return code.
    """
    returnCode = self.__process.returncode
    if self.codePointer != None:
      if 'checkForOutputFailure' in dir(self.codePointer):
        if  self.codePointerFailed == None:
          self.codePointerFailed = self.codePointer.checkForOutputFailure(self.output,self.getWorkingDir())
      if self.codePointerFailed:
        returnCode = 1
    return returnCode

  def getEvaluation(self):
    """
      Function to return the External runner evaluation (outcome/s). Since in process, return None
      @ In, None
      @ Out, evaluation, tuple, the evaluation or None if run failed
    """
    return None

  def expandForWindows(self, origCommand):
    """
      Function to expand a command that has a #! to a windows runnable command
      @ In, origCommand, string, The command to check for expantion
      @ Out, commandSplit, string or String List, the expanded command or the original if not expanded.
    """
    if origCommand.strip() == '':
      return origCommand
    # In Windows Python, you can get some backslashes in your paths
    commandSplit = shlex.split(origCommand.replace("\\","/"))
    executable = commandSplit[0]

    if os.path.exists(executable):
      executableFile = open(executable, "r")

      firstTwoChars = executableFile.read(2)

      if firstTwoChars == "#!":
        realExecutable = shlex.split(executableFile.readline())
        self.raiseAMessage("reading #! to find executable:" + repr(realExecutable))
        # The below code should work, and would be better than findMsys,
        # but it doesn't work.
        # winExecutable = subprocess.check_output(['cygpath','-w',realExecutable[0]],shell=True).rstrip()
        # print("winExecutable",winExecutable)
        # realExecutable[0] = winExecutable
        def findMsys():
          """
            Function to try and figure out where the MSYS64 is.
            @ In, None
            @ Out, dir, String, If not None, the directory where msys is.
          """
          dir = os.getcwd()
          head, tail = os.path.split(dir)
          while True:
            if tail.lower().startswith("msys"):
              return dir
            dir = head
            head, tail = os.path.split(dir)
          return None
        msysDir = findMsys()
        if msysDir is not None:
          beginExecutable = realExecutable[0]
          if beginExecutable.startswith("/"):
            beginExecutable = beginExecutable.lstrip("/")
          winExecutable = os.path.join(msysDir, beginExecutable)
          self.raiseAMessage("winExecutable" + winExecutable)
          realExecutable[0] = winExecutable
        else:
          self.raiseAWarning("Could not find msys in "+os.getcwd())
        commandSplit = realExecutable + [executable] + commandSplit[1:]
        return commandSplit
    return origCommand


  def start(self):
    """
      Function to run the driven code
      @ In, None
      @ Out, None
    """
    oldDir = os.getcwd()
    os.chdir(self.workingDir)
    localenv = dict(os.environ)
    outFile = open(self.output,'w', self.bufferSize)
    if platform.system() == 'Windows':
      self.command = self.expandForWindows(self.command)
      self.raiseAMessage("modified command to" + repr(self.command))
    self.__process = utils.pickleSafeSubprocessPopen(self.command,shell=True,stdout=outFile,stderr=outFile,cwd=self.workingDir,env=localenv)
    os.chdir(oldDir)
    self.started = True

  def kill(self):
    """
      Function to kill the subprocess of the driven code
      @ In, None
      @ Out, None
    """
    self.raiseAWarning("Terminating "+self.__process.pid+' '+self.command)
    self.__process.terminate()

  def getWorkingDir(self):
    """
      Function to get the working directory path
      @ In, None
      @ Out, workingDir, string, working directory
    """
    return self.workingDir

  def getOutputFilename(self):
    """
      Function to get the output filenames
      @ In, None
      @ Out, self.output, string, output filename root
    """
    return os.path.join(self.workingDir,self.output)
