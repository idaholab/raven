"""
Created on September 12, 2016
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import collections
import subprocess
# try               : import Queue as queue
# except ImportError: import queue
import os
import signal
import copy
import abc
#import logging, logging.handlers
import threading

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
from BaseClasses import BaseType
import MessageHandler
from .InternalRunner import InternalRunner
#Internal Modules End--------------------------------------------------------------------------------

class SharedMemoryRunner(InternalRunner):
  """
    Class for running internal objects in a threaded fashion using the built-in
    threading library
  """
  def __init__(self, messageHandler, Input, functionToRun, identifier=None, metadata=None, uniqueHandler = "any"):
    """
      Init method
      @ In, messageHandler, MessageHandler object, the global RAVEN message
        handler object
      @ In, Input, list, list of inputs that are going to be passed to the
        function as *args
      @ In, functionToRun, method or function, function that needs to be run
      @ In, identifier, string, optional, id of this job
      @ In, metadata, dict, optional, dictionary of metadata associated with
        this run
      @ In, functionToSkip, list, optional, list of functions, classes and
        modules that need to be skipped in pickling the function dependencies
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ Out, None
    """
    ## First, allow the base class handle the commonalities
    # we keep the command here, in order to have the hook for running exec code into internal models
    super(SharedMemoryRunner, self).__init__(messageHandler, Input, functionToRun, identifier, metadata, uniqueHandler)

    ## Other parameters manipulated internally
    self.subque = collections.deque()
    #self.subque = queue.Queue()

    self.skipOnCopy.append('subque')

  def isDone(self):
    """
      Method to check if the calculation associated with this Runner is finished
      @ In, None
      @ Out, finished, bool, is it finished?
    """
    ## If the process has not been started yet, then return False
    if not self.started:
      return False

    if self.thread is None:
      return True
    else:
      return not self.thread.is_alive()

  def getReturnCode(self):
    """
      Returns the return code from running the code.  If return code not yet
      set, then set it.
      @ In, None
      @ Out, returnCode, int,  the return code of this evaluation
    """
    if not self.hasBeenAdded:
      self._collectRunnerResponse()
    ## Is this necessary and sufficient for all failed runs?
    if len(self.subque) == 0 and self.runReturn is None:
      self.runReturn = None
      self.returnCode = -1

    return self.returnCode

  def _collectRunnerResponse(self):
    """
      Method to add the process response in the internal variable (pointer)
      self.runReturn
      @ In, None
      @ Out, None
    """
    if not self.hasBeenAdded:
      if len(self.subque) == 0:
        ## Queue is empty!
        self.runReturn = None
      else:
        self.runReturn = self.subque.popleft()

      self.hasBeenAdded = True

  def start(self):
    """
      Method to start the job associated to this Runner
      @ In, None
      @ Out, None
    """
    try:
      if len(self.input) == 1:
        self.thread = threading.Thread(target = lambda q,  arg : q.append(self.functionToRun(arg)), name = self.identifier, args=(self.subque,self.input[0]))
      else:
        self.thread = threading.Thread(target = lambda q, *arg : q.append(self.functionToRun(*arg)), name = self.identifier, args=(self.subque,)+tuple(self.input))

      self.thread.daemon = True
      self.thread.start()
      self.started = True
    except Exception as ae:
      self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
      self.returnCode = -1

  def kill(self):
    """
      Method to kill the job associated to this Runner
      @ In, None
      @ Out, None
    """
    self.raiseAWarning("Terminating "+self.thread.pid+ " Identifier " + self.identifier)
    os.kill(self.thread.pid, signal.SIGTERM)
