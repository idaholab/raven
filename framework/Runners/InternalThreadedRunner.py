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

class InternalThreadedRunner(InternalRunner):
  """
    Class for running internal objects in a threaded fashion
  """
  def __init__(self, messageHandler, Input, functionToRun, frameworkModules = [], identifier=None, metadata=None, functionToSkip = None, uniqueHandler = "any"):
    """
      Init method
      @ In, messageHandler, MessageHandler object, the global RAVEN message handler object
      @ In, Input, list, list of inputs that are going to be passed to the function as *args
      @ In, functionToRun, method or function, function that needs to be run
      @ In, frameworkModules, list, optional, list of modules that need to be imported for internal parallelization (parallel python).
                                             this list should be generated with the method returnImportModuleString in utils.py
      @ In, identifier, string, optional, id of this job
      @ In, metadata, dict, optional, dictionary of metadata associated with this run
      @ In, functionToSkip, list, optional, list of functions, classes and modules that need to be skipped in pickling the function dependencies
      @ In, uniqueHandler, string, optional, it is a special keyword attached to this runner. For example, if present, to retrieve this runner using the method jobHandler.getFinished, the uniqueHandler needs to be provided.
                                            if uniqueHandler == 'any', every "client" can get this runner
      @ Out, None
    """
    ## First, allow the base class handle the commonalities
    # we keep the command here, in order to have the hook for running exec code into internal models
    super(InternalThreadedRunner, self).__init__(messageHandler, None, Input, functionToRun, frameworkModules, identifier, metadata, functionToSkip, uniqueHandler)

    ## Other parameters manipulated internally
    self.subque = collections.deque()
    #self.subque = queue.Queue()

  def __deepcopy__(self,memo):
    """
      This is the method called with copy.deepcopy.  Overwritten to remove some keys.
      @ In, memo, dict, dictionary required by deepcopy method
      @ Out, newobj, object, deep copy of this object
    """
    cls = self.__class__
    newobj = cls.__new__(cls)
    memo[id(self)] = newobj
    ### these things can't be deepcopied ###
    toRemove = ['functionToRun','subque','thread','__queueLock']
    for k,v in self.__dict__.items():
      if k not in toRemove:
        setattr(newobj,k,copy.deepcopy(v,memo))
    return newobj

  def isDone(self):
    """
      Method to check if the calculation associated with this InternalRunner is finished
      @ In, None
      @ Out, finished, bool, is it finished?
    """
    if self.thread is None:
      return True
    else:
      return not self.thread.is_alive()

  def getReturnCode(self):
    """
      Returns the return code from running the code.  If return code not yet set, set it.
      @ In, None
      @ Out, returnCode, int,  the return code of this evaluation
    """
    if not self.hasBeenAdded:
      self.__collectRunnerResponse()
    ## Is this necessary and sufficient for all failed runs?
    if len(self.subque) == 0 and self.runReturn is None:
      self.runReturn = None
      self.returnCode = -1

    return self.returnCode

  def __collectRunnerResponse(self):
    """
      Method to add the process response in the internal variable (pointer) self.runReturn
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
      Method to start the job associated to this InternalRunner
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
    except Exception as ae:
      self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
      self.returnCode = -1

  def kill(self):
    """
      Method to kill the job associated to this InternalRunner
      @ In, None
      @ Out, None
    """
    self.raiseAWarning("Terminating "+self.thread.pid+ " Identifier " + self.identifier)
    os.kill(self.thread.pid, signal.SIGTERM)