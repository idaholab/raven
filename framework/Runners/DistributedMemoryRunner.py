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
import signal
import copy
import sys
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
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
from .InternalRunner import InternalRunner
#Internal Modules End--------------------------------------------------------------------------------

class DistributedMemoryRunner(InternalRunner):
  """
    Class for running internal objects in distributed memory fashion using
    ppserver
  """
  def __init__(self, messageHandler, ppserver, Input, functionToRun, frameworkModules = [], identifier=None, metadata=None, functionToSkip = None, uniqueHandler = "any"):
    """
      Init method
      @ In, messageHandler, MessageHandler object, the global RAVEN message
        handler object
      @ In, ppserver, ppserver, instance of the ppserver object
      @ In, Input, list, list of inputs that are going to be passed to the
        function as *args
      @ In, functionToRun, method or function, function that needs to be run
      @ In, frameworkModules, list, optional, list of modules that need to be
        imported for internal parallelization (parallel python). This list
        should be generated with the method returnImportModuleString in utils.py
      @ In, identifier, string, optional, id of this job
      @ In, metadata, dict, optional, dictionary of metadata associated with
        this run
      @ In, functionToSkip, list, optional, list of functions, classes and
        modules that need to be skipped in pickling the function dependencies
      @ In, forceUseThreads, bool, optional, flag that, if True, is going to
        force the usage of multi-threading even if parallel python is activated
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ Out, None
    """

    ## First, allow the base class to handle the commonalities
    ##   We keep the command here, in order to have the hook for running exec
    ##   code into internal models
    super(DistributedMemoryRunner, self).__init__(messageHandler, Input, functionToRun, frameworkModules, identifier, metadata, functionToSkip, uniqueHandler)

    ## Other parameters passed at initialization
    self.__ppserver = ppserver

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
      return self.thread.finished

  def _collectRunnerResponse(self):
    """
      Method to add the process response in the internal variable (pointer)
      self.__runReturn
      @ In, None
      @ Out, None
    """
    if not self.hasBeenAdded:
      for row in self.__ppserver.collect_stats_in_list():
        self.raiseADebug(row)
      if self.thread is not None:
        self.runReturn = self.thread()
      else:
        self.runReturn = None
      self.hasBeenAdded = True

  def start(self):
    """
      Method to start the job associated to this Runner
      @ In, None
      @ Out, None
    """
    try:
      if len(self.input) == 1:
        self.thread = self.__ppserver.submit(self.functionToRun, args= (self.input[0],), depfuncs=(), modules = tuple(list(set(self.frameworkMods))),functionToSkip=self.functionToSkip)
      else:
        self.thread = self.__ppserver.submit(self.functionToRun, args= self.input, depfuncs=(), modules = tuple(list(set(self.frameworkMods))),functionToSkip=self.functionToSkip)
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
    self.raiseAWarning("Terminating " + self.thread.tid + " Identifier " + self.identifier)
    os.kill(self.thread.tid, signal.SIGTERM)
