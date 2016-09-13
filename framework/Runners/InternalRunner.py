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
from .Runner import Runner
#Internal Modules End--------------------------------------------------------------------------------

class InternalRunner(Runner):
  """
    Class for running internal objects
  """
  def __init__(self, messageHandler, ppserver, Input, functionToRun, frameworkModules = [], identifier=None, metadata=None, functionToSkip = None, uniqueHandler = "any"):
    """
      Init method
      @ In, messageHandler, MessageHandler object, the global RAVEN message handler object
      @ In, ppserver, ppserver, instance of the ppserver object
      @ In, Input, list, list of inputs that are going to be passed to the function as *args
      @ In, functionToRun, method or function, function that needs to be run
      @ In, frameworkModules, list, optional, list of modules that need to be imported for internal parallelization (parallel python).
                                             this list should be generated with the method returnImportModuleString in utils.py
      @ In, identifier, string, optional, id of this job
      @ In, metadata, dict, optional, dictionary of metadata associated with this run
      @ In, functionToSkip, list, optional, list of functions, classes and modules that need to be skipped in pickling the function dependencies
      @ In, forceUseThreads, bool, optional, flag that, if True, is going to force the usage of multi-threading even if parallel python is activated
      @ In, uniqueHandler, string, optional, it is a special keyword attached to this runner. For example, if present, to retrieve this runner using the method jobHandler.getFinished, the uniqueHandler needs to be provided.
                                            if uniqueHandler == 'any', every "client" can get this runner
      @ Out, None
    """

    ## First, allow the base class to handle the commonalities
    ##   We keep the command here, in order to have the hook for running exec
    ##   code into internal models
    super(InternalRunner, self).__init__(messageHandler, 'internal', identifier, metadata, uniqueHandler)

    ## Other parameters passed at initialization
    self.__ppserver     = ppserver
    self.input          = copy.copy(Input)
    self.functionToRun  = functionToRun
    self.frameworkMods  = copy.copy(frameworkModules)
    self.functionToSkip = functionToSkip

    ## Other parameters manipulated internally
    self.thread         = None
    self.runReturn      = None
    self.hasBeenAdded   = False
    self.returnCode     = 0

    ## The Input needs to be a tuple. The first entry is the actual input (what
    ## is going to be stored here), the others are other arg the function needs
    if type(Input) != tuple:
      self.raiseAnError(IOError,"The input for " + self.__class__.__name__ + " should be a tuple. Instead received " + Input.__class__.__name__)

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
    toRemove = ['functionToRun','thread','__queueLock']
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
      return self.thread.finished

  def getReturnCode(self):
    """
      Returns the return code from running the code.  If return code not yet set, set it.
      @ In, None
      @ Out, returnCode, int,  the return code of this evaluation
    """
    return self.returnCode

  def __collectRunnerResponse(self):
    """
      Method to add the process response in the internal variable (pointer) self.__runReturn
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

  def getEvaluation(self):
    """
      Method to return the results of the function evaluation associated with this InternalRunner
      @ In, None
      @ Out, (Input,response), tuple, tuple containing the results of the evaluation (list of Inputs, function return value)
    """
    if self.isDone():
      self.__collectRunnerResponse()
      if self.runReturn is None:
        self.returnCode = -1
        return self.returnCode
      return (self.input[0],self.runReturn)
    else:
      return -1 #control return code

  def start(self):
    """
      Method to start the job associated to this InternalRunner
      @ In, None
      @ Out, None
    """
    try:
      if len(self.input) == 1:
        self.thread = self.__ppserver.submit(self.functionToRun, args= (self.input[0],), depfuncs=(), modules = tuple(list(set(self.frameworkMods))),functionToSkip=self.functionToSkip)
      else:
        self.thread = self.__ppserver.submit(self.functionToRun, args= self.input, depfuncs=(), modules = tuple(list(set(self.frameworkMods))),functionToSkip=self.functionToSkip)
    except Exception as ae:
      self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
      self.returnCode = -1

  def kill(self):
    """
      Method to kill the job associated to this InternalRunner
      @ In, None
      @ Out, None
    """
    self.raiseAWarning("Terminating " + self.thread.tid + " Identifier " + self.identifier)
    os.kill(self.thread.tid, signal.SIGTERM)
