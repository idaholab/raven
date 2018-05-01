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
Created on September 12, 2016
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
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
from utils import utils
from BaseClasses import BaseType
import MessageHandler
from .InternalRunner import InternalRunner

#Internal Modules End--------------------------------------------------------------------------------


class SharedMemoryRunner(InternalRunner):
  """
    Class for running internal objects in a threaded fashion using the built-in
    threading library
  """

  def __init__(self,
               messageHandler,
               args,
               functionToRun,
               identifier=None,
               metadata=None,
               uniqueHandler="any",
               profile=False):
    """
      Init method
      @ In, messageHandler, MessageHandler object, the global RAVEN message
        handler object
      @ In, args, dict, this is a list of arguments that will be passed as
        function parameters into whatever method is stored in functionToRun.
        e.g., functionToRun(*args)
      @ In, functionToRun, method or function, function that needs to be run
      @ In, identifier, string, optional, id of this job
      @ In, metadata, dict, optional, dictionary of metadata associated with
        this run
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ In, clientRunner, bool, optional,  Is this runner needed to be executed in client mode? Default = False
      @ In, profile, bool, optional, if True then at deconstruction timing statements will be printed
      @ Out, None
    """
    ## First, allow the base class handle the commonalities
    # we keep the command here, in order to have the hook for running exec code into internal models
    super(SharedMemoryRunner, self).__init__(messageHandler, args, functionToRun, identifier,
                                             metadata, uniqueHandler, profile)

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
      self.thread = threading.Thread(
          target=lambda q, *arg: q.append(self.functionToRun(*arg)),
          name=self.identifier,
          args=(self.subque, ) + tuple(self.args))

      self.thread.daemon = True
      self.thread.start()
      self.trackTime('runner_started')
      self.started = True
    except Exception as ae:
      self.raiseAWarning(self.__class__.__name__ + " job " + self.identifier +
                         " failed with error:" + str(ae) + " !", 'ExceptedError')
      self.returnCode = -1

  def kill(self):
    """
      Method to kill the job associated to this Runner
      @ In, None
      @ Out, None
    """
    self.raiseAWarning("Terminating " + self.thread.pid + " Identifier " + self.identifier)
    os.kill(self.thread.pid, signal.SIGTERM)
    self.trackTime('runner_killed')
