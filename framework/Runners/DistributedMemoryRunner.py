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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import signal
import copy
import sys
import abc
import ray
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from BaseClasses import BaseType
import MessageHandler
from .InternalRunner import InternalRunner
#Internal Modules End--------------------------------------------------------------------------------

waitTimeOut = 1e-10 # timeout to check for job to finish

class DistributedMemoryRunner(InternalRunner):
  """
    Class for running internal objects in distributed memory fashion using
    ppserver
  """
  def __init__(self, messageHandler, args, functionToRun,
                    identifier=None, metadata=None,
                     uniqueHandler = "any", profile = False):
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
      @ In, forceUseThreads, bool, optional, flag that, if True, is going to
        force the usage of multi-threading even if parallel python is activated
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ In, profile, bool, optional, if True then timing statements are printed
        during deconstruction.
      @ Out, None
    """
    ## First, allow the base class to handle the commonalities
    ##   We keep the command here, in order to have the hook for running exec
    ##   code into internal models
    super(DistributedMemoryRunner, self).__init__(messageHandler, args, functionToRun, identifier, metadata, uniqueHandler,profile)

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
      return self.thread in ray.wait([self.thread], timeout=waitTimeOut)[0]

  def _collectRunnerResponse(self):
    """
      Method to add the process response in the internal variable (pointer)
      self.__runReturn
      @ In, None
      @ Out, None
    """
    if not self.hasBeenAdded:
      if self.thread is not None:
        self.runReturn = ray.get(self.thread)
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
      self.thread = self.functionToRun(*self.args)
      self.trackTime('runner_started')
      self.started = True
    except Exception as ae:
      #Uncomment if you need the traceback
      #exc_type, exc_value, exc_traceback = sys.exc_info()
      #import traceback
      #traceback.print_exception(exc_type, exc_value, exc_traceback)
      self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
      self.returnCode = -1

  def kill(self):
    """
      Method to kill the job associated to this Runner
      @ In, None
      @ Out, None
    """
    del self.thread
    self.thread = None
    self.returnCode = -1
    self.trackTime('runner_killed')
