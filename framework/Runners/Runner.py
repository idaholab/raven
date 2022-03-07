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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import abc
import copy
import time
import datetime
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import utils
from ..BaseClasses import BaseType, MessageUser
from .Error import Error
#Internal Modules End--------------------------------------------------------------------------------

class Runner(MessageUser):
  """
    Generic base class for running codes and models in parallel environments
    both internally (shared data) and externally.
  """
  def __init__(self, identifier=None, metadata=None, uniqueHandler="any", profile=False):
    """
      Initialize command variable
      @ In, identifier, string, optional, id of this job
      @ In, metadata, dict, optional, dictionary of metadata associated with this Runner
      @ In, uniqueHandler, string, optional, it is a special keyword attached to this runner. For example, if present, to retrieve this runner using the method jobHandler.getFinished, the uniqueHandler needs to be provided.
                                             if uniqueHandler == 'any', every "client" can get this runner
      @ In, profile, bool, optional, if True then timing statements will be printed during garbage collection
      @ Out, None
    """
    super().__init__()
    self.timings = {}
    self.timings['created'] = time.time()
    self.__printTimings = profile
    self.identifier     = 'generalOut'  ## Default identifier name
    self.metadata       = copy.copy(metadata)
    self.uniqueHandler  = uniqueHandler
    self.groupId        = None  # the id of the group this run belong to (batching, if activated)
    self.started        = False

    ## First attempt to use a user-specified identifier name
    if identifier is not None:
      self.identifier =  str(identifier).split("~",1)[-1]

    self.identifier = self.identifier.strip()

  def __del__(self):
    """
      Deconstructor.
      @ In, None
      @ Out, None
    """
    if self.__printTimings:
      # print timing history
      pairs = list(self.timings.items())
      pairs.sort(key=lambda x:x[1])
      prof = ""
      prof += 'TIMINGS for job "{}":'.format(self.identifier)
      for e,(event,time) in enumerate(pairs):
        if e == 0:
          _, msg = self.messageHandler._printMessage(self,'TIMINGS ... {:^20s} at {} ({})'.format(event,time,datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')) ,'DEBUG',3,None)
          last = time
        else:
          _, msg = self.messageHandler._printMessage(self,'TIMINGS ... {:^20s} elapsed {:10.6f} s'.format(event,time-last),'DEBUG',3,None)
          last = time
        prof +=  "\n"+msg
      self.raiseADebug(prof)

  def isDone(self):
    """
      Function to inquire the process to check if the calculation is finished
      @ In, None
      @ Out, finished, bool, is this run finished?
    """
    ## If the process has not been started yet, then return False
    if not self.started:
      return False

    return True

  def getReturnCode(self):
    """
      Function to inquire the process to get the return code
      @ In, None
      @ Out, returnCode, int, return code.  1 if the checkForOutputFailure is true, otherwise the process return code.
    """
    return 0

  def getEvaluation(self):
    """
      Function to return the External runner evaluation (outcome/s). Since in process, return None
      @ In, None
      @ Out, returnValue, object or Error, whatever the method that this
        instance is executing returns, or if the job failed, will return an
        Error
    """
    return Error()

  def getMetadata(self):
    """
      Function to return the Internal runner metadata
      @ In, None
      @ Out, metadata, dict, return the dictionary of metadata associated with this ExternalRunner
    """
    return self.metadata

  def trackTime(self,event):
    """
      Records the time under 'event'.
      @ In, event, string, the label under which to store the timing
    """
    self.timings[event] = time.time()

  def start(self):
    """
      Function to run the driven code
      @ In, None
      @ Out, None
    """
    self.started = True

  def kill(self):
    """
      Function to kill the subprocess of the driven code
      @ In, None
      @ Out, None
    """
    pass
