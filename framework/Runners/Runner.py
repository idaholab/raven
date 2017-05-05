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
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import abc
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from BaseClasses import BaseType
import MessageHandler
from .Error import Error
#Internal Modules End--------------------------------------------------------------------------------

class Runner(MessageHandler.MessageUser):
  """
    Generic base class for running codes and models in parallel environments
    both internally (shared data) and externally.
  """
  def __init__(self, messageHandler, identifier = None, metadata = None, uniqueHandler = "any"):
    """
      Initialize command variable
      @ In, messageHandler, MessageHandler instance, the global RAVEN message handler instance
      @ In, identifier, string, optional, id of this job
      @ In, metadata, dict, optional, dictionary of metadata associated with this Runner
      @ In, uniqueHandler, string, optional, it is a special keyword attached to this runner. For example, if present, to retrieve this runner using the method jobHandler.getFinished, the uniqueHandler needs to be provided.
                                             if uniqueHandler == 'any', every "client" can get this runner
      @ Out, None
    """
    self.messageHandler = messageHandler
    self.identifier     = 'generalOut'  ## Default identifier name
    self.metadata       = copy.copy(metadata)
    self.uniqueHandler  = uniqueHandler
    self.started        = False

    ## First attempt to use a user-specified identifier name
    if identifier is not None:
      self.identifier =  str(identifier).split("~",1)[-1]

    self.identifier = self.identifier.strip()

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
