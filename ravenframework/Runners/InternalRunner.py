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
import copy

from .Runner import Runner
from .Error import Error

class InternalRunner(Runner):
  """
    Generic base Class for running internal objects
  """
  def __init__(self, args, functionToRun, **kwargs):
    """
      Init method
      @ In, args, dict, this is a list of arguments that will be passed as
        function parameters into whatever method is stored in functionToRun.
        e.g., functionToRun(*args)
      @ In, functionToRun, method or function, function that needs to be run
      @ Out, None
    """
    ## First, allow the base class to handle the commonalities
    ##   We keep the command here, in order to have the hook for running exec
    ##   code into internal models
    super().__init__(**kwargs)

    ## Other parameters passed at initialization
    self.args = copy.copy(args)
    self.functionToRun = functionToRun

    ## Other parameters manipulated internally
    self.runReturn = None
    self.hasBeenAdded = False
    self.returnCode = 0
    self.exceptionTrace = None    # sys.exc_info() if an error occurred while running

    ## These things cannot be deep copied
    self.skipOnCopy = ['functionToRun','thread','__queueLock', '_InternalRunner__queueLock']

  def __deepcopy__(self,memo):
    """
      This is the method called with copy.deepcopy.  Overwritten to remove some keys.
      @ In, memo, dict, dictionary required by deepcopy method
      @ Out, newobj, object, deep copy of this object
    """
    cls = self.__class__
    newobj = cls.__new__(cls)
    memo[id(self)] = newobj
    for k,v in self.__dict__.items():
      if k not in self.skipOnCopy:
        setattr(newobj,k,copy.deepcopy(v,memo))
    return newobj

  def _collectRunnerResponse(self):
    """
      Method to add the process response in the internal variable (pointer)
      self.runReturn
      @ In, None
      @ Out, None
    """
    pass

  def getReturnCode(self):
    """
      Returns the return code from running the code.
      @ In, None
      @ Out, returnCode, int,  the return code of this evaluation
    """
    return self.returnCode

  def getEvaluation(self):
    """
      Method to return the results of the function evaluation associated with
      this Runner
      @ In, None
      @ Out, returnValue, object or Error, whatever the method that this
        instance is executing returns, or if the job failed, will return an
        Error
    """
    if self.isDone():
      self._collectRunnerResponse()
      if self.runReturn is None:
        self.returnCode = -1
        return Error()
      return self.runReturn
    else:
      return Error()
