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
Created on Mar 13, 2023

@author: alfoa, cogljj, crisr, talbpw, maljdp
"""
#External Modules------------------------------------------------------------------------------------
import sys
import gc
import copy
import threading
from ..utils import importerUtils as im
from ..utils.utils import ParallelLibEnum


## check which libraries are available.
if im.isLibAvail("dask"):
  import dask
  import dask.distributed
  parallelLib = ParallelLibEnum.dask
elif im.isLibAvail("ray"):
  import ray
  parallelLib = ParallelLibEnum.ray
else:
  import inspect
  parallelLib = ParallelLibEnum.pp
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import utils
from .InternalRunner import InternalRunner
#Internal Modules End--------------------------------------------------------------------------------


class DaskRunner(InternalRunner):
  """
    Class for running internal objects in distributed memory fashion using
    dask.
  """
  def __init__(self, args, functionToRun, **kwargs):
    """
      Init method
      @ In, args, list, this is a list of arguments that will be passed as
        function parameters into whatever method is stored in functionToRun.
        e.g., functionToRun(*args)
      @ In, functionToRun, method or function, function that needs to be run
      @ In, kwargs, dict, additional arguments to base class
      @ Out, None
    """
    ## First, allow the base class to handle the commonalities
    ##   We keep the command here, in order to have the hook for running exec
    ##   code into internal models
    if not parallelLib == ParallelLibEnum.dask:
      raise Exception("in DaskRunner, but using"+str(parallelLib))
    #XXX weird way to transfer client
    self.__client, args = args[0], args[1:]
    super().__init__(args, functionToRun, **kwargs)
    # __func (when using ray) is the object ref
    self.__func = None
    # __funcLock is needed because if isDone and kill are called at the
    # same time, isDone might end up trying to use __func after it is deleted
    self.__funcLock = threading.RLock()

  def __getstate__(self):
    """
      This function return the state of the DistributedMemoryRunner
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    state = copy.copy(self.__dict__)
    state.pop('_DaskRunner__funcLock')
    return state

  def __setstate__(self, d):
    """
      Initialize the DistributedMemoryRunner with the data contained in newstate
      @ In, d, dict, it contains all the information needed by the DistributedMemoryRunner to be initialized
      @ Out, None
    """
    self.__dict__.update(d)
    self.__funcLock = threading.RLock()

  def isDone(self):
    """
      Method to check if the calculation associated with this Runner is finished
      @ In, None
      @ Out, finished, bool, is it finished?
    """
    ## If the process has not been started yet, then return False
    if not self.started:
      return False

    with self.__funcLock:
      if self.__func is None:
        return True
      elif self.hasBeenAdded:
        return True
      else:
        return self.__func.done()

  def getReturnCode(self):
    """
      Returns the return code from running the code.  If return code not yet
      set, then set it.
      @ In, None
      @ Out, returnCode, int,  the return code of this evaluation
    """
    if not self.hasBeenAdded:
      try:
        self._collectRunnerResponse()
      except Exception as ae:
        self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
        self.returnCode = -1

    return self.returnCode

  def _collectRunnerResponse(self):
    """
      Method to add the process response in the internal variable (pointer)
      self.__runReturn
      @ In, None
      @ Out, None
    """
    with self.__funcLock:
      if not self.hasBeenAdded:
        if self.__func is not None:
          #if the function threw an exception, result will rethrow it.
          try:
            self.runReturn = self.__func.result()
          except Exception as ae:
            self.runReturn = None
            self.hasBeenAdded = True
            self.returnCode = -1
            self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
            raise ae
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
      self.__func = self.__client.submit(self.functionToRun, *self.args, retries=0)
      self.trackTime('runner_started')
      self.started = True
      gc.collect()
      return

    except Exception as ae:
      #Uncomment if you need the traceback
      self.exceptionTrace = sys.exc_info()
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
    with self.__funcLock:
      del self.__func
      self.__func = None
    self.returnCode = -1
    self.trackTime('runner_killed')
