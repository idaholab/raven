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
import collections
import sys
import time
import ctypes
import inspect
import threading
import traceback

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .InternalRunner import InternalRunner
#Internal Modules End--------------------------------------------------------------------------------

class SharedMemoryRunner(InternalRunner):
  """
    Class for running internal objects in a threaded fashion using the built-in
    threading library
  """
  def __init__(self, args, functionToRun, **kwargs):
    """
      Init method
      @ In, args, dict, this is a list of arguments that will be passed as
        function parameters into whatever method is stored in functionToRun.
        e.g., functionToRun(*args)
      @ In, functionToRun, method or function, function that needs to be run
      @ In, kwargs, dict, additional arguments to pass to base
      @ Out, None
    """
    ## First, allow the base class handle the commonalities
    # we keep the command here, in order to have the hook for running exec code into internal models
    super().__init__(args, functionToRun, **kwargs)

    ## Other parameters manipulated internally
    self.subque = collections.deque()
    #self.subque = queue.Queue()

    self.skipOnCopy.append('subque')
    self.thread = None

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
    if self.runSucceeded is False:
      self.returnCode = -1
    elif self.runReturn is None:
      ## Either the wrapper never recorded an outcome (e.g. the thread was
      ## killed before completing), or the wrapped function returned None as
      ## its own failure signal without raising (e.g. Models.Code.evaluateSample
      ## on a non-zero process return code). Treat as failed either way.
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
    def _runFunction(q, *arg):
      """
        Thread target: runs the function, capturing exceptions and recording
        the outcome explicitly (so that a legitimate None return value is not
        mistaken for a failure, and the traceback is preserved).
        @ In, q, collections.deque, queue collecting the result
        @ In, arg, tuple, arguments for the function
        @ Out, None
      """
      try:
        result = self.functionToRun(*arg)
      except Exception:
        self.exceptionTrace = sys.exc_info()
        self.failureInfo = traceback.format_exc()
        self.runSucceeded = False
        self.returnCode = -1
        self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier
                           +" failed with error:\n"+self.failureInfo, 'ExceptedError')
        return
      q.append(result)
      self.runSucceeded = True

    try:
      self.thread = InterruptibleThread(target = _runFunction,
                                     name = self.identifier,
                                     args=(self.subque,) + tuple(self.args))

      self.thread.daemon = True
      self.thread.start()
      self.trackTime('runner_started')
      self.started = True
    except Exception as ae:
      self.exceptionTrace = sys.exc_info()
      self.failureInfo = traceback.format_exc()
      self.runSucceeded = False
      self.raiseAWarning(self.__class__.__name__ + " job "+self.identifier+" failed with error:"+ str(ae) +" !",'ExceptedError')
      self.returnCode = -1

  def kill(self):
    """
      Method to kill the job associated to this Runner
      @ In, None
      @ Out, None
    """
    if self.thread is not None:
      self.raiseADebug('Terminating job thread "{}" and RAVEN identifier "{}"'.format(self.thread.ident, self.identifier))
      ## NOTE: raising an asynchronous exception in a thread is inherently
      ## unreliable: it is silently ignored while the thread is blocked inside
      ## C extension code or system calls (exactly where external models spend
      ## most of their time). The previous unbounded loop could therefore spin
      ## forever. Bound the attempts and, if the thread will not die, warn and
      ## move on: the thread is a daemon, so it cannot keep the process alive.
      killTimeout = 10.0  # seconds
      waited = 0.0
      while self.thread is not None and self.thread.is_alive() and waited < killTimeout:
        self.thread.kill()
        time.sleep(0.1)
        waited += 0.1
      if self.thread is not None and self.thread.is_alive():
        self.raiseAWarning('Job thread "{}" (RAVEN identifier "{}") did not terminate '
                           'within {} s; it is likely blocked in native code. '
                           'Abandoning it as a daemon thread.'.format(
                           self.thread.ident, self.identifier, killTimeout))
      self.runSucceeded = False
      self.returnCode = -1
    self.trackTime('runner_killed')

## The following code is extracted from stack overflow with some minor cosmetic
## changes in order to adhere to RAVEN code standards:
## https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
def _asyncRaise(tid, exceptionType):
  """
    Raises an exception in the threads with id tid
    @ In, tid, integer, this variable represents the id of the thread to raise an exception
    @ In, exceptionType, Exception, the type of exception to throw
    @ Out, None
  """
  if not inspect.isclass(exceptionType):
    raise TypeError("Only types can be raised (not instances)")
  res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exceptionType))
  if res == 0:
    raise ValueError("invalid thread id")
  elif res != 1:
    # "if it returns a number greater than one, you're in trouble,
    # and you should call it again with exc=NULL to revert the effect"
    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
    raise SystemError("PyThreadState_SetAsyncExc failed")

class InterruptibleThread(threading.Thread):
  """
    A thread class that supports raising exception in the thread from another thread.
  """
  def kill(self):
    """
      Method to kill the thread raising an exception
      @ In, None
      @ Out, None
    """
    try:
      self.raiseException(RuntimeError)
    except ValueError:
      pass

  def raiseException(self, exceptionType):
    """
      Raises the given exception type in the context of this thread.
      If the thread is busy in a system call (time.sleep(), socket.accept(), ...), the exception is simply ignored.
      If you are sure that your exception should terminate the thread, one way to ensure that it works is:
       t = InterruptibleThread( ... )
        ...
        t.raiseException( SomeException )
        while t.is_alive():
          time.sleep( 0.1 )
          t.raiseException( SomeException )
      If the exception is to be caught by the thread, you need a way to check that your thread has caught it.
      CAREFUL : this function is executed in the context of the caller thread, to raise an excpetion in the context of the
                thread represented by this instance.
      @ In, exceptionType, Exception, the type of exception to raise in this thread
      @ Out, None
    """
    if self.is_alive():
      ## Assuming Python 2.6+, we can remove the need for the _get_my_tid as
      ## specifed in the Stack Overflow answer
      _asyncRaise( self.ident, exceptionType )
