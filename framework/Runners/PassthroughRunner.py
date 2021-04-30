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
  Module for Passthrough Runner class, which skips evaluation. Used particularly
  for restarting Samplers from existing data currently.
"""
import numpy as np
from .Runner import Runner

class PassthroughRunner(Runner):
  """
    A runner for when we already have the answer, but need to go through the mechanics.
  """
  def __init__(self, data, func, **kwargs):
    """
      Construct
      @ In, data, dict, fully-evaluated realization
      @ In, func, None, placeholder for consistency with other runners
      @ In, kwargs, dict, additional arguments to pass to base
      @ Out, None
    """
    super().__init__(**kwargs)
    self._data = data   # realization with completed data
    self.returnCode = 0 # passthrough was born successful

  def isDone(self):
    """
      Method to check if the calculation associated with this Runner is finished
      @ In, None
      @ Out, isDone, bool, is it finished?
    """
    return True # passthrough was born done

  def getReturnCode(self):
    """
      Returns the return code from "running the code."
      @ In, None
      @ Out, returnCode, int,  the return code of this evaluation
    """
    return self.returnCode

  def getEvaluation(self):
    """
      Return solution.
      @ In, None
      @ Out, result, dict, results
    """
    result = {}
    result.update(dict((key, np.atleast_1d(value)) for key, value in self._data['inputs'].items()))
    result.update(dict((key, np.atleast_1d(value)) for key, value in self._data['outputs'].items()))
    result.update(dict((key, np.atleast_1d(value)) for key, value in self._data['metadata'].items()))
    return result

  def start(self):
    """
      Method to start the job associated to this Runner
      @ In, None
      @ Out, None
    """
    pass # passthrough was born done

  def kill(self):
    """
      Method to kill the job associated to this Runner
      @ In, None
      @ Out, None
    """
    pass # passthrough was born done; you can't kill it
