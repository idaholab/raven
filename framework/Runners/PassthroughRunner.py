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
Created on July 17, 2020
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

import numpy as np
from .Runner import Runner

class PassthroughRunner(Runner):
  """
    A runner for when we already have the answer, but need to go through the mechanics.
  """
  def __init__(self, messageHandler, data, metadata=None, uniqueHandler="any", profile=False):
    """
      Init method
      @ In, messageHandler, MessageHandler object, the global RAVEN message
        handler object
      @ In, args, dict, this is a list of arguments that will be passed as
        function parameters into whatever method is stored in functionToRun.
        e.g., functionToRun(*args)
      @ In, metadata, dict, optional, dictionary of metadata associated with
        this run
      @ In, uniqueHandler, string, optional, it is a special keyword attached to
        this runner. For example, if present, to retrieve this runner using the
        method jobHandler.getFinished, the uniqueHandler needs to be provided.
        If uniqueHandler == 'any', every "client" can get this runner
      @ In, profile, bool, optional, if True then at deconstruction timing statements will be printed
      @ Out, None
    """
    super(PassthroughRunner, self).__init__(messageHandler, metadata=metadata, uniqueHandler=uniqueHandler, profile=profile)
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