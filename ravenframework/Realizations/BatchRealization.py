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
  Collection of Realizations for convenient co-packaging
"""
from . import Realization

class BatchRealization:
  """
    A container for groups of Realization objects, that should mostly invisibly work like a realization
  """
  def __init__(self, batchSize):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    # TODO are any of these shared across realizations?
    # self._values = {}    # mapping of variables to their values
    # self.inputInfo = {'SampledVars': {},  # additional information about this realization
    #                   'SampledVarsPb': {},
    #                   'crowDist': {}
    # }
    # self.indexMap = {}   # information about dimensionality of variables
    self.batchSize = batchSize # number of realizations that are part of this object
    self._realizations = [Realization() for _ in range(min(batchSize, 1))]


  ########
  #
  # other useful methods
  #

  ########
  #
  # dict-like members
  #
  def __len__(self):
    """
      Python built-in for realization length.
      @ In, None
      @ Out, len, int, number of realizations in batch
    """
    return len(self._realizations)

  def __getitem__(self, index):
    """
      Python built-in for acquiring values.
      @ In, index, int, index of desired item
      @ Out, item, any, contents of realization corresponding to variable
    """
    return self._realizations[index]

  def __setitem__(self, index, value):
    """
      Python built-in for setting values.
      @ In, index, int, index of desired item
      @ In, value, any, corresponding value
      @ Out, None
    """
    raise IndexError('Tried to overwrite a Realization object in a Batch!')

  def pop(self, *args):
    """
      Python built-in for removing and returning entry in realization
      @ In, None
      @ Out, pop, any, value of corresponding index
    """
    return self._realizations.pop(*args)

