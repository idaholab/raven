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
  Realizations carry sampled information between entities in RAVEN
"""

class Realization:
  """
    A mapping container specifically for carrying data between entities in RAVEN, such
    as the Sampler and Step.
    See https://docs.python.org/3/reference/datamodel.html?emulating-container-types=#emulating-container-types
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self._values = {}    # mapping of variables to their values
    self.inputInfo = {}  # additional information about this realization
    self.indexMap = {}   # information about dimensionality of variables
    self.labels = {}     # custom labels for tracking, set externally

  ########
  #
  # dict-like members
  #
  def __len__(self):
    """
      Python built-in for realization length.
      @ In, None
      @ Out, len, number of variables in realization
    """
    return len(self._values)

  def __getitem__(self, key):
    """
      Python built-in for acquiring values.
      @ In, key, str, variable name
      @ Out, item, any, contents of realization corresponding to variable
    """
    return self._values[key]

  def __setitem__(self, key, value):
    """
      Python built-in for setting values.
      @ In, key, str, variable name
      @ In, value, any, corresponding value
      @ Out, None
    """
    self._values[key] = value

  def __delitem__(self, key):
    """
      Python built-in for removing values.
      @ In, key, str, variable name
      @ Out, None
    """
    # TODO also remove from inputInfo and indexMap?
    del self._values[key]

  # TODO needed? Used when getitem is called and it's not present
  #def __missing__(self, key):
  #  return self._values.__missing__(key)

  def __iter__(self):
    """
      Python built-in for providing iterator for keys.
      @ In, None
      @ Out, iter, iterable, variable names
    """
    return iter(self._values)

  def __contains__(self, item):
    """
      Python built-in for "in" patterns such as "x in d"
      @ In, item, str, variable name
      @ Out, in, bool, True if variable name in realization
    """
    return item in self._values

  def update(self, *args, **kwargs):
    """
      Python built-in for updating many key-value pairs at once
      @ In, args, list, list arguments
      @ In, kwargs, dict, dictionary arguments
      @ Out, None
    """
    return self._values.update(*args, **kwargs)

  def keys(self):
    """
      Python built-in for acquiring variable names
      @ In, None
      @ Out, keys, list, list of var names
    """
    return self._values.keys()

  def values(self):
    """
      Python built-in for acquiring variable values
      @ In, None
      @ Out, keys, list, list of var values
    """
    return self._values.values()

  def items(self):
    """
      Python built-in for acquiring variable (name, value) pairs
      @ In, None
      @ Out, keys, list, list of var (name, value) tuples
    """
    return self._values.items()

  def pop(self, *args):
    """
      Python built-in for removing and returning entry in realization
      @ In, None
      @ Out, pop, any, value of corresponding key
    """
    # TODO extend to other info dicts?
    return self._values.pop(*args)


