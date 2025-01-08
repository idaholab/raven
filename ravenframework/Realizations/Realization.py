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
import numpy as np
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
    self._values = {}      # mapping of variables to their values
    self.indexMap = {}     # information about dimensionality of variables
    self.labels = {}       # custom labels for tracking, set externally
    self.isRestart = False # True if model was not run, but data was taken from restart
    self.inputInfo = {'SampledVars': {},   # additional information about this realization
                      'SampledVarsPb': {}, # point probability information for this realization
    }

  ########
  #
  # other useful methods
  #
  def setRestart(self, varVals):
    """
      Sets this Realization to have values coming from a restart point.
      @ In, varVals, dict, new var-value mapping
      @ Out, None
    """
    self.update(varVals)
    self.isRestart = True

  def asDict(self):
    """
      Collects all the information this Realization knows about and returns it
      Also assures that all entries are at least 1d np arrays
      @ In, None
      @ Out, info, dict, all the things
    """
    # TODO this is one-way, no easy way to unpack labels and input info back into rlz form
    # TODO any deep copies needed? Let's assume no.
    info = dict((var, np.atleast_1d(val)) for var, val in self._values.items())
    info['_indexMap'] = np.atleast_1d(self.indexMap)
    info.update(dict((key, np.atleast_1d(val)) for key, val in self.inputInfo.items()))
    info.update(dict((label, np.atleast_1d(val)) for label, val in self.labels.items()))
    return info

  def createSubsetRlz(self, targetVars, ignoreMissing=True):
    """
      Creates a realization, retaining the data in this realization but with only a subset
      of variables. Ignores any targetVars that aren't part of this rlz.
      @ In, targetVars, list(str), list of variable names to retain
      @ In, ignoreMissing, bool, if True then don't error if some entries missing
      @ Out, new, Realization, new realization instance
    """
    new = Realization()
    varKeyedEntries = []
    oneVar = next(iter(self._values))
    for key, entry in self.inputInfo.items():
      # assuming the only entries relevant to variables are first-layer dicts in inputInfo ...
      if isinstance(entry, dict) and oneVar in entry:
        new[key] = {}
        varKeyedEntries.append(key)
      # TODO other exceptions to handle?
      else:
        new.inputInfo[key] = entry
    # fill values from this rlz into the new one
    print('DEBUGG RLZ targetVars:', targetVars)
    for tvar in targetVars:
      if tvar in self._values:
        new[tvar] = self._values[tvar]
        for key in varKeyedEntries:
          if key in self.inputInfo[key]:
            new.inputInfo[key][tvar] = self.inputInfo[key][tvar]
      elif not ignoreMissing:
        raise KeyError(f'Desired variable "{tvar}" missing from source Realization!')
    return new



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

  def get(self, key, default=None):
    """
      Accessor for acquiring values.
      @ In, key, str, variable name
      @ Out, item, any, contents of realization corresponding to variable
    """
    if default is None:
      return self._values.get(key)
    else:
      return self._values.get(key, default)


