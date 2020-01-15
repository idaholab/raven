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
  This module containes the base class fo all the Adaptive Sampling Strategies

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa (2/16/2013)
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal
#Modules------------------------------------------------------------------------------------
from utils import mathUtils
from .Sampler import Sampler
#Internal Modules End--------------------------------------------------------------------------------

class AdaptiveSampler(Sampler):
  """
    This is a general adaptive sampler
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Sampler.__init__(self)
    self._registeredIdentifiers = set() # tracks job identifiers used for this adaptive sampler and its inheritors
    self._prefixToIdentifiers = {}      # tracks the mapping of run prefixes to particular identifiers
    self._inputIdentifiers = {}         # identifiers for a single realization

  def _registerSample(self, prefix, info):
    """ TODO """
    self.checkIdentifiersPresent(info)
    self._prefixToIdentifiers[prefix] = info

  def _checkSample(self):
    """
      Check sample consistency.
      @ In, None
      @ Out, None
    """
    Sampler._checkSample(self)
    # make sure the prefix is registered for tracking
    ## but if there's no identifying information, skip this check
    if self._registeredIdentifiers:
      prefix = self.inputInfo['prefix']
      if not prefix in self._prefixToIdentifiers:
        self.raiseAnError(RuntimeError, 'Prefix "{p}" has not been tracked in adaptive sampling!'.format(p=prefix))

  ##########################################
  # Utilities for Prefix-Identifier System #
  ##########################################
  def registerIdentifier(self, name):
    """
      Establishes an identifying attribute for a job run.
      Assures no conflicts with existing identifiers.
      @ In, name, str, identifier to register
      @ Out, None
    """
    assert mathUtils.isAString(name)
    assert name not in self._registeredIdentifiers
    # don't allow adding identifiers if existing jobs are already running, I think?
    assert not self._prefixToIdentifiers
    self._registeredIdentifiers.add(name)

  def checkIdentifiersPresent(self, checkDict):
    """ TODO checks that all identifiers registered have values """
    assert self._registeredIdentifiers.issubset(set(checkDict.keys()))

  def getIdentifierFromPrefix(self, prefix, pop=False):
    """ TODO """
    if pop:
      return self._prefixToIdentifiers.pop(prefix, None)
    else:
      return self._prefixToIdentifiers.get(prefix, None)

  def getPrefixFromIdentifier(self, idDict, pop=False):
    """ TODO get a prefix given identifying information """
    # make sure the request matches the expected form
    self.checkIdentifiersPresent(idDict)
    # find the match
    for prefix, info in self._prefixToIdentifiers.items():
      if info == idDict:
        if pop:
          self._prefixToIdentifiers.pop(prefix)
        return prefix
    # if no matches found ...
    return None




