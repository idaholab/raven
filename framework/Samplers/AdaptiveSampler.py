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

#Internal Modules
from utils import mathUtils
from .Sampler import Sampler


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
    self._targetEvaluation = None       # data object with feedback from sample realizations
    self._solutionExport = None         # data object for solution printing

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    Sampler.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    self._targetEvaluation = self.assemblerDict['TargetEvaluation'][0][3]
    self._solutionExport = solutionExport

  def _registerSample(self, prefix, info):
    """
      Register a sample's prefix info before submitting as job
      @ In, prefix, str, string integer prefix
      @ In, info, dict, unique information to record associated with the prefix
      @ Out, None
    """
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

  def stillLookingForPrefix(self, prefix):
    """
      Checks if a prefix is still registered for collection.
      @ In, prefix, str, job prefix
      @ Out, looking, bool, True if still waiting for prefix else False
    """
    return prefix in self._prefixToIdentifiers

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
    """
      checks that all registered identifiers have values
      @ In, checkDict, dict, dictionary of identifying information for a realization
      @ Out, None
    """
    assert self._registeredIdentifiers.issubset(set(checkDict.keys())), 'missing identifiers: {}'.format(self._registeredIdentifiers - set(checkDict.keys()))

  def getIdentifierFromPrefix(self, prefix, pop=False):
    """
      Obtains the identifying info dict given a prefix
      @ In, prefix, str, identifying prefix
      @ In, pop, bool, optional, if True then stop tracking prefix after providing it
      @ Out, ID, dict, identifying information (or None if not present)
    """
    if pop:
      return self._prefixToIdentifiers.pop(prefix, None)
    else:
      return self._prefixToIdentifiers.get(prefix, None)

  def getPrefixFromIdentifier(self, idDict, pop=False, getAll=False):
    """
      Obtains a prefix given identifying information
      @ In, idDict, dict, identifying information about a realization
      @ In, pop, bool, optional, if True then stop tracking prefix after providing
      @ In, getAll, bool, optional, if True then get all matching items instead of the first
      @ Out, prefix, str, identifying prefix (or None if not found)
    """
    # make sure the request matches the expected form
    if getAll:
      found = []
    else:
      found = None
    # if not collecting many, check all identifiers used
    if not getAll:
      self.checkIdentifiersPresent(idDict)
    # find the match
    toPop = []
    for prefix, info in self._prefixToIdentifiers.items():
      if all(list((v == info[k]) for k, v in idDict.items())):
        if pop:
          toPop.append(prefix)
        if getAll:
          found.append(prefix)
        else:
          found = prefix
          break
    for p in toPop:
      self._prefixToIdentifiers.pop(p)
    return found
  
