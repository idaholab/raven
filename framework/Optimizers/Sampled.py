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
  Base class for Optimizers using RAVEN's internal sampling mechanics.

  Created 2020-01
  @author: talbpaul
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import abc
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils, randomUtils, InputData, InputTypes
from BaseClasses import BaseType
from Assembler import Assembler
from .Optimizer import Optimizer
#Internal Modules End--------------------------------------------------------------------------------

class Sampled(Optimizer):
  """
    Base class for Optimizers using RAVEN's internal sampling mechanics.
    Handles the following:
     - Maintain queue for required realizations
     - Label and retrieve realizations given labels
     - Establish API for convergence checking
     - Establish API to extend labels for particular implementations
     - Implements constraint checking
     - Implements model evaluation limitations
     - Implements rejection strategy (?)
     - Implements convergence persistence
     - Establish API for iterative sample output to solution export
     - Implements specific sampling methods from Sampler (when not present in Optimizer)
  """

  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(Sampled, cls).getInputSpecification()
    # initialization: add sampling-based options
    whenSolnExpEnum = InputTypes.makeEnumType('whenWriteEnum', 'whenWriteType', ['final', 'every'])
    init = specs.getSub('samplerInit')
    #specs.addSub(init)
    limit = InputData.parameterInputFactory('limit', contentType=InputTypes.IntegerType)
    write = InputData.parameterInputFactory('writeSteps', contentType=whenSolnExpEnum)
    init.addSub(limit)
    init.addSub(write)
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Optimizer.__init__(self)
    # TODO

    ## Instance Variable Initialization
    # public
    self.limit = None

    # _protected
    self._writeSteps = 'final'

    # __private

    # additional methods

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    Optimizer.handleInput(self, paramInput)
    # samplerInit
    init = paramInput.findFirst('samplerInit')
    if init is not None:
      # limit
      limit = init.findFirst('limit')
      if limit is not None:
        self.limit = limit.value
      # writeSteps
      writeSteps = init.findFirst('writeSteps')
      if writeSteps is not None:
        self._writeSteps = writeSteps.value


  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    # TODO

  ###############
  # Run Methods #
  ###############
  # TODO

  ###################
  # Utility Methods #
  ###################
  def _createPrefix(self, **kwargs):
    """
      Creates a unique ID to identifiy particular realizations as they return from the JobHandler.
      Expandable by inheritors.
      @ In, args, list, list of arguments
      @ In, kwargs, dict, dictionary of keyword arguments
      @ Out, identifiers, list(str), the evaluation identifiers
    """
    # allow other identifiers as well
    otherInfo = kwargs.get('info', None) # TODO deepcopy?
    if otherInfo is None:
      otherInfo = []
    # add the iteration (or step)
    step = kwargs['step']
    otherInfo.append(step)
    # allow base class to contribute
    return Optimizer._createPrefix(self, info=otherInfo, **kwargs)

def _deconstructPrefix(self, prefix):
  """
    Deconstruct a prefix as far as this instance knows how.
    @ In, prefix, str, label for a realization
    @ Out, info, dict, {traj: #, resample: #}, information about the realization
    @ Out, rem, str, remainder of the prefix (with prior information removed)
  """
  # allow base class to peel off it's part
  info, prefix = Optimizer._createPrefix(self, prefix)
  asList = prefix.split('_')
  # get the iteration (or step, if you will)
  info['iteration'] = int(asList[0])
  rem = asList[1:].join('_')
  return info, rem