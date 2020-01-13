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
  Base class for Sampled Optimizers using gradient descent optimization methods.

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
from .Sampled import Sampled
#Internal Modules End--------------------------------------------------------------------------------

class GradientDescent(Sampled):
  """
    Base class for Sampled Optimizers using gradient descent optimization methods.
    Handles the following:
     - Implements API for step size handling
       - Initialization, iteration, on constrain violation, etc
     - Implements API for gradient handling
       - Algorithm for estimating local/global gradient
       - Perturbation distance
       - Perturbation direction
     - Implements method(s) for stepping around constraints
     - Implements history tracking
       - evaluations, gradients, step sizes
     - Implements trajectory handling
       - Initial points (user or sampled)
       - Trajectory removal criteria (using Optimizer API)
     - Implement iterative step limit checking
     - Implement relative/absolute convergence
       - converge on gradient magnitude, change in evaluation, min step size
     - Implement summary of step iteration to SolutionExport
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
    # TODO

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

    # _protected
    
    # __private

    # additional methods

  def handleInput(self, TODO):
    """ TODO """

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
  # add the opt/grad point identifier
  optOrGrad = kwargs['optOrGrad']
  otherInfo.append(optOrGrad)
  # allow base class to contribute
  return Sampled._createPrefix(self, info=otherInfo, **kwargs)

def _deconstructPrefix(self, prefix):
  """
    Deconstruct a prefix as far as this instance knows how.
    @ In, prefix, str, label for a realization
    @ Out, info, dict, {traj: #, resample: #}, information about the realization
    @ Out, rem, str, remainder of the prefix (with prior information removed)
  """
  # allow base class to peel off it's part
  info, prefix = Sampled._createPrefix(self, prefix)
  asList = prefix.split('_')
  # get the opt (0) or grad ID (1:N)
  info['optOrGrad'] = int(asList[0])
  rem = asList[1:].join('_')
  return info, rem