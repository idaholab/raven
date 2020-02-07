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
  Class to maintain APIs with other optimization libraries

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

class Interfaced(Optimizer):
  """
    Sets up interface to external libraries for optimization.
    TODO should this inherit from sampler? Or set up its own API?
    Handles the following:
     - API for interfacing with external libraries for optimization
     - Handles variables to and from Optimizer interface
     - Handles interpreting constraints for external libraries
     - Establishes API for wrapping Models (from input creation through running)
     - Sets up compatability with generateInput from Optimizer/Sampler
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
  # TODO