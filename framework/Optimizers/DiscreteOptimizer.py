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
  This module contains the base class for all discrete optimizers

  Created on January 9, 2019
  @author: mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
import abc
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Optimizer import Optimizer
from Assembler import Assembler
from utils import utils,cached_ndarray,mathUtils
#Internal Modules End--------------------------------------------------------------------------------

class DiscreteOptimizer(Optimizer):
  """
    This is the base class for all discrete optimizers. The following methods need to be overridden by all derived class
    self.localLocalInputAndChecks(self, xmlNode)
    self.localLocalInitialize(self, solutionExport)
    self.localLocalGenerateInput(self,model,oldInput)
  """

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Optimizer.__init__(self)

  def localInputAndChecks(self, xmlNode):
    """
      Method to read the portion of the xml input that belongs to all gradient based optimizer only
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    pass

  def localInitialize(self):
    """
      Method to initialize settings that belongs to all gradient based optimizer
      @ In, solutionExport, DataObject, a PointSet to hold the solution
      @ Out, None
    """
    pass

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    pass
