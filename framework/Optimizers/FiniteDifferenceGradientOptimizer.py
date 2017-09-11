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
  This module contains the Finite Difference Gradient Optimization strategy

  Created on Sept 10, 2017
  @ author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
import numpy as np
from numpy import linalg as LA
import scipy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SPSA import SPSA
from utils import mathUtils,randomUtils
#Internal Modules End--------------------------------------------------------------------------------

class FiniteDifferenceGradientOptimizer(SPSA):
  """
    Finite Difference Gradient Optimizer
    This class currently inherits from the SPSA (since most of the gradient based machinery is there).
    TODO: Move the SPSA machinery here (and GradientBasedOptimizer) and make the SPSA just take care of the
    random perturbation approach
  """
  def __init__(self):
    """
      Default Constructor
    """
    SPSA.__init__(self)

  def localInputAndChecks(self, xmlNode):
    """
      Local method for additional reading.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    SPSA.localInputAndChecks(self, xmlNode)
    self.paramDict['pertSingleGrad'] = len(self.fullOptVars)

  #def localLocalInitialize(self, solutionExport):
  #  """
  #    Method to initialize local settings.
  #    @ In, solutionExport, DataObject, a PointSet to hold the solution
  #    @ Out, None
  #  """
  #  SPSA.localLocalInitialize(solutionExport)

  def _getPerturbationDirection(self,perturbationIndex):
    """
      This method is aimed to get the perturbation direction (i.e. in this case the finite difference versor)
      @ In, perturbationIndex, int, the perturbation index (stored in self.perturbationIndeces)
      @ Out, direction, list, the versor for each optimization dimension
    """
    if perturbationIndex == self.perturbationIndeces[0]:
      direction = self.stochasticEngine()
      self.currentDirection = direction
    else:
      # in order to perform the de-noising we keep the same perturbation direction and we repeat the evaluation multiple times
      direction = self.currentDirection
    return direction
