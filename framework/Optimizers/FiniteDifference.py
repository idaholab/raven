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

class FiniteDifference(SPSA):
  """
    Finite Difference Gradient Optimizer
    This class currently inherits from the SPSA (since most of the gradient based machinery is there).
    TODO: Move the SPSA machinery here (and GradientBasedOptimizer) and make the SPSA just take care of the
    random perturbation approach
  """
  ##########################
  # Initialization Methods #
  ##########################
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
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * (self.paramDict['pertSingleGrad']+1)

  ###############
  # Run Methods #
  ###############

  ###################
  # Utility Methods #
  ###################
  def _getPerturbationDirection(self,perturbationIndex):
    """
      This method is aimed to get the perturbation direction (i.e. in this case the random perturbation versor)
      @ In, perturbationIndex, int, the perturbation index (stored in self.perturbationIndices)
      @ Out, direction, list, the versor for each optimization dimension
    """
    optVars = self.getOptVars()
    if len(optVars) == 1:
      if self.currentDirection:
        factor = np.sum(self.currentDirection)*-1.0
      else:
        factor = 1.0
      direction = [factor]
    else:
      if perturbationIndex == self.perturbationIndices[0]:
        direction = np.zeros(len(self.getOptVars())).tolist()
        if self.currentDirection:
          factor = np.sum(self.currentDirection)*-1.0
        else:
          factor = 1.0
        direction[0] = factor
      else:
        index = self.currentDirection.index(1.0) if self.currentDirection.count(1.0) > 0 else self.currentDirection.index(-1.0)
        direction = self.currentDirection
        newIndex = 0 if index+1 == len(direction) else index+1
        direction[newIndex],direction[index] = direction[index], 0.0
    self.currentDirection = direction
    return direction

  def localEvaluateGradient(self, traj):
    """
      Local method to evaluate gradient.
      @ In, traj, int, the trajectory id
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    gradient = {}
    inVars = self.getOptVars()
    opt = self.realizations[traj]['denoised']['opt'][0]
    allGrads = self.realizations[traj]['denoised']['grad']
    for g,pert in enumerate(allGrads):
      var = inVars[g]
      lossDiff = mathUtils.diffWithInfinites(pert[self.objVar],opt[self.objVar])
      # unlike SPSA, keep the loss diff magnitude so we get the exact right direction
      if self.optType == 'max':
        lossDiff *= -1.0
      dh = pert[var] - opt[var]
      if abs(dh) < 1e-15:
        self.raiseADebug('Checking Var:',var)
        self.raiseADebug('Opt point   :',opt)
        self.raiseADebug('Grad point  :',pert)
        self.raiseAnError(RuntimeError,'While calculating the gradArray a "dh" very close to zero was found for var:',var)
      gradient[var] = np.atleast_1d(lossDiff / dh)
    return gradient
