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
    self.resampleSwitch = False
    self.useCentralDiff = True


  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Local method for additional reading.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    SPSA.localInputAndChecks(self, xmlNode, paramInput)
    # need extra eval for central Diff, using boolean in math
    self.paramDict['pertSingleGrad'] = (1 + self.useCentralDiff) * len(self.fullOptVars)
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * (self.paramDict['pertSingleGrad']+1)
    if self.useCentralDiff:
      self.raiseADebug('Central differencing activated!')

  ###############
  # Run Methods #
  ###############

  ###################
  # Utility Methods #
  ###################
  def _getPerturbationDirection(self, perturbationIndex, step = None):
    """
      This method is aimed to get the perturbation direction (i.e. in this case the random perturbation versor)
      @ In, perturbationIndex, int, the perturbation index (stored in self.perturbationIndices)
      @ In, step, int, the step index, zero indexed, if not using central gradient, then passing the step index to flip the sign of the direction for FD optimizer.
      @ Out, direction, list, the versor for each optimization dimension
    """
    _, varId, denoId, cdId = self._identifierToLabel(perturbationIndex)
    direction = np.zeros(len(self.getOptVars())).tolist()
    if cdId == 0:
      direction[varId] = 1.0
    else:
      direction[varId] = -1.0
    if step:
      if step % 2 == 0:
        factor = 1.0
      else:
        # flip the sign of the direction for FD optimizer, this step will not affect central differancing
        # but will make the direction between forward and backward
        # for example of 2 variables 3 denoise W/O central diff:
        # step 0 directions are ([1 0],[0 1])*3
        # step 1 directions are ([-1 0],[0 -1])*3
        # with central diff:
        # step 0 directions are ([1 0],[0 1],[-1,0],[0,-1])*3
        # step 1 directions are ([-1,0],[0,-1],[1 0],[0 1])*3
        factor = -1.0
      direction = [var * factor for var in direction]

    self.currentDirection = direction
    return direction

  def localEvaluateGradient(self, traj, gradHist = False):
    """
      Local method to evaluate gradient.
      @ In, traj, int, the trajectory id
      @ In, gradHist, bool, optional, whether store  self.counter['gradientHistory'] in this step.
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    gradient = {}
    inVars = self.getOptVars()
    opt = self.realizations[traj]['denoised']['opt'][0]
    allGrads = self.realizations[traj]['denoised']['grad']
    gi = {}
    for g,pert in enumerate(allGrads):
      varId = g % len(inVars)
      var = inVars[varId]
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
      if gradient.get(var) == None:
        gi[var] = 0
        gradient[var] = np.atleast_1d(lossDiff / dh)
      else:
        gi[var] += 1
        gradient[var] = (gradient[var] + np.atleast_1d(lossDiff / dh))* gi[var]/(gi[var]  + 1)
    if gradHist:
      try:
        self.counter['gradientHistory'][traj][1] = self.counter['gradientHistory'][traj][0]
      except IndexError:
        pass # don't have a history on the first pass
      self.counter['gradientHistory'][traj][0] = gradient
    return gradient
