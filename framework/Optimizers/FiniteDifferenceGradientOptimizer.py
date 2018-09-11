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
    self.gradDict['pertNeeded']      = self.gradDict['numIterForAve'] * (self.paramDict['pertSingleGrad']+1)

  def _getPerturbationDirection(self,perturbationIndex, traj):
    """
      This method is aimed to get the perturbation direction (i.e. in this case the random perturbation versor)
      @ In, perturbationIndex, int, the perturbation index (stored in self.perturbationIndices)
      @ In, traj, int, the trajectory id
      @ Out, direction, list, the versor for each optimization dimension
    """
    optVars = self.getOptVars(traj)
    if len(optVars) == 1:
      if self.currentDirection:
        factor = np.sum(self.currentDirection)*-1.0
      else:
        factor = 1.0
      direction = [factor]
    else:
      if perturbationIndex == self.perturbationIndices[0]:
        direction = np.zeros(len(self.getOptVars(traj))).tolist()
        factor = 1.0
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

  def localEvaluateGradient(self, optVarsValues, traj, gradient = None):
    """
      Local method to evaluate gradient.
      @ In, optVarsValues, dict, dictionary containing perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ In, traj, int, the trajectory id
      @ In, gradient, dict, optional, dictionary containing gradient estimation by the caller.
                                      gradient should have the form {varName: gradEstimation}
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    gradArray = {}
    optVars = self.getOptVars(traj=traj)
    numRepeats = self.gradDict['numIterForAve']
    for var in optVars:
      gradArray[var] = np.zeros(self.gradDict['numIterForAve'],dtype=object)
    # optVarsValues:
    #  - the first <numRepeats> entries are the opt point (from 0 to numRepeats-1)
    #  - the next <numRepeats> entries are one each in each direction in turns (dx1, dy1, dx2, dy2, etc)
    #      dx are [lastOpt +1, lastOpt +3, lastOpt +5, etc] -> [lastOpt + <#var>*<index repeat>+1]
    #      dy are [lastOpt +2, lastOpt +4, lastOpt +6, etc]
    # Evaluate gradient at each point
    for i in range(numRepeats):
      opt  = optVarsValues[i] #the latest opt point
      for j in range(self.paramDict['pertSingleGrad']): # AKA for each input variable
        # loop over the perturbation to construct the full gradient
        ## first numRepeats are all the opt point, not the perturbed point
        ## then, need every Nth entry, where N is the number of variables
        pert = optVarsValues[numRepeats + i*len(optVars) + j] #the perturbed point
        #calculate grad(F) wrt each input variable
        lossDiff = mathUtils.diffWithInfinites(pert['output'],opt['output'])
        #cover "max" problems
        # TODO it would be good to cover this in the base class somehow, but in the previous implementation this
        #   sign flipping was only called when evaluating the gradient.
        if self.optType == 'max':
          lossDiff *= -1.0
        var = optVars[j]
        # gradient is calculated in normalized space
        dh = pert['inputs'][var] - opt['inputs'][var]
        if abs(dh) < 1e-15:
          self.raiseADebug('Values:',pert['inputs'][var],opt['inputs'][var])
          self.raiseAnError(RuntimeError,'While calculating the gradArray a "dh" very close to zero was found for var:',var)
        gradArray[var][i] = lossDiff/dh
    gradient = {}
    for var in optVars:
      gradient[var] = np.atleast_1d(gradArray[var].mean())
    return gradient
