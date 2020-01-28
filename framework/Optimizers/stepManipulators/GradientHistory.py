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
  Step size manipulations based on gradient history

  Created 2020-01
  @author: talbpaul
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import InputData, InputTypes, mathUtils
from .StepManipulator import StepManipulator
#Internal Modules End--------------------------------------------------------------------------------

class GradientHistory(StepManipulator):
  """
    Changes step size depending on history of gradients
  """
  requiredInformation = ['gradientHist', 'prevStepSize']
  optionalInformation = ['recommend']

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
    specs = super(GradientHistory, cls).getInputSpecification()
    specs.addSub(InputData.parameterInputFactory('growthFactor', contentType=InputTypes.FloatType))
    specs.addSub(InputData.parameterInputFactory('shrinkFactor', contentType=InputTypes.FloatType))
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    StepManipulator.__init__(self)
    # TODO
    ## Instance Variable Initialization
    # public
    # _protected
    self._optVars = None
    self._growth = 1.25
    self._shrink = 1.15
    # __private
    # additional methods

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    #specs = specs
    growth = specs.findFirst('growthFactor')
    if growth is not None:
      self._growth = growth.value
    shrink = specs.findFirst('shrinkFactor')
    if shrink is not None:
      self._shrink = shrink.value

  def initialize(self, optVars, **kwargs):
    """ TODO """
    self._optVars = optVars
    StepManipulator.initialize(self, optVars, **kwargs)


  ###############
  # Run Methods #
  ###############
  def initialStepSize(self, numOptVars=None, scaling=0.05, **kwargs):
    """
      Provides an initial step size
      @ In, numOptVars, int, number of optimization variables
      @ In, scaling, float, optional, scaling factor
    """
    return mathUtils.hyperdiagonal(np.ones(numOptVars) * scaling)

  def step(self, prevOpt, gradientHist=None, prevStepSize=None, recommend=None, **kwargs):
    """
      calculates the step size and direction to take
      @ In, prevOpt, dict, previous opt point
      @ In, gradientHist, deque, list of gradient dictionaries with 0 being oldest; versors
      @ In, prevStepSize, deque, list of float step sizes
      @ In, recommend, str, optional, override to 'grow' or 'shrink' step size
      @ In, kwargs, dict, keyword-based specifics as required by individual step sizers
      @ Out, newOpt, dict, new opt point
      @ Out, stepSize, float, new step size
    """
    stepSize = self._stepSize(gradientHist=gradientHist, prevStepSize=prevStepSize,
                              recommend=recommend, **kwargs)
    gradient = gradientHist[-1][1]
    # use gradient, prev point, and step size to choose new point
    newOpt = {}
    for var in self._optVars:
      newOpt[var] = prevOpt[var] - stepSize * gradient[var]
    return newOpt, stepSize

  def _stepSize(self, gradientHist=None, prevStepSize=None, recommend=None, **kwargs):
    """
      Calculates a new step size to use in the optimization path.
      @ In, gradientHist, deque, list of gradient dictionaries with 0 being oldest; versors
      @ In, prevStepSize, deque, list of float step sizes
      @ In, recommend, str, optional, override to 'grow' or 'shrink' step size
      @ In, kwargs, dict, keyword-based specifics as required by individual step sizers
      @ Out, stepSize, float, new step size
    """
    grad0 = gradientHist[-1][1]
    grad1 = gradientHist[-2][1] if len(gradientHist) > 1 else None
    gainFactor = self._fractionalStepChange(grad0, grad1, recommend=recommend)
    stepSize = gainFactor * prevStepSize[-1]
    return stepSize

  ###################
  # Utility Methods #
  ###################
  def _fractionalStepChange(self, grad0, grad1, recommend=None):
    """
      Calculates fractional step change based on gradient history
      @ In, grad0, dict, most recent gradient direction (versor)
      @ In, grad1, dict, next recent gradient direction (versor)
      @ In, recommend, str, optional, can override gradient-based suggestion to either cut or grow
      @ Out, factor, multiplicitave factor to use on step size
    """
    assert grad0 is not None
    # grad1 can be None if only one point has been taken
    assert recommend in [None, 'shrink', 'grow']
    if recommend:
      if recommend == 'shrink':
        factor = 1. / self._shrink
      else:
        factor = self._growth
      return factor
    # if history is only a single gradient, then keep step size the same for now
    if grad1 is None:
      return 1.0
    # otherwise, figure it out based on the gradient history
    # scalar product
    prod = np.sum([np.sum(grad0[v] * grad1[v]) for v in grad0.keys()])
    if prod > 0:
      factor = self._growth ** prod
    else:
      # NOTE prod is negative, so this is like 1 / (shrink ^ abs(prod))
      factor = self._shrink ** prod
    return factor

