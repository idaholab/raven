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
from .gradients import knownTypes as gradKnownTypes
from .gradients import returnInstance as gradReturnInstance
from .gradients import returnClass as gradReturnClass
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
    specs = super(GradientDescent, cls).getInputSpecification()
    # gradient estimation options
    grad = InputData.parameterInputFactory('gradientEstimation', strictMode=True)
    specs.addSub(grad)

    ## common options to all gradient descenders
    # TODO grad.addSub(InputData.parameterInputFactory('proximity', contentType=InputTypes.FloatType))

    ## get specs for each gradient subclass, and add them to this class's options
    for option in gradKnownTypes():
      subSpecs = gradReturnClass(option, cls).getInputSpecification()
      grad.addSub(subSpecs)
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Sampled.__init__(self)
    # TODO

    ## Instance Variable Initialization
    # public

    # _protected
    self._gradientInstance = None # instance of GradientApproximater

    # __private

    # additional methods

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    Sampled.handleInput(self, paramInput)
    # grad strategy
    gradNode = paramInput.findFirst('gradient')
    # TODO do I need to check for the node's existence?
    if len(gradNode.subs) != 1:
      self.raiseAnError('The <gradient> node requires exactly one gradient strategy! Choose from: ', gradKnownTypes())
    gradNode = gradNode.subs.keys()[0]
    gradType = gradNode.getName()
    self._gradientInstance = gradReturnInstance(gradType, self)
    self._gradientInstance.handleInput(gradNode)

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    Sampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)

  ###############
  # Run Methods #
  ###############
  def localGenerateInput(self, model, input):
    """
      TODO
    """

  def checkConvergence(self):
    """
      TODO
    """
  ###################
  # Utility Methods #
  ###################