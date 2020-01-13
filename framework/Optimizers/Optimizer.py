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
  The Optimizer is a specialization of adaptive sampling.
  This base class defines the principle methods required for optimizers and provides some general utilities.

  Reworked 2020-01
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
from collections import deque
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils, randomUtils, InputData, InputTypes
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
from Samplers import Sampler
#Internal Modules End--------------------------------------------------------------------------------

class Optimizer(Sampler):
  """
    The Optimizer is a specialization of adaptive sampling.
    This base class defines the principle methods required for optimizers and provides some general utilities.
    This base class is responsible for:
     - Implementing Sampler API
     - Handling stochastic resampling
     - Establishing "trajectory" counter
     - Handling Constant, Function variables
     - Specifying objective variable
     - Assembling constraints
     - API for adding, removing trajectories
     - Prefix handling for trajectory, denoising
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
    Sampler.__init__(self)
    ## Instance Variable Initialization
    # public

    # _protected
    self._activeTraj = []      # tracks live trajectories
    self._numRepeatSamples = 1 # number of times to repeat sampling (e.g. denoising)
    self._objectiveVar = None  # objective variable for optimization
    
    # __private

    # additional methods
    self.addAssemblerObject('TargetEvaluation','1') # Place where realization evaluations go
    self.addAssemblerObject('Constraint','-1')      # Explicit (input-based) constraints
    self.addAssemblerObject('Sampler','-1')         # This Sampler can be used to initialize the optimization initial points (e.g. partially replace the <initial> blocks for some variables)

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      Overloads the base Sampler class since optimizer has different requirements
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    self.assemblerDict['Functions'    ] = []
    self.assemblerDict['Distributions'] = []
    self.assemblerDict['DataObjects'  ] = []
    for mainClass in ['Functions','Distributions','DataObjects']:
      for funct in initDict[mainClass]:
        self.assemblerDict[mainClass].append([mainClass,initDict[mainClass][funct].type,funct,initDict[mainClass][funct]])

  def _localWhatDoINeed(self):
    """
      Identifies needed distributions and functions.
      Overloads Sampler base implementation because of unique needs.
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {}
    needDict['Distributions'] = [(None,'all')]
    needDict['Functions'    ] = [(None,'all')]
    needDict['DataObjects'  ] = [(None,'all')]
    return needDict

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
  def amIreadyToProvideAnInput(self):
    """
      This is a method that should be called from any user of the optimizer before requiring the generation of a new input.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of model evaluation, convergence criteria met, etc.
      @ In, None
      @ Out, ready, bool, indicating the readiness of the optimizer to generate a new input.
    """
    # TODO

  ###################
  # Utility Methods #
  ###################
  def checkConstraint(self, optVars):
    """
      Method to check whether a set of decision variables satisfy the constraint or not in UNNORMALIZED input space
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ Out, satisfaction, tuple, (bool,list) => (variable indicating the satisfaction of constraints at the point optVars, masks for the under/over violations)
    """
    # TODO

  @abc.abstractmethod
  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, none,
      @ Out, convergence, bool, variable indicating whether the convergence criteria has been met.
    """
    # TODO

  def checkIfBetter(self, a, b):
    """
      Checks if a is preferable to b for this optimization problem.  Helps mitigate needing to keep
      track of whether a minimization or maximation problem is being run.
      @ In, a, float, value to be compared
      @ In, b, float, value to be compared against
      @ Out, checkIfBetter, bool, True if a is preferable to b for this optimization
    """
    if self.optType == 'min':
      return a <= b
    elif self.optType == 'max':
      return a >= b

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
    # prefixes start with the trajectory
    traj = kwargs['traj']
    otherInfo.insert(0, traj)
    # prefixes end with the resample number
    resample = kwargs['resample']
    otherInfo.append(resample)
    return otherInfo

  def _deconstructPrefix(self, prefix):
    """
      Deconstruct a prefix as far as this instance knows how.
      @ In, prefix, str, label for a realization
      @ Out, info, dict, {traj: #, resample: #}, information about the realization
      @ Out, rem, str, remainder of the prefix (with prior information removed)
    """
    asList = prefix.split('_')
    # since this is the base class, create the info
    ## include the trajectory number
    ## and also the stochastic denoising/resampling number
    info = {'traj': int(asList[0]),
            'resample': int(asList[-1]),
           }
    rem = asList[1:-1].join('_')
    return info, rem

  def denormalizeData(self, normalized):
    """
      Method to normalize the data
      @ In, normalized, dict, dictionary containing the value of decision variables to be deormalized, in form of {varName: varValue}
      @ Out, denormed, dict, dictionary containing the value of denormalized decision variables, in form of {varName: varValue}
    """
    # TODO

  def normalizeData(self, denormed):
    """
      Method to normalize the data
      @ In, denormed, dict, dictionary containing the value of decision variables to be normalized, in form of {varName: varValue}
      @ Out, normalized, dict, dictionary containing the value of normalized decision variables, in form of {varName: varValue}
    """
    # TODO
