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
  Base class for step sizing strategies in optimization paths

  Created 2020-01
  @author: talbpaul
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import utils, InputData, InputTypes
from ...BaseClasses import MessageUser
#Internal Modules End--------------------------------------------------------------------------------


class StepManipulator(utils.metaclass_insert(abc.ABCMeta, object), MessageUser):
  """
    Base class for handling step sizing in optimization paths
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
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    specs.description = 'Base class for Step Manipulation algorithms in the GradientDescent Optimizer.'
    specs.addSub(InputData.parameterInputFactory('initialStepScale', contentType=InputTypes.FloatType,
        descr=r"""specifies the scale of the initial step in the optimization, in percent of the
              size of the problem. The size of the problem is defined as the hyperdiagonal of the
              input space, composed of the input variables. A value of 1 indicates the first step
              can reach from the lowest value of all inputs to the highest point of all inputs,
              which is too large for all problems with more than one optimization variable. In general this
              should be smaller as the number of optimization variables increases, but large enough
              that the first step is meaningful for the problem. This scaling factor should always
              be less than $1/\sqrt{N}$, where $N$ is the number of optimization variables. \default{0.05} """))

    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, vars, dict, acceptable variable names and descriptions
    """
    return {}

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    # TODO
    # Instance Variable Initialization
    # public
    self.type = self.__class__.__name__
    self.needsAccessToAcceptance = False # if True, then this stepManip may need to modify opt point acceptance criteria
    # _protected
    self._optVars = None                 # optimization variable names (e.g. input space vars)
    self._initialStepScaling = 0.05      # scale the size of the initial step, in % (where 1 is the length of hyperdiagonal of hypercube)
    # __private
    # additional methods

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    initialScaling = specs.findFirst('initialStepScale')
    if initialScaling is not None:
      self._initialStepScaling = initialScaling.value

  def initialize(self, optVars, **kwargs):
    """
      initializes this object
      @ In, optVars, list(str), optimization variables (e.g. input space)
      @ In, kwargs, dict, additional arguments
      @ Out, None
    """
    self._optVars = optVars

  ###############
  # Run Methods #
  ###############
  @abc.abstractmethod
  def initialStepSize(self, **kwargs):
    """
      Calculates the first step size to use in the optimization path.
      @ In, kwargs, dict, keyword-based specifics as required by individual step sizers
      @ Out, stepSize, float, new step size
    """

  @abc.abstractmethod
  def step(self, prevOpt, **kwargs):
    """
      Calculates a new step size to use in the optimization path.
      @ In, prevOpt, dict, previous optimal point
      @ In, kwargs, dict, keyword-based specifics as required by individual step sizers
      @ Out, newOpt, dict, new optimal point
      @ Out, stepSize, float, new step size
      @ Out, stepInfo, dict, additional information about this step to store
    """

  @abc.abstractmethod
  def fixConstraintViolations(self, proposed, previous, fixInfo):
    """
      Given constraint violations, update the desired optimal point to consider.
      @ In, proposed, dict, proposed new optimal point
      @ In, previous, dict, previous optimal point
      @ In, fixInfo, dict, contains record of progress in fixing search
      @ Out, proposed, new proposed point
      @ Out, stepSize, new step size taken
      @ Out, fixInfo, updated fixing info
    """

  @abc.abstractmethod
  def trajIsFollowing(self, traj, opt, info, data):
    """
      Determines if the current trajectory is following another trajectory.
      @ In, traj, int, integer identifier for trajectory that needs to be checked
      @ In, opt, dict, most recent optimal point for trajectory
      @ In, info, dict, additional information about optimal point
      @ In, data, DataObjects.DataSet, data collected through optimization so far (SolutionExport)
    """

  def modifyAcceptance(self, oldPoint, oldVal, newPoint, newVal):
    """
      Allows modification of acceptance criteria.
      Note this is only called if self.needsAccessToAcceptance is True.
      @ In, oldPoint, dict, old opt point
      @ In, oldVal, float, old objective function value
      @ In, newPoint, dict, new opt point
      @ In, newVal, float, new objective function value
    """

  def needDenormalized(self):
    """
      Determines if this algorithm needs denormalized input spaces
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    return False

  def updateSolutionExport(self, stepHistory):
    """
      Prints information to the solution export.
      @ In, stepHistory, list, (magnitude, versor, info) for each step entry
      @ Out, info, dict, realization of data to go in the solutionExport object
    """
    # overload in inheriting classes at will
    return {}
  ###################
  # Utility Methods #
  ###################
  def flush(self):
    """
      Reset StepManipulator attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    self._optVars = None
