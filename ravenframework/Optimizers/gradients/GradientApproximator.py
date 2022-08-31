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
  TODO

  Reworked 2020-01
  @author: talbpaul
"""
import abc

from ...utils import utils, InputData, InputTypes

class GradientApproximator(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    GradientApproximators use provided information to both select points
    required to estimate gradients as well as calculate the estimates.
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    specs.description = 'Base class for gradient approximation methods used in the GradientDescent Optimizer.'
    specs.addSub(InputData.parameterInputFactory('gradDistanceScalar', contentType=InputTypes.FloatType,
        descr=r"""a scalar for the distance away from an optimal point candidate in the optimization
        search at which points should be evaluated to estimate the local gradient. This scalar is a
        multiplier for the step size used to reach this optimal point candidate from the previous
        optimal point, so this scalar should generally be a small percent. \default{0.01}"""))

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
    ## Instance Variable Initialization
    # public
    # _protected
    self._optVars = None   # list(str) of opt variables
    self._proximity = 0.01 # float, scaling for perturbation distance
    self.N = None          # int, dimensionality
    # __private
    # additional methods

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    proximity = specs.findFirst('gradDistanceScalar')
    if proximity is not None:
      self._proximity = proximity.value

  def initialize(self, optVars):
    """
      After construction, finishes initialization of this approximator.
      @ In, optVars, list(str), list of optimization variable names
      @ In, proximity, float, percentage of step size away that neighbor samples should be taken
      @ Out, None
    """
    self._optVars = optVars
    self.N = len(self._optVars)

  ###############
  # Run Methods #
  ###############
  @abc.abstractmethod
  def chooseEvaluationPoints(self, opt, stepSize):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """

  @abc.abstractmethod
  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """

  @abc.abstractmethod
  def evaluate(self, opt, grads, infos, objVar):
    """
      Approximates gradient based on evaluated points.
      @ In, opt, dict, current opt point (normalized)
      @ In, grads, list(dict), evaluated neighbor points
      @ In, infos, list(dict), info about evaluated neighbor points
      @ In, objVar, string, objective variable
      @ Out, magnitude, float, magnitude of gradient
      @ Out, direction, dict, versor (unit vector) for gradient direction
      @ Out, foundInf, bool, if True then infinity calculations were used
    """

  def needDenormalized(self):
    """
      Determines if this algorithm needs denormalized input spaces
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    return False

  def updateSolutionExport(self, grads, gradInfos):
    """
      Prints information to the solution export.
      @ In, grads, list, list of gradient magnitudes and versors
      @ In, gradInfos, list, list of identifying information for each grad entry
      @ Out, info, dict, realization of data to go in the solutionExport object
    """
    # overload in inheriting classes at will
    return {}
  ###################
  # Utility Methods #
  ###################

  def flush(self):
    """
      Reset GradientApproximater attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    self._optVars = None
    self.N = None
