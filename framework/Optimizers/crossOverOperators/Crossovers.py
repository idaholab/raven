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
  Mutation methods base class
  Created May,13,2020
  @author: Mohammad Abdo
"""
import abc

from utils import utils, InputData, InputTypes

class Crossovers(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Crossovers class control the chromosomal crossover process via several
    implemented mechanisms. Currently, the crossover options include:

    1.
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
    specs.description = 'Base class for crossover methods used in the Genetic Algorithm Optimizer.'
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

    # __private

    # additional methods

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    pass

  def initialize(self, optVars, proximity):
    """
      After construction, finishes initialization of this approximator.
      @ In, optVars, list(str), list of optimization variable names
      @ In, proximity, float, percentage of step size away that neighbor samples should be taken
      @ Out, None
    """
    pass