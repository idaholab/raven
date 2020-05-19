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
import numpy as np
from copy import deepcopy

from utils import utils, randomUtils, InputData, InputTypes

class Crossovers(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Crossovers class control the chromosomal crossover process via several
    implemented mechanisms. Currently, the crossover options include:

    1. One Point
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

  def _onePoint(self,parent1,parent2,crossoverProb,point):
    """
      One Point crossover.
      @ In, parent1, 1D array, parent1 in the current mating process. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
      @ In, parent2, 1D array, parent2 in the current mating process. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
      @ In, crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
      @ In, point, integer, point at which the cross over happens, default is random
      @ Out, child1, 1D array, child1 resulting from the crossover. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
      @ Out, child2, 1D array, child2 resulting from the crossover. Shape is 1 x len(chromosome) i.e, number of Genes/Vars
    """
    nGenes = np.shape(parent1)[1]
    # defaults
    if point is None:
      point = randomUtils.randomIntegers(1,nGenes-1)
    if crossoverProb is None:
      crossoverProb = randomUtils.random(dim=1, samples=1)
    # create children
    if randomUtils.random(dim=1,samples=1) < crossoverProb:
      ## TODO create n children, where n is equal to number of parents
      ## add code here

      for i in range(nGenes):
        if i<point:
          child1[1,i]=parent1[1,i]
          child2[1,i]=parent2[1,i]
        else:
          child1[1,i]=parent2[1,i]
          child2[1,i]=parent1[1,i]
    else:
      # Each child is just a copy of the parents
      child1 = deepcopy(parent1)
      child2 = deepcopy(parent2)
    return child1,child2