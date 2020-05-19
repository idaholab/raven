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
from utils import utils, randomUtils, InputData, InputTypes

class ParentSelectors(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    ParentSelectors control the parent selection process via several
    implemented mechanisms. Currently, the parent selection options include:

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
    specs.description = 'Base class for parent selection methods used in the Genetic Algorithm Optimizer.'
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

  def initialize(self):
    """
      After construction, finishes initialization of this approximator.
      @ In,
      @ Out, None
    """
    pass

  def _rouletteSelection(population, fitnesses):
    """
      Roulette Selection mechanism for parent selection
      @ In, population, array, population is a pool of chromosomes (individuals), i.e., np.shape(population) = population size x nGenes
      @ In, fitnesses, a list or a 1D array, fitness of each chromosome in the population
      @ Out, selectedParent, a list or a 1D array, selected Parent
    """
    selectionProb = fitnesses/np.sum(fitnesses)
    # imagine a wheel that is partitioned according to the selection probabilities

    # set a random pointer
    roulettePointer = randomUtils.random(dim=1, samples=1)
    # Rotate the wheel
    counter = 0
    # intialize Probability
    sumProb = selectionProb[counter]
    while sumProb < roulettePointer:
      counter += 1
      sumProb += selectionProb[counter]
    selectedParent = population[counter]
    return selectedParent