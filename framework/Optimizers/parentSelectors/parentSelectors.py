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
  Implementation of parentSelctors for selection process of Genetic Algorithm
  currently the implemented parent selection algorithms are:
  1.  rouletteWheel

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

import numpy as np
from utils import randomUtils

def rouletteWheel(**kwargs):
  """
    Roulette Selection mechanism for parent selection
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          fitnesses, np.array, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          population, np.array, all chromosomes (idividuals) candidate to be parents, i.e. np.shape(population) = populationSize x nGenes.
    @ Out, counter, integer, the id of the selected parent
    @ Out, selectedParents, np.array, selected parents, i.e. np.shape(selectedParents) = nParents x nGenes.
  """
  fitnesses = kwargs['fitnesses']
  population = kwargs['population']
  selectionProb = fitnesses/np.sum(fitnesses) # Share of the pie (rouletteWheel)
  # imagine a wheel that is partitioned according to the selection
  # probabilities

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
  return counter, selectedParent

__parentSelectors = {}
__parentSelectors['rouletteWheel'] = rouletteWheel


def returnInstance(cls, name):
  if name not in __parentSelectors:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __parentSelectors[name]
