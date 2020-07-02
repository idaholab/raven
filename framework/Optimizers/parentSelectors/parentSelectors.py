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
import xarray as xr
from utils import randomUtils

def rouletteWheel(population,**kwargs):
  """
    Roulette Selection mechanism for parent selection
    @ In, population, xr.DataArray, populations containing all chromosomes (individuals) candidate to be parents, i.e. population.values.shape = populationSize x nGenes.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          fitness, np.array, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          nParents, int, number of required parents.
    @ Out, selectedParents, np.array, selected parents, i.e. np.shape(selectedParents) = nParents x nGenes.
  """
  # Arguments
  pop = population.copy()
  fitness = kwargs['fitness'].copy()
  nParents= kwargs['nParents']

  # if nparents = population size then do nothing (whole population are parents)
  if nParents == pop.shape[0]:
    return population
<<<<<<< HEAD
  # begin the roulette selection algorithm
  selectionProb = fitness/np.sum(fitness) # Share of the pie (rouletteWheel)
=======
  elif nParents > pop.shape[0]:
    raise IOError('Number of parents is greater than population size')
  # begine the roulette selection algorithm
>>>>>>> changing genetic operators againnnnnnn
  selectedParent = xr.DataArray(
        np.zeros((nParents,np.shape(pop)[1])),
        dims=['chromosome','Gene'],
        coords={'chromosome':np.arange(nParents),
                'Gene': ['x1','x2','x3','x4','x5','x6']})#np.arange(np.shape(pop)[1]
  # imagine a wheel that is partitioned according to the selection
  # probabilities

  for i in range(nParents):
<<<<<<< HEAD
    # initialize Probability
=======
    # set a random pointer
    roulettePointer = randomUtils.random(dim=1, samples=1)
    # Rotate the wheel

    # intialize Probability
>>>>>>> changing genetic operators againnnnnnn
    counter = 0
    selectionProb = fitness.data/np.sum(fitness.data) # Share of the pie (rouletteWheel)
    sumProb = selectionProb[counter]

    while sumProb < roulettePointer :
      counter += 1
      sumProb += selectionProb[counter]
    selectedParent[i,:] = pop.values[counter,:]
    pop = np.delete(pop, counter, axis=0)
    fitness = np.delete(fitness,counter,axis=0)
  return selectedParent

__parentSelectors = {}
__parentSelectors['rouletteWheel'] = rouletteWheel


def returnInstance(cls, name):
  if name not in __parentSelectors:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __parentSelectors[name]
