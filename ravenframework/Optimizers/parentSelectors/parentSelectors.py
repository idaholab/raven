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
  2.  tournamentSelection
  3.  rankSelection

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

import numpy as np
import xarray as xr
from ...utils import randomUtils

# For mandd: to be updated with RAVEN official tools
from itertools import combinations

# @profile
def rouletteWheel(population,**kwargs):
  """
    Roulette Selection mechanism for parent selection
    @ In, population, xr.DataArray, populations containing all chromosomes (individuals) candidate to be parents, i.e. population.values.shape = populationSize x nGenes.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          fitness, xr.DataArray, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          variables, list, variable names.
          nParents, int, number of required parents.
    @ Out, selectedParents, xr.DataArray, selected parents, i.e. np.shape(selectedParents) = nParents x nGenes.
  """
  # Arguments
  pop = population
  fitness = kwargs['fitness']
  nParents= kwargs['nParents']
  # if nparents = population size then do nothing (whole population are parents)
  if nParents == pop.shape[0]:
    return population
  elif nParents > pop.shape[0]:
    raise IOError('Number of parents is greater than population size')
  # begin the roulette selection algorithm
  selectedParent = xr.DataArray(
        np.zeros((nParents,np.shape(pop)[1])),
        dims=['chromosome','Gene'],
        coords={'chromosome':np.arange(nParents),
                'Gene': kwargs['variables']})
  # imagine a wheel that is partitioned according to the selection probabilities

  for i in range(nParents):
    # set a random pointer
    roulettePointer = randomUtils.random(dim=1, samples=1)
    # initialize Probability
    counter = 0
    if np.all(fitness.data>=0) or np.all(fitness.data<=0):
      selectionProb = fitness.data/np.sum(fitness.data) # Share of the pie (rouletteWheel)
    else:
      # shift the fitness to be all positive
      shiftedFitness = fitness.data + abs(min(fitness.data))
      selectionProb = shiftedFitness/np.sum(shiftedFitness) # Share of the pie (rouletteWheel)
    sumProb = selectionProb[counter]

    while sumProb < roulettePointer :
      counter += 1
      sumProb += selectionProb[counter]
    selectedParent[i,:] = pop.values[counter,:]
    pop = np.delete(pop, counter, axis=0)
    fitness = np.delete(fitness,counter,axis=0)
  return selectedParent

def tournamentSelection(population,**kwargs):
  """
    Tournament Selection mechanism for parent selection
    @ In, population, xr.DataArray, populations containing all chromosomes (individuals) candidate to be parents, i.e. population.values.shape = populationSize x nGenes.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          fitness, np.array, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          nParents, int, number of required parents
          variables, list, variable names
    @ Out, newPopulation, xr.DataArray, selected parents,
  """
  fitness = kwargs['fitness']
  nParents= kwargs['nParents']
  pop = population
  popSize = population.values.shape[0]

  if 'rank' in kwargs:
    # the key rank is used in multi-objective optimization where rank identifies which front the point belongs to
    rank = kwargs['rank']
    multiObjectiveRanking = True
    matrixOperationRaw = np.zeros((popSize,3))
    matrixOperationRaw[:,0] = np.transpose(np.arange(popSize))
    matrixOperationRaw[:,1] = np.transpose(fitness.data)
    matrixOperationRaw[:,2] = np.transpose(rank.data)
    matrixOperation = np.zeros((popSize,3))
  else:
    multiObjectiveRanking = False
    matrixOperationRaw = np.zeros((popSize,2))
    matrixOperationRaw[:,0] = np.transpose(np.arange(popSize))
    matrixOperationRaw[:,1] = np.transpose(fitness.data)
    matrixOperation = np.zeros((popSize,2))

  indexes = list(np.arange(popSize))
  indexesShuffled = randomUtils.randomChoice(indexes, size=popSize, replace=False, engine=None)

  if popSize<2*nParents:
    raise ValueError('In tournamentSelection the number of parents cannot be larger than half of the population size.')

  for idx, val in enumerate(indexesShuffled):
    matrixOperation[idx,:] = matrixOperationRaw[val,:]

  selectedParent = xr.DataArray(
    np.zeros((nParents,np.shape(pop)[1])),
    dims=['chromosome','Gene'],
    coords={'chromosome':np.arange(nParents),
            'Gene': kwargs['variables']})

  if not multiObjectiveRanking: # single-objective implementation of tournamentSelection
    for i in range(nParents):
      if matrixOperation[2*i,1] > matrixOperation[2*i+1,1]:
        index = int(matrixOperation[2*i,0])
      else:
        index = int(matrixOperation[2*i+1,0])
      selectedParent[i,:] = pop.values[index,:]
  else: # multi-objective implementation of tournamentSelection
    for i in range(nParents-1):
      if matrixOperation[2*i,2] > matrixOperation[2*i+1,2]:
        index = int(matrixOperation[i,0])
      elif matrixOperation[2*i,2] < matrixOperation[2*i+1,2]:
        index = int(matrixOperation[i+1,0])
      else: # same rank case
        if matrixOperation[2*i,1] > matrixOperation[2*i+1,1]:
          index = int(matrixOperation[i,0])
        else:
          index = int(matrixOperation[i+1,0])
      selectedParent[i,:] = pop.values[index,:]

  return selectedParent


def rankSelection(population,**kwargs):
  """
    Rank Selection mechanism for parent selection
    @ In, population, xr.DataArray, populations containing all chromosomes (individuals) candidate to be parents,
                                    i.e. population.values.shape = populationSize x nGenes.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          fitness, np.array, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          nParents, int, number of required parents.
    @ Out, newPopulation, xr.DataArray, selected parents,
  """
  fitness = kwargs['fitness']
  pop = population

  index = np.arange(0,pop.shape[0])
  rank = np.arange(0,pop.shape[0])

  data = np.vstack((fitness,index))
  dataOrderedByDecreasingFitness = data[:,(-data[0]).argsort()]
  dataOrderedByDecreasingFitness[0,:] = rank
  dataOrderedByIncreasingPos = dataOrderedByDecreasingFitness[:,dataOrderedByDecreasingFitness[1].argsort()]
  orderedRank = dataOrderedByIncreasingPos[0,:]

  rank = xr.DataArray(orderedRank,
                      dims=['chromosome'],
                      coords={'chromosome': np.arange(np.shape(orderedRank)[0])})

  selectedParent = rouletteWheel(population, fitness=rank , nParents=kwargs['nParents'],variables=kwargs['variables'])

  return selectedParent

__parentSelectors = {}
__parentSelectors['rouletteWheel'] = rouletteWheel
__parentSelectors['rankSelection'] = rankSelection
__parentSelectors['tournamentSelection'] = tournamentSelection

def returnInstance(cls, name):
  """
    Method designed to return class instance
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __parentSelectors:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __parentSelectors[name]
