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
# External Modules----------------------------------------------------------------------------------
import numpy as np
import xarray as xr
from ...utils import randomUtils
# External Modules----------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils.gaUtils import dataArrayToDict, datasetToDataArray
# Internal Modules End------------------------------------------------------------------------------

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
  fitness = np.array([item for sublist in datasetToDataArray(kwargs['fitness'], list(kwargs['fitness'].keys())).data for item in sublist])
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
    if np.all(fitness>=0) or np.all(fitness<=0):
      selectionProb = fitness/np.sum(fitness) # Share of the pie (rouletteWheel)
    else:
      # shift the fitness to be all positive
      shiftedFitness = fitness + abs(min(fitness))
      selectionProb = shiftedFitness/np.sum(shiftedFitness) # Share of the pie (rouletteWheel)
    sumProb = selectionProb[counter]

    while sumProb <= roulettePointer :
      counter += 1
      sumProb += selectionProb[counter]
    selectedParent[i,:] = pop.values[counter,:]
    pop = np.delete(pop, counter, axis=0)
    fitness = np.delete(fitness,counter,axis=0)
  return selectedParent

def countConstViolation(const):
  """
    Counts the number of constraints that are violated
    @ In, const, list, list of constraints
    @ Out, count, int, number of constraints that are violated
  """
  count = sum(1 for i in const if i < 0)
  return count

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

  nParents = kwargs['nParents']
  nObjVal  = len(kwargs['objVal'])
  kSelect = kwargs['kSelection']
  pop = population
  popSize = population.values.shape[0]

  selectedParent = xr.DataArray(np.zeros((nParents,np.shape(pop)[1])),
                                dims=['chromosome','Gene'],
                                coords={'chromosome':np.arange(nParents),
                                        'Gene': kwargs['variables']})

  if nObjVal == 1: # single-objective Case
    fitness = np.array([item for sublist in datasetToDataArray(kwargs['fitness'], list(kwargs['fitness'].keys())).data for item in sublist])
    for i in range(nParents):
      matrixOperationRaw = np.zeros((kSelect,2))
      selectChromoIndexes = list(np.arange(len(pop)))  #NOTE: JYK - selectChromoIndexes should cover all chromosomes in population.
      selectedChromo = randomUtils.randomChoice(selectChromoIndexes, size=kSelect, replace=False, engine=None) #NOTE: JYK - randomly select several indices with size of kSelect.
      matrixOperationRaw[:,0] = selectedChromo
      matrixOperationRaw[:,1] = np.transpose(fitness[selectedChromo])
      tournamentWinnerIndex = int(matrixOperationRaw[np.argmax(matrixOperationRaw[:,1]),0])
      selectedParent[i,:] = pop.values[tournamentWinnerIndex,:]

  else: # multi-objective Case
    # the key rank is used in multi-objective optimization where rank identifies which front the point belongs to.
    rank = kwargs['rank']
    crowdDistance = kwargs['crowdDistance']
    for i in range(nParents):
      matrixOperationRaw = np.zeros((kSelect,3))
      selectChromoIndexes = list(np.arange(kSelect))
      selectedChromo = randomUtils.randomChoice(selectChromoIndexes, size=kSelect, replace=False, engine=None)
      matrixOperationRaw[:,0] = selectedChromo
      matrixOperationRaw[:,1] = np.transpose(rank.data[selectedChromo])
      matrixOperationRaw[:,2] = np.transpose(crowdDistance.data[selectedChromo])
      minRankIndex = list(np.where(matrixOperationRaw[:,1] == matrixOperationRaw[:,1].min())[0])
      if len(minRankIndex) != 1: # More than one chrosome having same rank.
        minRankNmaxCDIndex = list(np.where(matrixOperationRaw[minRankIndex,2] == matrixOperationRaw[minRankIndex,2].max())[0])
      else:
        minRankNmaxCDIndex = minRankIndex
      tournamentWinnerIndex = minRankNmaxCDIndex[0]
      selectedParent[i,:] = pop.values[tournamentWinnerIndex,:]

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

  data = np.vstack((np.array(fitness.variables['test_RankSelection']),index))
  dataOrderedByDecreasingFitness = data[:,(-data[0]).argsort()]
  dataOrderedByDecreasingFitness[0,:] = rank
  dataOrderedByIncreasingPos = dataOrderedByDecreasingFitness[:,dataOrderedByDecreasingFitness[1].argsort()]
  orderedRank = dataOrderedByIncreasingPos[0,:]

  rank = xr.DataArray(orderedRank,
                      dims=['chromosome'],
                      coords={'chromosome': np.arange(np.shape(orderedRank)[0])})

  rank = rank.to_dataset(name = 'test_RankSelection')
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
