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
from ...utils.gaUtils import datasetToDataArray
from ... import MessageHandler # makes sure getMessageHandler is defined
# Internal Modules End------------------------------------------------------------------------------
mh = getMessageHandler()

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
  fitness = np.array([item for sublist in datasetToDataArray(kwargs['fitness'], list(kwargs['fitness'].keys())).data for item in sublist],dtype=np.float64)
  nParents= kwargs['nParents']
  # if nparents = population size then do nothing (whole population are parents)
  if nParents == pop.shape[0]:
    return population
  elif nParents > pop.shape[0]:
    mh.error('parentSelectors', IOError, 'Number of parents is greater than population size')
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
    try:
      if np.all(fitness>=0) or np.all(fitness<=0):
        selectionProb = fitness/np.sum(fitness) # Share of the pie (rouletteWheel)
      else:
        # shift the fitness to be all positive
        shiftedFitness = fitness + abs(min(fitness))
        selectionProb = shiftedFitness/np.sum(shiftedFitness) # Share of the pie (rouletteWheel)
    except (ZeroDivisionError, RuntimeWarning):
      #shift the fitnesses to be all positive (adds min and epsilon)
      shiftedFitness = fitness + abs(min(fitness))+1e-10
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

def tournamentSelection(population, **kwargs):
  """
    Tournament Selection mechanism for parent selection
    @ In, population, xr.DataArray, populations containing all chromosomes (individuals) candidate to be parents, i.e. population.values.shape = populationSize x nGenes.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          fitness, xr.DataArray, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          variables, list, variable names.
          nParents, int, number of required parents.
    @ Out, selectedParents, xr.DataArray, selected parents, i.e. np.shape(selectedParents) = nParents x nGenes.
  """
  nParents = kwargs['nParents']
  nObjVal = len(kwargs['objVar'])
  fitnessProvided = 'fitness' in kwargs
  selectedParent = xr.DataArray(np.zeros((nParents, np.shape(population.values)[1])),
                                dims=['chromosome', 'Gene'],
                                coords={'chromosome': np.arange(nParents),
                                        'Gene': kwargs['variables']})
  if not kwargs['isMultiObjective']:
    # Single-objective case
    if not fitnessProvided and nParents > 0:
      mh.error('parentSelectors', ValueError, "Fitness must be provided for single-objective selection")
    else:
      fitness = kwargs['fitness']

    allSelected = set()
    for i in range(nParents):
      matrixOperationRaw = np.zeros((kwargs['kSelection'], 2))
      selectChromoIndexes = list(set(population.indexes['chromosome']) - allSelected)
      selectedChromo = randomUtils.randomChoice(selectChromoIndexes, size=kwargs['kSelection'],
                                                replace=False, engine=None)
      # Extract relevant information
      if fitnessProvided:
        matrixOperationRaw[:, 0] = selectedChromo
        matrixOperationRaw[:, 1] = np.transpose(fitness[kwargs['objVar'][0]][selectedChromo].values)

      tournamentWinnerIndex = int(matrixOperationRaw[np.argmax(matrixOperationRaw[:, 1]), 0])
      allSelected.add(tournamentWinnerIndex)
      selectedParent[i, :] = population.values[tournamentWinnerIndex, :]
  else: # Multi-objective case

    rankProvided = 'rank' in kwargs
    crowdDistanceProvided = 'crowdDistance' in kwargs

    if not rankProvided or not crowdDistanceProvided or 'fitness' not in kwargs:
      # Handle cases where neither fitness nor rank are provided
      mh.error('parentSelectors',ValueError, 'At least one of "fitness" or "rank" must be provided for multi-objective selection')
    allSelected = set()
    for i in range(nParents):
      if rankProvided and crowdDistanceProvided:
      # If both rank and crowd distance are provided, use them directly as per NSGA-II
        matrixOperationRaw = np.zeros((kwargs['kSelection'], 2))
        selectChromoIndexes = list(set(population.indexes['chromosome']) - allSelected)
        selectedChromo = randomUtils.randomChoice(selectChromoIndexes, size=kwargs['kSelection'],
                                                  replace=False, engine=None)
        # Extract relevant information
        matrixOperationRaw[:, 0] = np.transpose(kwargs['rank'].data[selectedChromo])
        matrixOperationRaw[:, 1] = np.transpose(kwargs['crowdDistance'].data[selectedChromo])
        # Stage 1: Select based on rank and crowding distance
        minRankIndex = list(np.where(matrixOperationRaw[:, 0] == matrixOperationRaw[:, 0].min())[0])
        if len(minRankIndex) != 1:
          # Handle cases where more than one chromosome has the same rank
          minRankNmaxCDIndex = list(np.where((matrixOperationRaw[minRankIndex, 1] ==
                                              matrixOperationRaw[minRankIndex, 1].max()) &
                                             (matrixOperationRaw[minRankIndex, 0] ==
                                              matrixOperationRaw[minRankIndex, 0].min()))[0])
        else:
          minRankNmaxCDIndex = minRankIndex
        # Stage 2: Select the individual with the highest crowding distance within their rank group
        tournamentWinnerIndex = int(minRankNmaxCDIndex[0])
      elif rankProvided and not crowdDistanceProvided:
        # If only rank is provided (without crowd distance), calculate a default crowding distance
        matrixOperationRaw = np.zeros((kwargs['kSelection'], 1))
        selectChromoIndexes = list(set(population.indexes['chromosome']) - allSelected)
        selectedChromo = randomUtils.randomChoice(selectChromoIndexes, size=kwargs['kSelection'],
                                                  replace=False, engine=None)
        # Extract relevant information
        matrixOperationRaw[:, 0] = np.transpose(kwargs['rank'].data[selectedChromo])
        # Stage 1: Select based on rank
        minRankIndex = list(np.where(matrixOperationRaw[:, 0] == matrixOperationRaw[:, 0].min())[0])
        if len(minRankIndex) != 1:
          # Handle cases where more than one chromosome has the same rank
          minRankNmaxCDIndex = list(np.where((matrixOperationRaw[minRankIndex, 0] ==
                                              matrixOperationRaw[minRankIndex, 0].min()))[0])
        else:
          minRankNmaxCDIndex = minRankIndex
        # Stage 2: Select the individual with the highest rank within their group
        tournamentWinnerIndex = int(minRankNmaxCDIndex[0])
      elif 'fitness' in kwargs and not rankProvided:
        # If only fitness is provided (without rank), calculate a default rank
        matrixOperationRaw = np.zeros((kwargs['kSelection'], 2))
        selectChromoIndexes = list(set(population.indexes['chromosome']) - allSelected)
        selectedChromo = randomUtils.randomChoice(selectChromoIndexes, size=kwargs['kSelection'],
                                                  replace=False, engine=None)
        # Extract relevant information
        matrixOperationRaw[:, 0] = selectedChromo
        matrixOperationRaw[:, 1] = np.transpose(kwargs['fitness'][selectedChromo])
        # Stage 1: Select based on fitness
        tournamentWinnerIndex = int(matrixOperationRaw[np.argmax(matrixOperationRaw[:, 1]), 0])
      allSelected.add(selectedChromo[tournamentWinnerIndex])
      selectedParent[i, :] = population.values[selectedChromo[tournamentWinnerIndex], :]
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

  data = np.vstack((np.array(fitness.variables['test_RankSelection'],dtype=np.float64),index))
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
