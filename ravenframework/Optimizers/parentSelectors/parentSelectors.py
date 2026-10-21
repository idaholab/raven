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
    @ In, kwargs, dict, dictionary of parameters for this parent selection method:
          fitness, xr.DataArray, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
                    (single-objective only)
          rank, xr.DataArray, non-dominated rank of each chromosome (multi-objective only)
          crowdDistance, xr.DataArray, crowding distance of each chromosome (multi-objective only)
          variables, list, variable names.
          nParents, int, number of required parents.
    @ Out, selectedParents, xr.DataArray, selected parents, i.e. np.shape(selectedParents) = nParents x nGenes.
  """
  # Arguments
  pop = population
  fitValsInput = kwargs.get('popFitVals', kwargs.get('fitness'))
  fitVals = np.array([item for sublist in datasetToDataArray(fitValsInput, list(fitValsInput.keys())).data for item in sublist], dtype=np.float64)
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
      if np.all(fitVals>=0) or np.all(fitVals<=0):
        selectionProb = fitVals/np.sum(fitVals) # Share of the pie (rouletteWheel)
      else:
        # shift the fitness to be all positive
        shiftedFitness = fitVals + abs(min(fitVals))
        selectionProb = shiftedFitness/np.sum(shiftedFitness) # Share of the pie (rouletteWheel)
    except (ZeroDivisionError, RuntimeWarning):
      #shift the fitnesses to be all positive (adds min and epsilon)
      shiftedFitness = fitVals + abs(min(fitVals))+1e-10
      selectionProb = shiftedFitness/np.sum(shiftedFitness) # Share of the pie (rouletteWheel)

    sumProb = selectionProb[counter]

    while sumProb <= roulettePointer :
      counter += 1
      sumProb += selectionProb[counter]
    selectedParent[i,:] = pop.values[counter,:]
    pop = np.delete(pop, counter, axis=0)
    fitVals = np.delete(fitVals, counter, axis=0)
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
    @ In, kwargs, dict, dictionary of parameters for this parent selection method:
          fitness, xr.DataArray, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          variables, list, variable names.
          nParents, int, number of required parents.
    @ Out, selectedParents, xr.DataArray, selected parents, i.e. np.shape(selectedParents) = nParents x nGenes.
    @ Note, tournaments sample candidates without replacement within each tournament, while parent slots are selected
      with replacement across tournaments.
  """
  nParents = kwargs['nParents']
  fitVals = kwargs.get('popFitVals', kwargs.get('fitness'))
  fitnessProvided = fitVals is not None
  selectedParent = xr.DataArray(np.zeros((nParents, np.shape(population.values)[1])),
                                dims=['chromosome', 'Gene'],
                                coords={'chromosome': np.arange(nParents),
                                        'Gene': kwargs['variables']})
  popSize = population.sizes['chromosome']
  if kwargs['kSelection'] > popSize:
    mh.error('parentSelectors', ValueError, 'Tournament size cannot be greater than population size')
  candidatePositions = list(range(popSize))
  if not kwargs['isMultiObjective']:
    # Single-objective case
    if not fitnessProvided and nParents > 0:
      mh.error('parentSelectors', ValueError, "Fitness must be provided for single-objective selection")
    else:
      fitness = fitVals

    for i in range(nParents):
      selectedChromo = np.asarray(randomUtils.randomChoice(candidatePositions.copy(),
                                                           size=kwargs['kSelection'],
                                                           replace=False,
                                                           engine=None), dtype=int)
      # Extract relevant information
      if fitnessProvided:
        tournamentFitness = np.asarray(fitness[kwargs['objVar'][0]].data, dtype=float)[selectedChromo]

      tournamentWinnerIndex = int(selectedChromo[np.argmax(tournamentFitness)])
      selectedParent[i, :] = population.values[tournamentWinnerIndex, :]
  else: # Multi-objective case: NSGA-II binary tournament on (rank, crowding distance)
    rank = kwargs.get('rank')
    crowdDistance = kwargs.get('crowdDistance')
    if rank is None or crowdDistance is None:
      mh.error('parentSelectors', ValueError, 'Multi-objective tournament selection requires both "rank" '
               'and "crowdDistance" (NSGA-II crowded-comparison operator).')
    rankValues = np.asarray(getattr(rank, 'data', rank))
    crowdingValues = np.asarray(getattr(crowdDistance, 'data', crowdDistance))
    for i in range(nParents):
      selectedChromo = np.asarray(randomUtils.randomChoice(candidatePositions.copy(),
                                                           size=kwargs['kSelection'],
                                                           replace=False,
                                                           engine=None), dtype=int)
      tournamentRank = rankValues[selectedChromo]
      tournamentCrowding = crowdingValues[selectedChromo]
      # Stage 1: lowest (best) non-dominated rank wins
      minRankIndex = np.where(tournamentRank == tournamentRank.min())[0]
      # Stage 2: among equal ranks, the largest crowding distance (least crowded) wins
      tournamentWinnerIndex = int(selectedChromo[minRankIndex[np.argmax(tournamentCrowding[minRankIndex])]])
      selectedParent[i, :] = population.values[tournamentWinnerIndex, :]
  return selectedParent

def rankSelection(population,**kwargs):
  """
    Rank Selection mechanism for parent selection
    @ In, population, xr.DataArray, populations containing all chromosomes (individuals) candidate to be parents,
                                    i.e. population.values.shape = populationSize x nGenes.
    @ In, kwargs, dict, dictionary of parameters for this parent selection method:
          fitness, np.array, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          nParents, int, number of required parents.
    @ Out, newPopulation, xr.DataArray, selected parents,
  """
  fitness = kwargs.get('popFitVals', kwargs.get('fitness'))
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
    @ Out, __parentSelectors[name], the registered parent selection operator
  """
  if name not in __parentSelectors:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __parentSelectors[name]
