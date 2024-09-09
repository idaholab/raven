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
  Implementation of survivorSelctors (Elitism) for new generation
  selection process of Genetic Algorithm. Currently the implemented
  survivorSelctors algorithms are:
  1.  ageBased
  2.  fitnessBased

  Created June,16,2020
  @authors: Mohammad Abdo, Junyung Kim, Diego Mandelli, Andrea Alfonsi
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
import xarray as xr
from ravenframework.utils import frontUtils
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils.gaUtils import dataArrayToDict, datasetToDataArray
# Internal Modules End------------------------------------------------------------------------------

# @profile

def ageBased(newRlz,**kwargs):
  """
    ageBased survivorSelection mechanism for new generation selection.
    It replaces the oldest parents with the new children regardless of the fitness.
    @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          age, list, age list for each chromosome of the previous population
          variables, list of variable names to be sampled
          fitness, xr.DataArrays, fitness of the previous generation
          offSpringsFitness, xr.DataArray, fitness of each new child, i.e., np.shape(offSpringsFitness) = nChildren x nGenes
          population, xr.DataArray, population from previous generation
    @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newFitness, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
  """
  popSize = np.shape(kwargs['population'])[0]
  if ('age' not in kwargs.keys() or kwargs['age']== None):
    popAge = [0]*popSize
  else:
    popAge = kwargs['age']
  offSpringsFitness = np.atleast_1d(kwargs['offSpringsFitness'])
  offSprings = xr.DataArray(np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose()),
                            dims=['chromosome','Gene'],
                            coords={'chromosome':np.arange(np.shape(np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose()))[0]),
                                    'Gene': kwargs['variables']})
  population = np.atleast_2d(kwargs['population'].data)
  popFitness = np.atleast_1d(kwargs['fitness'].data)
  # sort population, popFitness according to age
  sortedAge,sortedPopulation,sortedFitness = zip(*[[x,y,z] for x,y,z in sorted(zip(popAge,population,popFitness),key=lambda x: (x[0], -x[2]))])# if equal age then use descending fitness
  sortedAge,sortedPopulation,sortedFitness = list(sortedAge),np.atleast_1d(list(sortedPopulation)),np.atleast_1d(list(sortedFitness))
  newPopulation = sortedPopulation
  newFitness    = sortedFitness
  newAge = list(map(lambda x:x+1, sortedAge))
  newPopulation[-1:-np.shape(offSprings)[0]-1:-1] = offSprings
  newFitness[-1:-np.shape(offSprings)[0]-1:-1] = offSpringsFitness
  newAge[-1:-np.shape(offSprings)[0]-1:-1] = [0]*np.shape(offSprings)[0]
  # converting back to DataArrays
  newPopulation = xr.DataArray(newPopulation,
                               dims=['chromosome','Gene'],
                               coords={'chromosome':np.arange(np.shape(newPopulation)[0]),
                                       'Gene': kwargs['variables']})
  newFitness = xr.DataArray(newFitness,
                               dims=['chromosome'],
                               coords={'chromosome':np.arange(np.shape(newFitness)[0])})
  return newPopulation,newFitness,newAge,kwargs['popObjectiveVal']


# @profile
def fitnessBased(newRlz,**kwargs):
  """
    fitnessBased survivorSelection mechanism for new generation selection
    It combines the parents and children/offsprings then keeps the fittest individuals
    to revert to the same population size.
    @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this survivor slection method:
          age, list, ages of each chromosome in the population of the previous generation
          offSpringsFitness, xr.DataArray, fitness of each new child, i.e., np.shape(offSpringsFitness) = nChildren x nGenes
          variables
          population
          fitness
    @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newFitness, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
  """
  popSize = np.shape(kwargs['population'])[0]
  if ('age' not in kwargs.keys() or kwargs['age'] == None):
    popAge = [0]*popSize
  else:
    popAge = kwargs['age']

  offSpringsFitness = datasetToDataArray(kwargs['offSpringsFitness'], list(kwargs['offSpringsFitness'].keys())).data
  offSpringsFitness = np.array([item for sublist in offSpringsFitness for item in sublist])
  offSprings = np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose().data)
  population = np.atleast_2d(kwargs['population'].data)
  popFitness = datasetToDataArray(kwargs['fitness'], list(kwargs['fitness'].keys())).data
  popFitness = np.array([item for sublist in popFitness for item in sublist])
  newPopulation = population
  newFitness = popFitness
  newAge = list(map(lambda x:x+1, popAge))
  newPopulationMerged = np.concatenate([newPopulation,offSprings])
  newFitness = np.concatenate([newFitness,offSpringsFitness])
  newAge.extend([0]*len(offSpringsFitness))

  # sort population, popFitness according to age
  sortedFitness,sortedAge,sortedPopulation = zip(*[(x,y,z) for x,y,z in sorted(zip(newFitness,newAge,newPopulationMerged),reverse=True,key=lambda x: (x[0], -x[1]))])
  sortedFitnessT,sortedAgeT,sortedPopulationT = np.atleast_1d(list(sortedFitness)),list(sortedAge),np.atleast_1d(list(sortedPopulation))
  newPopulationSorted = sortedPopulationT[:-len(offSprings)]
  newFitness = sortedFitnessT[:-len(offSprings)]
  newAge = sortedAgeT[:-len(offSprings)]

  newPopulationArray = xr.DataArray(newPopulationSorted,
                                    dims=['chromosome','Gene'],
                                    coords={'chromosome':np.arange(np.shape(newPopulationSorted)[0]),
                                            'Gene': kwargs['variables']})
  newFitness = xr.DataArray(newFitness,
                            dims=['chromosome'],
                            coords={'chromosome':np.arange(np.shape(newFitness)[0])})
  newFitness = newFitness.to_dataset(name = list(kwargs['variables'])[0])

  #return newPopulationArray,newFitness,newAge
  return newPopulationArray,newFitness,newAge,kwargs['popObjectiveVal']

# @profile
def rankNcrowdingBased(offsprings, **kwargs):
  """
    rankNcrowdingBased survivorSelection mechanism for new generation selection
    It combines the parents and children/offsprings then calculates their rank and crowding distance.
    After having ranks and crowding distance, it keeps the lowest ranks (and highest crowding distance if indivisuals have same rank.
    @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this survivor slection method:
          variables
          population
    @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newRank, xr.DataArray, rank of each chromosome in the new population
    @ Out, newCD, xr.DataArray, crowding distance of each chromosome in the new population.
  """
  popSize = np.shape(kwargs['population'])[0]
  if ('age' not in kwargs.keys() or kwargs['age'] == None):
    popAge = [0]*popSize
  else:
    popAge = kwargs['age']

  population = np.atleast_2d(kwargs['population'].data)
  offSprings = np.atleast_2d(offsprings[kwargs['variables']].to_array().transpose().data)
  popObjectiveVal = kwargs['popObjectiveVal']
  offObjectiveVal = kwargs['offObjectiveVal']
  popFit = kwargs['popFit']
  popFitArray = []
  offFit = kwargs['offFit']
  offFitArray = []
  for i in list(popFit.keys()): #NOTE popFit.keys() and offFit.keys() must be same.
    popFitArray.append(popFit[i].data.tolist())
    offFitArray.append(offFit[i].data.tolist())

  newFitMerged      = np.array([i + j for i, j in zip(popFitArray, offFitArray)])
  newFitMerged_pair = [list(ele) for ele in list(zip(*newFitMerged))]

  popConstV = kwargs['popConstV'].data
  offConstV = kwargs['offConstV'].data
  newConstVMerged = np.array(popConstV.tolist() + offConstV.tolist())

  newObjectivesMerged = np.array([i + j for i, j in zip(popObjectiveVal, offObjectiveVal)])
  newObjectivesMerged_pair = [list(ele) for ele in list(zip(*newObjectivesMerged))]

  newPopRank = frontUtils.rankNonDominatedFrontiers(np.array(newFitMerged_pair))
  newPopRank = xr.DataArray(newPopRank,
                            dims=['rank'],
                            coords={'rank': np.arange(np.shape(newPopRank)[0])})

  newPopCD = frontUtils.crowdingDistance(rank=newPopRank, popSize=len(newPopRank), objectives=np.array(newFitMerged_pair))
  newPopCD = xr.DataArray(newPopCD,
                          dims=['CrowdingDistance'],
                          coords={'CrowdingDistance': np.arange(np.shape(newPopCD)[0])})

  newAge = list(map(lambda x:x+1, popAge))
  newPopulationMerged = np.concatenate([population,offSprings])
  newAge.extend([0]*len(offSprings))

  sortedRank,sortedCD,sortedAge,sortedPopulation,sortedFit,sortedObjectives,sortedConstV = \
    zip(*[(x,y,z,i,j,k,a) for x,y,z,i,j,k,a in \
      sorted(zip(newPopRank.data, newPopCD.data, newAge, newPopulationMerged.tolist(), newFitMerged_pair, newObjectivesMerged_pair, newConstVMerged),reverse=False,key=lambda x: (x[0], -x[1]))])
  sortedRankT, sortedCDT, sortedAgeT, sortedPopulationT, sortedFitT, sortedObjectivesT, sortedConstVT = \
    np.atleast_1d(list(sortedRank)), list(sortedCD), list(sortedAge),np.atleast_1d(list(sortedPopulation)),np.atleast_1d(list(sortedFit)),np.atleast_1d(list(sortedObjectives)),np.atleast_1d(list(sortedConstV))

  newPopulation = sortedPopulationT[:-len(offSprings)]
  newObjectives = sortedObjectivesT[:-len(offSprings)]
  newFit        = sortedFitT[:-len(offSprings)]

  newRank = frontUtils.rankNonDominatedFrontiers(newObjectives)
  newRank = xr.DataArray(newRank,
                         dims=['rank'],
                         coords={'rank': np.arange(np.shape(newRank)[0])})

  newObjectivesP = [list(ele) for ele in list(zip(*newObjectives))]
  newCD = frontUtils.crowdingDistance(rank=newRank, popSize=len(newRank), objectives=newObjectives)
  newCD = xr.DataArray(newCD,
                       dims=['CrowdingDistance'],
                       coords={'CrowdingDistance': np.arange(np.shape(newCD)[0])})

  newAge = sortedAgeT[:-len(offSprings)]
  newConstV = sortedConstVT[:-len(offSprings)]

  for i in range(len(list(popFit.keys()))):
    fitness = xr.DataArray(newFit[:,i],
                           dims=['chromosome'],
                           coords={'chromosome': np.arange(len(newFit[:,i]))})
    if i == 0:
      newFitnessSet = fitness.to_dataset(name = list(popFit.keys())[i])
    else:
      newFitnessSet[list(popFit.keys())[i]] = fitness

  newPopulationArray = xr.DataArray(newPopulation,
                                    dims=['chromosome','Gene'],
                                    coords={'chromosome':np.arange(np.shape(newPopulation)[0]),
                                            'Gene': kwargs['variables']})
  newConstV = xr.DataArray(newConstV,
                           dims=['chromosome','ConstEvaluation'],
                           coords={'chromosome':np.arange(np.shape(newPopulation)[0]),
                                   'ConstEvaluation':np.arange(np.shape(newConstV)[1])})

  return newPopulationArray,newRank,newAge,newCD,newObjectivesP,newFitnessSet,newConstV

__survivorSelectors = {}
__survivorSelectors['ageBased'] = ageBased
__survivorSelectors['fitnessBased'] = fitnessBased
__survivorSelectors['rankNcrowdingBased'] = rankNcrowdingBased

def returnInstance(cls, name):
  """
    Method designed to return class instance
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __survivorSelectors:
    cls.raiseAnError (IOError, "{} is not an valid option for survivor selector. Please review the spelling of the survivor selector. ".format(name))
  return __survivorSelectors[name]
