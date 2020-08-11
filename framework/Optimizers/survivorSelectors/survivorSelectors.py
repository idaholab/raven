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
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

import numpy as np
import xarray as xr

def ageBased(self,newRlz,**kwargs):
  """
    ageBased survivorSelection mechanism for new generation selection.
    It replaces the oldest parents with the new children regardless of the fitness.
    @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          age
          popSize
          variables
          offSpringsFitness, xr.DataArray, fitness of each new child, i.e., np.shape(offSpringsFitness) = nChildren x nGenes
          population
    @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newFitness, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
  """
  popAge = kwargs['age']
  if popAge == None:
    popAge = np.zeros(kwargs['popSize'],dtype='int')
  offSpringsFitness = kwargs['offSpringsFitness']
  offSprings = xr.DataArray(newRlz[kwargs['variables']].to_array().transpose(),
        dims=['chromosome','Gene'],
        coords={'chromosome':np.arange(np.shape(newRlz[list(self.toBeSampled)].to_array().transpose())[0]),
                'Gene': kwargs['variables']})
  population = kwargs['population']
  popFitness = kwargs['fitness']
  # sort population, popFitness according to age
  sortedAge,sortedPopulation,sortedFitness = zip(*[[x,y,z] for x,y,z in sorted(zip(popAge,population,popFitness),key=lambda x: (x[0], -x[2]))])# if equal age then use descending fitness
  sortedAge,sortedPopulation,sortedFitness = list(sortedAge),np.atleast_1d(list(sortedPopulation)),np.atleast_1d(list(sortedFitness))
  newPopulation = sortedPopulation.copy()
  newFitness = sortedFitness.copy()
  newAge = list(map(lambda x:x+1, sortedAge.copy()))
  newPopulation[-1:-np.shape(offSprings)[0]-1:-1] = offSprings
  newFitness[-1:-np.shape(offSprings)[0]-1:-1] = offSpringsFitness
  newAge[-1:-np.shape(offSprings)[0]-1:-1] = np.zeros(np.shape(offSprings)[0],dtype='int')
  # converting back to DataArrays
  newPopulation = xr.DataArray(newPopulation,
                               dims=['chromosome','Gene'],
                               coords={'chromosome':np.arange(np.shape(newPopulation)[0]),
                                       'Gene': kwargs['variables']})
  newFitness = xr.DataArray(newFitness,
                               dims=['chromosome'],
                               coords={'chromosome':np.arange(np.shape(newFitness)[0])})
  return newPopulation,newFitness,newAge

def fitnessBased(newRlz,**kwargs):
  """
    fitnessBased survivorSelection mechanism for new generation selection
    It combines the parents and children/offsprings then keeps the fittest individuals
    to revert to the same population size.
    @ In, newRlz, xr.DataSet, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          age, list, ages of each chromosome in the population of the previous generation
          popSize, int, number of chromosomes (individuals) in each population.
          offSpringsFitness, xr.DataArray, fitness of each new child, i.e., np.shape(offSpringsFitness) = nChildren x nGenes
          variables
          population
          fitness
    @ Out, newPopulation, xr.DataArray, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newFitness, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
  """
  popAge = kwargs['age']
  if popAge == None:
    popAge = [0]*kwargs['popSize']
  offSpringsFitness = kwargs['offSpringsFitness'].data
  offSprings = newRlz[kwargs['variables']].to_array().transpose().data
  population = kwargs['population'].data
  popFitness = kwargs['fitness'].data
  offSpringsAge = [0]*(len(offSpringsFitness))
  newPopulation = population.copy()
  newFitness = popFitness.copy()
  newAge = list(map(lambda x:x+1, popAge.copy()))
  newPopulation = np.concatenate([newPopulation,offSprings])
  newFitness = np.concatenate([newFitness,offSpringsFitness])
  newAge.extend([0]*len(offSprings))
  # sort population, popFitness according to age
  sortedFitness,sortedAge,sortedPopulation = zip(*[(x,y,z) for x,y,z in sorted(zip(newFitness,newAge,newPopulation),reverse=True,key=lambda x: (x[0], -x[1]))])
  sortedFitness,sortedAge,sortedPopulation = np.atleast_1d(list(sortedFitness)),list(sortedAge),np.atleast_1d(list(sortedPopulation))
  newPopulation = sortedPopulation[:-len(offSprings)]
  newFitness = sortedFitness[:-len(offSprings)]
  newAge = sortedAge[:-len(offSprings)]
  newPopulation = xr.DataArray(newPopulation,
                               dims=['chromosome','Gene'],
                               coords={'chromosome':np.arange(np.shape(newPopulation)[0]),
                                       'Gene': kwargs['variables']})
  newFitness = xr.DataArray(newFitness,
                               dims=['chromosome'],
                               coords={'chromosome':np.arange(np.shape(newFitness)[0])})
  return newPopulation,newFitness,newAge

__survivorSelectors = {}
__survivorSelectors['ageBased'] = ageBased
__survivorSelectors['fitnessBased'] = fitnessBased

def returnInstance(cls, name):
  if name not in __survivorSelectors:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __survivorSelectors[name]
if __name__ == '__main__':
  # I am leaving this part right now for the sake of testing,
  # TODO REMOVE THIS IF BLOCK
  population =[[1,2,3,4,5,6],[2,1,3,4,6,5],[6,5,4,3,2,1],[3,5,6,2,1,4]]
  popFitness = [7.2,1.3,9.5,2.0]
  popAge = [3,1,7,1]
  offSprings = [[2,3,4,5,6,1],[1,3,5,2,4,6],[1,2,4,3,6,5]]
  offSpringsFitness = [1.1,2.0,3.2]
  newPop,newFit,newAge = fitnessBased(population=population,popAge=popAge,popFitness=popFitness,offSprings=offSprings,offSpringsFitness=offSpringsFitness)
  print('Fitness Based Selection')
  print('*'*23)
  print('new population: {}, \n new Fitness {}, \n new Age {}'.format(newPop,newFit,newAge))
  print('Note that the last parent and second offSpring had the same fitness, but the fitness based mechanism ommited the oldest one')
  newPop2,newFit2,newAge2 = ageBased(population=population,popAge=popAge,popFitness=popFitness,offSprings=offSprings,offSpringsFitness=offSpringsFitness)
  print('Age Based Selection')
  print('*'*19)
  print('new population: {}, \n new Fitness {}, \n new age'.format(newPop2,newFit2,newAge2))
  print('Note that the second and forth chromosome had the same age, but for the age based mechanism it ommited the one with the lowest fitness')