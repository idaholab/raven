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

def ageBased(**kwargs):
  """
    ageBased survivorSelection mechanism for new generation selection.
    It replaces the oldest parents with the new children regardless of the fitness.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          population, list, all chromosomes (idividuals) from previous generation, i.e. np.shape(population) = populationSize x nGenes.
          popFitnesses, list, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          popAge, list integers, number of generations each chromosome lasted
          offSprings, list, children coming from crossovers and mutations
          offSpringsFitnesses, list, fitness of each child in the population, i.e., np.shape(offSpringsFitnesses) = nChildren x nGenes
    @ Out, newPopulation, list, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
  """
  population = kwargs['population']
  popFitnesses = kwargs['popFitnesses']
  popAge = kwargs['popAge']
  offSpringsFitnesses = kwargs['offSpringsFitnesses']
  offSprings = kwargs['offSprings']
  # sort population, popFitnesses according to age
  sortedAge,sortedPopulation,sortedFitnesses = zip(*[[x,y,z] for x,y,z in sorted(zip(popAge,population,popFitnesses),key=lambda x: (x[0], -x[2]))])# if equal age then use descending fitness
  sortedAge,sortedPopulation,sortedFitnesses = list(sortedAge),list(sortedPopulation),list(sortedFitnesses)
  newPopulation = sortedPopulation.copy()
  newFitnesses = sortedFitnesses.copy()
  newAge = list(map(lambda x:x+1, sortedAge.copy()))
  newPopulation[-1:-np.shape(offSprings)[0]-1:-1] = offSprings
  newFitnesses[-1:-np.shape(offSprings)[0]-1:-1] = offSpringsFitnesses
  newAge[-1:-np.shape(offSprings)[0]-1:-1] = np.zeros(np.shape(offSprings)[0],dtype='int')
  return newPopulation,newFitnesses,newAge

def fitnessBased(**kwargs):
  """
    fitnessBased survivorSelection mechanism for new generation selection
    It combines the parents and children/offsprings then keeps the fittest indivduals
    to revert to the same population size.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          population, list, all chromosomes (idividuals) from previous generation, i.e. np.shape(population) = populationSize x nGenes.
          popFitnesses, list, fitness of each chromosome (individual) in the population, i.e., np.shape(fitness) = 1 x populationSize
          popAge, list integers, number of generations each chromosome lasted
          offSprings, list, children coming from crossovers and mutations
          offSpringsFitnesses, list, fitness of each new child, i.e., np.shape(offSpringsFitnesses) = nChildren x nGenes
    @ Out, newPopulation, list, newPopulation for the new generation, i.e. np.shape(newPopulation) = populationSize x nGenes.
    @ Out, newFitnesses, list, fitness of the new population
  """
  population = kwargs['population']
  popFitnesses = kwargs['popFitnesses']
  offSpringsFitnesses = kwargs['offSpringsFitnesses']
  offSprings = kwargs['offSprings']
  popAge = kwargs['popAge']
  offSpringsAge = [0]*(len(offSpringsFitnesses))
  newPopulation = population.copy()
  newFitnesses = popFitnesses.copy()
  newAge = list(map(lambda x:x+1, popAge.copy()))
  newPopulation.extend(offSprings)
  newFitnesses.extend(offSpringsFitnesses)
  newAge.extend([0]*len(offSprings))
  # sort population, popFitnesses according to age
  sortedFitnesses,sortedAge,sortedPopulation = zip(*[(x,y,z) for x,y,z in sorted(zip(newFitnesses,newAge,newPopulation),reverse=True,key=lambda x: (x[0], -x[1]))])
  sortedFitnesses,sortedAge,sortedPopulation = list(sortedFitnesses),list(sortedAge),list(sortedPopulation)
  newPopulation = sortedPopulation[:-len(offSprings)]
  newFitnesses = sortedFitnesses[:-len(offSprings)]
  newAge = sortedAge[:-len(offSprings)]
  return newPopulation,newFitnesses,newAge

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
  popFitnesses = [7.2,1.3,9.5,2.0]
  popAge = [3,1,7,1]
  offSprings = [[2,3,4,5,6,1],[1,3,5,2,4,6],[1,2,4,3,6,5]]
  offSpringsFitnesses = [1.1,2.0,3.2]
  newPop,newFit,newAge = fitnessBased(population=population,popAge=popAge,popFitnesses=popFitnesses,offSprings=offSprings,offSpringsFitnesses=offSpringsFitnesses)
  print('Fitness Based Selection')
  print('*'*23)
  print('new population: {}, \n new Fitness {}, \n new Age {}'.format(newPop,newFit,newAge))
  print('Note that the last parent and second offSpring had the same fitness, but the fitness based mechanism ommited the oldest one')
  newPop2,newFit2,newAge2 = ageBased(population=population,popAge=popAge,popFitnesses=popFitnesses,offSprings=offSprings,offSpringsFitnesses=offSpringsFitnesses)
  print('Age Based Selection')
  print('*'*19)
  print('new population: {}, \n new Fitness {}, \n new age'.format(newPop2,newFit2,newAge2))
  print('Note that the second and forth chromosome had the same age, but for the age based mechanism it ommited the one with the lowest fitness')