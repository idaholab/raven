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
  Implementation of crossovers for crossover process of Genetic Algorithm
  currently the implemented crossover algorithms are:
  1.  OnePoint Crossover
  2.  TwoPoints Crossover
  3.  Uniform Crossover
  4.  TwoPoints Partially Mapped Crossover (PMX)

  Created June,16,2020
  Last update July,8,2024
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi, Juan Luque-Gutierrez
"""

import numpy as np
from scipy.special import comb
from itertools import combinations
import xarray as xr
from ...utils import randomUtils


# @profile
def onePointCrossover(parents,**kwargs):
  """
    Method designed to perform crossover by swapping chromosome portions before/after specified or sampled location
    @ In, parents, xr.DataArray, parents involved in the mating process.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
          variables, list, variables names.
    @ Out, children, np.array, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(parents)
  # Number of children = 2* (nParents choose 2)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)),nGenes)),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(int(2*comb(nParents,2))),
                                  'Gene':kwargs['variables']})


  # defaults
  if (kwargs['crossoverProb'] == None) or ('crossoverProb' not in kwargs.keys()):
    crossoverProb = randomUtils.random(dim=1, samples=1)
  else:
    crossoverProb = kwargs['crossoverProb']

  # create children
  parentsPairs = list(combinations(parents,2))

  for ind,parent in enumerate(parentsPairs):
    parent = np.array(parent).reshape(2,-1) # two parents at a time

    if randomUtils.random(dim=1,samples=1) <= crossoverProb:
      if (kwargs['points'] == None) or ('points' not in kwargs.keys()):
        point = list([randomUtils.randomIntegers(1,nGenes-1,None)])
      elif (any(i>=nGenes-1 for i in kwargs['points'])):
        raise ValueError('Crossover point cannot be larger than number of Genes (variables)')
      else:
        point = kwargs['points']
      for i in range(nGenes):
        if len(point)>1:
          raise ValueError('In one Point Crossover a single crossover location should be provided!')
        children[2*ind:2*ind+2,i] = parent[np.arange(0,2)*(i<point[0])+np.arange(-1,-3,-1)*(i>=point[0]),i]
    else:
      # Each child is just a copy of the parents
      children[2*ind:2*ind+2,:] = parent

  return children

def uniformCrossover(parents,**kwargs):
  """
    Method designed to perform crossover by swapping genes one by one
    @ In, parents, xr.DataArray, parents involved in the mating process.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          parents, 2D array, parents in the current mating process.
          Shape is nParents x len(chromosome) i.e, number of Genes/Vars
    @ Out, children, xr.DataArray, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(parents)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)),np.shape(parents)[1])),
                              dims=['chromosome','Gene'],
                              coords={'chromosome': np.arange(int(2*comb(nParents,2))),
                                      'Gene':parents.coords['Gene'].values})

  if (kwargs['crossoverProb'] == None) or ('crossoverProb' not in kwargs.keys()):
    crossoverProb = randomUtils.random(dim=1, samples=1)
  else:
    crossoverProb = kwargs['crossoverProb']

  index = 0
  parentsPairs = list(combinations(parents,2))
  for parentPair in parentsPairs:
    parent1 = parentPair[0].values
    parent2 = parentPair[1].values
    children1,children2 = uniformCrossoverMethod(parent1,parent2,crossoverProb)
    children[index]   = children1
    children[index+1] = children2
    index +=  2
  return children


def twoPointsCrossover(parents, **kwargs):
  """
    Method designed to perform a two point crossover on 2 parents:
    Partition each parents in three sequences (A,B,C):
    parent1 = A1 B1 C1
    parent2 = A2 B2 C2
    Then:
    children1 = A1 B2 C1
    children2 = A2 B1 C2
    @ In, parents, xr.DataArray, parents involved in the mating process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          parents, 2D array, parents in the current mating process.
          Shape is nParents x len(chromosome) i.e, number of Genes/Vars
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
    @ Out, children, xr.DataArray, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(parents)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)),np.shape(parents)[1])),
                              dims=['chromosome','Gene'],
                              coords={'chromosome': np.arange(int(2*comb(nParents,2))),
                                      'Gene':parents.coords['Gene'].values})
  parentPairs = list(combinations(parents,2))
  index = 0
  if nGenes<=2:
    ValueError('In Two point Crossover the number of genes should be >=3!')
  for couples in parentPairs:
    [loc1,loc2] = randomUtils.randomChoice(list(range(1,nGenes)), size=2, replace=False, engine=None)
    if loc1 > loc2:
      locL = loc2
      locU = loc1
    else:
      locL=loc1
      locU=loc2
    parent1 = couples[0]
    parent2 = couples[1]
    children1,children2 = twoPointsCrossoverMethod(parent1,parent2,locL,locU)

    children[index]   = children1
    children[index+1] = children2
    index = index + 2

  return children

def partiallyMappedCrossover(parents, **kwargs):
  """
    Method designed to perform a two point partially mapped crossover (MPX) on 2 parents:
    Partition each parents in three sequences (A,B,C):
    parent1 = A1 B1 C1
    parent2 = A2 B2 C2
    Then:
    children1 = A1* B2 C1*
    children2 = A2* B1 C2*
    Children should have the same elements as their parents, but in different order.
    This crossover preserves the genes in a chromosome.
    @ In, parents, xr.DataArray, parents involved in the mating process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          parents, 2D array, parents in the current mating process.
          Shape is nParents x len(chromosome) i.e, number of Genes/Vars
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
    @ Out, children, xr.DataArray, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents, nGenes = np.shape(parents)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)), np.shape(parents)[1])),
                          dims = ['chromosome', 'Gene'],
                          coords = {'chromosome': np.arange(int(2*comb(nParents, 2))),
                                    'Gene':parents.coords['Gene'].values})
  parentPairs = list(combinations(parents, 2))
  index = 0
  if nGenes <= 2:
    ValueError('The number of genes should be >= 3')
  for couples in parentPairs:
    [loc1, loc2] = randomUtils.randomChoice(list(range(1, nGenes)), size = 2, replace=False, engine=None)
    if loc1 > loc2:
      locL = loc2
      locU = loc1
    else:
      locL = loc1
      locU = loc2
    parent1 = couples[0]
    parent2 = couples[1]
    children1, children2 = twoPointsPMXMethod(parent1, parent2, locL, locU)

    children[index] = children1
    children[index + 1] = children2
    index = index + 2

  return children

__crossovers = {}
__crossovers['onePointCrossover']  = onePointCrossover
__crossovers['twoPointsCrossover'] = twoPointsCrossover
__crossovers['uniformCrossover']   = uniformCrossover
__crossovers['partiallyMappedCrossover'] = partiallyMappedCrossover 

def returnInstance(cls, name):
  """
    Method designed to return class instance
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __crossovers:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __crossovers[name]

def twoPointsCrossoverMethod(parent1,parent2,locL,locU):
  """
    Method designed to perform a twopoint crossover on 2 arrays:
    Partition each array in three sequences (A,B,C):
    parent1 = A1 B1 C1
    parent2 = A2 B2 C2
    Then:
    children1 = A1 B2 C1
    children2 = A2 B1 C2
    @ In, parent1: first array
    @ In, parent2: second array
    @ In, LocL: first location
    @ In, LocU: second location
    @ Out, children1: first generated array
    @ Out, children2: second generated array
  """
  children1 = parent1.copy(deep=True)
  children2 = parent2.copy(deep=True)

  seqB1 = parent1.values[locL:locU]
  seqB2 = parent2.values[locL:locU]

  children1[locL:locU] = seqB2
  children2[locL:locU] = seqB1
  return children1,children2

def uniformCrossoverMethod(parent1,parent2,crossoverProb):
  """
    Method designed to perform a uniform crossover on 2 arrays
    @ In, parent1: first array
    @ In, parent2: second array
    @ In, crossoverProb: crossover probability for each gene
    @ Out, children1: first generated array
    @ Out, children2: second generated array
  """
  children1 = np.zeros(parent1.size)
  children2 = np.zeros(parent2.size)

  for pos in range(parent1.size):
    if randomUtils.random(dim=1,samples=1)<crossoverProb:
      children1[pos] = parent2[pos]
      children2[pos] = parent1[pos]
    else:
      children1[pos] = parent1[pos]
      children2[pos] = parent2[pos]

  return children1,children2

def twoPointsPMXMethod(parent1, parent2, locL, locU):
  """
    Method designed to perform a two point Partially Mapped Crossover (PMX) on 2 arrays:
    Partition each array into three sequences (A, B, C):
    parent1 = A1 B1 C1
    parent2 = A2 B2 C2
    We map the values contained in B1 to B2.
    Then:
    children1 = X B2 X
    children2 = X B1 X
    We verify if the values in A and C are found in B for each children. If so, we 
    replace such values for the ones in the map.
    children1 = A1* B2 C1*
    children2 = A2* B1 C2*
    Children should have the same elements as their parents, but in different order. 
    @ In, parent1, first array
    @ In, parent2, second array
    @ In, locL: first location
    @ In, LocU: second location
    @ Out, children1: first generated array
    @ Out, children2: second generated array
  """
  
  size = len(parent1)
  
  children1 = parent1.copy(deep=True)
  children2 = parent2.copy(deep=True)

  seqB1 = parent1.values[locL:locU]
  seqB2 = parent2.values[locL:locU]

  children1[locL:locU] = seqB2
  children2[locL:locU] = seqB1

  # Determine mapping relationship
  mapping1 = {parent2.values[i]: parent1.values[i] for i in range(locL, locU)}
  mapping2 = {parent1.values[i]: parent2.values[i] for i in range(locL, locU)}

  for i in list(range(locL)) + list(range(locU, size)):
    if children1.values[i] in mapping1:
      while children1.values[i] in mapping1:
        children1.values[i] = mapping1[children1.values[i]]
    if children2.values[i] in mapping2:
      while children2.values[i] in mapping2:
        children2.values[i] = mapping2[children2.values[i]]

  return children1, children2