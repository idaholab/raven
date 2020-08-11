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

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

import numpy as np
import xarray as xr
import copy
from utils import randomUtils
from copy import deepcopy
from scipy.special import comb
from itertools import combinations

def onePointCrossover(parents,**kwargs):
  """
    @ In, parents, xr.DataArray, parents involved in the mating process.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
          variables, list, variables names.
    @ Out, children, np.array, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  # parents = kwargs['parents']
  nParents,nGenes = np.shape(parents)
  # Number of children = 2* (nParents choose 2)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)),np.shape(parents)[1])),
                              dims=['chromosome','Gene'],
                              coords={'chromosome': np.arange(int(2*comb(nParents,2))),
                                      'Gene':kwargs['variables']})


  # defaults
  if kwargs['points'] is None:
    point = randomUtils.randomIntegers(1,nGenes-1)
  else:
    point = kwargs['points']

  if kwargs['crossoverProb'] is None:
    crossoverProb = randomUtils.random(dim=1, samples=1)
  else:
    crossoverProb = kwargs['crossoverProb']

  # create children
  parentsPairs = list(combinations(parents,2))
  for ind,parent in enumerate(parentsPairs):
    parent = np.array(parent).reshape(2,-1) # two parents at a time
    if randomUtils.random(dim=1,samples=1) < crossoverProb:
      for i in range(nGenes):
        children[2*ind:2*ind+2,i] = parent[np.arange(0,2)*(i<point[0])+np.arange(-1,-3,-1)*(i>=point[0]),i]
    else:
      # Each child is just a copy of the parents
      children[2*ind:2*ind+2,:] = deepcopy(parent)
  return children

def uniformCrossover(parents, parentIndexes,**kwargs):
  """
    Method designed
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
  index = 0
  for couples in parentIndexes:
    parent1 = parents[couples[0]].values
    parent2 = parents[couples[1]].values
    children1,children2 = uniformCrossoverMethod(parent1,parent2)

    children[index]=copy.deepcopy(children1)
    children[index+1]=copy.deepcopy(children2)
    index = index + 2

  return children


def twoPointsCrossover(parents, parentIndexes,**kwargs):
  """
    Method designed to perform a twopoint crossover on 2 parents:
    Partition each parents in three sequences (A,B,C):
    parent1 = A1 B1 C1
    parent2 = A2 B2 C2
    Then:
    children1 = A1 B2 C1
    children2 = A2 B1 C2
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
  index = 0
  for couples in parentIndexes:
    locRangeList = list(range(0,nGenes))
    index1 = randomUtils.randomIntegers(0, len(locRangeList), caller=None, engine=None)
    loc1 = locRangeList[index1]
    locRangeList.pop(loc1)
    index2 = randomUtils.randomIntegers(0, len(locRangeList), caller=None, engine=None)
    loc2 = locRangeList[index2]
    if loc1>loc2:
      locL=loc2
      locU=loc1
    elif loc1<loc2:
      locL=loc1
      locU=loc2

    parent1 = parents[couples[0]].values
    parent2 = parents[couples[1]].values
    children1,children2 = twoPointsCrossoverMethod(parent1,parent2,locL,locU)

    children[index]=copy.deepcopy(children1)
    children[index+1]=copy.deepcopy(children2)
    index = index + 2

  return children

__crossovers = {}
__crossovers['onePointCrossover']  = onePointCrossover
__crossovers['twoPointsCrossover'] = twoPointsCrossover
__crossovers['uniformCrossover']   = uniformCrossover


def returnInstance(cls, name):
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
  children1 = copy.deepcopy(parent1)
  children2 = copy.deepcopy(parent2)

  seqB1 = parent1.values[locL:locU+1]
  seqB2 = parent2.values[locL:locU+1]

  children1[locL:locU+1] = seqB2
  children2[locL:locU+1] = seqB1
  return children1,children2

def uniformCrossoverMethod(parent1,parent2):
  children1 = np.zeros(parent1.size)
  children2 = np.zeros(parent2.size)

  for pos in range(parent1.size):
    if randomUtils.random(dim=1,samples=1)>0.5:
      children1[pos] = parent1[pos]
      children2[pos] = parent2[pos]
    else:
      children1[pos] = parent2[pos]
      children2[pos] = parent1[pos]

  return children1,children2




