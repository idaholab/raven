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
  1.  onePointCrossover
  2.  uniformCrossover
  3.  twoPointsCrossover

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

import numpy as np
from scipy.special import comb
from itertools import combinations
import xarray as xr
from ...utils import randomUtils, gaUtils


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
  if nGenes <= 2:
    raise ValueError('In Two point Crossover the number of genes should be >=3!')
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

def _sbxBetaq(rand, alpha, eta):
  """
    Spread factor for Simulated Binary Crossover (Deb & Agrawal, 1995).
    @ In, rand, float, uniform random number in [0,1).
    @ In, alpha, float, SBX alpha term derived from the bounded beta.
    @ In, eta, float, crossover distribution index (larger -> children closer to parents).
    @ Out, betaq, float, SBX spread factor.
  """
  if rand <= 1.0 / alpha:
    return (rand * alpha) ** (1.0 / (eta + 1.0))
  return (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))


def sbxCrossover(parents, **kwargs):
  """
    Simulated Binary Crossover (SBX) for real-valued decision variables (Deb & Agrawal, 1995).
    SBX is the canonical NSGA-II continuous recombination operator: it produces two children
    distributed around the two parents with a spread controlled by the distribution index eta,
    so offspring can both interpolate between and extrapolate beyond the parents while
    respecting each variable's bounds. This is what makes NSGA-II competitive on continuous
    (ZDT/DTLZ/engineering) problems, unlike the gene-swapping crossovers.
    @ In, parents, xr.DataArray, parents involved in the mating process (nParents x nGenes).
    @ In, kwargs, dict, dictionary of parameters for this crossover method:
          crossoverProb, float, probability that crossover occurs for a parent pair (default 0.9 if None).
          distDict, dict, distribution per gene, used to obtain decision-variable bounds.
          eta, float, SBX distribution index (default 15.0).
          variables, list, variable names.
    @ Out, children, xr.DataArray, children resulting from the crossover (2*comb(nParents,2) x nGenes).
  """
  nParents, nGenes = np.shape(parents)
  geneNames = parents.coords['Gene'].values
  distDict = kwargs.get('distDict', {}) or {}
  crossoverProb = kwargs.get('crossoverProb', None)
  if crossoverProb is None:
    crossoverProb = 0.9
  eta = float(kwargs.get('eta', 15.0))
  children = xr.DataArray(np.zeros((int(2 * comb(nParents, 2)), nGenes)),
                          dims=['chromosome', 'Gene'],
                          coords={'chromosome': np.arange(int(2 * comb(nParents, 2))),
                                  'Gene': geneNames})
  parentPairs = list(combinations(parents, 2))
  index = 0
  for pair in parentPairs:
    p1 = np.array(pair[0].values, dtype=float)
    p2 = np.array(pair[1].values, dtype=float)
    c1 = p1.copy()
    c2 = p2.copy()
    if float(randomUtils.random(dim=1, samples=1)) <= crossoverProb:
      for g in range(nGenes):
        x1 = float(p1[g])
        x2 = float(p2[g])
        if abs(x1 - x2) < 1e-14:
          continue  # identical genes -> nothing to recombine
        xl, xu = gaUtils.finiteGeneBounds(distDict.get(geneNames[g]), x1, x2)
        y1, y2 = (x1, x2) if x1 < x2 else (x2, x1)
        rand = float(randomUtils.random(dim=1, samples=1))
        # child 1 (closer to lower parent)
        beta = 1.0 + (2.0 * (y1 - xl) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        ch1 = 0.5 * ((y1 + y2) - _sbxBetaq(rand, alpha, eta) * (y2 - y1))
        # child 2 (closer to upper parent)
        beta = 1.0 + (2.0 * (xu - y2) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        ch2 = 0.5 * ((y1 + y2) + _sbxBetaq(rand, alpha, eta) * (y2 - y1))
        ch1 = min(max(ch1, xl), xu)
        ch2 = min(max(ch2, xl), xu)
        # randomly assign the two children to the two slots (standard SBX)
        if float(randomUtils.random(dim=1, samples=1)) <= 0.5:
          c1[g], c2[g] = ch2, ch1
        else:
          c1[g], c2[g] = ch1, ch2
    children[index] = c1
    children[index + 1] = c2
    index += 2
  return children


__crossovers = {}
__crossovers['onePointCrossover']  = onePointCrossover
__crossovers['twoPointsCrossover'] = twoPointsCrossover
__crossovers['uniformCrossover']   = uniformCrossover
__crossovers['sbxCrossover']       = sbxCrossover


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
