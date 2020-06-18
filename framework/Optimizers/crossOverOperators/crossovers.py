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
from utils import randomUtils
from copy import deepcopy

def onePointCrossover(**kwargs):
  """
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          parents, 2D array, parents in the current mating process. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
    @ Out, children, np.array, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(kwargs['parents'])
  children = np.zeros((np.shape(kwargs['parents'])))
  parents = kwargs['parents']
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
  if randomUtils.random(dim=1,samples=1) < crossoverProb:
    # TODO right nChildren is equal to nParents whereas it should be nChildren = 2 x nParentsChoose2
    for i in range(nGenes):
      if i<point:
        children[:,i]=parents[:,i]
      else:
        for j in range(np.shape(parents)[0]):
          children[j,i]=parents[np.shape(parents)[0]-j-1,i]
  else:
    # Each child is just a copy of the parents
    children = deepcopy(parents)
  return children

__crossovers = {}
__crossovers['onePointCrossover'] = onePointCrossover

def returnInstance(cls, name):
  if name not in __crossovers:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __crossovers[name]