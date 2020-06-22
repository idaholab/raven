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
from scipy.special import comb
from itertools import combinations
def onePointCrossover(**kwargs):
  """
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          parents, 2D array, parents in the current mating process. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
    @ Out, children, np.array, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(kwargs['parents'])
  children = np.zeros((int(2*comb(nParents,2)),np.shape(kwargs['parents'])[1]))
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
  parentsPairs = list(combinations(parents,2))
  for ind,parent in enumerate(parentsPairs):
    parent = np.array(parent).reshape(2,-1)
    if randomUtils.random(dim=1,samples=1) < crossoverProb:
      for i in range(nGenes):
        children[2*ind:2*ind+2,i]=parent[np.arange(0,2)*(i<point[0])+np.arange(-1,-3,-1)*(i>=point[0]),i]
    else:
      # Each child is just a copy of the parents
      children[2*ind:2*ind+2,:] = deepcopy(parent)
  return children

__crossovers = {}
__crossovers['onePointCrossover'] = onePointCrossover

def returnInstance(cls, name):
  if name not in __crossovers:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __crossovers[name]