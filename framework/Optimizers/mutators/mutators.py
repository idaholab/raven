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
  Implementation of mutators for Mutation process of Genetic Algorithm
  currently the implemented mutation algorithms are:
  1.  Swap Mutator
  2.  Scramble Mutator

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""
import numpy as np
import xarray as xr
from operator import itemgetter
from utils import randomUtils

def swapMutator(offSprings,**kwargs):
  """
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          locs, list, the 2 locations of the genes to be swapped
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, child, numpy.array, the mutated chromosome, i.e., the child.
  """
  loc1 = kwargs['locs'][0]
  loc2 = kwargs['locs'][1]
  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':['x1','x2','x3','x4','x5','x6']})
  for i in range(np.shape(offSprings)[0]):
    children[i] = offSprings[i].copy()
    ## TODO What happens if loc1 or 2 is out of range?! should we raise an error?
    if randomUtils.random(dim=1,samples=1)>kwargs['mutationProb']:
      children[i,loc1] = offSprings[i,loc2]
      children[i,loc2] = offSprings[i,loc1]
  return children

def scrambleMutator(offSprings,**kwargs):
  """
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          chromosome, numpy.array, the chromosome that will mutate to the new child
          locs, list, the locations of the genes to be randomly scrambled
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, child, np.array, the mutated chromosome, i.e., the child.
  """
  locs = kwargs['locs']
  nMutable = len(locs)
    # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':['x1','x2','x3','x4','x5','x6']})
  for i in range(np.shape(offSprings)[0]):
    children[i] = offSprings[i].copy()
    new = list(itemgetter(*locs)(offSprings[i].values))
    for ind,element in enumerate(locs):
      if randomUtils.random(dim=1,samples=1)>0.0001:#kwargs['mutationProb']:
        ## TODO: use randomUtils instead
        children[i,locs[0]:locs[-1]+1] = np.random.permutation(offSprings[i,locs[0]:locs[-1]+1])
  return children


__mutators = {}
__mutators['swapMutator'] = swapMutator
__mutators['scrambleMutator'] = scrambleMutator


def returnInstance(cls, name):
  if name not in __mutators:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __mutators[name]
