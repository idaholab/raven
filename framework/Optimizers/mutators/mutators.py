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
  1.  swapMutator
  2.  scrambleMutator
  3.  bitFlipMutator
  4.  inversionMutator

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
          variables, list, variables names.
    @ Out, children, xr.DataArray, the mutated chromosome, i.e., the child.
  """
  loc1 = kwargs['locs'][0]
  loc2 = kwargs['locs'][1]
  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})
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
          variables, list, variables names.
    @ Out, child, np.array, the mutated chromosome, i.e., the child.
  """
  locs = kwargs['locs']
  nMutable = len(locs)
    # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})
  for i in range(np.shape(offSprings)[0]):
    children[i] = offSprings[i].copy()
    new = list(itemgetter(*locs)(offSprings[i].values))
    for ind,element in enumerate(locs):
      if randomUtils.random(dim=1,samples=1)< kwargs['mutationProb']:
        ## TODO: use randomUtils instead
        children[i,locs[0]:locs[-1]+1] = randomUtils.randomPermutation(list(offSprings.data[i,locs[0]:locs[-1]+1]),None)
  return children

def bitFlipMutator(offSprings,**kwargs):
  """
    This method is designed to flip a single gene in each chromosome with probability = mutationProb.
    E.g. gene at location loc is flipped from current value to newValue
    The gene to be flipped is completely random.
    The new value of the flipped gene is is completely random.
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene location to be flipped: i.e., determine loc
      chromosomeSize = child.values.shape[0]
      loc = randomUtils.randomIntegers(0, chromosomeSize, caller=None, engine=None)
      ##############
      # sample value: i.e., determine newValue
      if kwargs['sampleRange']=='local':
        rangeValues = list(set(offSprings[:,loc].values))
      else: #kwargs['sampleRange']=='global'
        rangeValues = offSprings.values.ravel().tolist()
      rangeValues.pop(child.values[loc])
      newValuePos = randomUtils.randomIntegers(0, len(rangeValues), caller=None, engine=None)
      newValue = rangeValues[newValuePos]
      ##############
      # gene at location loc is flipped from current value to newValue
      child.values[loc] = newValue

  return offSprings

def inversionMutator(offSprings,**kwargs):
  """
    This method is designed mirror a sequence of genes in each chromosome with probability = mutationProb.
    The sequence of genes to be mirrored is completely random.
    E.g. given chromosome C = [0,1,2,3,4,5,6,7,8,9] and sampled locL=2 locU=6;
         New chromosome  C' = [0,1,6,5,4,3,2,7,8,9]
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene locations: i.e., determine loc1 and loc2
      locRangeList = list(range(0,child.values.shape[0]))
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
      ##############
      # select sequence to be mirrored and mirror it
      seq=child.values[locL:locU+1]
      mirrSeq = seq[::-1]
      ##############
      # insert mirrored sequence into child
      child.values[locL:locU+1]=mirrSeq

  return offSprings

__mutators = {}
__mutators['swapMutator']       = swapMutator
__mutators['scrambleMutator']   = scrambleMutator
__mutators['bitFlipMutator']    = bitFlipMutator
__mutators['inversionMutator']  = inversionMutator


def returnInstance(cls, name):
  if name not in __mutators:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __mutators[name]
