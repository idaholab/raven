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
  5.  randomMutator

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""
import numpy as np
import xarray as xr
from operator import itemgetter
from ...utils import utils, randomUtils

def swapMutator(offSprings, distDict, **kwargs):
  """
    This method performs the swap mutator. For each child, two genes are sampled and switched
    E.g.:
    child=[a,b,c,d,e] --> b and d are selected --> child = [a,d,c,b,e]
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          locs, list, the 2 locations of the genes to be swapped
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
          variables, list, variables names.
    @ Out, children, xr.DataArray, the mutated chromosome, i.e., the child.
  """
  loc1,loc2 = locationsGenerator(offSprings, kwargs['locs'])

  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})
  for i in range(np.shape(offSprings)[0]):
    children[i] = offSprings[i]
    ## TODO What happens if loc1 or 2 is out of range?! should we raise an error?
    if randomUtils.random(dim=1,samples=1)<=kwargs['mutationProb']:
      # convert loc1 and loc2 in terms on cdf values
      cdf1 = distDict[offSprings.coords['Gene'].values[loc1]].cdf(float(offSprings[i,loc1].values))
      cdf2 = distDict[offSprings.coords['Gene'].values[loc2]].cdf(float(offSprings[i,loc2].values))
      children[i,loc1] = distDict[offSprings.coords['Gene'].values[loc1]].ppf(cdf2)
      children[i,loc2] = distDict[offSprings.coords['Gene'].values[loc2]].ppf(cdf1)
  return children

# @profile
def scrambleMutator(offSprings, distDict, **kwargs):
  """
    This method performs the scramble mutator. For each child, a subset of genes is chosen
    and their values are shuffled randomly.
    @ In, offSprings, xr.DataArray, offsprings after crossover
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          chromosome, numpy.array, the chromosome that will mutate to the new child
          locs, list, the locations of the genes to be randomly scrambled
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
          variables, list, variables names.
    @ Out, child, np.array, the mutated chromosome, i.e., the child.
  """
  loc1,loc2 = locationsGenerator(offSprings, kwargs['locs'])

  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})

  for i in range(np.shape(offSprings)[0]):
    for j in range(np.shape(offSprings)[1]):
      children[i,j] = distDict[offSprings[i].coords['Gene'].values[j]].cdf(float(offSprings[i,j].values))

  for i in range(np.shape(offSprings)[0]):
    for ind,element in enumerate([loc1,loc2]):
      if randomUtils.random(dim=1,samples=1)< kwargs['mutationProb']:
        children[i,loc1:loc2+1] = randomUtils.randomPermutation(list(children.data[i,loc1:loc2+1]),None)

  for i in range(np.shape(offSprings)[0]):
    for j in range(np.shape(offSprings)[1]):
      children[i,j] = distDict[offSprings.coords['Gene'].values[j]].ppf(float(children[i,j].values))

  return children

def bitFlipMutator(offSprings, distDict, **kwargs):
  """
    This method is designed to flip a single gene in each chromosome with probability = mutationProb.
    E.g. gene at location loc is flipped from current value to newValue
    The gene to be flipped is completely random.
    The new value of the flipped gene is is completely random.
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  if kwargs['locs'] is not None and 'locs' in kwargs.keys():
    raise ValueError('Locs arguments are not being used by bitFlipMutator')

  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene location to be flipped: i.e., determine loc
      chromosomeSize = child.values.shape[0]
      loc = randomUtils.randomIntegers(0, chromosomeSize, caller=None, engine=None)
      # gene at location loc is flipped from current value to newValue
      geneIDToBeChanged = child.coords['Gene'].values[loc-1]
      oldCDFvalue = distDict[geneIDToBeChanged].cdf(child.values[loc-1])
      newCDFValue = 1.0 - oldCDFvalue
      newValue = distDict[geneIDToBeChanged].ppf(newCDFValue)
      child.values[loc-1] = newValue
  return offSprings

def randomMutator(offSprings, distDict, **kwargs):
  """
    This method is designed to randomly mutate a single gene in each chromosome with probability = mutationProb.
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  if kwargs['locs'] is not None and 'locs' in kwargs.keys():
    raise ValueError('Locs arguments are not being used by randomMutator')
  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene location to be flipped: i.e., determine loc
      chromosomeSize = child.values.shape[0]
      loc = randomUtils.randomIntegers(0, chromosomeSize, caller=None, engine=None)
      # gene at location loc is flipped from current value to newValue
      geneIDToBeChanged = child.coords['Gene'].values[loc-1]
      newCDFValue = randomUtils.random()
      newValue = distDict[geneIDToBeChanged].ppf(newCDFValue)
      child.values[loc-1] = newValue
  return offSprings

def inversionMutator(offSprings, distDict, **kwargs):
  """
    This method is designed mirror a sequence of genes in each chromosome with probability = mutationProb.
    The sequence of genes to be mirrored is completely random.
    E.g. given chromosome C = [0,1,2,3,4,5,6,7,8,9] and sampled locL=2 locU=6;
         New chromosome  C' = [0,1,6,5,4,3,2,7,8,9]
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  # sample gene locations: i.e., determine locL and locU
  locL,locU = locationsGenerator(offSprings, kwargs['locs'])

  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # select sequence to be mirrored and mirror it
      seq = np.arange(locL,locU+1)
      allElems = []
      for i,elem in enumerate(seq):
        allElems.append(distDict[child.coords['Gene'].values[i]].cdf(float(child[elem].values)))

      mirrSeq = allElems[::-1]
      mirrElems = []
      for elem in mirrSeq:
        mirrElems.append(distDict[child.coords['Gene'].values[i]].ppf(elem))
      # insert mirrored sequence into child
      child.values[locL:locU+1]=mirrElems

  return offSprings

def locationsGenerator(offSprings,locs):
  """
  Methods designed to process the locations for the mutators. These locations can be either user specified or
  randomly generated.
  @ In, offSprings, xr.DataArray, children resulting from the crossover process
  @ In, locs, list, the two locations of the genes to be swapped
  @ Out, loc1, loc2, int, the two ordered processed locations required by the mutators
  """
  if locs == None:
    locs = list(set(randomUtils.randomChoice(list(np.arange(offSprings.data.shape[1])),size=2,replace=False)))
    loc1 = np.minimum(locs[0], locs[1])
    loc2 = np.maximum(locs[0], locs[1])
  else:
    loc1 = np.minimum(locs[0], locs[1])
    loc2 = np.maximum(locs[0], locs[1])
  return loc1, loc2

__mutators = {}
__mutators['swapMutator']       = swapMutator
__mutators['scrambleMutator']   = scrambleMutator
__mutators['bitFlipMutator']    = bitFlipMutator
__mutators['inversionMutator']  = inversionMutator
__mutators['randomMutator']     = randomMutator


def returnInstance(cls, name):
  """
    Method designed to return class instance:
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __mutators:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __mutators[name]
