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
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi, Junyung Kim
"""
import numpy as np
import xarray as xr
from ...utils import randomUtils, gaUtils

def swapMutator(offspring, distDict, **kwargs):
  """
    This method performs the swap mutator. For each child, two genes are sampled and switched
    E.g.:
    child=[a,b,c,d,e] --> b and d are selected --> child = [a,d,c,b,e]
    @ In, offspring, xr.DataArray, children resulting from the crossover process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          locs, list, the 2 locations of the genes to be swapped
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
          variables, list, variables names.
    @ Out, children, xr.DataArray, the mutated chromosome, i.e., the child.
  """
  loc1, loc2 = locationsGenerator(offspring, kwargs['locs'])

  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offspring))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offspring)[0]),
                                  'Gene':kwargs['variables']})
  for i in range(np.shape(offspring)[0]):
    children[i] = offspring[i]
    ## TODO What happens if loc1 or 2 is out of range?! should we raise an error?
    if randomUtils.random(dim=1,samples=1)<=kwargs['mutationProb']:
      # convert loc1 and loc2 in terms on cdf values
      cdf1 = distDict[offspring.coords['Gene'].values[loc1]].cdf(float(offspring[i,loc1].values))
      cdf2 = distDict[offspring.coords['Gene'].values[loc2]].cdf(float(offspring[i,loc2].values))
      children[i,loc1] = distDict[offspring.coords['Gene'].values[loc1]].ppf(cdf2)
      children[i,loc2] = distDict[offspring.coords['Gene'].values[loc2]].ppf(cdf1)
  return children

# @profile
def scrambleMutator(offspring, distDict, **kwargs):
  """
    This method performs the scramble mutator. For each child, a subset of genes is chosen
    and their values are shuffled randomly.
    @ In, offspring, xr.DataArray, offspring after crossover
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          chromosome, numpy.array, the chromosome that will mutate to the new child
          locs, list, the locations of the genes to be randomly scrambled
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
          variables, list, variables names.
    @ Out, child, np.array, the mutated chromosome, i.e., the child.
  """
  locs = locationsGenerator(offspring, kwargs['locs'])

  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offspring))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offspring)[0]),
                                  'Gene':kwargs['variables']})

  for i in range(np.shape(offspring)[0]):
    for j in range(np.shape(offspring)[1]):
      children[i,j] = distDict[offspring[i].coords['Gene'].values[j]].cdf(float(offspring[i,j].values))

  for i in range(np.shape(offspring)[0]):
    if randomUtils.random(dim=1, samples=1) < kwargs['mutationProb']:
      children[i, locs[0]:locs[-1]+1] = randomUtils.randomPermutation(list(children.data[i, locs[0]:locs[-1]+1]), None)

  for i in range(np.shape(offspring)[0]):
    for j in range(np.shape(offspring)[1]):
      children[i,j] = distDict[offspring.coords['Gene'].values[j]].ppf(float(children[i,j].values))

  return children

def bitFlipMutator(offspring, distDict, **kwargs):
  """
    This method is designed to flip a single gene in each chromosome with probability = mutationProb.
    E.g. gene at location loc is flipped from current value to newValue
    The gene to be flipped is completely random.
    The new value of the flipped gene is is completely random.
    @ In, offspring, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offspring, xr.DataArray, children resulting from the crossover process
  """
  if kwargs['locs'] is not None and 'locs' in kwargs.keys():
    raise ValueError('Locs arguments are not being used by bitFlipMutator')

  for child in offspring:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene location to be flipped: i.e., determine loc
      chromosomeSize = child.values.shape[0]
      loc = randomUtils.randomIntegers(0, chromosomeSize - 1, caller=None, engine=None)
      # gene at location loc is flipped from current value to newValue
      geneIDToBeChanged = child.coords['Gene'].values[loc]
      oldCDFvalue = distDict[geneIDToBeChanged].cdf(child.values[loc])
      newCDFValue = 1.0 - oldCDFvalue
      newValue = distDict[geneIDToBeChanged].ppf(newCDFValue)
      child.values[loc] = newValue
  return offspring

def randomMutator(offspring, distDict, **kwargs):
  """
    This method is designed to randomly mutate a single gene in each chromosome with probability = mutationProb.
    @ In, offspring, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offspring, xr.DataArray, children resulting from the crossover process
  """
  if kwargs['locs'] is not None and 'locs' in kwargs.keys():
    raise ValueError('Locs arguments are not being used by randomMutator')
  for child in offspring:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene location to be flipped: i.e., determine loc
      chromosomeSize = child.values.shape[0]
      loc = randomUtils.randomIntegers(0, chromosomeSize - 1, caller=None, engine=None)
      # gene at location loc is flipped from current value to newValue
      geneIDToBeChanged = child.coords['Gene'].values[loc]
      newCDFValue = randomUtils.random()
      newValue = distDict[geneIDToBeChanged].ppf(newCDFValue)
      child.values[loc] = newValue
  return offspring

def inversionMutator(offspring, distDict, **kwargs):
  """
    This method is designed mirror a sequence of genes in each chromosome with probability = mutationProb.
    The sequence of genes to be mirrored is completely random.
    E.g. given chromosome C = [0,1,2,3,4,5,6,7,8,9] and sampled locL=2 locU=6;
         New chromosome  C' = [0,1,6,5,4,3,2,7,8,9]
    @ In, offspring, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offspring, xr.DataArray, children resulting from the crossover process
  """
  # sample gene locations: i.e., determine locL and locU
  locL, locU = locationsGenerator(offspring, kwargs['locs'])

  for child in offspring:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # select sequence to be mirrored and mirror it
      seq = np.arange(locL,locU+1)
      cdfValues = []
      genes = child.coords['Gene'].values
      for elem in seq:
        cdfValues.append(distDict[genes[elem]].cdf(float(child[elem].values)))

      mirroredCdfValues = cdfValues[::-1]
      mirroredValues = []
      for elem, cdfValue in zip(seq, mirroredCdfValues):
        mirroredValues.append(distDict[genes[elem]].ppf(cdfValue))
      # insert mirrored sequence into child
      child.values[locL:locU+1] = mirroredValues

  return offspring

def locationsGenerator(offspring,locs):
  """
  Methods designed to process the locations for the mutators. These locations can be either user specified or
  randomly generated.
  @ In, offspring, xr.DataArray, children resulting from the crossover process
  @ In, locs, list, the two locations of the genes to be swapped
  @ Out, loc1, loc2, int, the two ordered processed locations required by the mutators
  """
  if locs is None:
    locs = list(set(randomUtils.randomChoice(list(np.arange(offspring.data.shape[1])),size=2,replace=False)))
  loc1 = np.minimum(locs[0], locs[1])
  loc2 = np.maximum(locs[0], locs[1])
  return loc1, loc2

def polynomialMutator(offspring, distDict, **kwargs):
  """
    Polynomial mutation for real-valued decision variables (Deb & Goyal, 1996).
    Each gene is perturbed, with probability mutationProb, by a polynomial-distributed
    step bounded by the decision-variable limits; the distribution index eta controls how
    local the perturbation is (larger eta -> smaller perturbations near the current value).
    Together with SBX this is the canonical NSGA-II continuous variation pair and provides
    the directed, bound-respecting local search that gene-resampling mutators lack.
    @ In, offspring, xr.DataArray, children resulting from the crossover process.
    @ In, distDict, dict, distribution per gene, used to obtain decision-variable bounds.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, per-gene probability that a gene is mutated.
          eta, float, mutation distribution index (default 20.0).
          variables, list, variable names.
    @ Out, children, xr.DataArray, the mutated children.
  """
  mutationProb = kwargs['mutationProb']
  eta = float(kwargs.get('eta', 20.0))
  geneNames = offspring.coords['Gene'].values
  children = offspring.copy(deep=True)
  numChildren, numGenes = np.shape(offspring)
  for i in range(numChildren):
    for g in range(numGenes):
      if float(randomUtils.random(dim=1, samples=1)) < mutationProb:
        x = float(offspring[i, g].values)
        low, high = gaUtils.finiteGeneBounds(distDict.get(geneNames[g]), x)
        spread = high - low
        if spread <= 0.0:
          continue
        delta1 = (x - low) / spread
        delta2 = (high - x) / spread
        rand = float(randomUtils.random(dim=1, samples=1))
        mutPow = 1.0 / (eta + 1.0)
        if rand < 0.5:
          xy = 1.0 - delta1
          val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
          deltaq = val ** mutPow - 1.0
        else:
          xy = 1.0 - delta2
          val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
          deltaq = 1.0 - val ** mutPow
        xNew = min(max(x + deltaq * spread, low), high)
        children.values[i, g] = xNew
  return children


__mutators = {}
__mutators['swapMutator']       = swapMutator
__mutators['scrambleMutator']   = scrambleMutator
__mutators['bitFlipMutator']    = bitFlipMutator
__mutators['inversionMutator']  = inversionMutator
__mutators['randomMutator']     = randomMutator
__mutators['polynomialMutator'] = polynomialMutator


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
