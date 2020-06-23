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

from operator import itemgetter
from utils import randomUtils

def swapMutator(**kwargs):
  """
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          chromosome, numpy.array, the chromosome that will mutate to the new child
          locs, list, the 2 locations of the genes to be swapped
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, child, numpy.array, the mutated chromosome, i.e., the child.
  """
  loc1 = kwargs['locs'][0]
  loc2 = kwargs['locs'][1]
  chromosome = kwargs['chromosome']
  ## TODO What happens if loc1 or 2 is out of range?! should we raise an error?
  child = chromosome.copy()
  if randomUtils.random(dim=1,samples=1)>kwargs['mutationProb']:
    child[loc1] = chromosome[loc2]
    child[loc2] = chromosome[loc1]
  return child

def scrambleMutator(**kwargs):
  """
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          chromosome, numpy.array, the chromosome that will mutate to the new child
          locs, list, the locations of the genes to be randomly scrambled
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, child, np.array, the mutated chromosome, i.e., the child.
  """
  chromosome = kwargs['chromosome']
  locs = kwargs['locs']
  nMutable = len(locs)
  child = chromosome.copy()

  new = list(itemgetter(*locs)(chromosome))
  for ind,element in enumerate(locs):
    if randomUtils.random(dim=1,samples=1)>kwargs['mutationProb']:
      child[element] = new[ind]
  return child


__mutators = {}
__mutators['swapMutator'] = swapMutator
__mutators['scrambleMutator'] = scrambleMutator


def returnInstance(cls, name):
  if name not in __mutators:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __mutators[name]
