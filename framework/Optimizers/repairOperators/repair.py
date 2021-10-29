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
  Implementation of gene repair operators for Genetic Algorithm
  currently the implemented repair algorithms are:
  1.  replacementRepair

  Created July,11,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

import numpy as np

# @profile
def replacementRepair(offSprings,**kwargs):
  """
    @ In, offSprings, xr.DataArray, distorted offSprings resulting from the mating process.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          variables, list, variables names.
          distInfo, dict, Distribution information.
          distDict, dict, Distribution dictionary.
    @ Out, children, np.array, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nChildren,nGenes = np.shape(offSprings)
  children = offSprings
  # read distribution info
  distInfo = kwargs['distInfo']

  # create children
  for chrom in range(nChildren):
    duplicated = set()
    unique = set(offSprings.data[chrom,:])
    if len(offSprings.data[chrom,:]) != len(unique):
      for ind,x in enumerate(offSprings.data[chrom,:]):
        if x not in duplicated:
          children[chrom,ind] = x
          duplicated.add(x)
        else:
          if (distInfo[offSprings['Gene'].data[ind]].strategy == 'withoutReplacement'):
            y = distInfo[kwargs['variables'][ind]].selectedRvs(list(duplicated))
            children[chrom,ind] = y
            duplicated.add(y)
  return children

__repairs = {}
__repairs['replacementRepair']  = replacementRepair

def returnInstance(cls, name):
  """
    Method designed to return class instance
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __repairs:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __repairs[name]
