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
  Implementation of fitness function for Genetic Algorithm
  currently the implemented fitness function is a linear combination of the objective function and penalty function for constraint violation:

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""
# External Imports
import numpy as np
import xarray as xr
# Internal Imports
from utils import randomUtils

def invLinear(rlz,**kwargs):
  """
    .. math::

    fitness = \\frac{1}{a * obj + b * penalty}

    @ In, rlz, xr.Dataset, containing the evaluation of a certain
              set of individuals (can be the initial population for the very first iteration,
              or a population of offsprings)
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          objVar, string, the name of the objective variable
          a, float, linear coefficient for the objective function (default = 1.0)
          penalty, float, measuring the severity of the constraint violation. (default = 1.0)
          b, float, linear coefficient for the penalty measure. (default = 1.0)
    @ Out, fitness, float, the fitness function of the given objective corresponding to a specific chromosome.
  """
  if kwargs['a'] == None:
    a = 1.0
  else:
    a = kwargs['a']
  if kwargs['b'] == None:
    b = 1.0
  else:
    b = kwargs['b']
  if kwargs['penalty'] == None:
    penalty = 0.0
  else:
    penalty = kwargs['penalty']

  objVar = kwargs['objVar']
  # Initializing fitness
  fitness = xr.DataArray(np.zeros((eval('rlz[\'' + objVar + '\'].size'))),
                              dims=['chromosome'],
                              coords={'chromosome': np.arange(eval('rlz[\'' + objVar + '\'].size'))})
  for i in range(eval('rlz[\'' + objVar + '\'].size')):
    obj = eval('rlz[\'' + objVar + '\'].data[i]')*randomUtils.random(dim=1,samples=1)
    fitness[i] = 1.0/(a * obj + b * penalty)
  return fitness

__fitness = {}
__fitness['invLinear'] = invLinear


def returnInstance(cls, name):
  if name not in __fitness:
    cls.raiseAnError (IOError, "{} FITNESS FUNCTION NOT IMPLEMENTED!!!!!".format(name))
  return __fitness[name]
