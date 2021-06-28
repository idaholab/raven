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

# [MANDD] Note: the fitness function are bounded by 2 parameters: a and b
#               We should make this method flexible to accept different set of params

# @profile
def invLinear(rlz,**kwargs):
  """
    Inverse linear fitness method
    .. math::

    fitness = \\frac{1}{a * obj + b * penalty}

    @ In, rlz, xr.Dataset, containing the evaluation of a certain
              set of individuals (can be the initial population for the very first iteration,
              or a population of offsprings)
    @ In, kwargs, dict, dictionary of parameters for this fitness method:
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
  fitness = -1.0*(a * rlz[objVar] + b * penalty)
  return fitness

def feasibleFirst(rlz,**kwargs):
  """
    Efficient Parameter-less Feasible First Penalty Fitness method
    .. math::

    fitness = \[ \\begin{cases}
                                                                      -obj & g_j(x)\\geq 0 \\forall j \\
                                                                      -obj_{worst} - \\Sigma_{j=1}^{J}<g_j(x)> & otherwise \\
                                                                    \\end{cases}
                                                                \]

    @ In, rlz, xr.Dataset, containing the evaluation of a certain
              set of individuals (can be the initial population for the very first iteration,
              or a population of offsprings)
    @ In, kwargs, dict, dictionary of parameters for this fitness method:
          objVar, string, the name of the objective variable
          'constraintFunction', xr.Dataarray, containing all constraint functions (explicit and implicit) evaluations for the whole population
    @ Out, fitness, xr.Dataarray, the fitness function of the given objective corresponding to a specific chromosome.
  """
  objVar = kwargs['objVar']
  g = kwargs['constraintFunction']
  worstObj = max(rlz[objVar].data)
  fitness = []
  for ind in range(len(rlz[objVar].data)):
    # fit = 0
    if np.all(g.data[ind, :]>=0):
      fit=(rlz[objVar].data[ind])
    else:
      fit = worstObj
      for constInd,constraint in enumerate(g['Constraint'].data):
        fit+=(max(0,-1 * g.data[ind, constInd]))
    fitness.append(-1 * fit)
  fitness = xr.DataArray(np.array(fitness),
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(len(rlz[objVar].data))})
  return fitness

def logistic(rlz,**kwargs):
  """
    Logistic fitness method
    .. math::

    fitness = \frac{1}{1+e^{-a(x-b)}}

    @ In, rlz, xr.Dataset, containing the evaluation of a certain
              set of individuals (can be the initial population for the very first iteration,
              or a population of offsprings)
    @ In, kwargs, dict, dictionary of parameters for this fitness method:
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
    b = 0.0
  else:
    b = kwargs['b']

  objVar = kwargs['objVar']
  val = rlz[objVar]
  denom = 1.0 + np.exp(-a * (val - b))
  fitness = 1.0 / denom

  return fitness


__fitness = {}
__fitness['invLinear'] = invLinear
__fitness['logistic']  = logistic
__fitness['feasibleFirst'] = feasibleFirst


def returnInstance(cls, name):
  """
    Method designed to return class instance:
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __fitness:
    cls.raiseAnError (IOError, "{} FITNESS FUNCTION NOT IMPLEMENTED!!!!!".format(name))
  return __fitness[name]