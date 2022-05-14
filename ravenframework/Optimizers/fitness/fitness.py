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
  r"""
    Inverse linear fitness method requires that the fitness value is inversely proportional to the objective function
    This method is designed such that:
    For minimization Problems:
    1.  As the objective function decreases (comes closer to the min value), the fitness value increases
    2.  As the objective function increases (away from the min value), the fitness value decreases
    3.  As the solution violates the constraints the fitness should decrease and hence the solution is less favored by the algorithm.

    For maximization problems the objective value is multiplied by -1 and hence the previous trends are inverted.
    A great quality of this fitness is that if the objective value is equal for multiple solutions it selects the furthest from constraint violation.

    .. math::

    fitness = -a * obj - b * \Sum_{j=1}^{nConstraint} max(0,-penalty_j)
    @ In, rlz, xr.Dataset, containing the evaluation of a certain
              set of individuals (can be the initial population for the very first iteration,
              or a population of offsprings)
    @ In, kwargs, dict, dictionary of parameters for this fitness method:
          objVar, string, the name of the objective variable
          a, float, linear coefficient for the objective function (default = 1.0)
          b, float, linear coefficient for the penalty measure. (default = 10.0)
          ConstraintFunction, xr.DataArray, measuring the severity of the constraint violation. The higher positive value the individual  has,
          the farthest from violating the constraint it is, The highest negative value it have the largest the violation is.
    @ Out, fitness, xr.DataArray, the fitness function of the given objective corresponding to a specific chromosome.
  """
  if kwargs['a'] == None:
    a = 1.0
  else:
    a = kwargs['a']
  if kwargs['b'] == None:
    b = 10.0
  else:
    b = kwargs['b']
  if kwargs['constraintFunction'].all() == None:
    penalty = 0.0
  else:
    penalty = kwargs['constraintFunction'].data

  objVar = kwargs['objVar']
  data = np.atleast_1d(rlz[objVar].data)

  fitness = -a * (rlz[objVar].data).reshape(-1,1) - b * np.sum(np.maximum(0,-penalty),axis=-1).reshape(-1,1)
  fitness = xr.DataArray(np.squeeze(fitness),
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(len(data))})
  return fitness

def feasibleFirst(rlz,**kwargs):
  r"""
    Efficient Parameter-less Feasible First Penalty Fitness method
    This method is designed such that:
    For minimization Problems:
    1.  As the objective function decreases (comes closer to the min value), the fitness value increases
    2.  As the objective function increases (away from the min value), the fitness value decreases
    3.  As the solution violates the constraints the fitness should decrease and hence the solution is less favored by the algorithm.
    4.  For the violating solutions, the fitness is starts from the worst solution in the population
        (i.e., max objective in minimization problems and min objective in maximization problems)

    For maximization problems the objective value is multiplied by -1 and hence the previous trends are inverted.
    A great quality of this fitness is that if the objective value is equal for multiple solutions it selects the furthest from constraint violation.

    .. math::

    fitness = \[ \\begin{cases}
                  -obj & g_j(x)\\geq 0 \\forall j \\
                  -obj_{worst} - \\Sigma_{j=1}^{J}<g_j(x)> & otherwise \\
                  \\end{cases}
              \];

    @ In, rlz, xr.Dataset, containing the evaluation of a certain
              set of individuals (can be the initial population for the very first iteration,
              or a population of offsprings)
    @ In, kwargs, dict, dictionary of parameters for this fitness method:
          objVar, string, the name of the objective variable
          'constraintFunction', xr.Dataarray, containing all constraint functions (explicit and implicit) evaluations for the whole population
    @ Out, fitness, xr.DataArray, the fitness function of the given objective corresponding to a specific chromosome.
  """
  objVar = kwargs['objVar']
  g = kwargs['constraintFunction']
  data = np.atleast_1d(rlz[objVar].data)
  worstObj = max(data)
  fitness = []
  for ind in range(data.size):
    if np.all(g.data[ind, :]>=0):
      fit=(data[ind])
    else:
      fit = worstObj
      for constInd,_ in enumerate(g['Constraint'].data):
        fit+=(max(0,-1 * g.data[ind, constInd]))
    fitness.append(-1 * fit)
  fitness = xr.DataArray(np.array(fitness),
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(len(data))})
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
    @ Out, fitness, xr.DataArray, the fitness function of the given objective corresponding to a specific chromosome.
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
  data = np.atleast_1d(rlz[objVar].data)
  denom = 1.0 + np.exp(-a * (val - b))
  fitness = 1.0 / denom
  fitness = xr.DataArray(np.array(fitness),
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(len(data))})

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
