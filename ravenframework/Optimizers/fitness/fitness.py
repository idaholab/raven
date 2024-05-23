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
  Updated September,17,2023
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi, Junyung Kim
"""
# Internal Modules----------------------------------------------------------------------------------
from ...utils import frontUtils
from ..parentSelectors.parentSelectors import countConstViolation

# External Imports
import numpy as np
import xarray as xr
import sys
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
  #NOTE invLinear is not yet support Multi-objective optimization problem solving. Further literature reivew applying invLinear method to multi-objective optimization
  #     needs to be involved. Potentially, applying obj_Worst in fitness function (i.e., -a[j] * (rlz[objVar][objVar[j]].data).reshape(-1,1) - b[j] * np.sum(np.maximum(0,-penalty),axis=-1).reshape(-1,1))
  #     should be considerd .
  a = [1.0] if kwargs['a'] == None else kwargs['a']
  b = [10.0] if kwargs['b'] == None else kwargs['b']
  penalty = 0.0 if kwargs['constraintFunction'].all() == None  else kwargs['constraintFunction'].data
  objVar = [kwargs['objVar']] if isinstance(kwargs['objVar'], str) == True else kwargs['objVar']
  for j in range(len(objVar)):
    data = np.atleast_1d(rlz[objVar][objVar[j]].data)
    fitness = -a[j] * (rlz[objVar][objVar[j]].data).reshape(-1,1) - b[j] * np.sum(np.maximum(0,-penalty),axis=-1).reshape(-1,1)
    fitness = xr.DataArray(np.squeeze(fitness),
                           dims=['chromosome'],
                           coords={'chromosome': np.arange(len(data))})
    if j == 0:
        fitnessSet = fitness.to_dataset(name = objVar[j])
    else:
        fitnessSet[objVar[j]] = fitness
  return fitnessSet


def feasibleFirst(rlz,**kwargs):
  r"""
    Efficient Parameter-less Feasible First Penalty Fitness method
    This method is designed such that:
    For minimization Problems:
    1.  As the objective function decreases (comes closer to the min value), the fitness value increases
    2.  As the objective function increases (away from the min value), the fitness value decreases
    3.  As the solution violates the constraints the fitness should decrease and hence the solution is less favored by the algorithm.
    4.  For the violating solutions, the fitness starts from the worst solution in the population
        (i.e., max objective in minimization problems and min objective in maximization problems)

    For maximization problems the objective value is multiplied by -1 and hence the previous trends are inverted.
    A great quality of this fitness is that if the objective value is equal for multiple solutions it selects the furthest from constraint violation.

    Reference: Deb, Kalyanmoy. "An efficient constraint handling method for genetic algorithms." Computer methods in applied mechanics and engineering 186.2-4 (2000): 311-338.

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
  objVar = [kwargs['objVar']] if isinstance(kwargs['objVar'], str) == True else kwargs['objVar']
  a = [1.0]*len(objVar) if kwargs['a'] == None else kwargs['a']
  if kwargs['constraintNum'] == 0:
    pen = kwargs['b']
  else:
    g = kwargs['constraintFunction']
    penalty = kwargs['b']
    pen = [penalty[i:i+len(g['Constraint'].data)] for i in range(0, len(penalty), len(g['Constraint'].data))]
  objPen = dict(map(lambda i,j : (i,j), objVar, pen))

  for i in range(len(objVar)):
    data = np.atleast_1d(rlz[objVar][objVar[i]].data)
    worstObj = max(data)
    fitness = []
    for ind in range(data.size):
      if kwargs['constraintNum'] == 0 or np.all(g.data[ind, :]>=0):
        fit=(a[i]*data[ind])
      else:
        fit = a[i]*worstObj
        for constInd,_ in enumerate(g['Constraint'].data):
          fit = a[i]*fit + objPen[objVar[i]][constInd]*(max(0,-1*g.data[ind, constInd])) #NOTE: objPen[objVar[i]][constInd] is "objective & Constraint specific penalty."
      if len(kwargs['type']) == 1:
        fitness.append(-1*fit)
      else:
        fitness.append(fit)

    fitness = xr.DataArray(np.array(fitness),
                          dims=['chromosome'],
                          coords={'chromosome': np.arange(len(data))})
    if i == 0:
      fitnessSet = fitness.to_dataset(name = objVar[i])
    else:
      fitnessSet[objVar[i]] = fitness

  return fitnessSet

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
    a = [1.0]
  else:
    a = kwargs['a']
  if kwargs['b'] == None:
    b = [0.0]
  else:
    b = kwargs['b']
  if isinstance(kwargs['objVar'], str) == True:
    objVar = [kwargs['objVar']]
  else:
    objVar = kwargs['objVar']
  for i in range(len(objVar)):
    val = rlz[objVar][objVar[i]].data
    data = np.atleast_1d(rlz[objVar][objVar[i]].data)
    denom = 1.0 + np.exp(-a[0] * (val - b[0]))
    fitness = 1.0 / denom
    fitness = xr.DataArray(fitness.data,
                           dims=['chromosome'],
                           coords={'chromosome': np.arange(len(data))})
    if i == 0:
      fitnessSet = fitness.to_dataset(name = objVar[i])
    else:
      fitnessSet[objVar[i]] = fitness

  return fitnessSet


__fitness = {}
__fitness['invLinear'] = invLinear
__fitness['logistic']  = logistic
__fitness['feasibleFirst'] = feasibleFirst
#NOTE hardConstraint method will be used later once constraintHandling is realized. Until then, it will be commented. @JunyungKim
# __fitness['hardConstraint'] = hardConstraint


def returnInstance(cls, name):
  """
    Method designed to return class instance:
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __fitness:
    cls.raiseAnError (IOError, "{} is not a supported fitness function. ".format(name))
  return __fitness[name]
