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
# Internal Imports
from ... import MessageHandler # makes sure getMessageHandler is defined
mh = getMessageHandler()

# [MANDD] Note: the fitness function are bounded by 2 parameters: a and b
#               We should make this method flexible to accept different set of params

_defaultObjectiveScaling = 1.0
_defaultPenaltyScaling = 10.0

# @profile
def invLinear(rlz, **kwargs):
  r"""
  Inverse linear fitness method where the fitness value is inversely proportional to the objective function.
  For minimization problems:
  1. As the objective decreases (closer to min), the fitness value increases.
  2. As the objective increases (away from the min), the fitness value decreases.
  3. If the solution violates the constraints, the fitness decreases, making it less favorable.
  For maximization problems, the objective value is negated, inverting the trends.
  Formula:
  fitness = -a * obj - b * \Sum_{j=1}^{nConstraint} max(0, -penalty_j)
  @ In, rlz, xr.Dataset, containing the evaluation of a set of individuals
  @ In, kwargs, dict, dictionary of parameters:
        objVar, list of strings or single string, name(s) of the objective variable(s)
        a, list of floats, linear coefficient(s) for the objective function (default = 1.0 for each objective)
        b, list of floats, linear coefficient(s) for the penalty measure (default = 10.0 for each objective)
        constraintFunction, xr.DataArray, measuring the severity of the constraint violation.
        type, list of strings, indicating 'min' or 'max' for each objective
  @ Out, fitnessSet, xr.Dataset, the fitness function for the given population.
  """
  objVar = kwargs['objVar']
  a = [_defaultObjectiveScaling] * len(objVar) if kwargs.get('a') is None else kwargs['a']  # Scaling factors for objectives
  b = [_defaultPenaltyScaling] * len(objVar) if kwargs.get('b') is None else kwargs['b']  # Penalty scaling factors
  if len(a) != len(objVar):
    mh.error("fitness", IOError, f"Objective scaling factors {a} should have length {len(objVar)}")
  if len(b) != len(objVar):
    mh.error("fitness", IOError, f"Penalty scaling factors {b} should have length {len(objVar)}")
  g = kwargs['constraintFunction'] if 'constraintFunction' in kwargs else None  # Constraint evaluations
  fitnessSet = xr.Dataset()
  for i, obj in enumerate(objVar):
      data = np.atleast_1d(rlz[obj].data)  # Objective values
      fitness = np.zeros(data.shape)
      for ind in range(data.shape[0]):
          # Calculate base fitness: Inversely proportional to the objective value
          fit = -a[i] * data[ind]
          # Apply penalties for constraint violations, if any
          if g is not None and np.any(g.data[ind, :] < 0):  # Violating constraints
              for constInd in range(g.data.shape[1]):
                  fit -= b[i] * max(0, -g.data[ind, constInd])  # Apply penalty for violation
          fitness[ind] = fit
      # Add the fitness for the current objective to the dataset
      fitnessSet[obj] = xr.DataArray(fitness, dims=['chromosome'], coords={'chromosome': np.arange(len(data))})
  return fitnessSet

def feasibleFirst(rlz, **kwargs):
  r"""
  Efficient Parameter-less Feasible First Penalty Fitness method
  This method is designed for minimization problems. For maximization, the fitness values are negated.
  For minimization problems:
  1.  As the objective decreases, the fitness value increases.
  2.  As the objective increases, the fitness value decreases.
  3.  If a solution violates constraints, the fitness decreases, making it less favorable.
  For maximization problems, the objective value is negated, inverting the trends.
  Reference: Deb, Kalyanmoy. "An efficient constraint handling method for genetic algorithms."

  .. math::
  fitness = \[ \\begin{cases}
                -obj & g_j(x)\\geq 0 \\forall j \\
                -obj_{worst} - \\Sigma_{j=1}^{J}<g_j(x)> & otherwise \\
                \\end{cases}
            \];
  @ In, rlz, xr.Dataset, containing the evaluation of a set of individuals
  @ In, kwargs, dict, dictionary of parameters:
        objVar, list of strings, the names of objective variables
        'constraintFunction', xr.DataArray, containing all constraint evaluations for the population
        'constraintNum', int, number of constraints
        'a', list of floats, scaling factors for the objectives
        'b', list of floats, penalty factors for constraint violations
        'type', list of strings, indicating 'min' or 'max' for each objective
  @ Out, fitnessSet, xr.Dataset, the fitness function for the given population.
  """
  objVar = kwargs['objVar']
  a = [_defaultObjectiveScaling] * len(objVar) if kwargs.get('a') is None else kwargs['a']  # Scaling factors for objectives
  b = [_defaultPenaltyScaling] * len(objVar) if kwargs.get('b') is None else kwargs['b']  # Penalty scaling factors
  if len(a) != len(objVar):
    mh.error("fitness", IOError, f"Objective scaling factors {a} should have length {len(objVar)}")
  if len(b) != len(objVar):
    mh.error("fitness", IOError, f"Penalty scaling factors {b} should have length {len(objVar)}")
  constraintNum = kwargs['constraintNum']
  g = kwargs['constraintFunction'] if constraintNum > 0 else None  # Constraint evaluations
  fitnessSet = xr.Dataset()
  # For each objective
  for i, obj in enumerate(objVar):
      data = np.atleast_1d(rlz[obj].data)
      worstObj = max(data) # Worst objective value for penalizing violating solutions
      fitness = np.zeros(data.shape)
      for ind in range(data.shape[0]):
          # If no contraints or all constraints are satisfied
          if constraintNum == 0 or np.all(g.data[ind, :] >= 0):  # Feasible solutions
              fit = -a[i] * data[ind]
          # if constraints are violated
          else:  # Penalize constraint violations
              fit = -a[i] * worstObj  # Start with the worst objective value
              for constInd in range(g.data.shape[1]):
                  violation = max(0, -g.data[ind, constInd])
                  fit -= b[i] * violation
          fitness[ind] = fit
      # Add the fitness for the current objective to the dataset
      fitnessSet[obj] = xr.DataArray(fitness, dims=['chromosome'], coords={'chromosome': np.arange(len(data))})
  return fitnessSet

def logistic(rlz, **kwargs):
  r"""
  Logistic fitness method for multi-objective optimization with constraint handling.
  For minimization problems:
  1. As the objective decreases, the fitness value increases.
  2. As the objective increases, the fitness value decreases.
  3. If the solution violates the constraints, the fitness decreases, making it less favorable.
  For maximization problems, the objective value is negated, inverting the trends.
  math::
    fitness = \frac{1}{1 + e^{-scale(x - shift)}} - penalty terms for constraint violations.
  @ In, rlz, xr.Dataset, containing the evaluation of a set of individuals
  @ In, kwargs, dict, dictionary of parameters:
        objVar, list of strings or single string, name(s) of the objective variable(s)
        scale, list of floats, scaling coefficient(s) for the objective function (default = 1.0 for each objective)
        shift, list of floats, coefficient(s) for shifting the objective value (default = 0.0 for each objective)
        constraintFunction, xr.DataArray, measuring the severity of the constraint violation
        penalty, list of floats, penalties for constraint violations (default = 1.0 for each objective)
        type, list of strings, indicating 'min' or 'max' for each objective
  @ Out, fitnessSet, xr.Dataset, the fitness function for the given population.
  """
  objVar = kwargs['objVar']
  scale = kwargs.get('scale', [_defaultObjectiveScaling] * len(objVar))  # Scaling factors for objectives
  shift = kwargs.get('shift', [0.0] * len(objVar))  # Shifting value for each objective
  penalty = kwargs.get('penalty', [_defaultPenaltyScaling] * len(objVar))  # Penalty for constraint violations
  g = kwargs.get('constraintFunction', None)  # Constraint evaluations (if any)
  fitnessSet = xr.Dataset()
  for i, obj in enumerate(objVar):
      data = np.atleast_1d(rlz[obj].data)  # Objective values
      fitness = np.zeros(data.shape)
      for ind in range(data.shape[0]):
          # Base logistic fitness calculation
          denom = 1.0 + np.exp(-scale[i] * (data[ind] - shift[i]))
          fit = 1.0 / denom
          # Apply penalties for constraint violations, if any
          if g is not None and np.any(g.data[ind, :] < 0):  # Constraint violation
              for constInd in range(g.data.shape[1]):
                  fit -= penalty[i] * max(0, -g.data[ind, constInd])
          # Adjust for maximization problems by negating the fitness value
          if kwargs['type'][i] == 'max':
              fit = 1.0 - fit  # Adjust the logistic fitness for maximization
          fitness[ind] = fit
      # Store fitness in the dataset
      fitnessSet[obj] = xr.DataArray(fitness, dims=['chromosome'], coords={'chromosome': np.arange(len(data))})
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
