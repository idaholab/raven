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
from ... import MessageHandler # makes sure getMessageHandler is defined
mh = getMessageHandler()

# [MANDD] Note: the fitness function are bounded by 2 parameters: a and b
#               We should make this method flexible to accept different set of params

_defaultObjectiveScaling = 1.0
_defaultPenaltyScaling = 10.0          # default <b> for invLinear/feasibleFirst (devel used 1.0)
_defaultLogisticPenalty = 1.0          # default constraint penalty for logistic (unchanged from devel)


def _scalingFactors(objVar, constraintNum, a=None, b=None):
  """
    Resolve the objective (a) and penalty (b) scaling factors to their full shapes.
    'a' may be a scalar or have one entry per objective. 'b' may be a scalar, have one entry per
    constraint (shared by every objective, as in earlier RAVEN versions), or have one entry per
    (objective, constraint) pair, given objective-major, i.e. b = [b_(1,1), ..., b_(1,nConst), b_(2,1), ...].
    @ In, objVar, list(str), objective variable names
    @ In, constraintNum, int, number of constraints
    @ In, a, float or list(float), optional, objective scaling factors
    @ In, b, float or list(float), optional, penalty scaling factors
    @ Out, a, np.array, shape (len(objVar),), objective scaling factors
    @ Out, b, np.array, shape (len(objVar), constraintNum), penalty scaling factors
  """
  nObj = len(objVar)
  a = np.atleast_1d(np.asarray(a if a is not None else _defaultObjectiveScaling, dtype=float))
  if a.size == 1:
    a = np.full(nObj, a[0])
  if a.shape != (nObj,):
    mh.error("fitness", IOError, f"Objective scaling factors {a.tolist()} should have length {nObj}")
  b = np.atleast_1d(np.asarray(b if b is not None else _defaultPenaltyScaling, dtype=float)).ravel()
  if b.size == 1:
    b = np.full((nObj, constraintNum), b[0])
  elif b.size == constraintNum:
    b = np.tile(b, (nObj, 1))   # one per constraint, shared by every objective
  elif b.size == nObj * constraintNum:
    b = b.reshape(nObj, constraintNum)
  else:
    mh.error("fitness", IOError, f"Penalty scaling factors {b.tolist()} must have length 1, {constraintNum} (one per "
             f"constraint), or {nObj * constraintNum} (one per objective and constraint)")
  return a, b

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
  fitness = -a_i * obj - \Sum_{j=1}^{nConstraint} b_(i,j) * max(0, -penalty_j)
  @ In, rlz, xr.Dataset, containing the evaluation of a set of individuals
  @ In, kwargs, dict, dictionary of parameters:
        objVar, list of strings or single string, name(s) of the objective variable(s)
        a, list of floats, linear coefficient(s) for the objective function (default = 1.0 for each objective)
        b, list of floats, penalty coefficient(s): one value, one per constraint, or one per (objective, constraint) pair
           (default = 10.0; this default was 1.0 before the NSGA-II refactor, so constrained inputs without <b> change results)
        constraintFunction, xr.DataArray, measuring the severity of the constraint violation.
        type, list of strings, indicating 'min' or 'max' for each objective
  @ Out, fitnessSet, xr.Dataset, the fitness function for the given population.
  """
  objVar = kwargs['objVar']
  g = kwargs['constraintFunction'] if kwargs.get('constraintFunction') is not None else None  # Constraint evaluations
  constraintNum = kwargs.get('constraintNum', g.data.shape[1] if g is not None else 0)
  a, b = _scalingFactors(objVar, constraintNum, a=kwargs.get('a'), b=kwargs.get('b'))
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
                  fit -= b[i][constInd] * max(0, -g.data[ind, constInd])  # Apply penalty for violation
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

  This is a weighted generalization of Deb (2000): with a_i = 1 and b_i = 1 the two branches reduce
  exactly to Deb's parameter-less scheme; a_i and b_i let the user re-weight the objective and the
  penalty. Following Deb, obj_{worstFeasible} is the worst objective among the FEASIBLE individuals
  (not the whole population), which guarantees every feasible solution outranks every infeasible one
  and orders infeasibles purely by total violation. When no feasible individual exists yet, the
  whole-population worst objective is used as a fallback anchor.

  .. math::
  fitness = \[ \\begin{cases}
                -a_i obj & g_j(x)\\geq 0 \\forall j \\
                -a_i obj_{worstFeasible} - b_i \\Sigma_{j=1}^{J}<g_j(x)> & otherwise \\
                \\end{cases}
            \];
  @ In, rlz, xr.Dataset, containing the evaluation of a set of individuals
  @ In, kwargs, dict, dictionary of parameters:
        objVar, list of strings, the names of objective variables
        'constraintFunction', xr.DataArray, containing all constraint evaluations for the population
        'constraintNum', int, number of constraints
        'a', list of floats, scaling factors for the objectives (default = 1.0 for each objective)
        'b', list of floats, penalty factors: one value, one per constraint, or one per (objective, constraint) pair
             (default = 10.0; this default was 1.0 before the NSGA-II refactor)
        'type', list of strings, indicating 'min' or 'max' for each objective
  @ Out, fitnessSet, xr.Dataset, the fitness function for the given population.
  """
  objVar = kwargs['objVar']
  constraintNum = kwargs['constraintNum']
  a, b = _scalingFactors(objVar, constraintNum, a=kwargs.get('a'), b=kwargs.get('b'))
  g = kwargs['constraintFunction'] if constraintNum > 0 else None  # Constraint evaluations
  fitnessSet = xr.Dataset()
  # Feasibility mask over the population: True where the individual satisfies all constraints.
  # Deb (2000) anchors the infeasible penalty base at the WORST FEASIBLE objective, not the worst
  # objective over the whole population, so that (a) every feasible solution dominates every infeasible
  # one and (b) infeasibles are ordered purely by total violation. When no feasible solution exists yet
  # (common in early generations) worst-feasible is undefined, so we fall back to the whole-population
  # worst objective as a sane anchor.
  if constraintNum == 0:
    feasibleMask = np.ones(np.atleast_1d(rlz[objVar[0]].data).shape[0], dtype=bool)
  else:
    feasibleMask = np.all(g.data >= 0, axis=1)
  # For each objective
  for i, obj in enumerate(objVar):
      data = np.atleast_1d(rlz[obj].data)
      worstObj = max(data[feasibleMask]) if np.any(feasibleMask) else max(data) # Worst FEASIBLE objective (Deb 2000); whole-population worst as all-infeasible fallback
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
                  fit -= b[i][constInd] * violation
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
        penalty, list of floats or None, penalties for constraint violations (default = 1.0 for each objective; None uses the default)
        type, list of strings, indicating 'min' or 'max' for each objective
  @ Out, fitnessSet, xr.Dataset, the fitness function for the given population.
  """
  objVar = kwargs['objVar']
  # the GA passes None for options the input omits, so fall back on None as well as on a missing key
  scale = kwargs.get('scale')
  if scale is None:
    scale = [_defaultObjectiveScaling] * len(objVar)  # Scaling factors for objectives
  shift = kwargs.get('shift')
  if shift is None:
    shift = [0.0] * len(objVar)  # Shifting value for each objective
  # the GA passes penalty=None, so fall back to the default instead of only when the key is absent
  penalty = kwargs.get('penalty')
  if penalty is None:
    penalty = [_defaultLogisticPenalty] * len(objVar)  # Penalty for constraint violations
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


def returnInstance(cls, name):
  """
    Method designed to return class instance:
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __fitness[name], the registered fitness operator
  """
  if name not in __fitness:
    cls.raiseAnError (IOError, "{} is not a supported fitness function. ".format(name))
  return __fitness[name]
