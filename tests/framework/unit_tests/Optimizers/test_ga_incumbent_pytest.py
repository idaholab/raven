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
  Tests for how the single-objective GeneticAlgorithm keeps its incumbent (the reported optimum).
  Fitness is the selection criterion and already carries the constraint penalty, so the incumbent is
  replaced whenever a generation's best fitness is higher. Earlier versions also required a lower raw
  objective, so an infeasible first-generation point with a very good objective stayed the reported
  optimum for the whole run.

  @author: Mohammad Abdo (@Jimmy-INL)
"""
import numpy as np
import xarray as xr

from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm


def _makeGA():
  """
    Bare single-objective GA with only the attributes _collectOptPoint reads.
    @ In, None
    @ Out, ga, GeneticAlgorithm, minimal instance
  """
  ga = GeneticAlgorithm()
  ga.toBeSampled = {'x': 'dist', 'y': 'dist'}
  ga._objectiveVar = ['ans']
  ga._solutionExport = None
  ga.popAges = None
  ga.batchId = 1
  return ga


def _generation(ga, counter, genes, minObj, fitness, constraint):
  """
    Feed one generation's survivors to _collectOptPoint.
    @ In, ga, GeneticAlgorithm, the optimizer
    @ In, counter, int, generation counter (1 = first generation)
    @ In, genes, list, one [x, y] row per chromosome
    @ In, minObj, list, minimization-space objective of each chromosome
    @ In, fitness, list, fitness of each chromosome
    @ In, constraint, list, constraint value g of each chromosome (g < 0 is a violation)
    @ Out, None
  """
  ga.counter = counter
  population = xr.DataArray(np.array(genes, dtype=float), dims=['chromosome', 'Gene'],
                            coords={'chromosome': np.arange(len(genes)), 'Gene': ['x', 'y']})
  fitVals = xr.Dataset({'ans': xr.DataArray(np.array(fitness, dtype=float), dims=['chromosome'])})
  constraintVals = xr.DataArray(np.array(constraint, dtype=float).reshape(-1, 1), dims=['chromosome', 'Constraint'],
                                coords={'chromosome': np.arange(len(genes)), 'Constraint': ['g']})
  ga._collectOptPoint(None, fitVals, list(minObj), constraintVals, population=population)


def test_higher_fitness_replaces_infeasible_incumbent():
  """A feasible point with higher fitness becomes the incumbent even though its raw objective is worse."""
  ga = _makeGA()
  # generation 1: the best fitness belongs to an infeasible point with a very good objective
  _generation(ga, 1, [[1.2, -0.8], [0.0, 0.0]], minObj=[-629.0, -0.7], fitness=[0.51, 0.33], constraint=[-0.49, 2.0])
  assert ga.bestPoint['x'] == 1.2
  # generation 2: a feasible point has higher fitness but a worse (larger) minimization-space objective
  _generation(ga, 2, [[1.1, -0.8], [0.0, 0.0]], minObj=[-454.0, -0.7], fitness=[0.99, 0.33], constraint=[0.01, 2.0])
  assert ga.bestPoint['x'] == 1.1
  assert ga.bestFitVal == 0.99
  assert ga.bestPoint['ConstraintEvaluation_g'] >= 0


def test_lower_fitness_keeps_incumbent():
  """A generation whose best fitness is lower does not replace the incumbent."""
  ga = _makeGA()
  _generation(ga, 1, [[1.1, -0.8]], minObj=[-454.0], fitness=[0.99], constraint=[0.01])
  _generation(ga, 2, [[1.5, -1.4]], minObj=[-1375.0], fitness=[-1.4], constraint=[-2.4])
  assert ga.bestPoint['x'] == 1.1


def test_tie_refreshes_incumbent_record():
  """An equal-fitness, equal-objective survivor refreshes the incumbent record (e.g. its batch)."""
  ga = _makeGA()
  _generation(ga, 1, [[1.0, 1.0]], minObj=[2.0], fitness=[-2.0], constraint=[1.0])
  ga.batchId = 2
  _generation(ga, 2, [[1.0, 1.0]], minObj=[2.0], fitness=[-2.0], constraint=[1.0])
  assert ga._bestSnapshot['batchId'] == 2


if __name__ == '__main__':
  from pytest_runner import run_module
  run_module(__file__)
