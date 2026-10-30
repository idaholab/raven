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
  Unit tests for the multi-objective GA convergence (stopping) criteria defined on
  MultiObjectiveGeneticAlgorithm: hypervolume, spread, maxSpread, and rank1Ratio. Each test drives
  the _checkConv* method directly on an NSGAII instance (the concrete multi-objective leaf) with
  hand-built population state, and asserts the method reports convergence on a converged input and
  does NOT report it on a non-converged one (so an always-True check cannot pass).
"""
import numpy as np
import xarray as xr

from ravenframework.Optimizers.NSGAII import NSGAII


def _makeOptimizer(objectiveVars):
  """
    Build a bare NSGAII instance with just the attributes the convergence checks read. The optimizer
    is not initialized through RAVEN input; only the fields each _checkConv* consults are set.
    @ In, objectiveVars, list, objective-variable names
    @ Out, optimizer, NSGAII, a minimally-populated optimizer instance
  """
  optimizer = NSGAII()
  optimizer.raiseADebug = lambda *args, **kwargs: None
  optimizer.raiseAWarning = lambda *args, **kwargs: None
  optimizer._objectiveVar = list(objectiveVars)
  return optimizer


def _setPopulation(optimizer, ranks, objectiveRows):
  """
    Set the current-population rank and minimization-space objective fields the checks read.
    @ In, optimizer, NSGAII, the instance to populate
    @ In, ranks, list, per-individual non-dominated rank (1 = best front)
    @ In, objectiveRows, list, one list of per-individual values per objective
    @ Out, None
  """
  optimizer.popRanks = xr.DataArray(np.array(ranks), dims=['chromosome'])
  optimizer.popMinObjVals = [np.array(row, dtype=float) for row in objectiveRows]


def test_convergence_spread():
  """
    spread: converges when the rank-1 front's Deb spread falls below the threshold. Single call, needs
    at least three rank-1 points.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  _setPopulation(optimizer,
                 ranks=[1, 1, 1, 1],
                 objectiveRows=[[1.0, 0.8, 0.6, 0.4], [0.4, 0.6, 0.8, 1.0]])
  optimizer._convergenceCriteria = {'spread': 10.0}
  assert optimizer._checkConvSpread(0)


def test_convergence_rank1_ratio():
  """
    rank1Ratio: converges once the rank-1 fraction has been high and stable for three generations.
    The check appends a ratio each call, so it is called three times.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  optimizer._populationSize = 4
  optimizer._convergenceCriteria = {'rank1Ratio': 0.6}
  converged = False
  for _ in range(3):
    _setPopulation(optimizer,
                   ranks=[1, 1, 1, 1],
                   objectiveRows=[[1.0, 0.8, 0.6, 0.4], [0.4, 0.6, 0.8, 1.0]])
    converged = optimizer._checkConvRank1Ratio(0)
  assert converged


def test_convergence_max_spread():
  """
    maxSpread: converges when the rank-1 front's maximum spread changes little between generations.
    Requires _optPointHistory[traj] with at least two entries; element [-2] is a (optDict, info)
    tuple whose optDict carries 'rank' and one key per objective variable.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  optimizer._convergenceCriteria = {'maxSpread': 10.0}
  previousOpt = {'rank': np.array([1, 1, 1]),
                 'obj1': np.array([1.0, 0.6, 0.2]),
                 'obj2': np.array([0.2, 0.6, 1.0])}
  optimizer._optPointHistory = {0: [(previousOpt, None), (previousOpt, None)]}
  _setPopulation(optimizer,
                 ranks=[1, 1, 1],
                 objectiveRows=[[1.0, 0.6, 0.2], [0.2, 0.6, 1.0]])
  assert optimizer._checkConvMaxSpread(0)


def test_convergence_hypervolume():
  """
    hypervolume: converges when the rank-1 front's hypervolume (vs a common reference) changes little
    between generations. The first call records the front and returns False, so it is called twice
    with the same front to drive the relative change to zero.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  optimizer._convergenceCriteria = {'hypervolume': 0.05}
  front = ([1.0, 0.6, 0.2], [0.2, 0.6, 1.0])
  converged = False
  for _ in range(2):
    _setPopulation(optimizer, ranks=[1, 1, 1], objectiveRows=[front[0], front[1]])
    converged = optimizer._checkConvHypervolume(0)
  assert converged


def test_no_convergence_spread():
  """
    spread: an uneven front with a tight threshold must not converge.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  _setPopulation(optimizer,
                 ranks=[1, 1, 1, 1],
                 objectiveRows=[[1.0, 0.95, 0.9, 0.1], [0.1, 0.15, 0.2, 1.0]])
  optimizer._convergenceCriteria = {'spread': 1e-6}
  assert not optimizer._checkConvSpread(0)


def test_no_convergence_rank1_ratio():
  """
    rank1Ratio: a population that is only 25% rank-1 never reaches a 0.6 ratio.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  optimizer._populationSize = 4
  optimizer._convergenceCriteria = {'rank1Ratio': 0.6}
  converged = True
  for _ in range(3):
    _setPopulation(optimizer,
                   ranks=[1, 2, 2, 2],
                   objectiveRows=[[0.4, 0.8, 0.9, 1.0], [0.4, 0.8, 0.9, 1.0]])
    converged = optimizer._checkConvRank1Ratio(0)
  assert not converged


def test_no_convergence_max_spread():
  """
    maxSpread: a front whose extent changes a lot between generations must not converge.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  optimizer._convergenceCriteria = {'maxSpread': 1e-3}
  previousOpt = {'rank': np.array([1, 1, 1]),
                 'obj1': np.array([0.6, 0.5, 0.4]),
                 'obj2': np.array([0.4, 0.5, 0.6])}
  optimizer._optPointHistory = {0: [(previousOpt, None), (previousOpt, None)]}
  _setPopulation(optimizer,
                 ranks=[1, 1, 1],
                 objectiveRows=[[1.0, 0.6, 0.2], [0.2, 0.6, 1.0]])
  assert not optimizer._checkConvMaxSpread(0)


def test_no_convergence_hypervolume():
  """
    hypervolume: a front that improves (shifts toward the origin) between generations must not converge.
  """
  optimizer = _makeOptimizer(['obj1', 'obj2'])
  optimizer._convergenceCriteria = {'hypervolume': 0.05}
  converged = True
  for shift in (0.0, 0.3):
    _setPopulation(optimizer, ranks=[1, 1, 1],
                   objectiveRows=[[1.0 - shift, 0.6 - shift, 0.2 - shift], [0.2 - shift, 0.6 - shift, 1.0 - shift]])
    converged = optimizer._checkConvHypervolume(0)
  assert not converged


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
