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
  Unit tests for tournament parent selection on a very small population. A population of 4 gives
  nParents = ceil(1/2 + sqrt(1 + 4*4)/2) = 3 (GeneticAlgorithm.handleInput), which errored in an
  earlier version. Tournaments draw kSelection candidates without replacement, so any kSelection up
  to the population size must work, and a larger one must be rejected with a clear error.
"""
import numpy as np
import pytest
import xarray as xr

from ravenframework.Optimizers.parentSelectors.parentSelectors import tournamentSelection
from ravenframework.utils import randomUtils

POP_SIZE = 4
N_PARENTS = int(np.ceil(1/2 + np.sqrt(1 + 4*POP_SIZE)/2))  # same formula as GeneticAlgorithm.handleInput
VARIABLES = ['x1', 'x2']


def _population():
  """
    Four chromosomes with distinct genes.
    @ In, None
    @ Out, population, xr.DataArray, (4, 2) population
  """
  return xr.DataArray(np.arange(8.0).reshape(POP_SIZE, 2), dims=['chromosome', 'Gene'],
                      coords={'chromosome': np.arange(POP_SIZE), 'Gene': VARIABLES})


def _fitness():
  """
    Single-objective fitness for the four chromosomes.
    @ In, None
    @ Out, fitness, xr.Dataset, fitness per chromosome
  """
  return xr.Dataset({'ans': xr.DataArray(np.array([1.0, 4.0, 2.0, 3.0]), dims=['chromosome'],
                                         coords={'chromosome': np.arange(POP_SIZE)})})


def test_population_four_gives_three_parents():
  """The reported configuration: population 4 -> nParents 3."""
  assert N_PARENTS == 3


@pytest.mark.parametrize('kSelection', [2, 3, 4])
def test_single_objective_tournament_population_four(kSelection):
  """Single-objective tournaments work for every kSelection up to the population size."""
  randomUtils.randomSeed(1)
  parents = tournamentSelection(_population(), variables=VARIABLES, fitness=_fitness(), nParents=N_PARENTS,
                                objVar=['ans'], kSelection=kSelection, isMultiObjective=False)
  assert parents.shape == (N_PARENTS, len(VARIABLES))
  # every selected parent is a member of the population
  rows = {tuple(row) for row in _population().values}
  assert all(tuple(row) in rows for row in parents.values)


@pytest.mark.parametrize('kSelection', [2, 3, 4])
def test_multi_objective_tournament_population_four(kSelection):
  """Multi-objective (rank, crowding distance) tournaments work for every kSelection up to the population size."""
  randomUtils.randomSeed(1)
  rank = xr.DataArray(np.array([1, 2, 1, 2]), dims=['rank'], coords={'rank': np.arange(POP_SIZE)})
  crowding = xr.DataArray(np.array([np.inf, 0.5, 1.0, np.inf]), dims=['CrowdingDistance'],
                          coords={'CrowdingDistance': np.arange(POP_SIZE)})
  parents = tournamentSelection(_population(), variables=VARIABLES, fitness=_fitness(), nParents=N_PARENTS,
                                objVar=['ans'], kSelection=kSelection, isMultiObjective=True,
                                rank=rank, crowdDistance=crowding)
  assert parents.shape == (N_PARENTS, len(VARIABLES))
  if kSelection == POP_SIZE:
    # the whole population is in every tournament: the best (rank 1, largest CD) always wins
    np.testing.assert_allclose(parents.values, np.tile(_population().values[0], (N_PARENTS, 1)))


def test_tournament_larger_than_population_raises():
  """A tournament larger than the population is rejected with a clear error, not an IndexError."""
  with pytest.raises(ValueError, match='Tournament size cannot be greater than population size'):
    tournamentSelection(_population(), variables=VARIABLES, fitness=_fitness(), nParents=N_PARENTS,
                        objVar=['ans'], kSelection=POP_SIZE + 1, isMultiObjective=False)


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
