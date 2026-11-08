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
  Selection probabilities of rouletteWheel and rankSelection (issues #2618 and #2619): the best
  individual always gets the largest share, for any sign of the fitness; an all-equal population is
  selected uniformly; rankSelection reads the real objective name and depends only on fitness order.

  @author: Mohammad Abdo (@Jimmy-INL)
"""
import collections

import numpy as np
import pytest
import xarray as xr

from ravenframework.Optimizers.parentSelectors import parentSelectors
from ravenframework.utils import randomUtils

VARIABLES = ['x1', 'x2']


def _population(n):
  """
    Population whose genes identify the individual (gene x1 of individual k is k).
    @ In, n, int, population size
    @ Out, population, xr.DataArray, (n, 2) population
  """
  data = np.array([[k, 0] for k in range(n)], dtype=float)
  return xr.DataArray(data, dims=['chromosome', 'Gene'], coords={'chromosome': np.arange(n), 'Gene': VARIABLES})


def _fitness(values, name='obj'):
  """
    @ In, values, list(float), fitness per individual
    @ In, name, str, optional, objective name
    @ Out, fitness, xr.Dataset, fitness dataset as the GA builds it
  """
  return xr.DataArray(np.asarray(values, dtype=float), dims=['chromosome'],
                      coords={'chromosome': np.arange(len(values))}).to_dataset(name=name)


def _firstPickFrequencies(selector, values, draws=4000, name='obj'):
  """
    Empirical probability of each individual being drawn as the first parent.
    @ In, selector, callable, parent selector
    @ In, values, list(float), fitness per individual
    @ In, draws, int, optional, number of draws
    @ In, name, str, optional, objective name
    @ Out, freq, np.array, frequency of each individual
  """
  randomUtils.randomSeed(2026)
  pop = _population(len(values))
  fitness = _fitness(values, name)
  counts = collections.Counter()
  for _ in range(draws):
    parents = selector(pop, fitness=fitness, nParents=1, variables=VARIABLES)
    counts[int(parents.values[0, 0])] += 1
  return np.array([counts[k] for k in range(len(values))]) / draws


# ---- rouletteWheel probabilities (#2619) --------------------------------------------------------
@pytest.mark.parametrize('values', [[1.0, 2.0, 4.0], [-4.0, -2.0, -1.0], [-1.0, 0.5, 3.0]])
def testRouletteBestGetsLargestShare(values):
  """Positive, all-negative and mixed fitness: the larger the fitness, the larger the share."""
  prob = parentSelectors._rouletteProbabilities(values)
  assert np.isclose(prob.sum(), 1.0)
  assert np.all(prob >= 0)
  assert list(np.argsort(prob)) == list(np.argsort(values))


def testRouletteAllNegativeNotInverted():
  """All-negative fitness used to give the worst individual the largest share."""
  prob = parentSelectors._rouletteProbabilities([-4.0, -2.0, -1.0])
  # shifted by the minimum: weights 0, 2, 3
  np.testing.assert_allclose(prob, [0.0, 0.4, 0.6])


def testRoulettePositiveIsProportional():
  """Positive fitness keeps the classic fitness-proportionate shares (no change from before)."""
  np.testing.assert_allclose(parentSelectors._rouletteProbabilities([1.0, 3.0]), [0.25, 0.75])


@pytest.mark.parametrize('values', [[0.0, 0.0, 0.0], [-2.0, -2.0, -2.0], [5.0, 5.0]])
def testRouletteAllEqualIsUniform(values):
  """All-zero (used to give NaN) or all-equal fitness: every individual is equally likely."""
  prob = parentSelectors._rouletteProbabilities(values)
  assert np.all(np.isfinite(prob))
  np.testing.assert_allclose(prob, np.full(len(values), 1.0 / len(values)))


def testRouletteWheelSamplesTheProbabilities():
  """The wheel draws parents with the computed probabilities (all-negative fitness)."""
  freq = _firstPickFrequencies(parentSelectors.rouletteWheel, [-4.0, -2.0, -1.0])
  np.testing.assert_allclose(freq, [0.0, 0.4, 0.6], atol=0.03)


def testRouletteWheelAllZeroDoesNotFail():
  """An all-zero population used to produce NaN probabilities; now it is drawn uniformly."""
  freq = _firstPickFrequencies(parentSelectors.rouletteWheel, [0.0, 0.0, 0.0, 0.0])
  np.testing.assert_allclose(freq, np.full(4, 0.25), atol=0.03)


# ---- rankSelection (#2618) ----------------------------------------------------------------------
def testRankSelectionReadsTheRealObjectiveName():
  """Any objective name works (it used to require the literal 'test_RankSelection')."""
  randomUtils.randomSeed(1)
  parents = parentSelectors.rankSelection(_population(4), fitness=_fitness([7.2, 1.3, 9.5, 2.0], name='MaxEFPD'),
                                          nParents=2, variables=VARIABLES)
  assert parents.shape == (2, 2)


def testRankSelectionBestIsMostLikely():
  """Linear ranking: weights 1..n from worst to best, so the best has probability n / sum(1..n)."""
  freq = _firstPickFrequencies(parentSelectors.rankSelection, [7.2, 1.3, 9.5, 2.0])
  # ranks: 9.5 -> 4, 7.2 -> 3, 2.0 -> 2, 1.3 -> 1; probabilities rank / 10
  np.testing.assert_allclose(freq, [0.3, 0.1, 0.4, 0.2], atol=0.03)


def testRankSelectionDependsOnlyOnOrder():
  """Rescaling or shifting the fitness (keeping its order) does not change the probabilities."""
  base = _firstPickFrequencies(parentSelectors.rankSelection, [1.0, 2.0, 3.0])
  scaled = _firstPickFrequencies(parentSelectors.rankSelection, [-1000.0, 0.0, 1e6])
  np.testing.assert_allclose(base, scaled)


def testRankSelectionTiesShareTheirRank():
  """Equal fitness values get the average of their ranks, so they are equally likely."""
  freq = _firstPickFrequencies(parentSelectors.rankSelection, [5.0, 5.0, 1.0])
  # ranks 1 (for 1.0) and 2.5, 2.5 (for the tied 5.0s): probabilities 0.4167, 0.4167, 0.1667
  np.testing.assert_allclose(freq, [2.5 / 6, 2.5 / 6, 1.0 / 6], atol=0.03)


if __name__ == '__main__':
  from pytest_runner import run_module
  run_module(__file__)
