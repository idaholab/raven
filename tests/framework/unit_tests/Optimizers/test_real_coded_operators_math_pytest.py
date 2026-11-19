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
  Math checks for the real-coded GA operators. The random draws are fixed, and each result is
  compared with the published formula evaluated independently in the test:
    - SBX crossover, bounded form (Deb & Agrawal 1995; Deb & Goyal NSGA-II reference code)
    - polynomial mutation (Deb & Goyal 1996)
  plus the SBX mean-preservation property c1 + c2 = p1 + p2 for interior parents.
"""
import numpy as np
import pytest
import xarray as xr

from ravenframework.Optimizers.crossOverOperators import crossovers
from ravenframework.Optimizers.mutators import mutators
from ravenframework.utils import randomUtils


class _Bounded:
  """
    Minimal distribution exposing finite bounds (what the operators read).
  """
  def __init__(self, lowerBound, upperBound):
    """
      Constructor.
      @ In, lowerBound, float, lower bound
      @ In, upperBound, float, upper bound
      @ Out, None
    """
    self.lowerBound = lowerBound
    self.upperBound = upperBound


def _fixedDraws(monkeypatch, draws):
  """
    Make randomUtils.random return the given values in order.
    @ In, monkeypatch, pytest fixture, patcher
    @ In, draws, list(float), values to return
    @ Out, None
  """
  sequence = iter(draws)
  monkeypatch.setattr(randomUtils, 'random', lambda *args, **kwargs: next(sequence))


def _referenceSbx(y1, y2, xl, xu, rand, eta):
  """
    Bounded SBX children for one gene, straight from Deb & Agrawal (1995).
    @ In, y1, float, smaller parent value
    @ In, y2, float, larger parent value
    @ In, xl, float, lower bound
    @ In, xu, float, upper bound
    @ In, rand, float, uniform draw
    @ In, eta, float, distribution index
    @ Out, (c1, c2), tuple(float, float), children (lower, upper)
  """
  def betaq(alpha):
    if rand <= 1.0 / alpha:
      return (rand * alpha) ** (1.0 / (eta + 1.0))
    return (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
  alphaLow = 2.0 - (1.0 + 2.0 * (y1 - xl) / (y2 - y1)) ** (-(eta + 1.0))
  alphaHigh = 2.0 - (1.0 + 2.0 * (xu - y2) / (y2 - y1)) ** (-(eta + 1.0))
  c1 = 0.5 * ((y1 + y2) - betaq(alphaLow) * (y2 - y1))
  c2 = 0.5 * ((y1 + y2) + betaq(alphaHigh) * (y2 - y1))
  return min(max(c1, xl), xu), min(max(c2, xl), xu)


def _parents(rows):
  """
    Parents DataArray for a single gene x1.
    @ In, rows, list(float), parent values
    @ Out, parents, xr.DataArray, (nParents, 1)
  """
  return xr.DataArray(np.array(rows, dtype=float).reshape(-1, 1), dims=['chromosome', 'Gene'],
                      coords={'chromosome': np.arange(len(rows)), 'Gene': ['x1']})


@pytest.mark.parametrize('rand', [0.1, 0.5, 0.9])
def test_sbx_matches_reference_formula(monkeypatch, rand):
  """SBX children equal the bounded Deb-Agrawal formula for the same draws."""
  eta, xl, xu, p1, p2 = 15.0, 0.0, 10.0, 3.0, 4.0
  # draws: crossover gate (always cross), the SBX rand, then the child-slot coin (0.9 -> keep order)
  _fixedDraws(monkeypatch, [0.0, rand, 0.9])
  children = crossovers.sbxCrossover(_parents([p1, p2]), crossoverProb=1.0, eta=eta,
                                     distDict={'x1': _Bounded(xl, xu)})
  expected = _referenceSbx(p1, p2, xl, xu, rand, eta)
  np.testing.assert_allclose(children.values[:, 0], expected, rtol=0, atol=1e-12)


def test_sbx_does_not_just_copy_parents(monkeypatch):
  """A crossover that copies its parents must fail this test."""
  _fixedDraws(monkeypatch, [0.0, 0.9, 0.9])
  children = crossovers.sbxCrossover(_parents([3.0, 4.0]), crossoverProb=1.0,
                                     distDict={'x1': _Bounded(0.0, 10.0)})
  assert not np.allclose(sorted(children.values[:, 0]), [3.0, 4.0])


@pytest.mark.parametrize('rand', [0.05, 0.3, 0.5, 0.7, 0.95])
def test_sbx_preserves_mean_for_interior_parents(monkeypatch, rand):
  """With parents far from both bounds, the two SBX children are symmetric about the parents' mean."""
  _fixedDraws(monkeypatch, [0.0, rand, 0.9])
  p1, p2 = 499.0, 501.0
  children = crossovers.sbxCrossover(_parents([p1, p2]), crossoverProb=1.0,
                                     distDict={'x1': _Bounded(0.0, 1000.0)})
  np.testing.assert_allclose(children.values[:, 0].sum(), p1 + p2, rtol=0, atol=1e-9)


@pytest.mark.parametrize('rand', [0.1, 0.4, 0.6, 0.9])
def test_polynomial_mutation_matches_reference_formula(monkeypatch, rand):
  """Polynomial mutation equals the Deb-Goyal formula for the same draw."""
  eta, low, high, x = 20.0, 0.0, 10.0, 3.0
  # draws: mutation gate (mutate), then the polynomial rand
  _fixedDraws(monkeypatch, [0.0, rand])
  offspring = _parents([x])
  mutated = mutators.polynomialMutator(offspring, {'x1': _Bounded(low, high)}, mutationProb=1.0, eta=eta)
  spread = high - low
  delta1, delta2 = (x - low) / spread, (high - x) / spread
  mutPow = 1.0 / (eta + 1.0)
  if rand < 0.5:
    deltaq = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta1) ** (eta + 1.0)) ** mutPow - 1.0
  else:
    deltaq = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - delta2) ** (eta + 1.0)) ** mutPow
  expected = min(max(x + deltaq * spread, low), high)
  assert mutated.values[0, 0] == pytest.approx(expected, abs=1e-12)
  assert mutated.values[0, 0] != pytest.approx(x)


def test_polynomial_mutation_direction_follows_draw(monkeypatch):
  """rand < 0.5 moves the gene down, rand > 0.5 moves it up."""
  bounds = {'x1': _Bounded(0.0, 10.0)}
  _fixedDraws(monkeypatch, [0.0, 0.2])
  down = mutators.polynomialMutator(_parents([5.0]), bounds, mutationProb=1.0).values[0, 0]
  _fixedDraws(monkeypatch, [0.0, 0.8])
  up = mutators.polynomialMutator(_parents([5.0]), bounds, mutationProb=1.0).values[0, 0]
  assert down < 5.0 < up


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
