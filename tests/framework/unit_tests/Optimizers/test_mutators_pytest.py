import numpy as np
import pytest
import xarray as xr

from ravenframework.Optimizers.mutators import mutators
from ravenframework.utils import randomUtils


class IdentityDistribution:
  def cdf(self, value):
    return float(value)

  def ppf(self, quantile):
    return float(quantile)


@pytest.fixture
def simple_offsprings():
  data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
  offsprings = xr.DataArray(
    data,
    dims=['chromosome', 'Gene'],
    coords={'chromosome': np.arange(2), 'Gene': ['x1', 'x2', 'x3']},
  )
  dist = {gene: IdentityDistribution() for gene in offsprings.coords['Gene'].values}
  return offsprings, dist


def test_swap_mutator_swaps_selected_locations(simple_offsprings, monkeypatch):
  offsprings, dist = simple_offsprings

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return 0.0

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  mutated = mutators.swapMutator(
    offsprings.copy(deep=True),
    dist,
    locs=[0, 2],
    mutationProb=0.5,
    variables=list(dist.keys()),
  )
  expected = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]])
  np.testing.assert_allclose(mutated.values, expected)


def test_scramble_mutator_uses_random_permutation(simple_offsprings, monkeypatch):
  offsprings, dist = simple_offsprings

  random_calls = iter([0.0, 1.0, 0.0, 1.0])

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    try:
      return next(random_calls)
    except StopIteration:
      return 1.0

  def fake_permutation(values, engine=None):
    return list(reversed(values))

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  monkeypatch.setattr(randomUtils, 'randomPermutation', fake_permutation)
  mutated = mutators.scrambleMutator(
    offsprings.copy(deep=True),
    dist,
    locs=[0, 2],
    mutationProb=0.5,
    variables=list(dist.keys()),
  )
  expected = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]])
  np.testing.assert_allclose(mutated.values, expected)


def test_bit_flip_mutator_flips_gene_using_cdf(simple_offsprings, monkeypatch):
  offsprings, dist = simple_offsprings
  calls = iter([1, 1])

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return 0.0

  def fake_integers(low, high, caller=None, engine=None):
    return next(calls)

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  monkeypatch.setattr(randomUtils, 'randomIntegers', fake_integers)
  mutated = mutators.bitFlipMutator(
    offsprings.copy(deep=True),
    dist,
    locs=None,
    mutationProb=1.0,
    variables=list(dist.keys()),
  )
  expected = np.array([[0.9, 0.2, 0.3], [0.6, 0.5, 0.6]])
  np.testing.assert_allclose(mutated.values, expected)


def test_bit_flip_mutator_rejects_locs_argument(simple_offsprings):
  offsprings, dist = simple_offsprings
  with pytest.raises(ValueError):
    mutators.bitFlipMutator(
      offsprings.copy(deep=True),
      dist,
      locs=[0, 1],
      mutationProb=1.0,
      variables=list(dist.keys()),
    )

def test_random_mutator_draws_new_values(simple_offsprings, monkeypatch):
  offsprings, dist = simple_offsprings
  rand_values = iter([0.0, 0.15, 0.0, 0.85])
  calls = iter([1, 1])

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    try:
      return next(rand_values)
    except StopIteration:
      return 1.0

  def fake_integers(low, high, caller=None, engine=None):
    return next(calls)

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  monkeypatch.setattr(randomUtils, 'randomIntegers', fake_integers)
  mutated = mutators.randomMutator(
    offsprings.copy(deep=True),
    dist,
    locs=None,
    mutationProb=1.0,
    variables=list(dist.keys()),
  )
  expected = np.array([[0.15, 0.2, 0.3], [0.85, 0.5, 0.6]])
  np.testing.assert_allclose(mutated.values, expected)


def test_random_mutator_rejects_locs_argument(simple_offsprings):
  offsprings, dist = simple_offsprings
  with pytest.raises(ValueError):
    mutators.randomMutator(
      offsprings.copy(deep=True),
      dist,
      locs=[0, 1],
      mutationProb=1.0,
      variables=list(dist.keys()),
    )


def test_inversion_mutator_reverses_segment(simple_offsprings, monkeypatch):
  offsprings, dist = simple_offsprings

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return 0.0

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  mutated = mutators.inversionMutator(
    offsprings.copy(deep=True),
    dist,
    locs=[0, 2],
    mutationProb=1.0,
    variables=list(dist.keys()),
  )
  expected = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]])
  np.testing.assert_allclose(mutated.values, expected)


def test_locations_generator_orders_random_choices(simple_offsprings, monkeypatch):
  offsprings, _ = simple_offsprings

  def fake_choice(array, size=1, replace=True, engine=None):
    return [2, 0]

  monkeypatch.setattr(randomUtils, 'randomChoice', fake_choice)
  loc1, loc2 = mutators.locationsGenerator(offsprings, None)
  assert (loc1, loc2) == (0, 2)


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
