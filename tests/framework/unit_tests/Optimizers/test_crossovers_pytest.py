import numpy as np
import pytest
import xarray as xr

from ravenframework.Optimizers.crossOverOperators import crossovers
from ravenframework.utils import randomUtils


@pytest.fixture
def simple_parents():
  data = np.array(
    [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2],
    ]
  )
  parents = xr.DataArray(
    data,
    dims=['chromosome', 'Gene'],
    coords={'chromosome': np.arange(3), 'Gene': ['x1', 'x2', 'x3', 'x4']},
  )
  return parents


def test_one_point_crossover_uses_provided_split(simple_parents, monkeypatch):
  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return 0.0

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  children = crossovers.onePointCrossover(
    simple_parents,
    crossoverProb=0.5,
    points=[2],
    variables=list(simple_parents.coords['Gene'].values),
  )
  expected = np.array(
    [
      [0.1, 0.2, 0.7, 0.8],
      [0.5, 0.6, 0.3, 0.4],
      [0.1, 0.2, 1.1, 1.2],
      [0.9, 1.0, 0.3, 0.4],
      [0.5, 0.6, 1.1, 1.2],
      [0.9, 1.0, 0.7, 0.8],
    ]
  )
  np.testing.assert_allclose(children.values, expected)


def test_uniform_crossover_method_swaps_by_probability(monkeypatch):
  parent1 = np.array([0.0, 0.1, 0.2])
  parent2 = np.array([1.0, 1.1, 1.2])
  probs = iter([0.0, 0.6, 0.4])

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return next(probs)

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  child1, child2 = crossovers.uniformCrossoverMethod(parent1, parent2, 0.5)
  assert np.allclose(child1, np.array([1.0, 0.1, 1.2]))
  assert np.allclose(child2, np.array([0.0, 1.1, 0.2]))


def test_uniform_crossover_for_parent_pairs(simple_parents, monkeypatch):
  calls = iter([0.0] * 12)

  def fake_random(dim=1, samples=1, keepMatrix=False, engine=None):
    return next(calls)

  monkeypatch.setattr(randomUtils, 'random', fake_random)
  children = crossovers.uniformCrossover(
    simple_parents,
    crossoverProb=0.5,
  )
  assert children.shape == (6, 4)
  assert np.all(children.values[:2, 0] == np.array([0.5, 0.1]))


def test_two_points_crossover_uses_sampled_points(simple_parents, monkeypatch):
  def fake_choice(array, size=1, replace=True, engine=None):
    return [1, 3]

  monkeypatch.setattr(randomUtils, 'randomChoice', fake_choice)
  children = crossovers.twoPointsCrossover(
    simple_parents,
    crossoverProb=1.0,
  )
  expected_first_pair = np.array(
    [
      [0.1, 0.6, 0.7, 0.4],
      [0.5, 0.2, 0.3, 0.8],
    ]
  )
  np.testing.assert_allclose(children.values[:2], expected_first_pair)


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
