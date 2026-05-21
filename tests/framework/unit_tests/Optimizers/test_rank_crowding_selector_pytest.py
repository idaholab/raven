import numpy as np

from ravenframework.Optimizers.survivorSelectors.survivorSelectors import (
  rankNcrowdingBased,
)
from ravenframework.utils import frontUtils


def test_rank_crowding_based_selects_by_front_and_distance(nsga_combined_data):
  params = nsga_combined_data["params"]
  expected_indices = nsga_combined_data["expected_indices"]
  population, ranks, ages, crowding, objectives, fitness_ds, constraints = (
    rankNcrowdingBased(**params)
  )

  expected_population = params["combinedPop"][expected_indices]
  assert population.shape == expected_population.shape
  assert np.allclose(population.data, expected_population)

  expected_ranks = np.array([params["combinedRanks"][i] for i in expected_indices])
  assert np.allclose(ranks.data, expected_ranks)

  expected_ages = [params["age"][i] for i in expected_indices]
  assert ages == expected_ages

  expected_cd = params["combinedCD"][expected_indices]
  assert np.allclose(crowding.data, expected_cd)

  for obj_idx, values in enumerate(objectives):
    expected = [
      params["combinedMinObjVals"][obj_idx][i]
      for i in expected_indices
    ]
    assert np.allclose(values, expected)

  for obj_name in fitness_ds.data_vars:
    idx = int(obj_name.replace("obj", ""))
    expected = [
      params["combinedFitVals"][i][idx]
      for i in expected_indices
    ]
    assert np.allclose(fitness_ds[obj_name].data, expected)

  expected_constraints = params["combinedConstraintVals"][expected_indices]
  assert constraints.shape == expected_constraints.shape
  assert np.allclose(constraints.data, expected_constraints)


def test_rank_fronts_use_original_objectives_with_directions_not_transformed_fitness():
  external_obj_vals = np.array(
    [
      [1.0, 10.0],
      [2.0, 9.0],
      [3.0, 11.0],
    ],
    dtype=float,
  )
  min_mask = np.array([True, False], dtype=bool)
  misleading_fit_vals = np.array(
    [
      [0.0, 0.0],
      [100.0, 100.0],
      [50.0, 50.0],
    ],
    dtype=float,
  )

  objective_ranks = frontUtils.rankNonDominatedFrontiers(external_obj_vals, minMask=min_mask)
  fitness_ranks = frontUtils.rankNonDominatedFrontiers(misleading_fit_vals, isFitness=True)

  assert objective_ranks == [1, 2, 1]
  assert fitness_ranks != objective_ranks


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
