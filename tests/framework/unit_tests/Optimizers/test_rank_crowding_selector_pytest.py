import numpy as np

from ravenframework.Optimizers.survivorSelectors.survivorSelectors import (
  rankNcrowdingBased,
)


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


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
