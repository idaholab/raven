import numpy as np
import xarray as xr

from ravenframework.Optimizers.survivorSelectors import survivorSelectors


def _dataset_from(values, variable):
  data = xr.Dataset()
  data[variable] = xr.DataArray(
    np.array(values, dtype=float),
    dims=['chromosome'],
    coords={'chromosome': np.arange(len(values))},
  )
  return data


def test_age_based_replaces_oldest_individuals():
  variables = ['x1', 'x2']
  population = xr.DataArray(
    np.array([[1, 1], [2, 2], [3, 3]], dtype=float),
    dims=['chromosome', 'Gene'],
    coords={'chromosome': np.arange(3), 'Gene': variables},
  )
  age = [0, 2, 1]
  fitness = _dataset_from([5.0, 3.0, 4.0], 'obj')
  offspring = xr.Dataset(
    {
      'x1': xr.DataArray([9.0], dims=['chromosome'], coords={'chromosome': [0]}),
      'x2': xr.DataArray([8.0], dims=['chromosome'], coords={'chromosome': [0]}),
    }
  )
  offspring_fitness = _dataset_from([6.0], 'obj')
  popMinObjVals = [5.0, 3.0, 4.0]
  new_pop, new_fit, new_age, popMinObjVals_vals = survivorSelectors.ageBased(
    offspring,
    age=age,
    variables=variables,
    popFitVals=fitness,
    offspringFitVals=offspring_fitness,
    population=population,
    objVar='obj',
    popMinObjVals=popMinObjVals,
  )
  assert np.allclose(new_pop.values[-1], np.array([9.0, 8.0]))
  assert new_age[-1] == 0
  assert np.allclose(new_fit['obj'].values[-1], 6.0)
  assert popMinObjVals_vals[-1] == popMinObjVals[-1]


def test_fitness_based_keeps_best_individuals():
  variables = ['x1', 'x2']
  population = xr.DataArray(
    np.array([[1, 1], [5, 5], [3, 3]], dtype=float),
    dims=['chromosome', 'Gene'],
    coords={'chromosome': np.arange(3), 'Gene': variables},
  )
  age = [1, 4, 2]
  fitness = _dataset_from([1.0, 6.0, 2.0], 'obj')
  offspring = xr.Dataset(
    {
      'x1': xr.DataArray([2.0, 8.0], dims=['chromosome'], coords={'chromosome': [0, 1]}),
      'x2': xr.DataArray([2.0, 8.0], dims=['chromosome'], coords={'chromosome': [0, 1]}),
      'obj': xr.DataArray([2.5, 7.5], dims=['chromosome'], coords={'chromosome': [0, 1]}),
    }
  )
  offspring_fitness = _dataset_from([2.5, 7.5], 'obj')
  popMinObjVals = [1.0, 6.0, 2.0]
  new_pop, new_fit, new_age, new_obj = survivorSelectors.fitnessBased(
    offspring,
    age=age,
    variables=variables,
    popFitVals=fitness,
    offspringFitVals=offspring_fitness,
    population=population,
    objVar='obj',
    popMinObjVals=popMinObjVals,
  )
  expected_pop = np.array([[8.0, 8.0], [5.0, 5.0], [2.0, 2.0]])
  np.testing.assert_allclose(new_pop.values, expected_pop)
  assert new_age == [0, 5, 0]
  np.testing.assert_allclose(new_fit['obj'].values, np.array([7.5, 6.0, 2.5]))
  assert new_obj == [7.5, 6.0, 2.5]


def test_rank_and_crowding_selector_respects_fronts(nsga_combined_data):
  result = survivorSelectors.rankNcrowdingBased(**nsga_combined_data['params'])
  selected = result[0].coords['chromosome'].values
  assert len(selected) == nsga_combined_data['params']['popSize']
  assert list(result[0].values[:, 0]) == [
    nsga_combined_data['params']['combinedPop'][i][0] for i in nsga_combined_data['expected_indices']
  ]


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
