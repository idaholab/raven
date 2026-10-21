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
  Unit tests for the GA survivor selectors (ageBased, fitnessBased, rankNcrowdingBased), including alignment of objectives and constraints with the selected population.
"""
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
      'obj': xr.DataArray([0.5], dims=['chromosome'], coords={'chromosome': [0]}),
    }
  )
  offspring_fitness = _dataset_from([6.0], 'obj')
  # objective/constraint values identify each individual: parent k has objective 10*k+1 and constraint -k
  popMinObjVals = [11.0, 21.0, 31.0]
  popConstraints = xr.DataArray(np.array([[-1.0], [-2.0], [-3.0]]), dims=['chromosome', 'Constraint'],
                                coords={'chromosome': np.arange(3), 'Constraint': ['c1']})
  offspringConstraints = xr.DataArray(np.array([[-9.0]]), dims=['chromosome', 'Constraint'],
                                      coords={'chromosome': [0], 'Constraint': ['c1']})
  new_pop, new_fit, new_age, new_obj, new_cons = survivorSelectors.ageBased(
    offspring,
    age=age,
    variables=variables,
    popFitVals=fitness,
    offspringFitVals=offspring_fitness,
    population=population,
    objVar='obj',
    popMinObjVals=popMinObjVals,
    popConstraintVals=popConstraints,
    offspringConstraintVals=offspringConstraints,
  )
  # youngest-first order is parents (age 0, 1) = ([1,1], [3,3]); the oldest ([2,2], age 2) is replaced
  np.testing.assert_allclose(new_pop.values, np.array([[1.0, 1.0], [3.0, 3.0], [9.0, 8.0]]))
  assert new_age == [1, 2, 0]
  np.testing.assert_allclose(new_fit['obj'].values, [5.0, 4.0, 6.0])
  # objectives and constraints follow their individuals, not their old positions
  np.testing.assert_allclose(new_obj, [11.0, 31.0, 0.5])
  np.testing.assert_allclose(new_cons.values[:, 0], [-1.0, -3.0, -9.0])


def test_age_based_unconstrained_returns_no_constraints():
  variables = ['x1']
  population = xr.DataArray(np.array([[1.0], [2.0]]), dims=['chromosome', 'Gene'],
                            coords={'chromosome': np.arange(2), 'Gene': variables})
  offspring = xr.Dataset({
    'x1': xr.DataArray([7.0], dims=['chromosome'], coords={'chromosome': [0]}),
    'obj': xr.DataArray([7.0], dims=['chromosome'], coords={'chromosome': [0]}),
  })
  *_, new_obj, new_cons = survivorSelectors.ageBased(
    offspring, age=[0, 1], variables=variables, popFitVals=_dataset_from([1.0, 2.0], 'obj'),
    offspringFitVals=_dataset_from([3.0], 'obj'), population=population, objVar='obj', popMinObjVals=[1.0, 2.0])
  assert new_cons is None
  np.testing.assert_allclose(new_obj, [1.0, 7.0])


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
  new_pop, new_fit, new_age, new_obj, new_cons = survivorSelectors.fitnessBased(
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
  # objectives stay paired with their genes: offspring1 (7.5), parent1 (6.0), offspring0 (2.5)
  np.testing.assert_allclose(new_obj, [7.5, 6.0, 2.5])
  assert new_cons is None
  np.testing.assert_allclose(new_fit['obj'].values, np.array([7.5, 6.0, 2.5]))


def test_fitness_based_carries_constraint_rows():
  """Constraint rows of the survivors (parents and offspring mixed) follow their individuals (I2)."""
  variables = ['x1']
  population = xr.DataArray(np.array([[1.0], [5.0]]), dims=['chromosome', 'Gene'],
                            coords={'chromosome': np.arange(2), 'Gene': variables})
  offspring = xr.Dataset({
    'x1': xr.DataArray([8.0, 0.0], dims=['chromosome'], coords={'chromosome': [0, 1]}),
    'obj': xr.DataArray([8.0, 0.0], dims=['chromosome'], coords={'chromosome': [0, 1]}),
  })
  popCons = xr.DataArray(np.array([[-1.0, 0.1], [-5.0, 0.5]]), dims=['chromosome', 'Constraint'],
                         coords={'chromosome': np.arange(2), 'Constraint': ['c1', 'c2']})
  offCons = xr.DataArray(np.array([[-8.0, 0.8], [0.0, 0.0]]), dims=['chromosome', 'Constraint'],
                         coords={'chromosome': np.arange(2), 'Constraint': ['c1', 'c2']})
  new_pop, _, _, new_obj, new_cons = survivorSelectors.fitnessBased(
    offspring, age=[0, 0], variables=variables, popFitVals=_dataset_from([1.0, 5.0], 'obj'),
    offspringFitVals=_dataset_from([8.0, 0.0], 'obj'), population=population, objVar='obj',
    popMinObjVals=[1.0, 5.0], popConstraintVals=popCons, offspringConstraintVals=offCons)
  np.testing.assert_allclose(new_pop.values[:, 0], [8.0, 5.0])
  np.testing.assert_allclose(new_obj, [8.0, 5.0])
  np.testing.assert_allclose(new_cons.values, [[-8.0, 0.8], [-5.0, 0.5]])
  assert list(new_cons.coords['Constraint'].values) == ['c1', 'c2']

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
