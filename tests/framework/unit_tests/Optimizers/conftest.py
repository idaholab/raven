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
  Shared pytest fixtures for the GA optimizer unit tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Ensure the repository root is discoverable when pytest is executed outside the managed environment.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

collect_ignore_glob = [
  "test1pointCrossover.py",
  "test2pointsCrossover.py",
  "test2pointsCrossoverSmallArray.py",
  "testAgeBased.py",
  "testBitFlipMutator.py",
  "testFiniteDifference.py",
  "testFitnessBased.py",
  "testInversionMutator.py",
  "testRandomMutator.py",
  "testRankSelection.py",
  "testRouletteWheel.py",
  "testScrambleMutator.py",
  "testSwapMutator.py",
  "testTournamentSelection.py",
  "testUniformCrossover.py",
  "tests",
]


@pytest.fixture
def fitness_inputs():
  """
    Three-individual, single-objective realization with two constraints.
    @ In, None
    @ Out, inputs, dict, {"rlz": xr.Dataset, "constraints": xr.DataArray, "obj_var": list}
  """
  rlz = xr.Dataset(
    {
      "obj": xr.DataArray(
        np.array([1.0, 2.0, 0.5]),
        dims=["chromosome"],
        coords={"chromosome": np.arange(3)},
      )
    }
  )
  constraints = xr.DataArray(
    np.array([[0.2, 0.0], [-0.3, 0.1], [-0.1, -0.4]], dtype=float),
    dims=["chromosome", "Constraint"],
    coords={"chromosome": np.arange(3), "Constraint": ["c1", "c2"]},
  )
  return {"rlz": rlz, "constraints": constraints, "obj_var": ["obj"]}


@pytest.fixture
def ga_dataset():
  """
    Two-chromosome, two-gene dataset.
    @ In, None
    @ Out, data, dict, {"dataset": xr.Dataset, "variables": list}
  """
  dataset = xr.Dataset(
    {
      "x1": xr.DataArray(
        np.array([1.0, 2.0]),
        dims=["chromosome"],
        coords={"chromosome": np.arange(2)},
      ),
      "x2": xr.DataArray(
        np.array([3.0, 4.0]),
        dims=["chromosome"],
        coords={"chromosome": np.arange(2)},
      ),
    }
  )
  return {"dataset": dataset, "variables": ["x1", "x2"]}


@pytest.fixture
def nsga_combined_data():
  """
    Combined parent+offspring pool R(t) for rank-and-crowding selection, with the expected survivors.
    @ In, None
    @ Out, data, dict, {"params": dict, "expected_indices": list}
  """
  combined_inputs = np.array(
    [
      [0.1, 1.0],
      [0.2, 0.9],
      [0.4, 0.7],
      [0.6, 0.5],
      [0.8, 0.3],
    ],
    dtype=float,
  )
  combined_ranks = [0, 0, 1, 1, 2]
  combined_cd = np.array([0.1, 0.3, 0.2, 0.5, 0.4], dtype=float)
  combined_objectives = [
    [1.0, 1.2, 1.1, 1.5, 2.0],
    [2.0, 1.8, 1.7, 1.2, 1.0],
  ]
  combined_fitness = [
    [0.9, 0.8],
    [0.85, 0.82],
    [0.8, 0.78],
    [0.7, 0.75],
    [0.6, 0.7],
  ]
  combined_constraints = np.array(
    [[0.1], [0.0], [-0.2], [-0.1], [0.3]],
    dtype=float,
  )
  age = [1, 2, 1, 0, 3]
  params = {
    "combinedPop": combined_inputs,
    "combinedRanks": combined_ranks,
    "combinedCD": combined_cd,
    "combinedMinObjVals": combined_objectives,
    "combinedFitVals": combined_fitness,
    "combinedConstraintVals": combined_constraints,
    "age": age,
    "popSize": 3,
    "variables": ["x1", "x2"],
  }
  expected_indices = [0, 1, 3]
  return {"params": params, "expected_indices": expected_indices}


@pytest.fixture
def repair_builder():
  """
    Factory that builds a chromosome array and stub withoutReplacement distributions for repair tests.
    @ In, None
    @ Out, builder, callable, builder(values) -> (array, distInfo, variables)
  """
  variables = ["x1", "x2", "x3"]
  candidate_pool = {
    "x1": [0.0, 1.0, 2.0, 3.0],
    "x2": [0.0, 1.0, 2.0, 3.0],
    "x3": [0.0, 1.0, 2.0, 3.0],
  }

  class DummyDistribution:
    """
      withoutReplacement distribution stand-in that draws the first allowed value not excluded.
    """
    def __init__(self, allowed):
      """
        Constructor.
        @ In, allowed, list, candidate values
        @ Out, None
      """
      self.strategy = "withoutReplacement"
      self._allowed = list(allowed)

    def selectedRvs(self, excluded):
      """
        Draw the first allowed value not in excluded.
        @ In, excluded, list, values already used
        @ Out, value, float, the drawn value
      """
      for value in self._allowed:
        if value not in excluded:
          return value
      raise RuntimeError("No available values for repair.")

  def builder(values):
    """
      Build the repair inputs for the given chromosome values.
      @ In, values, list, chromosome rows
      @ Out, (array, distInfo, variables), tuple, inputs for the repair operator
    """
    array = xr.DataArray(
      np.array(values, dtype=float),
      dims=["chromosome", "Gene"],
      coords={"chromosome": np.arange(len(values)), "Gene": variables},
    )
    dist_info = {
      name: DummyDistribution(candidates)
      for name, candidates in candidate_pool.items()
    }
    return array, dist_info, variables

  return builder
