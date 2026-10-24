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
  Unit tests for the GA withoutReplacement repair operator.
"""
import numpy as np

from ravenframework.Optimizers.repairOperators.repair import replacementRepair


def test_replacement_repair_replaces_duplicates(repair_builder):
  offsprings, dist_info, variables = repair_builder([[1.0, 1.0, 2.0]])
  repaired = replacementRepair(
    offsprings,
    variables=variables,
    distInfo=dist_info,
  )
  unique = np.unique(repaired.sel(chromosome=0).data)
  assert unique.size == len(variables)


def test_replacement_repair_keeps_unique_entries(repair_builder):
  offsprings, dist_info, variables = repair_builder([[0.0, 1.0, 2.0]])
  repaired = replacementRepair(
    offsprings,
    variables=variables,
    distInfo=dist_info,
  )
  assert np.allclose(repaired.data, [[0.0, 1.0, 2.0]])


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
