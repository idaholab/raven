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
  Unit tests for the Genetic Algorithm 'withoutReplacement' initial-population handling:
  the fail-fast feasibility guard and the generation-0 duplicate repair. The two methods
  under test (_checkWithoutReplacementFeasible, _repairInitialValue) depend only on
  self.toBeSampled and self.distDict, so they are exercised here on a light stand-in that
  borrows the real GeneticAlgorithm implementations, avoiding a full optimizer build.
"""
import pytest

from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm


class DummyUniformDiscrete:
  """Minimal UniformDiscrete stand-in exposing the attributes the guard/repair read."""
  def __init__(self, strategy, lowerBound=None, upperBound=None, nPoints=None):
    self.strategy = strategy
    self.lowerBound = lowerBound
    self.upperBound = upperBound
    self.nPoints = nPoints

  def selectedRvs(self, discardedElems):
    """Return the first in-range integer value not in discardedElems (mirrors the real draw)."""
    if self.nPoints is not None:
      pool = [self.lowerBound + i * (self.upperBound - self.lowerBound) / (self.nPoints - 1)
              for i in range(self.nPoints)]
    else:
      pool = list(range(int(self.lowerBound), int(self.upperBound) + 1))
    for value in pool:
      if value not in discardedElems:
        return float(value)
    raise RuntimeError("value pool exhausted")


class _GAStub:
  """Carries toBeSampled/distDict and borrows the real GA methods under test."""
  _withoutReplacementPoolSize = GeneticAlgorithm._withoutReplacementPoolSize
  _checkWithoutReplacementFeasible = GeneticAlgorithm._checkWithoutReplacementFeasible
  _repairInitialValue = GeneticAlgorithm._repairInitialValue

  def __init__(self, toBeSampled, distDict):
    self.toBeSampled = toBeSampled
    self.distDict = distDict

  def raiseAnError(self, etype, msg):
    """Match the BaseClasses signature so the guard's error path is testable."""
    raise etype(msg)


def _makeStub(nVars, strategy='withoutReplacement', lowerBound=1, upperBound=6, nPoints=None):
  dist = DummyUniformDiscrete(strategy, lowerBound, upperBound, nPoints)
  variables = [f'x{i+1}' for i in range(nVars)]
  toBeSampled = {var: 'woRep_dist' for var in variables}
  distDict = {var: dist for var in variables}
  return _GAStub(toBeSampled, distDict), variables


def test_pool_size_from_range():
  """Pool size for an integer-range UniformDiscrete is (upper - lower + 1)."""
  stub, _ = _makeStub(1, lowerBound=1, upperBound=6)
  assert stub._withoutReplacementPoolSize(stub.distDict['x1']) == 6


def test_pool_size_from_npoints():
  """Pool size for an nPoints grid is nPoints."""
  stub, _ = _makeStub(1, lowerBound=0, upperBound=10, nPoints=4)
  assert stub._withoutReplacementPoolSize(stub.distDict['x1']) == 4


def test_feasible_setup_passes():
  """vars == pool is feasible; the guard must not raise."""
  stub, _ = _makeStub(6, lowerBound=1, upperBound=6)
  stub._checkWithoutReplacementFeasible()  # no exception


def test_infeasible_setup_raises():
  """More variables than distinct values is infeasible; the guard must raise IOError."""
  stub, _ = _makeStub(3, lowerBound=1, upperBound=2)  # pool {1,2} < 3 vars
  with pytest.raises(IOError) as info:
    stub._checkWithoutReplacementFeasible()
  message = str(info.value)
  assert 'withoutReplacement' in message
  assert '2 distinct value' in message
  assert '3 variables' in message


def test_withreplacement_never_blocks():
  """A withReplacement distribution is never subject to the uniqueness guard."""
  stub, _ = _makeStub(10, strategy='withReplacement', lowerBound=1, upperBound=2)
  stub._checkWithoutReplacementFeasible()  # no exception even though 10 > pool of 2


def test_initial_value_deduplicated():
  """A gen-0 chromosome with duplicate withoutReplacement genes is repaired to be unique."""
  stub, variables = _makeStub(3, lowerBound=1, upperBound=6)
  init = {'x1': 5.0, 'x2': 5.0, 'x3': 6.0}  # x2 duplicates x1
  repaired = stub._repairInitialValue(init)
  values = [repaired[v] for v in variables]
  assert len(set(values)) == len(values)       # all distinct
  assert repaired['x1'] == 5.0                   # first occurrence kept
  assert repaired['x3'] == 6.0                   # untouched unique gene kept


def test_initial_value_unique_untouched():
  """A gen-0 chromosome already satisfying the constraint is returned unchanged."""
  stub, variables = _makeStub(3, lowerBound=1, upperBound=6)
  init = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0}
  repaired = stub._repairInitialValue(init)
  assert repaired == init


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
