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
  Unit tests for the optimizer 'withoutReplacement' initial-point handling shared by every
  optimizer (Optimizer base class): the fail-fast feasibility guard and the initial-point
  duplicate repair. The methods under test (_withoutReplacementGroups,
  _checkWithoutReplacementFeasible, _repairInitialValue) depend only on self.toBeSampled and
  self.distDict, so they are exercised on a light stand-in that borrows the real Optimizer
  implementations, with real UniformDiscrete distributions.
"""
import os
import sys
import xml.etree.ElementTree as ET

import pytest

ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir] * 4)))
if ravenDir not in sys.path:
  sys.path.append(ravenDir)

from ravenframework.utils.utils import find_crow
find_crow(os.path.join(ravenDir, 'framework'))

from ravenframework import MessageHandler
from ravenframework import Distributions
from ravenframework.Optimizers.Optimizer import Optimizer
from ravenframework.utils import randomUtils

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity': 'silent'})


def makeUniformDiscrete(lowerBound=None, upperBound=None, strategy='withoutReplacement', nPoints=None,
                        name='woRep_dist', step=None, values=None):
  """
    Build a real UniformDiscrete distribution through its input specification.
    @ In, lowerBound, int, optional, lower bound
    @ In, upperBound, int, optional, upper bound
    @ In, strategy, str, optional, sampling strategy
    @ In, nPoints, int, optional, number of grid points
    @ In, name, str, optional, distribution name
    @ In, step, float, optional, grid spacing
    @ In, values, list, optional, explicit outcomes (used without bounds)
    @ Out, dist, UniformDiscrete, the initialized distribution
  """
  node = ET.Element('UniformDiscrete', {'name': name})
  valuesText = None if values is None else ' '.join(str(v) for v in values)
  for tag, text in (('lowerBound', lowerBound), ('upperBound', upperBound), ('strategy', strategy), ('nPoints', nPoints),
                    ('step', step), ('values', valuesText)):
    if text is not None:
      sub = ET.SubElement(node, tag)
      sub.text = str(text)
  dist = Distributions.factory.returnInstance('UniformDiscrete')
  dist.setMessageHandler(mh)
  spec = dist.getInputSpecification()()
  spec.parseNode(node)
  dist._handleInput(spec)
  return dist


class _OptimizerStub:
  """
    Carries toBeSampled/distDict and borrows the real Optimizer methods under test.
  """
  _withoutReplacementGroups = Optimizer._withoutReplacementGroups
  _checkWithoutReplacementFeasible = Optimizer._checkWithoutReplacementFeasible
  _repairInitialValue = Optimizer._repairInitialValue

  def __init__(self, toBeSampled, distDict):
    """
      Constructor.
      @ In, toBeSampled, dict, {varName: distributionName}
      @ In, distDict, dict, {varName: distribution}
      @ Out, None
    """
    self.toBeSampled = toBeSampled
    self.distDict = distDict

  def raiseAnError(self, etype, msg):
    """
      Match the MessageUser signature so the guard's error path is testable.
      @ In, etype, Exception, error type
      @ In, msg, str, message
      @ Out, None
    """
    raise etype(msg)

  def raiseADebug(self, *args, **kwargs):
    """
      Silence debug output.
      @ In, args, list, unused
      @ In, kwargs, dict, unused
      @ Out, None
    """


def makeStub(nVars, **distKwargs):
  """
    Build a stub whose nVars variables all share one UniformDiscrete.
    @ In, nVars, int, number of variables
    @ In, distKwargs, dict, UniformDiscrete settings
    @ Out, (stub, variables), tuple, the stub and its variable names
  """
  dist = makeUniformDiscrete(**distKwargs)
  variables = [f'x{i+1}' for i in range(nVars)]
  return _OptimizerStub({var: dist.name for var in variables}, {var: dist for var in variables}), variables


def test_number_of_outcomes():
  """numberOfOutcomes counts the integer range or the nPoints grid."""
  assert makeUniformDiscrete(1, 6).numberOfOutcomes() == 6
  assert makeUniformDiscrete(0, 10, nPoints=4).numberOfOutcomes() == 4


def test_number_of_outcomes_explicit_values():
  """With <values> (no bounds) the count is the number of listed outcomes, not int(None)."""
  assert makeUniformDiscrete(values=[2, 3, 5, 7, 11]).numberOfOutcomes() == 5


def test_number_of_outcomes_step():
  """With <step> the count is the size of the stepped grid, not upper - lower + 1."""
  assert makeUniformDiscrete(0, 10, step=2.5).numberOfOutcomes() == 5


def test_number_of_outcomes_ignores_sampling_state():
  """Sampling without replacement shrinks the working pool; the support count must not change."""
  dist = makeUniformDiscrete(1, 6)
  dist.initializeDistribution()
  dist.selectedRvs([1, 2, 3])
  assert dist.numberOfOutcomes() == 6


def test_guard_with_explicit_values():
  """The feasibility guard works for <values>: 3 variables on 2 listed values is rejected."""
  stub, _ = makeStub(3, values=[4, 9])
  with pytest.raises(IOError) as info:
    stub._checkWithoutReplacementFeasible()
  assert '2 distinct value' in str(info.value)
  stub, _ = makeStub(2, values=[4, 9])
  stub._checkWithoutReplacementFeasible()


def test_guard_with_step():
  """The feasibility guard counts a stepped grid: {0, 5, 10} holds 3 variables but not 4."""
  stub, _ = makeStub(3, lowerBound=0, upperBound=10, step=5)
  stub._checkWithoutReplacementFeasible()
  stub, _ = makeStub(4, lowerBound=0, upperBound=10, step=5)
  with pytest.raises(IOError):
    stub._checkWithoutReplacementFeasible()


def test_feasible_setup_passes():
  """vars == pool is feasible; the guard must not raise."""
  stub, _ = makeStub(6, lowerBound=1, upperBound=6)
  stub._checkWithoutReplacementFeasible()


def test_infeasible_setup_raises():
  """More variables than distinct values is infeasible; the guard must raise IOError."""
  stub, _ = makeStub(3, lowerBound=1, upperBound=2)
  with pytest.raises(IOError) as info:
    stub._checkWithoutReplacementFeasible()
  message = str(info.value)
  assert 'withoutReplacement' in message
  assert '2 distinct value' in message
  assert '3 variables' in message


def test_khang_report_setup_is_infeasible():
  """The reported case: six variables on {5, 6, 7} without replacement is rejected up front."""
  stub, _ = makeStub(6, lowerBound=5, upperBound=7)
  with pytest.raises(IOError):
    stub._checkWithoutReplacementFeasible()


def test_withreplacement_never_blocks():
  """A withReplacement distribution is never subject to the uniqueness guard."""
  stub, _ = makeStub(10, lowerBound=1, upperBound=2, strategy='withReplacement')
  stub._checkWithoutReplacementFeasible()
  init = {f'x{i+1}': 1.0 for i in range(10)}
  assert stub._repairInitialValue(init) == init


def test_initial_value_deduplicated():
  """An initial point with duplicate withoutReplacement values is repaired to be unique and in range."""
  randomUtils.randomSeed(42)
  stub, variables = makeStub(3, lowerBound=1, upperBound=6)
  init = {'x1': 5.0, 'x2': 5.0, 'x3': 6.0}
  repaired = stub._repairInitialValue(init)
  values = [repaired[v] for v in variables]
  assert len(set(values)) == len(values)
  assert all(1 <= v <= 6 for v in values)
  assert repaired['x1'] == 5.0                   # first occurrence kept


def test_initial_value_unique_untouched():
  """An initial point already satisfying the constraint is returned unchanged."""
  stub, _ = makeStub(3, lowerBound=1, upperBound=6)
  init = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0}
  assert stub._repairInitialValue(init) == init


def test_full_pool_permutation_is_reachable():
  """With as many variables as values, repair always yields a permutation of the support."""
  randomUtils.randomSeed(7)
  stub, variables = makeStub(3, lowerBound=5, upperBound=7)
  for init in ({'x1': 5.0, 'x2': 5.0, 'x3': 5.0}, {'x1': 7.0, 'x2': 6.0, 'x3': 7.0}):
    repaired = stub._repairInitialValue(init)
    assert sorted(repaired[v] for v in variables) == [5.0, 6.0, 7.0]


def test_independent_distributions_repaired_separately():
  """Values are only required to be distinct among variables sharing the same distribution."""
  distA = makeUniformDiscrete(1, 3, name='a')
  distB = makeUniformDiscrete(1, 3, name='b')
  stub = _OptimizerStub({'x1': 'a', 'x2': 'a', 'y1': 'b', 'y2': 'b'},
                        {'x1': distA, 'x2': distA, 'y1': distB, 'y2': distB})
  repaired = stub._repairInitialValue({'x1': 1.0, 'x2': 2.0, 'y1': 1.0, 'y2': 2.0})
  assert repaired == {'x1': 1.0, 'x2': 2.0, 'y1': 1.0, 'y2': 2.0}


def test_selected_rvs_exhausted_raises():
  """Drawing with every value discarded raises a clear error instead of ZeroDivisionError."""
  dist = makeUniformDiscrete(5, 7)
  with pytest.raises(RuntimeError) as info:
    dist.selectedRvs([5.0, 6.0, 7.0])
  assert 'no values left to draw' in str(info.value)


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
