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
  Unit tests for MultiObjectiveGeneticAlgorithm._reconcileRankingVariant: the
  <rankingAlgorithm type> and the <constraintEpsilon>/<stochasticRanking> nodes must agree, so the
  selected type always matches the ranking that actually runs.
"""
import pytest

from ravenframework.Optimizers.MultiObjectiveGeneticAlgorithm import MultiObjectiveGeneticAlgorithm


class _MogaStub:
  """
    Carries the ranking settings and borrows the real reconciliation method.
  """
  _reconcileRankingVariant = MultiObjectiveGeneticAlgorithm._reconcileRankingVariant

  def __init__(self, rankType='constrainedDomination', epsilon=0.0, stochastic=False):
    """
      Constructor.
      @ In, rankType, str, optional, <rankingAlgorithm type>
      @ In, epsilon, float, optional, <constraintEpsilon> value
      @ In, stochastic, bool, optional, <stochasticRanking> value
      @ Out, None
    """
    self._rankingAlgorithmType = rankType
    self._constraintEpsilon = epsilon
    self._stochasticRanking = stochastic
    self._stochasticRankingPf = 0.45

  def raiseAnError(self, etype, msg):
    """
      Match the MessageUser signature.
      @ In, etype, Exception, error type
      @ In, msg, str, message
      @ Out, None
    """
    raise etype(msg)

  def raiseAMessage(self, *args, **kwargs):
    """
      Silence messages.
      @ In, args, list, unused
      @ In, kwargs, dict, unused
      @ Out, None
    """


def test_default_is_unchanged():
  """Plain Deb-2002 stays plain Deb-2002."""
  stub = _MogaStub()
  stub._reconcileRankingVariant()
  assert stub._rankingAlgorithmType == 'constrainedDomination'
  assert not stub._stochasticRanking


def test_epsilon_node_selects_epsilon_type():
  """<constraintEpsilon> with the default type reports the epsilon-constrained variant it runs."""
  stub = _MogaStub(epsilon=2.0)
  stub._reconcileRankingVariant()
  assert stub._rankingAlgorithmType == 'epsilonConstrained'


def test_epsilon_type_without_epsilon_raises():
  """type="epsilonConstrained" without a positive <constraintEpsilon> would silently be Deb-2002."""
  with pytest.raises(IOError):
    _MogaStub(rankType='epsilonConstrained')._reconcileRankingVariant()


def test_stochastic_type_enables_stochastic_ranking():
  """type="stochasticRanking" turns stochastic ranking on even without the node."""
  stub = _MogaStub(rankType='stochasticRanking')
  stub._reconcileRankingVariant()
  assert stub._stochasticRanking
  assert stub._rankingAlgorithmType == 'stochasticRanking'


def test_stochastic_node_selects_stochastic_type():
  """<stochasticRanking>True</...> with the default type reports the stochastic variant."""
  stub = _MogaStub(stochastic=True)
  stub._reconcileRankingVariant()
  assert stub._rankingAlgorithmType == 'stochasticRanking'


@pytest.mark.parametrize('kwargs', [
  dict(epsilon=1.0, stochastic=True),
  dict(rankType='feasibleFirstPenalty', epsilon=1.0),
  dict(rankType='feasibleFirstPenalty', stochastic=True),
])
def test_conflicting_settings_raise(kwargs):
  """Settings that ask for two different ranking variants are rejected."""
  with pytest.raises(IOError):
    _MogaStub(**kwargs)._reconcileRankingVariant()


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
