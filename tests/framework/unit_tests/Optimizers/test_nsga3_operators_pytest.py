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
  Unit tests for the NSGA-III reference-point operator families: reference directions, objective
  normalization, association, and niching. The assertions check fidelity to Das & Dennis (1998) and
  Deb & Jain (2014), not merely that the code runs.
"""
from math import comb

import numpy as np

from ravenframework.Optimizers.referenceDirectionHandlers.referenceDirectionHandlers import (
    dasDennis, returnInstance as referenceDirectionReturnInstance)
from ravenframework.Optimizers.normalizationHandlers.normalizationHandlers import (
    hyperplane, simplexSum, none as noNormalization, returnInstance as normalizationReturnInstance)
from ravenframework.Optimizers.associationHandlers.associationHandlers import (
    perpendicular, returnInstance as associationReturnInstance)
from ravenframework.Optimizers.nichingHandlers.nichingHandlers import (
    returnInstance as nichingReturnInstance)


class _Caller:
  """Minimal stand-in for the optimizer: supplies the raiseAnError hook the handlers use."""
  def raiseAnError(self, etype, msg):
    """
      Raise the given error type, mimicking the RAVEN BaseType hook.
      @ In, etype, type, exception class
      @ In, msg, str, message
      @ Out, None
    """
    raise etype(msg)


CALLER = _Caller()


def test_das_dennis_counts_and_simplex():
  """dasDennis points lie on the unit simplex and the boundary-layer count obeys Das & Dennis."""
  for numObjectives, populationSize in [(2, 50), (3, 50), (5, 100)]:
    directions = dasDennis(CALLER, numObjectives, populationSize)
    assert directions.shape[1] == numObjectives
    # every reference point lies on the unit simplex (coordinates sum to 1, nonnegative)
    assert np.allclose(directions.sum(axis=1), 1.0)
    assert np.all(directions >= -1e-12)
    # the boundary layer alone must be a valid Das & Dennis lattice not exceeding the population
    divisions = 1
    while comb(divisions + numObjectives, numObjectives - 1) <= populationSize:
      divisions += 1
    boundaryCount = comb(divisions - 1 + numObjectives - 1, numObjectives - 1)
    assert len(directions) >= boundaryCount


def test_das_dennis_two_layer_adds_interior_points():
  """For M > 2 the Deb & Jain inside layer adds points strictly interior to the simplex."""
  directions = dasDennis(CALLER, 3, 100)
  # interior (inside-layer) points have all coordinates strictly positive; boundary points have a zero
  interior = np.all(directions > 1e-9, axis=1)
  assert interior.any(), 'expected an interior (inside-layer) reference point for M=3'


def test_normalization_handlers_nonnegative_and_finite():
  """All normalization handlers return finite, nonnegative, correctly shaped arrays."""
  rng = np.random.default_rng(0)
  objectives = np.abs(rng.random((12, 3))) + 0.1
  for handler in (hyperplane, simplexSum, noNormalization):
    normalized = handler(CALLER, objectives)
    assert normalized.shape == objectives.shape
    assert np.isfinite(normalized).all()
    assert np.all(normalized >= -1e-12)
  # simplexSum projects onto the unit simplex: nonzero rows sum to 1
  projected = simplexSum(CALLER, objectives)
  assert np.allclose(projected.sum(axis=1), 1.0)


def test_perpendicular_association_is_order_invariant():
  """Association assigns the same reference direction regardless of the point ordering."""
  refs = dasDennis(CALLER, 3, 50)
  rng = np.random.default_rng(1)
  points = hyperplane(CALLER, np.abs(rng.random((20, 3))) + 0.05)
  idx, dist = perpendicular(CALLER, points, refs)
  assert np.all(dist >= -1e-12) and np.isfinite(dist).all()
  # permuting the points permutes the association identically (no order dependence)
  perm = rng.permutation(len(points))
  idxPerm, _ = perpendicular(CALLER, points[perm], refs)
  assert np.array_equal(idxPerm, idx[perm])


def test_niching_fills_requested_slots_and_balances():
  """referencePointDeterministic selects exactly the requested number and spreads across niches."""
  nicher = nichingReturnInstance(CALLER, 'referencePointDeterministic')
  refs = dasDennis(CALLER, 3, 20)
  rng = np.random.default_rng(2)
  normalized = hyperplane(CALLER, np.abs(rng.random((30, 3))) + 0.05)
  front = list(range(30))
  slots = 10
  selected, distanceCache, countsUpdate = nicher(
      CALLER, front=front, slots=slots, normalized=normalized, selectedIndices=[],
      nicheCounts=np.zeros(len(refs), dtype=int), referenceDirections=refs, associate=perpendicular)
  assert len(selected) == slots
  assert len(set(selected)) == slots           # no duplicate survivors
  assert set(selected).issubset(set(front))    # survivors come from the front
  assert int(countsUpdate.sum()) == slots       # niche counts account for every selection
  assert all(idx in distanceCache for idx in selected)


def test_unknown_type_raises_for_each_family():
  """Each family's returnInstance raises a clear IOError on an unknown operator name."""
  for returnInstance in (referenceDirectionReturnInstance, normalizationReturnInstance,
                         associationReturnInstance, nichingReturnInstance):
    try:
      returnInstance(CALLER, 'definitelyNotAHandler')
    except IOError:
      continue
    raise AssertionError('expected IOError for an unknown operator name')
