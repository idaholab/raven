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
  Unit tests for the GA fitness functions (invLinear, feasibleFirst, logistic) and the <a>/<b> scaling contract.
"""
import math

import numpy as np
import pytest

from ravenframework.Optimizers.fitness import fitness


def test_inv_linear_penalizes_constraint_violations(fitness_inputs):
  result = fitness.invLinear(
    fitness_inputs['rlz'],
    objVar=fitness_inputs['obj_var'],
    a=[1.0],
    b=[10.0],
    constraintFunction=fitness_inputs['constraints'],
    type=['min'],
  )
  values = result['obj'].values
  np.testing.assert_allclose(values, np.array([-1.0, -5.0, -5.5]))


def test_feasible_first_uses_worst_objective_for_penalty(fitness_inputs):
  result = fitness.feasibleFirst(
    fitness_inputs['rlz'],
    objVar=fitness_inputs['obj_var'],
    a=[1.0],
    b=[10.0],
    constraintFunction=fitness_inputs['constraints'],
    constraintNum=2,
    type=['min'],
  )
  values = result['obj'].values
  # Deb (2000) anchors each infeasible penalty at the WORST FEASIBLE objective, which here is
  # obj[0]=1.0 (the only feasible individual), not the whole-population worst (2.0).
  # ind0 feasible: -a*obj = -1.0; ind1 infeasible: -1*1.0 - 10*0.3 = -4.0;
  # ind2 infeasible: -1*1.0 - 10*0.5 = -6.0.
  np.testing.assert_allclose(values, np.array([-1.0, -4.0, -6.0]))


def _logistic(x):
  """
    Reference logistic function.
    @ In, x, float, argument
    @ Out, value, float, 1 / (1 + exp(-x))
  """
  return 1.0 / (1.0 + math.exp(-x))


def test_logistic_minimization_prefers_lower_objective(fitness_inputs):
  """For 'min' (the default) a lower objective gets a higher fitness, as the docstring and manual state."""
  result = fitness.logistic(fitness_inputs['rlz'], objVar=fitness_inputs['obj_var'], scale=[1.0], shift=[1.0],
                            penalty=[0.0], type=['min'])
  values = result['obj'].values
  # objectives are 1.0, 2.0, 0.5: fitness must order 0.5 > 1.0 > 2.0
  assert values[2] > values[0] > values[1]
  expected = np.array([_logistic(-(val - 1.0)) for val in fitness_inputs['rlz']['obj'].data])
  np.testing.assert_allclose(values, expected)


def test_logistic_maximization_prefers_higher_objective(fitness_inputs):
  """The GA passes maximized objectives negated; 'max' must rank by the original (un-negated) value."""
  rlz = fitness_inputs['rlz'].copy()
  original = rlz['obj'].data.copy()
  rlz['obj'] = -rlz['obj']  # minimization space, as the GA hands it over
  result = fitness.logistic(rlz, objVar=fitness_inputs['obj_var'], scale=[1.0], shift=[1.0], penalty=[0.0],
                            type=['max'])
  values = result['obj'].values
  assert values[1] > values[0] > values[2]
  np.testing.assert_allclose(values, [_logistic(val - 1.0) for val in original])


def test_logistic_shift_is_the_half_fitness_point(fitness_inputs):
  """At objective == shift the logistic fitness is 1/2 for both directions."""
  for minMax, sign in (('min', 1.0), ('max', -1.0)):
    rlz = fitness_inputs['rlz'].copy()
    rlz['obj'] = sign * xr_full(rlz['obj'], 3.0)
    result = fitness.logistic(rlz, objVar=fitness_inputs['obj_var'], scale=[2.0], shift=[3.0], penalty=[0.0],
                              type=[minMax])
    np.testing.assert_allclose(result['obj'].values, 0.5)


def test_logistic_violation_always_lowers_fitness(fitness_inputs):
  """A constraint violation lowers the fitness for both 'min' and 'max' (it used to raise it for 'max')."""
  for minMax in ('min', 'max'):
    common = dict(objVar=fitness_inputs['obj_var'], scale=[2.0], shift=[1.0], type=[minMax])
    clean = fitness.logistic(fitness_inputs['rlz'], penalty=[0.0], **common)['obj'].values
    penalized = fitness.logistic(fitness_inputs['rlz'], penalty=[0.5], constraintFunction=fitness_inputs['constraints'],
                                 **common)['obj'].values
    # individual 0 is feasible; 1 violates c1 by 0.3; 2 violates c1 by 0.1 and c2 by 0.4
    np.testing.assert_allclose(clean - penalized, [0.0, 0.5 * 0.3, 0.5 * 0.5])


def xr_full(array, value):
  """
    Copy of an xarray DataArray with every entry set to value.
    @ In, array, xr.DataArray, template
    @ In, value, float, fill value
    @ Out, filled, xr.DataArray, filled copy
  """
  filled = array.copy()
  filled.data[:] = value
  return filled


def _twoObjectiveInputs(fitness_inputs):
  """
    Two objectives over the shared 3-individual, 2-constraint fixture.
    @ In, fitness_inputs, dict, fixture
    @ Out, (rlz, constraints), tuple, two-objective dataset and constraint values
  """
  rlz = fitness_inputs['rlz'].copy()
  rlz['obj2'] = rlz['obj'] * 2.0
  return rlz, fitness_inputs['constraints']


def test_penalty_scalar_per_constraint_and_per_pair_agree(fitness_inputs):
  """b given as one value, one per constraint, or one per (objective, constraint) pair: equal weights agree."""
  rlz, constraints = _twoObjectiveInputs(fitness_inputs)
  common = dict(objVar=['obj', 'obj2'], a=[1.0, 1.0], constraintFunction=constraints, constraintNum=2, type=['min', 'min'])
  for func in (fitness.invLinear, fitness.feasibleFirst):
    scalar = func(rlz, b=[3.0], **common)
    perConst = func(rlz, b=[3.0, 3.0], **common)
    perPair = func(rlz, b=[3.0, 3.0, 3.0, 3.0], **common)
    for obj in common['objVar']:
      np.testing.assert_allclose(scalar[obj].values, perConst[obj].values)
      np.testing.assert_allclose(scalar[obj].values, perPair[obj].values)


def _threeObjectiveInputs(fitness_inputs):
  """
    Three objectives over the 2-constraint fixture, so nObj != nConst and a length-nConst b cannot
    be mistaken for a per-objective b.
    @ In, fitness_inputs, dict, fixture
    @ Out, (rlz, constraints), tuple, three-objective dataset and constraint values
  """
  rlz = fitness_inputs['rlz'].copy()
  rlz['obj2'] = rlz['obj'] * 2.0
  rlz['obj3'] = rlz['obj'] * 3.0
  return rlz, fitness_inputs['constraints']


def test_penalty_per_constraint_is_shared_across_objectives(fitness_inputs):
  """A length-nConst b weights constraint j by b_j for every objective (earlier RAVEN behavior)."""
  rlz, constraints = _threeObjectiveInputs(fitness_inputs)
  objVar = ['obj', 'obj2', 'obj3']
  common = dict(objVar=objVar, a=[1.0, 1.0, 1.0], constraintFunction=constraints, constraintNum=2, type=['min'] * 3)
  for func in (fitness.invLinear, fitness.feasibleFirst):
    perConst = func(rlz, b=[1.0, 5.0], **common)
    expanded = func(rlz, b=[1.0, 5.0] * 3, **common)   # the same weights written per (objective, constraint)
    for obj in objVar:
      np.testing.assert_allclose(perConst[obj].values, expanded[obj].values)
  # violations: ind1 -> c1=0.3; ind2 -> c1=0.1, c2=0.4; penalty = 1*c1 + 5*c2 for every objective
  result = fitness.invLinear(rlz, b=[1.0, 5.0], **common)
  np.testing.assert_allclose(result['obj'].values, np.array([-1.0, -2.0 - 0.3, -0.5 - 0.1 - 2.0]))
  np.testing.assert_allclose(result['obj3'].values, np.array([-3.0, -6.0 - 0.3, -1.5 - 0.1 - 2.0]))


def test_penalty_per_objective_length_is_rejected(fitness_inputs):
  """With nObj != nConst, a length-nObj b (neither 1, nConst nor nObj*nConst) is rejected."""
  rlz, constraints = _threeObjectiveInputs(fitness_inputs)
  with pytest.raises(IOError):
    fitness.invLinear(rlz, objVar=['obj', 'obj2', 'obj3'], b=[1.0, 2.0, 3.0],
                      constraintFunction=constraints, constraintNum=2, type=['min'] * 3)


def test_penalty_prlo_form_two_objectives_four_constraints():
  """The PRLO deck form (2 objectives, 4 constraints, one b per constraint) is accepted (B7 reproducer)."""
  a, b = fitness._scalingFactors(['MaxEFPD', 'FuelCycleCost'], 4, b=[0.0025, 6.0, 8.0, 0.2])
  np.testing.assert_allclose(b, np.array([[0.0025, 6.0, 8.0, 0.2]] * 2))
  np.testing.assert_allclose(a, np.ones(2))


def test_penalty_per_pair_is_objective_major(fitness_inputs):
  """A per-(objective, constraint) b is read objective-major: b = [b11, b12, b21, b22]."""
  rlz, constraints = _twoObjectiveInputs(fitness_inputs)
  result = fitness.invLinear(rlz, objVar=['obj', 'obj2'], a=[1.0, 1.0], b=[1.0, 2.0, 10.0, 20.0],
                             constraintFunction=constraints, constraintNum=2, type=['min', 'min'])
  # violations: ind1 -> c1=0.3; ind2 -> c1=0.1, c2=0.4
  # obj  (b=1, 2):   ind1 = -2.0 - 1*0.3 = -2.3;            ind2 = -0.5 - 1*0.1 - 2*0.4 = -1.4
  # obj2 (b=10, 20): ind1 = -4.0 - 10*0.3 = -7.0;           ind2 = -1.0 - 10*0.1 - 20*0.4 = -10.0
  np.testing.assert_allclose(result['obj'].values, np.array([-1.0, -2.3, -1.4]))
  np.testing.assert_allclose(result['obj2'].values, np.array([-2.0, -7.0, -10.0]))


def test_penalty_default_matches_explicit(fitness_inputs):
  """Without <b> the penalty weight is the documented default (10.0), not 1.0."""
  withDefault = fitness.invLinear(fitness_inputs['rlz'], objVar=fitness_inputs['obj_var'],
                                  constraintFunction=fitness_inputs['constraints'], constraintNum=2, type=['min'])
  explicitDefault = fitness.invLinear(fitness_inputs['rlz'], objVar=fitness_inputs['obj_var'], b=[fitness._defaultPenaltyScaling],
                                      constraintFunction=fitness_inputs['constraints'], constraintNum=2, type=['min'])
  np.testing.assert_allclose(withDefault['obj'].values, explicitDefault['obj'].values)
  # and it must NOT match b=1.0 (guards against the default silently reverting)
  explicitOne = fitness.invLinear(fitness_inputs['rlz'], objVar=fitness_inputs['obj_var'], b=[1.0],
                                  constraintFunction=fitness_inputs['constraints'], constraintNum=2, type=['min'])
  assert not np.allclose(withDefault['obj'].values, explicitOne['obj'].values)
  assert fitness._defaultPenaltyScaling == 10.0


def test_penalty_bad_length_raises(fitness_inputs):
  """A b whose length is neither 1, nConstraint, nor nObj*nConstraint is rejected."""
  rlz, constraints = _twoObjectiveInputs(fitness_inputs)
  with pytest.raises(IOError):
    fitness.invLinear(rlz, objVar=['obj', 'obj2'], b=[1.0, 2.0, 3.0],
                      constraintFunction=constraints, constraintNum=2, type=['min', 'min'])


def test_logistic_none_penalty_uses_default(fitness_inputs):
  """The GA passes penalty=None; logistic must fall back to its default (1.0) instead of crashing."""
  common = dict(objVar=fitness_inputs['obj_var'], scale=[2.0], shift=[1.0],
                constraintFunction=fitness_inputs['constraints'], type=['min'])
  withNone = fitness.logistic(fitness_inputs['rlz'], penalty=None, **common)
  explicitOne = fitness.logistic(fitness_inputs['rlz'], penalty=[1.0], **common)
  np.testing.assert_allclose(withNone['obj'].values, explicitOne['obj'].values)


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
