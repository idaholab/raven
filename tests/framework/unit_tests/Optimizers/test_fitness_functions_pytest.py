import numpy as np

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
  np.testing.assert_allclose(values, np.array([-1.0, -5.0, -7.0]))


def test_logistic_adjusts_for_maximization_and_penalties(fitness_inputs):
  result = fitness.logistic(
    fitness_inputs['rlz'],
    objVar=fitness_inputs['obj_var'],
    scale=[2.0],
    shift=[1.0],
    penalty=[0.5],
    constraintFunction=fitness_inputs['constraints'],
    type=['max'],
  )
  values = result['obj'].values
  expected = np.array([0.5, 0.269203, 0.981059])
  np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
