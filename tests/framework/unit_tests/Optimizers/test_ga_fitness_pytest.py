import math

import numpy as np
import pytest

from ravenframework.Optimizers.fitness import fitness


def test_inv_linear_penalizes_constraint_violations(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  constraints = fitness_inputs["constraints"]
  result = fitness.invLinear(
    rlz,
    objVar=["obj"],
    a=[1.0],
    b=[5.0],
    constraintFunction=constraints,
    type=["min"],
  )
  expected = np.array([-1.0, -3.5, -3.0])
  assert np.allclose(result["obj"].data, expected)


def test_feasible_first_uses_worst_objective_for_infeasible(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  constraints = fitness_inputs["constraints"]
  result = fitness.feasibleFirst(
    rlz,
    objVar=["obj"],
    constraintFunction=constraints,
    constraintNum=2,
    a=[1.0],
    b=[2.0],
    type=["min"],
  )
  expected = np.array([-1.0, -2.6, -3.0])
  assert np.allclose(result["obj"].data, expected)


def test_logistic_applies_penalty_and_respects_minimization(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  constraints = fitness_inputs["constraints"]
  result = fitness.logistic(
    rlz,
    objVar=["obj"],
    scale=[-1.0],
    shift=[1.0],
    penalty=[0.5],
    constraintFunction=constraints,
    type=["min"],
  )
  expected = np.array(
    [
      0.5,
      1.0 / (1.0 + math.exp(1.0)) - 0.5 * 0.3,
      1.0 / (1.0 + math.exp(-0.5)) - 0.5 * 0.5,
    ]
  )
  assert np.allclose(result["obj"].data, expected)


def test_logistic_flips_for_maximization(fitness_inputs):
  rlz = fitness_inputs["rlz"]
  base = [
    1.0 / (1.0 + math.exp(-(val - 1.0)))
    for val in rlz["obj"].data
  ]
  result = fitness.logistic(
    rlz,
    objVar=["obj"],
    scale=[1.0],
    shift=[1.0],
    penalty=[0.0],
    type=["max"],
  )
  assert np.allclose(result["obj"].data, 1.0 - np.array(base))


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
