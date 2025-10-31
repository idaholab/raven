import numpy as np
import xarray as xr

from ravenframework.Optimizers.constraintHandling import constraintHandling as constraint_module


class DummyConstraint:
  def __init__(self, name, parameters):
    self.name = name
    self._parameters = parameters

  def parameterNames(self):
    return list(self._parameters)


class DummyGeneticAlgorithm:
  def __init__(self):
    self._constraintFunctions = [DummyConstraint('explicitC', [])]
    self._impConstraintFunctions = [DummyConstraint('implicitC', ['extra'])]
    self._objectiveVar = ['obj']
    self.toBeSampled = {'x1': None}
    self._objMult = {'obj': 1.0}

  def _handleExplicitConstraints(self, newOpt, constraint):
    return 0.5 - float(newOpt.sel(Gene='x1').values)

  def _handleImplicitConstraints(self, newOpt, opt, constraint):
    return float(opt['extra'] - opt['obj'])


def _make_offsprings():
  return xr.DataArray(
    np.array([[0.2], [0.6]], dtype=float),
    dims=['chromosome', 'Gene'],
    coords={'chromosome': np.arange(2), 'Gene': ['x1']},
  )


def _make_rlz():
  return xr.Dataset(
    {
      'obj': xr.DataArray([0.4, 0.8], dims=['chromosome'], coords={'chromosome': np.arange(2)}),
      'extra': xr.DataArray([0.7, 0.4], dims=['chromosome'], coords={'chromosome': np.arange(2)}),
    }
  )


def test_constraint_handling_combines_explicit_and_implicit():
  ga = DummyGeneticAlgorithm()
  offsprings = _make_offsprings()
  rlz = _make_rlz()
  objective_values = [np.array([0.4, 0.8])]
  info = {'traj': 0}
  g = constraint_module.constraintHandling(ga, info, rlz, offsprings, objective_values)
  assert g.shape == (2, 2)
  expected = np.array([[0.3, 0.3], [-0.1, -0.4]])
  np.testing.assert_allclose(g.values, expected)
  assert list(g.coords['Constraint'].values) == ['explicitC', 'implicitC']


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
