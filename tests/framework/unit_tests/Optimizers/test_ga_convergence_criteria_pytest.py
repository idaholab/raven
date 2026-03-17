import numpy as np
import xarray as xr

from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm


def _make_ga_base():
  ga = GeneticAlgorithm()
  ga.raiseADebug = lambda *args, **kwargs: None
  ga.raiseAWarning = lambda *args, **kwargs: None
  ga.convFormat = '{name}:{conv}:{got}:{req}'
  ga._objectiveVar = ['obj1', 'obj2']
  ga._objMult = {'obj1': 1.0, 'obj2': 1.0}
  ga._isMultiObjective = True
  return ga


def test_convergence_hypervolume():
  ga = _make_ga_base()
  ranks = np.array([1, 1, 1, 2])
  ga.matingPopRanks = xr.DataArray(ranks, dims=['chromosome'])
  ga.matingPopObjVals = [
    np.array([1.0, 0.8, 0.6, 1.4]),
    np.array([0.5, 0.7, 0.9, 1.3]),
  ]
  prev = {
    'rank': np.array([1, 1, 1, 2]),
    'obj1': np.array([1.1, 0.9, 0.7, 1.5]),
    'obj2': np.array([0.6, 0.8, 1.0, 1.4]),
  }
  ga._optPointHistory = {0: [(prev, None), (prev, None)]}
  ga._convergenceCriteria = {'hypervolume': 0.05}
  ga._populationSize = len(ranks)

  assert ga._checkConvHypervolume(0)


def test_convergence_spread():
  ga = _make_ga_base()
  ranks = np.array([1, 1, 1, 1])
  ga.matingPopRanks = xr.DataArray(ranks, dims=['chromosome'])
  ga.matingPopObjVals = [
    np.array([1.0, 0.8, 0.6, 0.4]),
    np.array([0.5, 0.7, 0.9, 1.1]),
  ]
  ga._convergenceCriteria = {'spread': 0.8}
  ga._populationSize = len(ranks)

  assert ga._checkConvSpread(0)


def test_convergence_max_spread():
  ga = _make_ga_base()
  ranks = np.array([1, 1, 1, 2])
  ga.matingPopRanks = xr.DataArray(ranks, dims=['chromosome'])
  ga.matingPopObjVals = [
    np.array([1.0, 0.8, 0.6, 1.4]),
    np.array([0.5, 0.7, 0.9, 1.3]),
  ]
  prev = {
    'rank': np.array([1, 1, 1, 2]),
    'obj1': np.array([1.02, 0.82, 0.62, 1.4]),
    'obj2': np.array([0.52, 0.72, 0.92, 1.3]),
  }
  ga._optPointHistory = {0: [(prev, None), (prev, None)]}
  ga._convergenceCriteria = {'maxSpread': 0.1}
  ga._populationSize = len(ranks)

  assert ga._checkConvMaxSpread(0)


def test_convergence_rank1_ratio():
  ga = _make_ga_base()
  ranks = np.array([1, 1, 2, 2, 1])
  ga.matingPopRanks = xr.DataArray(ranks, dims=['chromosome'])
  ga._populationSize = len(ranks)
  ga._convergenceCriteria = {'rank1Ratio': 0.6}
  ga._rank1History = {0: [0.61, 0.6]}

  assert ga._checkConvRank1Ratio(0)


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
