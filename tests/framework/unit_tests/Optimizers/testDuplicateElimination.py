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
  Unit tests for NSGA-II duplicate elimination in R(t) (<duplicateElimination>, <DuplicateKey>) and for the
  per-variable mutation controls <excludeVariables> and <variableProbability> in randomMutator / bitFlipMutator.
"""
import os
import sys
import numpy as np
import xarray as xr

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()
from ravenframework.utils import randomUtils
from ravenframework.Optimizers.MultiObjectiveGeneticAlgorithm import MultiObjectiveGeneticAlgorithm
from ravenframework.Optimizers.mutators.mutators import randomMutator, bitFlipMutator, swapMutator
from ravenframework.Distributions import Uniform

results = {'pass': 0, 'fail': 0}

def checkSame(comment, value, expected, update=True):
  """
    Compares two values for equality.
    @ In, comment, str, description printed if it fails
    @ In, value, any, the value to compare
    @ In, expected, any, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking value', comment, '|', value, '!=', expected)
      results['fail'] += 1
  return res

class KeyFunction:
  """
    Stand-in for an External Function with a "duplicateKey" method: the design is (family, size); the other
    variable is inactive and ignored.
  """
  name = 'designKey'
  def evaluate(self, method, inputs):
    """
      Mimics Function.evaluate.
      @ In, method, str, method name (must be "duplicateKey")
      @ In, inputs, dict, decision variables
      @ Out, key, tuple, design key
    """
    assert method == 'duplicateKey'
    return (int(round(float(inputs['family'][0]))), int(round(float(inputs['size'][0]))))

def makeGA(popSize=4, tolerance=0.0, variables=None, key=None):
  """
    Minimal MultiObjectiveGeneticAlgorithm with only the state duplicate elimination uses.
    @ In, popSize, int, population size N
    @ In, tolerance, float, relative tolerance
    @ In, variables, list or None, compared variables
    @ In, key, object or None, DuplicateKey function
    @ Out, ga, MultiObjectiveGeneticAlgorithm, instance
  """
  ga = MultiObjectiveGeneticAlgorithm()
  ga.toBeSampled = {'family': 'd', 'size': 'd', 'other': 'd'}
  ga.constants = {}
  ga._populationSize = popSize
  ga._duplicateElimination = True
  ga._duplicateTolerance = tolerance
  ga._duplicateVariables = variables
  ga._duplicateKeyFunction = key
  ga._objectiveVar = ['o1', 'o2']
  ga._objMult = {'o1': 1, 'o2': 1}
  ga._minMax = ['min', 'min']
  return ga

# R(t): rows 0-2 are P(t), rows 3-6 Q(t). Row 3 duplicates row 0; row 4 equals row 1 within 1e-10; row 5 has the
# same family and size as row 2 but a different inactive "other"; row 6 is new.
pop = np.array([[2, 9, 0.10],
                [1, 3, 0.50],
                [3, 4, 0.20],
                [2, 9, 0.10],
                [1, 3 + 1e-10, 0.50],
                [3, 4, 0.99],
                [0, 1, 0.30]], dtype=float)
obj = [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.6]]
cons = np.ones((7, 1))

ga = makeGA(popSize=4)
checkSame('default (exact, all variables): only the exact copy is removed; the oldest is kept',
          ga._uniqueIndividuals(pop, obj, cons), [0, 1, 2, 4, 5, 6])
checkSame('default tolerance of the class is exact', MultiObjectiveGeneticAlgorithm()._duplicateTolerance, 0.0)
ga = makeGA(popSize=4, tolerance=1e-8)
checkSame('tolerance 1e-8: the within-tolerance copy is removed too', ga._uniqueIndividuals(pop, obj, cons), [0, 1, 2, 5, 6])
ga = makeGA(popSize=4, variables=['family', 'size'])
checkSame('variables subset: the copy differing only in an inactive variable is removed (size 3 vs 3+1e-10 differ)',
          ga._uniqueIndividuals(pop, obj, cons), [0, 1, 2, 4, 6])
ga = makeGA(popSize=4, key=KeyFunction())
checkSame('DuplicateKey function: same result as comparing family and size',
          ga._uniqueIndividuals(pop, obj, cons), [0, 1, 2, 6])

# population size preserved: 6 slots but only 4 unique designs -> 2 duplicates refilled, best rank first
ga = makeGA(popSize=6, variables=['family', 'size'])
keep = ga._uniqueIndividuals(pop, obj, cons)
checkSame('fewer unique than N: kept count equals N', len(keep), 6)
checkSame('fewer unique than N: all unique designs kept', set([0, 1, 2, 6]).issubset(keep), True)
# infeasible duplicates are refilled after feasible ones
consInf = np.ones((7, 1))
consInf[3] = -1.0
ga = makeGA(popSize=5, variables=['family', 'size'])
keep = ga._uniqueIndividuals(pop, obj, consInf)
checkSame('refill prefers a feasible duplicate over an infeasible one', 3 in keep, False)
# no duplicates: identity
ga = makeGA(popSize=3)
checkSame('no duplicates: all rows kept in order', ga._uniqueIndividuals(pop[[0, 1, 6]], [o[:3] for o in obj], cons[:3]),
          [0, 1, 2])

#
# excludeVariables in randomMutator / bitFlipMutator
#
dist = Uniform(0.0, 1.0)
dist.initializeDistribution()
distDict = {'family': dist, 'size': dist, 'other': dist}
def offspring(n=200, value=0.5):
  """
    Children with every gene at the same value, so a mutation is visible.
    @ In, n, int, number of children
    @ In, value, float, gene value (bitFlipMutator maps v to 1 - v on U(0, 1), so 0.5 is a fixed point for it)
    @ Out, children, xr.DataArray, offspring
  """
  return xr.DataArray(np.full((n, 3), value), dims=['chromosome', 'Gene'],
                      coords={'chromosome': np.arange(n), 'Gene': ['family', 'size', 'other']})

for name, mutator in (('randomMutator', randomMutator), ('bitFlipMutator', bitFlipMutator)):
  randomUtils.randomSeed(42)
  out = mutator(offspring(200, 0.25), distDict, locs=None, mutationProb=1.0, excludeVariables=['family'])
  checkSame(f'{name}: excluded gene never mutated', bool(np.all(out.sel(Gene='family').values == 0.25)), True)
  checkSame(f'{name}: other genes still mutated', bool(np.any(out.sel(Gene='size').values != 0.25)), True)
  # default (no exclusion) draws the same random stream as before the option existed
  randomUtils.randomSeed(7)
  a = mutator(offspring(20), distDict, locs=None, mutationProb=0.5, excludeVariables=None).values.copy()
  randomUtils.randomSeed(7)
  b = mutator(offspring(20), distDict, locs=None, mutationProb=0.5).values.copy()
  checkSame(f'{name}: excludeVariables=None is identical to omitting it', bool(np.array_equal(a, b)), True)
  try:
    mutator(offspring(2), distDict, locs=None, mutationProb=1.0, excludeVariables=['family', 'size', 'other'])
    checkSame(f'{name}: excluding every variable raises', False, True)
  except ValueError:
    checkSame(f'{name}: excluding every variable raises', True, True)
# per-variable probability: the selector is chosen in proportion to its weight
for name, mutator in (('randomMutator', randomMutator), ('bitFlipMutator', bitFlipMutator)):
  randomUtils.randomSeed(3)
  out = mutator(offspring(4000, 0.25), distDict, locs=None, mutationProb=1.0,
                variableProbabilities={'family': 0.1, 'size': 1.0, 'other': 1.0})
  frac = float(np.mean(out.sel(Gene='family').values != 0.25))
  checkSame(f'{name}: weight 0.1 vs 1, 1 gives the selector ~0.1/2.1 = 4.8 % of mutations (got {frac:.3f})',
            0.03 < frac < 0.07, True)
  randomUtils.randomSeed(3)
  out = mutator(offspring(50, 0.25), distDict, locs=None, mutationProb=1.0, variableProbabilities={'family': 0.0})
  checkSame(f'{name}: weight 0 is never mutated', bool(np.all(out.sel(Gene='family').values == 0.25)), True)
  randomUtils.randomSeed(11)
  a = mutator(offspring(30), distDict, locs=None, mutationProb=0.7,
              variableProbabilities={'family': 2.0, 'size': 2.0, 'other': 2.0}).values.copy()
  randomUtils.randomSeed(11)
  b = mutator(offspring(30), distDict, locs=None, mutationProb=0.7).values.copy()
  checkSame(f'{name}: equal weights reproduce the default stream exactly', bool(np.array_equal(a, b)), True)
try:
  swapMutator(offspring(2), distDict, locs=None, mutationProb=1.0, variableProbabilities={'family': 0.1})
  checkSame('swapMutator rejects variableProbabilities', False, True)
except ValueError:
  checkSame('swapMutator rejects variableProbabilities', True, True)
try:
  swapMutator(offspring(2), distDict, locs=None, mutationProb=1.0, excludeVariables=['family'])
  checkSame('swapMutator rejects excludeVariables', False, True)
except ValueError:
  checkSame('swapMutator rejects excludeVariables', True, True)

#
# end
#
print('Results:', results)
sys.exit(results['fail'])
"""
  <TestInfo>
    <name>framework.unit_tests.Optimizers.DuplicateElimination</name>
    <author>Jimmy-INL</author>
    <created>2026-11-07</created>
    <classesTested>Optimizers.MultiObjectiveGeneticAlgorithm, Optimizers.mutators</classesTested>
    <description>
       Duplicate elimination in the NSGA-II combined population (exact by default, optional tolerance, variable
       subset, DuplicateKey function, oldest copy kept, population size preserved with feasible duplicates first)
       and the per-variable mutation controls of randomMutator / bitFlipMutator: excludeVariables (never mutated)
       and variableProbability (relative weight; 0 = never; equal weights reproduce the default stream exactly);
       misuse with other mutators is rejected.
    </description>
  </TestInfo>
"""
