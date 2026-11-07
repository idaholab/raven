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
  Unit tests for the opt-in NSGA-II options whose logic lives on the optimizer:
    * adaptive (annealed) mutation probability  -> NSGAII._effectiveMutationProb
    * stochastic ranking (Runarsson & Yao)      -> NSGAII._rankingConstraintVals
  These exercise the per-generation decision logic directly (the end-to-end behavior is
  additionally covered by the NSGA-II_ZDT1_sbx_adaptmut and NSGA-II_MinwoRepMultiObjectiveStoch
  framework tests).
"""
import os, sys
import numpy as np
ravenDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(ravenDir)
from ravenframework.utils import utils
utils.find_crow(os.path.join(ravenDir, "ravenframework"))
from ravenframework.Optimizers.NSGAII import NSGAII

results = {"pass": 0, "fail": 0}

def checkAnswer(comment, value, expected, tol=1e-7):
  """
    Compare two floats within a tolerance.
    @ In, comment, str, message on failure
    @ In, value, float, computed value
    @ In, expected, float, expected value
    @ In, tol, float, tolerance
    @ Out, None
  """
  if abs(value - expected) > tol:
    print("checking answer", comment, value, "!=", expected)
    results["fail"] += 1
  else:
    results["pass"] += 1

def checkTrue(comment, cond):
  """
    Assert a boolean condition.
    @ In, comment, str, message on failure
    @ In, cond, bool, condition
    @ Out, None
  """
  if cond:
    results["pass"] += 1
  else:
    print("checking", comment, "-> expected True, got False")
    results["fail"] += 1

#############################
# Adaptive (annealed) mutation probability: NSGAII._effectiveMutationProb
opt = NSGAII()
opt.toBeSampled = {'x1': None, 'x2': None, 'x3': None, 'x4': None}  # 4 decision variables
opt._mutationProb = 0.5
opt.limit = 11           # 11 generations -> progress = (counter-1)/10
opt._adaptiveMutationFinal = 0.1

# Disabled -> always the configured constant probability regardless of generation.
opt._adaptiveMutation = False
opt.counter = 6
checkAnswer('adaptive disabled returns constant', opt._effectiveMutationProb(), 0.5)

# Enabled: first generation is the initial probability, last generation is the final value,
# midpoint is the linear interpolation.
opt._adaptiveMutation = True
opt.counter = 1
checkAnswer('adaptive gen 1 = initial', opt._effectiveMutationProb(), 0.5)
opt.counter = 11
checkAnswer('adaptive last gen = final', opt._effectiveMutationProb(), 0.1)
opt.counter = 6
checkAnswer('adaptive midpoint interpolates', opt._effectiveMutationProb(), 0.3)

# final omitted -> defaults to 1/nVariables (=0.25 for 4 variables) at the last generation.
opt._adaptiveMutationFinal = None
opt.counter = 11
checkAnswer('adaptive default final = 1/nVars', opt._effectiveMutationProb(), 0.25)

#############################
# Stochastic ranking: NSGAII._rankingConstraintVals
opt2 = NSGAII()
constraintData = np.array([[1.0], [-2.0], [0.5]])

# Disabled -> constraint data passes through unchanged (strict constrained ranking).
opt2._stochasticRanking = False
checkTrue('stochastic disabled passes constraints through',
          opt2._rankingConstraintVals(constraintData) is constraintData)

# Enabled with pf = 0 -> the objective-only coin never fires, so constraints are always used.
opt2._stochasticRanking = True
opt2._stochasticRankingPf = 0.0
checkTrue('stochastic pf=0 keeps constraints',
          opt2._rankingConstraintVals(constraintData) is constraintData)

# Enabled with pf = 1 -> the coin always fires, so this generation ignores constraints (None).
opt2._stochasticRankingPf = 1.0
checkTrue('stochastic pf=1 ignores constraints (objective-only ranking)',
          opt2._rankingConstraintVals(constraintData) is None)

print(results)
sys.exit(results["fail"])
