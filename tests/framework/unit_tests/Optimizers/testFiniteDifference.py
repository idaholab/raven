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
  Testing for FiniteDifference gradient approximation
"""
import os
import sys

import numpy as np

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5, 'framework'))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
import Driver
#from utils.utils import find_crow
#add_path(os.path.join(ravenPath, 'contrib'))
#add_path(os.path.join(ravenPath, 'contrib', 'AMSC'))
#add_path_recursively(os.path.join(ravenPath, 'contrib', 'pp3'))
#find_crow(ravenPath)

from Optimizers.gradients import factory # returnInstance

fd = factory.returnInstance('FiniteDifference')

#
#
# checkers
#
def checkFloat(comment, value, expected, tol=1e-10, update=True):
  """
    This method compares two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  if np.isnan(value) and np.isnan(expected):
    res = True
  elif np.isnan(value) or np.isnan(expected):
    res = False
  else:
    res = abs(value - expected) <= tol
  if update:
    if not res:
      print("checking float", comment, '|', value, "!=", expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

def checkSame(comment, value, expected, update=True):
  """
    This method compares two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking string", comment, '|', value, "!=", expected)
      results["fail"] += 1
  return res

results = {'pass': 0, 'fail': 0}

#
#
# formatters
#
def formatSample(vars):
  return dict ((x, np.atleast_1d(y)) for x, y in vars.items())

#
#
# initialization
#
optVars = ['a', 'b', 'c']
fd.initialize(optVars)

checkSame('Check num vars', fd.N, 3)

#
#
# choosing eval points
#

optPoint = {'a': 0.1,
            'b': 0.2,
            'c': 0.3}
stepSize = 0.5
pts, info = fd.chooseEvaluationPoints(optPoint, stepSize)
correct = [
    {'a': 0.095, 'b': 0.2, 'c': 0.3},
    {'a': 0.1, 'b': 0.205, 'c': 0.3},
    {'a': 0.1, 'b': 0.2, 'c': 0.295}]
cinfo = [
    {'type': 'grad', 'optVar': 'a', 'delta': -0.005},
    {'type': 'grad', 'optVar': 'b', 'delta':  0.005},
    {'type': 'grad', 'optVar': 'c', 'delta': -0.005}]

for p, pt in enumerate(pts):
  for v in ['a', 'b', 'c']:
    checkFloat('Point "{}" var "{}"'.format(p, v), pts[p][v], correct[p][v])
  for i in ['type', 'optVar']:
    checkSame('Point "{}" info "{}"'.format(p, i), info[p][i], cinfo[p][i])
  checkFloat('Point "{}" var "{}"'.format(p, 'delta'), info[p]['delta'], cinfo[p]['delta'])

#
#
# evaluating the gradient
#
def linearModel(vars):
  ans = 3 * vars['a'] + 2 * vars['b'] + vars['c']
  vars['ans'] = ans

opt = {'a': 0.1, 'b': 0.2, 'c': 0.3}
grads = [
    {'a': 0.095, 'b': 0.2, 'c': 0.3},
    {'a': 0.1, 'b': 0.205, 'c': 0.3},
    {'a': 0.1, 'b': 0.2, 'c': 0.295}]
for g, grad in enumerate(grads):
  grads[g] = formatSample(grad)
# fill in samples
linearModel(opt)
for g in grads:
  linearModel(g)

infos = [
    {'type': 'grad', 'optVar': 'a', 'delta': -0.005},
    {'type': 'grad', 'optVar': 'b', 'delta':  0.005},
    {'type': 'grad', 'optVar': 'c', 'delta': -0.005}]
mag, vsr, inf = fd.evaluate(opt, grads, infos, 'ans')
correctMag = [3.741657386773921]
correctVsr = [0.8017837257372723, 0.5345224838248521, 0.2672612419124202]
checkSame('Linear finite model, inf check', inf, False)
checkFloat('Linear finite model, magnitude', mag, correctMag)
for v, var in enumerate(['a', 'b', 'c']):
  checkFloat('Linear finite model, versor, var "{}"'.format(var), vsr[var], correctVsr[v])


#
# try some infinites
def infiniteModel(vs):
  a, b, c = vs['a'], vs['b'], vs['c']
  if a < 0.1 or b > 0.2:
    ans = np.atleast_1d(np.inf)
  else:
    ans = 3 * a + 2 * b + c
  vs['ans'] = ans

opt = formatSample({'a': 0.1, 'b': 0.2, 'c': 0.3})
grads = [
    {'a': 0.095, 'b': 0.2, 'c': 0.3},
    {'a': 0.1, 'b': 0.205, 'c': 0.3},
    {'a': 0.1, 'b': 0.2, 'c': 0.295}]
for g, grad in enumerate(grads):
  grads[g] = formatSample(grad)
infos = [
    {'type': 'grad', 'optVar': 'a', 'delta': -0.005},
    {'type': 'grad', 'optVar': 'b', 'delta':  0.005},
    {'type': 'grad', 'optVar': 'c', 'delta': -0.005}]
# fill in samples
infiniteModel(opt)
for g in grads:
  infiniteModel(g)
mag, vsr, inf = fd.evaluate(opt, grads, infos, 'ans')
correctMag = [1.4142135623730951]
correctVsr = [-0.7071067811865475, 0.7071067811865475, 0]
checkSame('Linear infinite model, inf check', inf, True)
checkFloat('Linear infinite model, magnitude', mag, correctMag)
for v, var in enumerate(['a', 'b', 'c']):
  checkFloat('Linear infinite model, versor, var "{}"'.format(var), vsr[var], correctVsr[v])

#
# number of required samples
#
optVars = ['a', 'b', 'c']
fd.initialize(optVars)
checkSame('Number of samples needed, 3 vars', fd.numGradPoints(), 3)
optVars = ['a', 'b', 'c', 'd', 'e']
fd.initialize(optVars)
checkSame('Number of samples needed, 5 vars', fd.numGradPoints(), 5)

#
# end
#
print('Results:', results)
sys.exit(results['fail'])
