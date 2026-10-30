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
  This Module performs Unit Tests for the DMDC (Dynamic Mode Decomposition with Control) ROM, in particular for
  the pooled identification (single model from all the training histories), the time constants export, the
  solution metadata and the evaluation of multiple requests.
  It can not be considered part of the active code but of the regression test system
"""
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4)))
if ravenDir not in sys.path:
  sys.path.append(ravenDir)

frameworkDir = os.path.join(ravenDir, "framework")
from ravenframework.utils.utils import find_crow
find_crow(frameworkDir)

from ravenframework.SupervisedLearning.DMD.DynamicModeDecompositionControl import DMDC

results = {"pass":0,"fail":0}

def checkFloat(comment, value, expected, tol=1e-10):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ Out, res, bool, True if same
  """
  res = abs(value - expected) <= tol
  if not res:
    print("checking float", comment, '|', value, "!=", expected)
  results["pass" if res else "fail"] += 1
  return res

def checkTrue(comment, res):
  """
    This method is a pass-through for consistency and updating
    @ In, comment, string, a comment printed out if it fails
    @ In, res, bool, the tested value
    @ Out, res, bool, True if test
  """
  if not res:
    print("checking bool", comment, '|', res, 'is not True!')
  results["pass" if res else "fail"] += 1
  return res

def checkFails(comment, errstr, function, args=None):
  """
    Checks if expected error occurs
    @ In, comment, string, a comment printed out if it fails
    @ In, errstr, str, expected fail message (contained in the error message)
    @ In, function, method, method to run to test for failure
    @ In, args, list, optional, arguments to pass to function
    @ Out, res, bool, True if failed as expected
  """
  try:
    function(*(args or []))
    res, msg = False, 'Function call did not error!'
  except Exception as e:
    res = errstr in str(e)
    msg = f'Unexpected error message: "{e}"'
  if not res:
    print("checking error", comment, '|', msg)
  results["pass" if res else "fail"] += 1
  return res

def buildDMDC(options, features='u,level,x1_init,x2_init,tEnd'):
  """
    Create a DMDC ROM from its XML input
    @ In, options, str, additional XML nodes
    @ In, features, str, optional, the features
    @ Out, rom, DMDC, the ROM
  """
  rom = DMDC()
  spec = DMDC.getInputSpecification()()
  spec.parseNode(ET.fromstring(f'<ROM name="dmdc" subType="DMDC"><Target>Time,x1,x2,y1</Target><Features>{features}</Features>'
                               '<actuators>u</actuators><stateVariables>x1,x2</stateVariables>'
                               '<initStateVariables>x1_init,x2_init</initStateVariables><pivotParameter>Time</pivotParameter>'
                               f'{options}</ROM>'))
  rom._handleInput(spec)
  return rom

# training data of the PooledDMDC framework test (linear system with time constants 15 s and 300 s,
# 4 histories of different lengths extended to a common time grid, see generatePooledDMDC.py)
dataDir = os.path.join(ravenDir, 'tests', 'framework', 'ROM', 'TimeSeries', 'DMD', 'DMDC', 'trainingData')
main = pd.read_csv(os.path.join(dataDir, 'Pooled_DMDC.csv'))
histories = [pd.read_csv(os.path.join(dataDir, fileName)) for fileName in main['filename']]
nSteps = len(histories[0])
trainingSet = {var: [hist[var].values for hist in histories] for var in ['Time', 'x1', 'x2', 'y1', 'u']}
for var in ['level', 'x1_init', 'x2_init', 'tEnd']:
  trainingSet[var] = [np.full(nSteps, value) for value in main[var]]
offsets = np.array([(0.02, -0.01), (0.04, -0.02), (0.06, -0.03), (0.08, -0.04)])
initStates = main[['x1_init', 'x2_init']].values
# A matrix of the generating system (time constants 15 s and 300 s, time step 2 s)
eigVectors = np.array([[1.0, 0.3], [0.5, 1.0]])
trueA = eigVectors @ np.diag(np.exp(-2.0 / np.array([15.0, 300.0]))) @ np.linalg.inv(eigVectors)
# with centered states and actuators (subtractNormUXY), c' = c + (A - I) x0 + B u0 (u0 = 0 here)
centeredOffsets = offsets + initStates @ (trueA - np.eye(2)).T

#####################################
#  Pooled identification            #
#####################################
rom = buildDMDC('<pooled>True</pooled><validUntil>tEnd</validUntil><subtractNormUXY>True</subtractNormUXY>')
checkTrue('validUntil is not a parameter', 'tEnd' not in rom.parametersIDs and 'level' in rom.parametersIDs)
checkTrue('default rankSVD is no truncation', rom.dmdParams['rankSVD'] == -1)
rom.train(trainingSet)
meta = rom.getSolutionMetadata()
checkTrue('one entry per history in the metadata', len(meta['dataBySamples']) == 4)
for smp, sampleData in enumerate(meta['dataBySamples']):
  taus = [float(tau) for tau in sampleData['timeConstants']['tau'].split()]
  checkFloat(f'pooled fast time constant (history {smp})', taus[0], 15.0, tol=1e-6)
  checkFloat(f'pooled slow time constant (history {smp})', taus[1], 300.0, tol=1e-4)
  checkTrue(f'non oscillatory modes (history {smp})', sampleData['timeConstants']['period'].split() == ['inf', 'inf'])
  for i in range(2):
    checkFloat(f'pooled offset {i} (history {smp})', sampleData['historyOffsets'][i], centeredOffsets[smp][i], tol=1e-9)
  checkTrue(f'A matrix in the metadata (history {smp})', sampleData['Atilde']['matrixShape'] == '2,2')
  checkTrue(f'C matrix in the metadata (history {smp})', sampleData['Ctilde']['matrixShape'] == '1,2')
checkTrue('XLast in the metadata', len(meta['dataBySamples'][0]['XLast']) == 2)

# evaluation of two requests at once (with centering of the actuators and states)
requests = [2, 3]
featureVals = np.zeros((len(requests), nSteps, len(rom.features)))
for r, hist in enumerate(requests):
  for var in rom.features:
    featureVals[r, :, rom.features.index(var)] = trainingSet[var][hist]
evaluation = rom.__evaluateLocal__(featureVals)
for r, hist in enumerate(requests):
  end = int(main['tEnd'][hist] / 2.0) + 1
  for var in ['x1', 'x2', 'y1']:
    error = np.abs(evaluation[var][r, :end] - histories[hist][var].values[:end]).max()
    checkFloat(f'evaluation of request {r} ({var})', error, 0.0, tol=1e-8)

# without centering, the offsets are the ones of the generating system
rom = buildDMDC('<pooled>True</pooled><validUntil>tEnd</validUntil>')
rom.train(trainingSet)
meta = rom.getSolutionMetadata()
for smp, sampleData in enumerate(meta['dataBySamples']):
  for i in range(2):
    checkFloat(f'pooled offset {i} without centering (history {smp})', sampleData['historyOffsets'][i], offsets[smp][i], tol=1e-9)

# without validUntil the extended portions bias the identification
rom = buildDMDC('<pooled>True</pooled>')
rom.train(trainingSet)
taus = [float(tau) for tau in rom.getSolutionMetadata()['dataBySamples'][0]['timeConstants']['tau'].split()]
checkTrue('the extended portions bias the slow time constant', abs(taus[1] - 300.0) > 10.0)

# model without output variables (only states)
rom = DMDC()
spec = DMDC.getInputSpecification()()
spec.parseNode(ET.fromstring('<ROM name="dmdc" subType="DMDC"><Target>Time,x1,x2</Target><Features>u,level,x1_init,x2_init,tEnd</Features>'
                             '<actuators>u</actuators><stateVariables>x1,x2</stateVariables>'
                             '<initStateVariables>x1_init,x2_init</initStateVariables><pivotParameter>Time</pivotParameter>'
                             '<pooled>True</pooled><validUntil>tEnd</validUntil></ROM>'))
rom._handleInput(spec)
rom.train({var: values for var, values in trainingSet.items() if var != 'y1'})
taus = [float(tau) for tau in rom.getSolutionMetadata()['dataBySamples'][0]['timeConstants']['tau'].split()]
checkFloat('pooled slow time constant without outputs', taus[1], 300.0, tol=1e-4)
evaluation = rom.__evaluateLocal__(featureVals[1:])
checkFloat('evaluation without outputs', np.abs(evaluation['x2'][:int(main['tEnd'][3] / 2.0) + 1] -
                                                 histories[3]['x2'].values[:int(main['tEnd'][3] / 2.0) + 1]).max(), 0.0, tol=1e-8)

#####################################
#  History-wise identification      #
#####################################
rom = buildDMDC('<subtractNormUXY>True</subtractNormUXY>')
rom.train(trainingSet)
meta = rom.getSolutionMetadata()
checkTrue('no offsets without pooled identification', 'historyOffsets' not in meta['dataBySamples'][0])
checkTrue('time constants also for the history-wise identification', len(meta['dataBySamples'][3]['timeConstants']['tau'].split()) == 2)

#####################################
#  Time constants helper            #
#####################################
eigs, taus, periods = DMDC._timeConstants(np.diag([np.exp(-0.5), 1.0, 0.0]), 2.0)
checkFloat('time constant of a decaying mode', taus[1], 4.0)
checkTrue('null eigenvalue gives zero time constant', taus[0] == 0.0)
checkTrue('non decaying mode gives infinite time constant', np.isinf(taus[2]))
rotation = 0.9 * np.array([[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]])
eigs, taus, periods = DMDC._timeConstants(rotation, 1.0)
checkFloat('period of an oscillatory mode', periods[0], 2.0 * np.pi / 0.1, tol=1e-8)

#####################################
#  Input errors                     #
#####################################
checkFails('validUntil not in the features', 'The validUntil variable "tEnd" must be listed among <Features>!',
           buildDMDC, args=['<pooled>True</pooled><validUntil>tEnd</validUntil>', 'u,level,x1_init,x2_init'])
rom = buildDMDC('<pooled>True</pooled>')
oneHistory = {var: values[:1] for var, values in trainingSet.items()}
checkFails('pooled identification with one history', 'requires at least 2 training histories', rom.train, args=[oneHistory])

print(results)

sys.exit(results["fail"])
