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
  Unit tests for RavenSampled checkpoint/restart: state round-trip (both the raw dict
  round-trip and a full HDF5 write/restore round-trip) and checkpoint validation failures.
"""
import os
import sys
import json
import tempfile
import importlib
import numpy as np

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()
from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm
# imported via importlib (not "import ... as"): the Optimizers package __init__ rebinds the
# attribute "RavenSampled" to the class, which would shadow the submodule otherwise
RavenSampledModule = importlib.import_module('ravenframework.Optimizers.RavenSampled')

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

def checkTrue(comment, res, update=True):
  """
    Records whether a boolean condition holds.
    @ In, comment, str, description printed if it fails
    @ In, res, bool, the condition to check
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, the same value passed in
  """
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking bool', comment, '| got False, expected True')
      results['fail'] += 1
  return res

def checkRaises(comment, callable, exceptionType, update=True):
  """
    Confirms that calling `callable` raises an exception of the given type.
    @ In, comment, str, description printed if it fails
    @ In, callable, callable, zero-argument callable to invoke
    @ In, exceptionType, type, expected exception type
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if the expected exception was raised
  """
  res = False
  try:
    callable()
  except exceptionType:
    res = True
  except Exception as err:
    print('checking raise', comment, '| wrong exception type raised:', type(err), err)
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking raise', comment, '| no', exceptionType, 'was raised')
      results['fail'] += 1
  return res

def makeGA(name='testOpt'):
  """
    Builds a minimal GeneticAlgorithm instance with just enough state set to exercise
    checkpoint logic directly, bypassing the full XML handleInput path.
    @ In, name, str, optional, name to assign to the optimizer instance
    @ Out, ga, GeneticAlgorithm, minimal instance
  """
  ga = GeneticAlgorithm()
  ga.name = name
  ga.toBeSampled = {'x1': 'dummyDist', 'x2': 'dummyDist'}
  ga._objectiveVar = ['obj1']
  ga._minMax = ['min']
  ga._solutionExport = None
  # normally set by handleInput; GA's checkpoint settings/state include these
  ga._isMultiObjective = False
  ga._populationSize = 4
  return ga

#
# Round-trip via _getCheckpointState/_restoreCheckpointState, through the same JSON
# encode/decode step _writeCheckpoint/_restoreFromCheckpoint use (exercises key-stringification
# handling, e.g. ast.literal_eval of dedup cache keys, without needing an actual HDF5 file).
#
ga = makeGA()
ga.counter = 5
ga.batchId = 2
ga._trajCounter = 1
ga._activeTraj = [0]
ga._cancelledTraj = {}
ga._convergedTraj = {}
ga._rerunsSinceAccept = {0: 3}
ga._stepTracker = {0: {'converge': False}}
ga._optPointHistory = {0: [{'x1': 0.5, 'x2': 0.25}]}
ga._RavenSampled__stepCounter = {0: 4}
ga._submissionQueue.append(({'x1': 0.1, 'x2': 0.2}, {'traj': 0}))
ga._deduplication = True
ga._cacheEvaluatedSubmissionPoints({'x1': np.atleast_1d(0.5), 'x2': np.atleast_1d(0.25), 'obj1': np.atleast_1d(1.0)})
dedupKey = ga._makeSubmissionKey({'x1': np.atleast_1d(0.5), 'x2': np.atleast_1d(0.25)})

state = ga._getCheckpointState()
encoded = json.loads(json.dumps(RavenSampledModule._encodeCheckpointState(state)))
decoded = RavenSampledModule._decodeCheckpointState(encoded)

ga2 = makeGA()
ga2._restoreCheckpointState(decoded)
checkSame('restored counter matches original', ga2.counter, 5)
checkSame('restored batchId matches original', ga2.batchId, 2)
checkSame('restored trajCounter matches original', ga2._trajCounter, 1)
checkSame('restored activeTraj matches original', ga2._activeTraj, [0])
checkSame('restored rerunsSinceAccept matches original', ga2._rerunsSinceAccept, {0: 3})
checkSame('restored stepCounter matches original', ga2._RavenSampled__stepCounter, {0: 4})
checkSame('restored submission queue matches original', list(ga2._submissionQueue), list(ga._submissionQueue))
checkTrue('restored dedup cache retains the cached point', dedupKey in ga2._evaluatedSubmissionData)

#
# Full round-trip through an actual HDF5 checkpoint file via _writeCheckpoint/_restoreFromCheckpoint
#
with tempfile.TemporaryDirectory() as tmpDir:
  checkpointPath = os.path.join(tmpDir, 'test.ravenrst')

  gaWrite = makeGA()
  gaWrite._checkpointFile = checkpointPath
  gaWrite._checkpointInterval = 1
  gaWrite.counter = 9
  gaWrite.batchId = 4
  gaWrite._trajCounter = 3
  gaWrite._activeTraj = []
  gaWrite._deduplication = True
  gaWrite._cacheEvaluatedSubmissionPoints({'x1': np.atleast_1d(0.7), 'x2': np.atleast_1d(0.3), 'obj1': np.atleast_1d(2.0)})
  gaWrite._writeCheckpoint()
  checkTrue('checkpoint file was written to disk', os.path.exists(checkpointPath))

  gaRead = makeGA()
  gaRead._restartFromFile = checkpointPath
  gaRead._restoreFromCheckpoint()
  checkSame('HDF5 round-trip restores counter', gaRead.counter, 9)
  checkSame('HDF5 round-trip restores batchId', gaRead.batchId, 4)
  checkSame('HDF5 round-trip restores trajCounter', gaRead._trajCounter, 3)
  checkTrue('HDF5 round-trip marks checkpoint as restored', gaRead._checkpointRestored)
  restoredKey = gaRead._makeSubmissionKey({'x1': np.atleast_1d(0.7), 'x2': np.atleast_1d(0.3)})
  checkTrue('HDF5 round-trip retains the dedup cache entry', restoredKey in gaRead._evaluatedSubmissionData)

#
# Validation failures: mismatched optimizer type, mismatched sampled-variable set
#
ga = makeGA()
matchingSettings = {'variables': sorted(ga.toBeSampled.keys()), 'objectiveVars': ga._objectiveVar, 'minMax': ga._minMax}
mismatchedTypeCheckpoint = {
  'version': RavenSampledModule._CHECKPOINT_VERSION,
  'optimizerType': 'SomeOtherOptimizerClass',
  'optimizerName': ga.name,
  'settings': matchingSettings,
}
checkRaises('validate: mismatched optimizer type is rejected',
            lambda: ga._validateCheckpoint(mismatchedTypeCheckpoint), IOError)

mismatchedVarsCheckpoint = {
  'version': RavenSampledModule._CHECKPOINT_VERSION,
  'optimizerType': ga.__class__.__name__,
  'optimizerName': ga.name,
  'settings': {'variables': ['x1'], 'objectiveVars': ga._objectiveVar, 'minMax': ga._minMax},
}
checkRaises('validate: mismatched sampled-variable set is rejected',
            lambda: ga._validateCheckpoint(mismatchedVarsCheckpoint), IOError)

validCheckpoint = {
  'version': RavenSampledModule._CHECKPOINT_VERSION,
  'optimizerType': ga.__class__.__name__,
  'optimizerName': ga.name,
  'settings': matchingSettings,
}
try:
  ga._validateCheckpoint(validCheckpoint)
  checkTrue('validate: matching checkpoint settings pass validation', True)
except Exception as err:
  print('checking bool', 'validate: matching checkpoint settings pass validation', '| unexpected exception:', err)
  results['fail'] += 1

checkTrue('older-but-supported checkpoint versions are recognized',
          '1.0' in RavenSampledModule._SUPPORTED_CHECKPOINT_VERSIONS)

olderSupportedVersionCheckpoint = {
  'version': '1.0',
  'optimizerType': ga.__class__.__name__,
  'optimizerName': ga.name,
  'settings': matchingSettings,
}
try:
  ga._validateCheckpoint(olderSupportedVersionCheckpoint)
  checkTrue('validate: older-but-supported checkpoint version passes validation', True)
except Exception as err:
  print('checking bool', 'validate: older-but-supported checkpoint version passes validation',
        '| unexpected exception:', err)
  results['fail'] += 1

#
# end
#
print('Results:', results)
sys.exit(results['fail'])
