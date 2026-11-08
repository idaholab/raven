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
  This Module performs Unit Tests for the TSA.ExponentialResponse class.
  It can not be considered part of the active code but of the regression test system
"""
import os
import sys
import numpy as np

ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4)))
if ravenDir not in sys.path:
  sys.path.append(ravenDir)

frameworkDir = os.path.join(ravenDir, "framework")
from ravenframework.utils.utils import find_crow
find_crow(frameworkDir)

from ravenframework.utils import xmlUtils
from ravenframework.TSA import ExponentialResponse as ER

print(f"\nModule undergoing testing:\n{ER}\n")

results = {"pass":0,"fail":0}

def checkFloat(comment, value, expected, tol=1e-10, update=True):
  """
    This method is aimed to compare two floats given a certain tolerance
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
      print("checking float",comment,'|',value,"!=",expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

def checkTrue(comment, res, update=True):
  """
    This method is a pass-through for consistency and updating
    @ In, comment, string, a comment printed out if it fails
    @ In, res, bool, the tested value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if test
  """
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking bool",comment,'|',res,'is not True!')
      results["fail"] += 1
  return res

def checkSame(comment, value, expected, update=True):
  """
    This method is aimed to compare two identical things
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
      print("checking string",comment,'|',value,"!=",expected)
      results["fail"] += 1
  return res

def checkArray(comment, first, second, dtype, tol=1e-10, update=True):
  """
    This method is aimed to compare two arrays
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = True
  if len(first) != len(second):
    res = False
    print("checking answer",comment,'|','lengths do not match:',len(first),len(second))
  else:
    for i in range(len(first)):
      if dtype == float:
        pres = checkFloat('',first[i],second[i],tol,update=False)
      elif dtype in (str,unicode):
        pres = checkSame('',first[i],second[i],update=False)
      if not pres:
        print('checking array',comment,'|','entry "{}" does not match: {} != {}'.format(i,first[i],second[i]))
        res = False
  if update:
    if res:
      results["pass"] += 1
    else:
      results["fail"] += 1
  return res

def checkNone(comment, entry, update=True):
  """
    Checks if entry is None.
    @ In, comment, string, a comment printed out if it fails
    @ In, entry, object, to test if against None
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if None
  """
  res = entry is None
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking answer",comment,'|','"{}" is not None!'.format(entry))
      results["fail"] += 1

def checkFails(comment, errstr, function, update=True, args=None, kwargs=None):
  """
    Checks if expected error occurs
    @ In, comment, string, a comment printed out if it fails
    @ In, errstr, str, expected fail message
    @ In, function, method, method to run to test for failure
    @ In, update, bool, optional, if False then don't update results counter
    @ In, args, list, arguments to pass to function
    @ In, kwargs, dict, keyword arguments to pass to function
    @ Out, res, bool, True if failed as expected
  """
  print('Error testing ...')
  if args is None:
    args = []
  if kwargs is None:
    kwargs = {}
  try:
    function(*args,**kwargs)
    res = False
    msg = 'Function call did not error!'
  except Exception as e:
    res = checkSame('',e.args[0],errstr,update=False)
    if not res:
      msg = 'Unexpected error message.  \n    Received: "{}"\n    Expected: "{}"'.format(e.args[0],errstr)
  if update:
    if res:
      results["pass"] += 1
      print(' ... end Error testing (PASSED)')
    else:
      print("checking error",comment,'|',msg)
      results["fail"] += 1
      print(' ... end Error testing (FAILED)')
  print('')
  return res


######################################
#            CONSTRUCTION            #
######################################
def createExpResponseXML(targets, nTerms=None, startTime=None, tauMin=None, tauMax=None, model=None):
  """
    Return ExponentialResponse XML node for RAVEN input.
    @ In, targets, list[str], list of strings describing targets for current RAVEN run.
    @ In, nTerms, int, optional, number of exponential terms
    @ In, startTime, float, optional, start time of the response
    @ In, tauMin, float, optional, lower bound of the time constants
    @ In, tauMax, float, optional, upper bound of the time constants
    @ In, model, str, optional, response model (exponential or lagged)
    @ Out, xml, xml.etree.ElementTree.Element, new node
  """
  xml = xmlUtils.newNode('ExponentialResponse', attrib={'target':','.join(targets)})
  for name, value in [('model', model), ('nTerms', nTerms), ('startTime', startTime), ('tauMin', tauMin), ('tauMax', tauMax)]:
    if value is not None:
      xml.append(xmlUtils.newNode(name, text=f'{value}'))
  return xml

def createFromXML(xml):
  """
    Return ExponentialResponse TSA object and its settings.
    @ In, xml, xml.etree.ElementTree.Element, RAVEN input xml.
    @ Out, (model, settings), (TSA.ExponentialResponse, dict), model object and settings
  """
  model = ER.ExponentialResponse()
  inputSpec = ER.ExponentialResponse.getInputSpecification()()
  inputSpec.parseNode(xml)
  settings = model.handleInput(inputSpec)
  return model, settings

def createExpResponse(targets, **kwargs):
  """
    Return ExponentialResponse object and its settings.
    @ In, targets, list[str], list of string describing targets.
    @ In, kwargs, dict, optional settings (see createExpResponseXML)
    @ Out, (model, settings), (TSA.ExponentialResponse, dict), model object and settings
  """
  return createFromXML(createExpResponseXML(targets, **kwargs))

pivot = np.arange(-6.0, 300.0, 2.0)

#####################################
#  Single mode, step at t=0         #
#####################################
# rising response, held at the initial value before the step
signalA = np.where(pivot < 0, 20.0, 100.0 - 80.0 * np.exp(-pivot / 12.0))
# decaying response
signalB = np.where(pivot < 0, 5.0, 2.0 + 3.0 * np.exp(-pivot / 40.0))
signals = np.column_stack([signalA, signalB])
targets = ['A', 'B']

model, settings = createExpResponse(targets, startTime=0.0)
checkTrue("model can characterize", model.canCharacterize())
checkTrue("model can generate", model.canGenerate())
checkTrue("model can transform", model.canTransform())
checkSame("default number of terms", settings['nTerms'], 1)
params = model.fit(signals, pivot, targets, settings)
checkFloat("single mode A steady", params['A']['model']['steady'], 100.0, tol=1e-6)
checkFloat("single mode A initial", params['A']['model']['initial'], 20.0, tol=1e-6)
checkFloat("single mode A tau", params['A']['model']['taus'][0], 12.0, tol=1e-6)
checkFloat("single mode A amplitude", params['A']['model']['amplitudes'][0], -80.0, tol=1e-6)
checkFloat("single mode B steady", params['B']['model']['steady'], 2.0, tol=1e-6)
checkFloat("single mode B tau", params['B']['model']['taus'][0], 40.0, tol=1e-5)
checkFloat("single mode B amplitude", params['B']['model']['amplitudes'][0], 3.0, tol=1e-6)

# generation, residual and composite
synthetic = model.generate(params, pivot, settings)
checkArray("single mode generate", synthetic[:, 0], signalA, float, tol=1e-5)
checkFloat("single mode hold before start", synthetic[0, 1], 5.0, tol=1e-6)
residual = model.getResidual(signals, params, pivot, settings)
checkTrue("single mode residual is zero", np.abs(residual).max() < 1e-5)
composite = model.getComposite(residual, params, pivot, settings)
checkArray("single mode composite", composite[:, 1], signalB, float, tol=1e-8)

# variable names
rlz = model.getParamsAsVars(params)
names = model.getParamNames(settings)
checkSame("parameter names match realization", sorted(names), sorted(rlz.keys()))
checkSame("number of parameter names", len(names), 12)
checkSame("realization at-bound flag", rlz['ExponentialResponse__A__tau1_atBound'], 0.0)
checkFloat("realization tau", rlz['ExponentialResponse__A__tau1'], 12.0, tol=1e-6)

#####################################
#  Two modes (fast + slow)          #
#####################################
signalC = np.where(pivot < 0, 20.0, 100.0 - 50.0 * np.exp(-pivot / 8.0) - 30.0 * np.exp(-pivot / 90.0))
model, settings = createExpResponse(['C'], nTerms=2, startTime=0.0)
params = model.fit(signalC[:, None], pivot, ['C'], settings)
checkFloat("two modes steady", params['C']['model']['steady'], 100.0, tol=1e-4)
checkFloat("two modes fast tau", params['C']['model']['taus'][0], 8.0, tol=1e-4)
checkFloat("two modes slow tau", params['C']['model']['taus'][1], 90.0, tol=1e-3)
checkFloat("two modes fast amplitude", params['C']['model']['amplitudes'][0], -50.0, tol=1e-4)
checkFloat("two modes slow amplitude", params['C']['model']['amplitudes'][1], -30.0, tol=1e-4)
checkFloat("two modes rmse", params['C']['model']['rmse'], 0.0, tol=1e-5)

#####################################
#  Noise and missing values         #
#####################################
rng = np.random.default_rng(42)
signalD = np.where(pivot < 0, 20.0, 100.0 - 80.0 * np.exp(-pivot / 12.0)) + rng.normal(0, 0.5, len(pivot))
signalD[[10, 50, 100]] = np.nan
model, settings = createExpResponse(['D'], startTime=0.0)
params = model.fit(signalD[:, None], pivot, ['D'], settings)
checkFloat("noisy single mode tau", params['D']['model']['taus'][0], 12.0, tol=0.3)
checkFloat("noisy single mode steady", params['D']['model']['steady'], 100.0, tol=0.3)
checkFloat("noisy single mode rmse", params['D']['model']['rmse'], 0.5, tol=0.1)

#####################################
#  Bounds and input errors          #
#####################################
# a mode much slower than the signal length cannot be identified: tau goes to the upper bound
model, settings = createExpResponse(['E'], startTime=0.0, tauMax=50.0)
ramp = np.where(pivot < 0, 0.0, pivot)
params = model.fit(ramp[:, None], pivot, ['E'], settings)
checkFloat("ramp gives tau at upper bound", params['E']['model']['taus'][0], 50.0, tol=1e-3)
checkSame("ramp tau flagged at bound", params['E']['model']['atBound'][0], 1.0)
checkFails("invalid number of terms", "ExponentialResponse: <nTerms> must be greater than 0. Got 0!",
           createExpResponse, args=[['A']], kwargs={'nTerms': 0})
model, settings = createExpResponse(['A'], startTime=290.0, nTerms=2)
checkFails("not enough points", 'ExponentialResponse: not enough valid points after startTime to fit 2 exponential term(s) for target "A"!',
           model.fit, args=[signalA[:, None], pivot, ['A'], settings])

#####################################
#  Lagged model (lags in series)    #
#####################################
# two first-order lags in series: S-shaped response with zero initial slope
tau1, tau2 = 9.0, 150.0
tt = np.maximum(pivot, 0.0)
signalF = 100.0 + (20.0 - 100.0) * (tau1 * np.exp(-tt / tau1) - tau2 * np.exp(-tt / tau2)) / (tau1 - tau2)
model, settings = createExpResponse(['F'], model='lagged', nTerms=2, startTime=0.0)
checkSame("lagged model setting", settings['model'], 'lagged')
params = model.fit(signalF[:, None], pivot, ['F'], settings)
checkFloat("lagged steady", params['F']['model']['steady'], 100.0, tol=1e-4)
checkFloat("lagged initial", params['F']['model']['initial'], 20.0, tol=1e-4)
checkFloat("lagged fast tau", params['F']['model']['taus'][0], tau1, tol=1e-3)
checkFloat("lagged slow tau", params['F']['model']['taus'][1], tau2, tol=1e-2)
checkFloat("lagged fast amplitude", params['F']['model']['amplitudes'][0], -80.0 * tau1 / (tau1 - tau2), tol=1e-3)
checkFloat("lagged rmse", params['F']['model']['rmse'], 0.0, tol=1e-5)
synthetic = model.generate(params, pivot, settings)[:, 0]
checkArray("lagged generate", synthetic, signalF, float, tol=1e-4)
slope = (synthetic[4] - synthetic[3]) / (pivot[4] - pivot[3])
checkTrue("lagged zero initial slope", abs(slope) < 0.6 * 80.0 * 2.0 / tau2)
# the free-amplitude model cannot represent the zero initial slope with the same quality
model, settings = createExpResponse(['F'], nTerms=1, startTime=0.0)
params1 = model.fit(signalF[:, None], pivot, ['F'], settings)
checkTrue("single exponential worse than lagged on S-shaped response", params1['F']['model']['rmse'] > 0.1)
# with one term the lagged model is the first-order response
model, settings = createExpResponse(['A'], model='lagged', startTime=0.0)
params = model.fit(signalA[:, None], pivot, ['A'], settings)
checkFloat("lagged single term tau", params['A']['model']['taus'][0], 12.0, tol=1e-6)
checkFloat("lagged weights sum to one", np.sum(ER.ExponentialResponse._lagWeights([3.0, 10.0, 40.0])), 1.0, tol=1e-12)
checkFails("invalid model", "ExponentialResponse.model: Value \"second\" unrecognized! Expected one of ['exponential', 'lagged'].",
           lambda: createExpResponse(['A'], model='second'), args=[])

print(results)

sys.exit(results["fail"])
