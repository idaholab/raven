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
  This Module performs Unit Tests for the TSA.Fourier class.
  It can not be considered part of the active code but of the regression test system
"""
import os
import sys
import copy
import numpy as np

# add RAVEN to path
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4 + ['framework'])))
if frameworkDir not in sys.path:
  sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)

from utils import xmlUtils

from TSA import Fourier

plot = False

print('Module undergoing testing:')
print(Fourier)
print('')

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
def createFourierXML(targets, periods):
  xml = xmlUtils.newNode('Fourier', attrib={'target':','.join(targets)})
  xml.append(xmlUtils.newNode('periods', text=','.join(str(k) for k in periods)))
  return xml

def createFromXML(xml):
  fourier = Fourier()
  inputSpec = Fourier.getInputSpecification()()
  inputSpec.parseNode(xml)
  fourier.handleInput(inputSpec)
  return fourier

def createFourier(targets, periods):
  xml = createFourierXML(targets, periods)
  fourier = createFromXML(xml)
  return fourier

def createFourierSignal(amps, periods, phases, pivot, intercept=0, plot=False):
  if plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
  signal = np.zeros(len(pivot)) + intercept
  for k, period in enumerate(periods):
    new = amps[k] * np.sin(2 * np.pi / period * pivot + phases[k])
    if plot:
      ax.plot(pivot, new, ':')
    signal += new
  if plot:
    ax.plot(pivot, signal, 'k-')
    plt.show()
  return signal


###################
#  Simple         #
###################
# generate signal
targets = ['A', 'B', 'C']
pivot = np.arange(100) / 10.
periods = [2, 5, 10]
amps = [0.5, 1, 2]

phasesA = [0, np.pi, 0]
signalA = createFourierSignal(amps, periods, phasesA, pivot, plot=plot)

phasesB = [np.pi, 0, np.pi/4]
signalB = createFourierSignal(amps, periods, phasesB, pivot, plot=plot)

phasesC = [np.pi, np.pi/4, -np.pi/4]
interceptC = 2
signalC = createFourierSignal(amps, periods, phasesC, pivot, intercept=interceptC, plot=plot)

signals = np.zeros((len(pivot), 3))
signals[:, 0] = signalA
signals[:, 1] = signalB
signals[:, 2] = signalC

fourier = createFourier(targets, periods)
settings = {'periods': periods}
params = fourier.characterize(signals, pivot, targets, settings)

checkTrue("fourier can generate", fourier.canGenerate())
checkTrue("fourier can characterize", fourier.canCharacterize())

# intercepts
checkFloat('Signal A intercept', params['A']['intercept'], 0)
checkFloat('Signal B intercept', params['B']['intercept'], 0)
checkFloat('Signal C intercept', params['C']['intercept'], interceptC)

# amplitudes
checkFloat('Signal A period 0 amplitude', params['A']['coeffs'][periods[0]]['amplitude'], amps[0])
checkFloat('Signal A period 1 amplitude', params['A']['coeffs'][periods[1]]['amplitude'], amps[1])
checkFloat('Signal A period 2 amplitude', params['A']['coeffs'][periods[2]]['amplitude'], amps[2])

checkFloat('Signal B period 0 amplitude', params['B']['coeffs'][periods[0]]['amplitude'], amps[0])
checkFloat('Signal B period 1 amplitude', params['B']['coeffs'][periods[1]]['amplitude'], amps[1])
checkFloat('Signal B period 2 amplitude', params['B']['coeffs'][periods[2]]['amplitude'], amps[2])

checkFloat('Signal C period 0 amplitude', params['C']['coeffs'][periods[0]]['amplitude'], amps[0])
checkFloat('Signal C period 1 amplitude', params['C']['coeffs'][periods[1]]['amplitude'], amps[1])
checkFloat('Signal C period 2 amplitude', params['C']['coeffs'][periods[2]]['amplitude'], amps[2])

# phases
# check absolute value of phase pi since -pi and pi are often converged on separately
checkFloat('Signal A period 0 phase',     params['A']['coeffs'][periods[0]]['phase'] , phasesA[0])
checkFloat('Signal A period 1 phase', abs(params['A']['coeffs'][periods[1]]['phase']), phasesA[1])
checkFloat('Signal A period 2 phase',     params['A']['coeffs'][periods[2]]['phase'] , phasesA[2])

checkFloat('Signal B period 0 phase', abs(params['B']['coeffs'][periods[0]]['phase']), phasesB[0])
checkFloat('Signal B period 1 phase',     params['B']['coeffs'][periods[1]]['phase'] , phasesB[1])
checkFloat('Signal B period 2 phase',     params['B']['coeffs'][periods[2]]['phase'] , phasesB[2])

checkFloat('Signal C period 0 phase', abs(params['C']['coeffs'][periods[0]]['phase']), phasesC[0])
checkFloat('Signal C period 1 phase',     params['C']['coeffs'][periods[1]]['phase'] , phasesC[1])
checkFloat('Signal C period 2 phase',     params['C']['coeffs'][periods[2]]['phase'] , phasesC[2])

# residual
## add constant to training, make sure we get constant back
const = 42.0
residSig = signals + const
resid = fourier.getResidual(residSig, params, pivot, settings)
checkFloat('Residual check', (resid-const).sum(), 0)

# recreate signals
res = fourier.generate(params, pivot, None)
for tg, target in enumerate(targets):
  checkArray(f'Signal {target} replication', res[:, tg], signals[:, tg], float)



##### now redo with non-simultaneous fitting
params = fourier.characterize(signals, pivot, targets, settings, simultFit=False)
# intercepts
checkFloat('Signal A intercept', params['A']['intercept'], 0)
checkFloat('Signal B intercept', params['B']['intercept'], 0)
checkFloat('Signal C intercept', params['C']['intercept'], interceptC)

# amplitudes
checkFloat('Signal A period 0 amplitude', params['A']['coeffs'][periods[0]]['amplitude'], amps[0])
checkFloat('Signal A period 1 amplitude', params['A']['coeffs'][periods[1]]['amplitude'], amps[1])
checkFloat('Signal A period 2 amplitude', params['A']['coeffs'][periods[2]]['amplitude'], amps[2])

checkFloat('Signal B period 0 amplitude', params['B']['coeffs'][periods[0]]['amplitude'], amps[0])
checkFloat('Signal B period 1 amplitude', params['B']['coeffs'][periods[1]]['amplitude'], amps[1])
checkFloat('Signal B period 2 amplitude', params['B']['coeffs'][periods[2]]['amplitude'], amps[2])

checkFloat('Signal C period 0 amplitude', params['C']['coeffs'][periods[0]]['amplitude'], amps[0])
checkFloat('Signal C period 1 amplitude', params['C']['coeffs'][periods[1]]['amplitude'], amps[1])
checkFloat('Signal C period 2 amplitude', params['C']['coeffs'][periods[2]]['amplitude'], amps[2])

# phases
# check absolute value of phase pi since -pi and pi are often converged on separately
checkFloat('Signal A period 0 phase',     params['A']['coeffs'][periods[0]]['phase'] , phasesA[0])
checkFloat('Signal A period 1 phase', abs(params['A']['coeffs'][periods[1]]['phase']), phasesA[1])
checkFloat('Signal A period 2 phase',     params['A']['coeffs'][periods[2]]['phase'] , phasesA[2])

checkFloat('Signal B period 0 phase', abs(params['B']['coeffs'][periods[0]]['phase']), phasesB[0])
checkFloat('Signal B period 1 phase',     params['B']['coeffs'][periods[1]]['phase'] , phasesB[1])
checkFloat('Signal B period 2 phase',     params['B']['coeffs'][periods[2]]['phase'] , phasesB[2])

checkFloat('Signal C period 0 phase', abs(params['C']['coeffs'][periods[0]]['phase']), phasesC[0])
checkFloat('Signal C period 1 phase',     params['C']['coeffs'][periods[1]]['phase'] , phasesC[1])
checkFloat('Signal C period 2 phase',     params['C']['coeffs'][periods[2]]['phase'] , phasesC[2])

# recreate signals
res = fourier.generate(params, pivot, settings)
for tg, target in enumerate(targets):
  checkArray(f'Signal {target} replication', res[:, tg], signals[:, tg], float)

# check residual
# -> generate random noise to add to signal, then check it is returned in residual
r = np.random.rand(pivot.size, len(targets))
new = r + signals
res = fourier.getResidual(new, params, pivot, None)
for tg, target in enumerate(targets):
  checkArray(f'Signal {target} residual', res[:, tg], r[:, tg], float)




print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.unit_tests.TSA.Fourier</name>
    <author>talbpaul</author>
    <created>2021-01-05</created>
    <classesTested>TSA.Fourier</classesTested>
    <description>
       This test is a Unit Test for the Fourier TimeSeriesAnalyzer classes.
    </description>
  </TestInfo>
"""
