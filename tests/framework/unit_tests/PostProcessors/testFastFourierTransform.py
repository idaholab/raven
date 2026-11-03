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
  This Module performs Unit Tests for the computeFFT function of the FastFourierTransform PostProcessor.
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

from ravenframework.Models.PostProcessors.FastFourierTransform import computeFFT

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
  res = (np.isinf(value) and np.isinf(expected)) or abs(value - expected) <= tol
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

def checkFails(comment, errstr, function, args=None, kwargs=None):
  """
    Checks if expected error occurs
    @ In, comment, string, a comment printed out if it fails
    @ In, errstr, str, expected fail message (contained in the error message)
    @ In, function, method, method to run to test for failure
    @ In, args, list, optional, arguments to pass to function
    @ In, kwargs, dict, optional, keyword arguments to pass to function
    @ Out, res, bool, True if failed as expected
  """
  try:
    function(*(args or []), **(kwargs or {}))
    res, msg = False, 'Function call did not error!'
  except Exception as e:
    res = errstr in str(e)
    msg = f'Unexpected error message: "{e}"'
  if not res:
    print("checking error", comment, '|', msg)
  results["pass" if res else "fail"] += 1
  return res

#####################################
#  Even number of values, on bins   #
#####################################
dt = 0.25
n = 128
t = np.arange(n) * dt
f1, f2 = 8 / (n * dt), 20 / (n * dt)          # frequencies on the FFT bins
signal = 3.0 + 2.0 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t + 0.3)
freq, period, amplitude, phase = computeFFT(signal, t)
checkTrue('number of frequencies (even n, Nyquist included)', len(freq) == n // 2 + 1)
checkFloat('frequency resolution in 1/pivot units', freq[1], 1.0 / (n * dt))
checkFloat('Nyquist frequency', freq[-1], 0.5 / dt)
checkFloat('zero frequency period is infinite', period[0], np.inf)
checkFloat('period is the inverse of the frequency', period[8], 1.0 / f1)
checkFloat('zero frequency amplitude is the mean', amplitude[0], 3.0)
checkFloat('sine amplitude', amplitude[8], 2.0)
checkFloat('sine phase', phase[8], -np.pi / 2)
checkFloat('cosine amplitude', amplitude[20], 0.5)
checkFloat('cosine phase', phase[20], 0.3)
checkTrue('no other components', np.all(np.delete(amplitude, [0, 8, 20]) < 1e-10))
# Nyquist component is not doubled
nyquist = np.cos(np.pi * np.arange(n))
checkFloat('Nyquist amplitude', computeFFT(nyquist, t)[2][-1], 1.0)

#####################################
#  Odd number of values             #
#####################################
n = 129
t = np.arange(n) * dt
fOdd = 10 / (n * dt)
freq, period, amplitude, phase = computeFFT(1.5 * np.sin(2 * np.pi * fOdd * t), t)
checkTrue('number of frequencies (odd n)', len(freq) == (n - 1) // 2 + 1)
checkFloat('sine amplitude (odd n)', amplitude[10], 1.5)
checkFloat('last frequency (odd n)', freq[-1], (n - 1) / 2 / (n * dt))

#####################################
#  Detrending and windows           #
#####################################
n = 256
t = np.arange(n) * dt
fw = 16 / (n * dt)
drifting = 5.0 + 0.4 * t + 1.0 * np.sin(2 * np.pi * fw * t)
_, _, amplitude, _ = computeFFT(drifting, t)
checkTrue('the trend dominates without detrending', amplitude[1] > amplitude[16])
_, _, amplitude, _ = computeFFT(drifting, t, detrend='constant')
checkFloat('constant detrend removes the mean', amplitude[0], 0.0, tol=1e-9)
_, _, amplitude, _ = computeFFT(drifting, t, detrend='linear')
checkFloat('linear detrend: sine amplitude', amplitude[16], 1.0, tol=0.02)
checkTrue('linear detrend: the sine is the largest component', np.argmax(amplitude) == 16)
for window in ['hann', 'hamming', 'blackman']:
  _, _, amplitude, _ = computeFFT(drifting, t, detrend='linear', window=window)
  checkFloat(f'{window} window: sine amplitude corrected for the coherent gain', amplitude[16], 1.0, tol=0.02)
# leakage: off-bin sine, the window reduces the far-away spectral leakage
offBin = np.sin(2 * np.pi * 16.5 / (n * dt) * t)
_, _, rect, _ = computeFFT(offBin, t)
_, _, hann, _ = computeFFT(offBin, t, window='hann')
checkTrue('hann window reduces the leakage far from the peak', hann[60:].max() < 0.1 * rect[60:].max())

#####################################
#  Input errors                     #
#####################################
checkFails('non uniform pivot', 'The pivot values must be uniformly spaced', computeFFT,
           args=[np.ones(4), np.array([0.0, 1.0, 2.5, 3.0])])
checkFails('decreasing pivot', 'The pivot values must be uniformly spaced', computeFFT,
           args=[np.ones(3), np.array([2.0, 1.0, 0.0])])
checkFails('too few values', 'At least 2 values are needed', computeFFT, args=[np.ones(1), np.zeros(1)])
checkFails('unknown detrend', 'Unknown detrend "quadratic"', computeFFT, args=[np.ones(4), np.arange(4.0)],
           kwargs={'detrend': 'quadratic'})
checkFails('unknown window', 'Unknown window "flattop"', computeFFT, args=[np.ones(4), np.arange(4.0)],
           kwargs={'window': 'flattop'})

print(results)

sys.exit(results["fail"])
