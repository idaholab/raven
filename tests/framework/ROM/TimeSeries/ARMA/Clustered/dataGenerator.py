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
  Generates a synthetic signal for testing ARMA operations
  Specific for this case, creates a time-dependent noise in addtion to Fourier signals
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PLAN: A B A B C A A B B C
## Fourier Full: (1/100, 1, 1, t)
## Fourier A: (1/5, 2, 0, t)
## Fourier B: (1/5, 0, 2, t)
## Fourier C: (1/3, 2, 2, t)
plan = ['A']*10 + ['B']*10 + ['A']*10 + ['B']*10 + ['C']*10 + \
       ['A']*10 + ['A']*10 + ['B']*10 + ['B']*10 + ['C']*10
plan = np.array(plan)
maskA = plan == 'A'
maskB = plan == 'B'
maskC = plan == 'C'

# script options
## number of elements in history
N = 100

## noise scale
noiseScale = 1.0

## make plots?
plot = True

def fourier(freq, a, b, x):
  """
    evaluates Fourier expression for a given frequency, amplitudes, and time series
    @ In, freq, float, frequency to evaluate
    @ In, a, float, sine coefficient
    @ In, b, float, cosine coefficient
    @ In, x, np.ndarray(float), independent parameter at which to evaluate
    @ Out, fourier, Fourier signal given prescribed parameters
  """
  sig = 2.0 * np.pi * freq * x
  return a * np.sin(sig) + b * np.cos(sig)

## Fourier
t = np.linspace(0, 100, N)
signal = np.zeros(N)
signal += fourier(1./100., 1., 1., t)
signal[maskA] += fourier(1./5., 2., 0., t[maskA])
signal[maskB] += fourier(1./5., 0., 1., t[maskB])
signal[maskC] += fourier(1./3., 3., 3., t[maskC])


if plot:
  fig, ax = plt.subplots(figsize=(12, 10))
  ax.plot(t, signal, '.-', label='Fourier')

# add some random noise
## pure noise
noise = np.random.rand(N) * noiseScale
## time-dependence
#noise *= fourier(1./5., 1.0, 0.0, t)**2
signal += noise

if plot:
  ax.plot(t, noise, '.-', label='Noise')
  ax.plot(t, signal, '.-', label='Full')


if plot:
  ax.set_title('Signal Construction')
  ax.legend(loc=0)
  ax.set_ylabel('Value')
  ax.set_xlabel('Time')

idx = pd.Index(t, name='Time')
df = pd.DataFrame(signal, index=idx, columns=['Signal'])
df.to_csv('signal.csv')

if plot:
  fig.savefig('generated_signal.png')
  plt.show()
