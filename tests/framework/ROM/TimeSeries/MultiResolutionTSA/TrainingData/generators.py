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
  Generates signals using predefined algorithms
"""
import numpy as np

def fourier(amps, periods, phases, pivot, mean=0):
  """
    Generates a signal using Fourier properties.
    @ In, amps, np.array, amplitudes of waves
    @ In, periods, np.array, periods to use
    @ In, phases, np.array, phase offsets to use
    @ In, pivot, np.array, time-like parameter
    @ In, mean, float, offset value
    @ Out, signal, np.array, generated signal
  """
  signal = np.zeros(len(pivot)) + mean
  for k, period in enumerate(periods):
    signal += amps[k] * np.sin(2 * np.pi / period * pivot + phases[k])
  return signal

def arma(slags, nlags, pivot, noise=None, intercept=0, plot=False):
  """
    Generates a signal using ARMA properties.
    @ In, slags, list, signal lag coefficients (aka AR coeffs, phi)
    @ In, nlags, list, noise lag coefficients (aka MA coeffs, theta)
    @ In, pivot, np.array, time-like array
    @ In, noise, np.array, optional, instead of sampling random noise will use this if provided
    @ In, intercept, float, optional, nominal level of signal
    @ In, plot, bool, optional, if True then produce a plot of generated signal
    @ Out, signal, np.array, generated signal
    @ Out, noise, np.array, noise signal used in generation (provided or sampled)
  """
  if plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
  signal = np.zeros(len(pivot)) + intercept
  if noise is None:
    noise = np.random.normal(loc=0, scale=1, size=len(pivot))
  signal += noise
  # moving average: random noise lag
  for q, theta in enumerate(nlags):
    signal[q+1:] += theta * noise[:-(q+1)]
  # autoregressive: signal lag
  for t, time in enumerate(pivot):
    for p, phi in enumerate(slags):
      if t > p:
        signal[t] += phi * signal[t - p - 1]
  if plot:
    ax.plot(pivot, noise, 'k:')
    ax.plot(pivot, signal, 'g.-')
    plt.show()
  return signal, noise

def toFile(signal, baseName, targets=None, pivotName=None):
  """
    writes signals to RAVEN CSV files
    @ In, signal, np.ndarray, signals shaped (time, targets+1) with pivot as first target
    @ In, baseName, str, base filename
    @ In, targets, list(str), optional, target names
    @ In, pivotName, str, optional, pivot parameter (time-like) name
    @ Ou, None
  """
  if len(signal.shape) < 2:
    signal.shape = (signal.size, 1)
  if targets is None:
    targets = [f'signal{i}' for i in range(signal.shape[1] - 1)]
  if pivotName is None:
    pivotName = 'pivot'
  subname = f'{baseName}_0.csv'
  np.savetxt(subname, signal, delimiter=',', header=','.join([pivotName] + targets), comments='')
  with open(f'{baseName}.csv', 'w') as f:
    f.writelines('scaling,filename\n')
    f.writelines(f'1,{subname}\n')
