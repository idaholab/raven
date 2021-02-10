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
  Generates training data in the form of Fourier signals.
"""
import numpy as np

plot = True

def createARMASignal(slags, nlags, pivot, noise=None, intercept=0, plot=False):
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

##########
# ARMA A #
##########
# normally-distributed noise with 0 loc, 1 scale

seconds = np.arange(1000) / 10. # 0 to 100 in 0.1 increments

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
signal0, _ = createARMASignal(slags, nlags, seconds, plot=plot)
slags = [0.5, 0.3]
nlags = [0.1, 0.05, 0.01]
signal1, _ = createARMASignal(slags, nlags, seconds, plot=plot)

if plot:
  import matplotlib.pyplot as plt
  plt.show()

out = np.zeros((len(seconds), 3))
out[:, 0] = seconds
out[:, 1] = signal0
out[:, 2] = signal1
# out[:, 2] = signal1
fname = 'ARMA_A'
subname = f'{fname}_0.csv'
np.savetxt(subname, out, delimiter=',', header='seconds,signal0,signal1', comments='')
with open(f'{fname}.csv', 'w') as f:
  f.writelines('scaling,filename\n')
  f.writelines(f'1,{subname}\n')
