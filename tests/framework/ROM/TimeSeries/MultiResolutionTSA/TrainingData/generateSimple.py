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
  Generates training data for a FilterBank multiresolution test, applying Fourier signals and
  ARMA-style signals at different resolutions.
"""
import numpy as np
import pandas as pd
import os.path as path
from generators import fourier, arma

plot = False

dirname = path.dirname(path.abspath(__file__))

# using a signal with length 2**j where j is a positive integer
pivot  = np.arange(0,2**8)
signal0 = np.zeros(len(pivot))
signal1 = np.zeros(len(pivot))

# adding sinusoidal trend for *one* signal
amps      = [    15,  5]
periods   = [    12,  20]
phases    = [np.pi, np.pi/3]
intercept = 10
f = fourier(amps, periods, phases, pivot, mean=intercept)

# adding ARMA-style noise with different resolutions for both signals
## SIGNAL0 with P=1 and Q=1
l = 1
slags = [0.4]
nlags = [0.3]
a0, _ = arma(slags, nlags, pivot[::l], plot=False)
a0 = np.repeat(a0,l)

l = 2
slags = [-0.5]
nlags = [0.1]
a1, _ = arma(slags, nlags, pivot[::l], plot=False)
a1 = np.repeat(a1,l)

l = 4
slags = [0.1]
nlags = [-0.25]
a2, _ = arma(slags, nlags, pivot[::l], plot=False)
a2 = np.repeat(a2,l)

l = 8
slags = [-0.4]
nlags = [0.15]
a3, _ = arma(slags, nlags, pivot[::l], plot=False)
a3 = np.repeat(a3,l)

signal0 += f
signal0 += a0
signal0 += a1
signal0 += a2
signal0 += a3

# adding ARMA-style noise with different resolutions for both signals
## SIGNAL1 with P=1 and Q=2
l = 1
slags = [0.4]
nlags = [0.3, 0.2]
a0, _ = arma(slags, nlags, pivot[::l], plot=False)
a0 = np.repeat(a0,l)

l = 2
slags = [-0.5]
nlags = [0.1, -0.05]
a1, _ = arma(slags, nlags, pivot[::l], plot=False)
a1 = np.repeat(a1,l)

l = 4
slags = [0.1]
nlags = [-0.25, 0.3]
a2, _ = arma(slags, nlags, pivot[::l], plot=False)
a2 = np.repeat(a2,l)

l = 8
slags = [-0.4]
nlags = [0.15, 0.2]
a3, _ = arma(slags, nlags, pivot[::l], plot=False)
a3 = np.repeat(a3,l)

# no fourier
signal1 += a0
signal1 += a1
signal1 += a2
signal1 += a3

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, signal0, '.-', label='0')
  ax.plot(pivot, signal1, '.-', label='1')
  ax.legend()
  plt.show()

out = np.zeros((len(pivot), 3))
out[:, 0] = pivot
out[:, 1] = signal0
out[:, 2] = signal1

# Write signals to file using pandas DataFrames
df = pd.DataFrame.from_dict({'pivot':   out[:, 0],
                             'signal0': out[:, 1],
                             'signal1': out[:, 2]})
df_filepath = path.join(dirname, 'simpleMR_A.csv')
df.to_csv(df_filepath, index=False)

# Create pointer CSV file
pointer = pd.DataFrame.from_dict({
    'scaling': [1],
    'macro':   [1],
    'filename': ['simpleMR_A.csv']
})
pointer_filepath = path.join(dirname, 'simpleMR.csv')
pointer.to_csv(pointer_filepath, index=False)
