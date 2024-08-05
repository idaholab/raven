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
import pandas as pd
from generators import fourier, arma

plot = False

pivot = np.arange(1000)


# macroStep 0
amps = [16, 8, 10, 12]
periods = [75, 125, 250, 500]
phases = [0, np.pi/4, np.pi/2, np.pi]
intercept = 42
f0 = fourier(amps, periods, phases, pivot, mean=intercept)

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
a0, _ = arma(slags, nlags, pivot, plot=False)

s0 = f0 + a0

# macroStep 1
amps = [4, 21, 7, 35]
periods = [75, 125, 250, 500]
phases = [0, np.pi/4, np.pi/2, np.pi]
intercept = 42
f1 = fourier(amps, periods, phases, pivot, mean=intercept)

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
a1, _ = arma(slags, nlags, pivot, plot=False)

s1 = f1 + a1

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, s0, '.-', label='0')
  ax.plot(pivot, s1, '.-', label='1')
  ax.legend()
  plt.show()


# Write signals to file using pandas DataFrames
df0 = pd.DataFrame({'seconds': pivot, 'signal0': s0})
df0.to_csv('multiYear_A.csv', index=False)

# Write signals to file using pandas DataFrames
df1 = pd.DataFrame({'seconds': pivot, 'signal0': s1})
df1.to_csv('multiYear_B.csv', index=False)

# Create pointer CSV file
pointer = pd.DataFrame({
    'scaling': [1, 1],
    'macro': [1, 2], # interpolation still needed... ?
    'filename': ['multiYear_A.csv', 'multiYear_B.csv']
})
pointer.to_csv('multiYear.csv', index=False)
