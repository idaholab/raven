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
  ARMA-style signals at different resolutions. Additionally generates multiyear signals.
"""
import numpy as np
import pandas as pd
import os.path as path
from generators import fourier, arma

plot = False

dirname = path.dirname(path.abspath(__file__))

# YEAR 0 signal
pivot  = np.arange(0,2**6)
year0 = np.zeros(len(pivot))
year1 = np.zeros(len(pivot))

amps      = [      8,  20]
periods   = [      4,  15]
phases    = [np.pi/6,  np.pi/2]
intercept = 3
f = fourier(amps, periods, phases, pivot, mean=intercept)

l = 1
slags = [-0.05, 0.1]
nlags = [0.05, 0.25, -0.12]
a0, _ = arma(slags, nlags, pivot[::l], plot=False)
a0 = np.repeat(a0,l)

l = 2
slags = [0.2, -0.03]
nlags = [-0.15, -0.2, 0.1]
a1, _ = arma(slags, nlags, pivot[::l], plot=False)
a1    = np.repeat(a1,l)

year0 += f
year0 += a0
year0 += a1

# YEAR 0 signal
year1 = np.zeros(len(pivot))
phases    = [np.pi/3,  np.pi/4]
intercept = 5
f = fourier(amps, periods, phases, pivot, mean=intercept)

l = 1
slags = [0.15, 0.03]
nlags = [-0.15, -0.2, 0.1]
a0, _ = arma(slags, nlags, pivot[::l], plot=False)
a0 = np.repeat(a0,l)

l = 2
slags = [-0.05, 0.2]
nlags = [0.05, 0.25, -0.12]
a1, _ = arma(slags, nlags, pivot[::l], plot=False)
a1    = np.repeat(a1,l)

year1 += f
year1 += a0
year1 += a1

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, year0, '.-', label='0')
  ax.plot(pivot, year1, '.-', label='1')
  ax.legend()
  plt.show()


# Write signals to file using pandas DataFrames
df0 = pd.DataFrame({'seconds': pivot, 'signal': year0})
df_filepath = path.join(dirname, 'multiYear_A.csv')
df0.to_csv(df_filepath, index=False)

# Write signals to file using pandas DataFrames
df1 = pd.DataFrame({'seconds': pivot, 'signal': year1})
df_filepath = path.join(dirname, 'multiYear_B.csv')
df1.to_csv(df_filepath, index=False)

# Create pointer CSV file
pointer = pd.DataFrame({
    'scaling': [1, 1],
    'macro':   [1, 2], # interpolation still needed... ?
    'filename': ['multiYear_A.csv', 'multiYear_B.csv']
})
pointer_filepath = path.join(dirname, 'multiYear.csv')
pointer.to_csv(pointer_filepath, index=False)
