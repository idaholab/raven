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
from generators import fourier, arma, toFile

plot = True

pivot = np.arange(1000)/10.

amps = [8, 10, 12]
periods = [2, 5, 10]
phases = [0, np.pi/4, np.pi]
intercept = 42
f = fourier(amps, periods, phases, pivot, mean=intercept)

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
a0, _ = arma(slags, nlags, pivot, plot=False)
slags = [0.5, 0.3]
nlags = [0.1, 0.05, 0.01]
a1, _ = arma(slags, nlags, pivot, plot=False)

s0 = f + a0
s1 = f + a1

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, s0, '.-', label='0')
  ax.plot(pivot, s1, '.-', label='1')
  ax.legend()
  plt.show()


out = np.zeros((len(pivot), 3))
out[:, 0] = pivot
out[:, 1] = f + a0
out[:, 2] = f + a1
toFile(out, 'FourierARMA_A')
