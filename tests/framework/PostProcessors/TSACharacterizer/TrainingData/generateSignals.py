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
  Generates training data in the form of Fourier/ARMA/etc signals.
"""
import numpy as np
from generators import fourier, arma, toFile

plot = False
outs = []

pivot = np.arange(1000)/10.

################################################
#
# sample 0
#
amps = [2, 3, 4]
periods = [2, 5, 10]
phases = [0, np.pi/4, np.pi]
intercept = 42
f = fourier(amps, periods, phases, pivot, mean=intercept)

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
a0, _ = arma(slags, nlags, pivot, plot=False)

s0 = f + a0

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, f, '.-', label='f')
  ax.plot(pivot, a0, '.-', label='a0')
  ax.plot(pivot, s0, '.-', label='f+a0')
  ax.legend()
  plt.show()

out = np.zeros((len(pivot), 4))
out[:, 0] = pivot
out[:, 1] = f
out[:, 2] = a0
out[:, 3] = f + a0
outs.append(out)

################################################
#
# sample 1
#
amps = [3, 1, 2]
periods = [2, 5, 10]
phases = [0, np.pi/4, np.pi]
intercept = 20
f = fourier(amps, periods, phases, pivot, mean=intercept)

slags = [0.4, 0.4]
nlags = [0.4, 0.3, 0.2]
a0, _ = arma(slags, nlags, pivot, plot=False)

s0 = f + a0
out = np.zeros((len(pivot), 4))
out[:, 0] = pivot
out[:, 1] = f
out[:, 2] = a0
out[:, 3] = f + a0
outs.append(out)

toFile(outs, 'signals', targets=['signal_f', 'signal_a', 'signal_fa'])
