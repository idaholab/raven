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
from generators import arma, toFile

np.random.seed(12345)

plot = True

pivot = np.arange(1000) / 10

slags = [0.5]
nlags = []
noise0 = np.random.normal(loc=0, scale=0.5, size=(len(pivot),))
signal0, _ = arma(slags, nlags, pivot, noise=noise0, plot=plot)
slags = [-0.5]
nlags = []
noise1 = np.random.normal(loc=0, scale=1.5, size=(len(pivot),))
signal1, _ = arma(slags, nlags, pivot, noise=noise1, plot=plot)

# Create two clusters by swapping the noise signals halfway through the series
split = int(len(pivot) // 2)
s0 = np.append(signal0[:split], signal1[split:])
s1 = np.append(signal1[:split], signal0[split:])

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, s0, '.-', label='0')
  ax.plot(pivot, s1, '.-', label='1')
  ax.legend()
  plt.show()


out = np.zeros((len(pivot), 3))
out[:, 0] = pivot
out[:, 1] = s0
out[:, 2] = s1
toFile(out, 'Clustered_A')
