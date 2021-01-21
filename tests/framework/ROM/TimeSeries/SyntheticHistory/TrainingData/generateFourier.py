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

#############
# Fourier A #
#############
seconds = np.arange(100) / 10. # 0 to 10 in 0.1 increments

periods = [2, 5, 10]
amps = [0.5, 1, 2]
phases = [0, np.pi/4, np.pi]
intercept = 42
signal0 = np.zeros(len(seconds))
for k, period in enumerate(periods):
  signal0 += amps[k] * np.sin(2*np.pi / period * seconds + phases[k])

periods = [3]
amps = [2]
phases = [np.pi]
intercept = 1
signal1 = np.zeros(len(seconds))
for k, period in enumerate(periods):
  signal1 += amps[k] * np.sin(2*np.pi / period * seconds + phases[k])

out = np.zeros((len(seconds), 3))
out[:, 0] = seconds
out[:, 1] = signal0
out[:, 2] = signal1
fname = 'FourierA'
subname = f'{fname}_0.csv'
np.savetxt(subname, out, delimiter=',', header='seconds,signal1,signal2', comments='')
with open(f'{fname}.csv', 'w') as f:
  f.writelines('scaling,filename\n')
  f.writelines(f'1,{subname}\n')
