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
  Generates the signals for the FastFourierTransform PostProcessor tests: two histories sampled every
  0.5 s (64 values and 63 values, to test even and odd lengths) of
    x = offset + a1 sin(2 pi f1 t) + a2 cos(2 pi f2 t)     (frequencies on the FFT bins of the 64-value history)
    y = drift t + a3 sin(2 pi f3 t + phi)                  (linearly drifting signal)
"""
import numpy as np

dt = 0.5
# (number of values, offset, a1, f1, a2, f2, drift, a3, f3, phi)
histories = [(64, 2.0, 1.5, 4/32, 0.5, 10/32, 0.3, 1.0, 5/32, 0.7),
             (63, 1.0, 0.8, 4/32, 0.2, 10/32, -0.2, 0.6, 5/32, -0.4)]

if __name__ == '__main__':
  with open('signals.csv', 'w') as main:
    main.write('case,filename\n')
    for h, (n, offset, a1, f1, a2, f2, drift, a3, f3, phi) in enumerate(histories):
      main.write(f'{h+1},signals_{h}.csv\n')
      t = np.arange(n) * dt
      x = offset + a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.cos(2 * np.pi * f2 * t)
      y = drift * t + a3 * np.sin(2 * np.pi * f3 * t + phi)
      with open(f'signals_{h}.csv', 'w') as sub:
        sub.write('time,x,y\n')
        for k in range(n):
          sub.write(f'{t[k]:.2f},{x[k]:.15e},{y[k]:.15e}\n')
