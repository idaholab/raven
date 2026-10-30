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
  Generates the training data for the pooled DMDC test: histories of a known linear system
    x[k+1] = A x[k] + B u[k] + c_s,  y[k] = C x[k]
  (time constants 15 s and 300 s) with different actuator levels, constant terms c_s and lengths.
  The histories are provided on a common time grid: after their end, they are extended with their last
  values (as the HistorySetSync PostProcessor does) and their end time is provided in the 'tEnd' variable.
  The input file of the test (pooled_dmdc.xml) uses the actuator signal of the longest history
  (printed by this script) for the evaluation of the ROM.
"""
import numpy as np

dt = 2.0
taus = np.array([15.0, 300.0])
V = np.array([[1.0, 0.3], [0.5, 1.0]])
A = V @ np.diag(np.exp(-dt / taus)) @ np.linalg.inv(V)
B = np.array([0.8, 0.2])
C = np.array([1.0, 2.0])
nSteps = 300
time = np.arange(nSteps) * dt
# (actuator level, history length in steps, constant term c_s, initial state)
histories = [(1.0, 60, (0.02, -0.01), (0.0, 0.0)),
             (1.2, 75, (0.04, -0.02), (0.5, 0.0)),
             (1.4, 100, (0.06, -0.03), (1.0, 0.2)),
             (1.6, 300, (0.08, -0.04), (1.5, 0.4))]

def simulate(level, offset, x0):
  """
    Simulate a history of the linear system
    @ In, level, float, final level of the actuator (first-order rise with time constant 10 s)
    @ In, offset, tuple, constant term c_s
    @ In, x0, tuple, initial state
    @ Out, (u, x, y), tuple(np.array), actuator, states and output
  """
  u = level * (1.0 - np.exp(-time / 10.0))
  x = np.zeros((nSteps, 2))
  x[0] = x0
  for k in range(nSteps - 1):
    x[k+1] = A @ x[k] + B * u[k] + np.asarray(offset)
  return u, x, x @ C

if __name__ == '__main__':
  with open('Pooled_DMDC.csv', 'w') as main:
    main.write('level,x1_init,x2_init,tEnd,filename\n')
    for h, (level, length, offset, x0) in enumerate(histories):
      main.write(f'{level},{x0[0]},{x0[1]},{time[length-1]:.1f},Pooled_DMDC_{h}.csv\n')
      u, x, y = simulate(level, offset, x0)
      with open(f'Pooled_DMDC_{h}.csv', 'w') as sub:
        sub.write('Time,u,x1,x2,y1\n')
        for k in range(nSteps):
          j = min(k, length - 1)
          sub.write(f'{time[k]:.1f},{u[j]:.12e},{x[j,0]:.12e},{x[j,1]:.12e},{y[j]:.12e}\n')
  print('A =', A.tolist())
  print('B =', B.tolist())
  print('u (longest history) =', ' '.join(f'{v:.12e}' for v in simulate(*histories[-1][0:1], histories[-1][2], histories[-1][3])[0]))
