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
  Generates step-response signals (sums of exponential modes plus noise) for the
  TSACharacterizer ExponentialResponse test
"""
import numpy as np

rng = np.random.default_rng(42)
rngLag = np.random.default_rng(7)
pivot = np.arange(-4.0, 400.0, 2.0)
# (steady, fast amplitude, fast tau, slow amplitude, slow tau) for temperature and power
cases = {1: [(1000.0, -600.0, 10.0, -300.0, 120.0), (5.0, 2.0, 6.0, 1.0, 80.0)],
         2: [(1100.0, -700.0, 12.0, -320.0, 150.0), (6.0, 2.5, 5.0, 1.2, 90.0)],
         3: [(1200.0, -750.0, 15.0, -350.0, 180.0), (7.0, 3.0, 4.0, 1.5, 100.0)]}
# (initial, steady, tau1, tau2) of two first-order lags in series for the "lagged" signal
lagCases = {1: (20.0, 100.0, 8.0, 120.0), 2: (20.0, 110.0, 9.0, 150.0), 3: (20.0, 120.0, 10.0, 180.0)}

def response(steady, b1, tau1, b2, tau2):
  """
    Two-mode step response, held at its initial value before the step at t=0
    @ In, steady, float, steady value
    @ In, b1, float, fast mode amplitude
    @ In, tau1, float, fast mode time constant
    @ In, b2, float, slow mode amplitude
    @ In, tau2, float, slow mode time constant
    @ Out, response, np.array, signal
  """
  t = np.maximum(pivot, 0.0)
  return steady + b1 * np.exp(-t / tau1) + b2 * np.exp(-t / tau2)

def laggedResponse(initial, steady, tau1, tau2):
  """
    Step response of two first-order lags in series (zero initial slope), held before the step at t=0
    @ In, initial, float, initial value
    @ In, steady, float, steady value
    @ In, tau1, float, first time constant
    @ In, tau2, float, second time constant
    @ Out, response, np.array, signal
  """
  t = np.maximum(pivot, 0.0)
  return steady + (initial - steady) * (tau1 * np.exp(-t / tau1) - tau2 * np.exp(-t / tau2)) / (tau1 - tau2)

with open('expResponse.csv', 'w') as main:
  main.write('case,filename\n')
  for c, (temp, power) in cases.items():
    main.write(f'{c},expResponse_{c-1}.csv\n')
    temperature = response(*temp) + rng.normal(0, 1.0, len(pivot))
    pw = response(*power) + rng.normal(0, 0.005, len(pivot))
    lagged = laggedResponse(*lagCases[c]) + rngLag.normal(0, 0.1, len(pivot))
    with open(f'expResponse_{c-1}.csv', 'w') as sub:
      sub.write('pivot,temperature,power,lagged\n')
      for p, tv, pv, lv in zip(pivot, temperature, pw, lagged):
        sub.write(f'{p:.1f},{tv:.6f},{pv:.6f},{lv:.6f}\n')
