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
  This Module performs Unit Tests for the TSA.Fourier class.
  It can not be considered part of the active code but of the regression test system
"""
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api

def createARMASignal(slags, nlags, pivot, noise=None, intercept=0, plot=False):
  if plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
  signal = np.zeros(len(pivot)) + intercept
  if noise is None:
    noise = np.random.normal(loc=0, scale=1, size=len(pivot))
  signal += noise
  # moving average: random noise lag
  for q, theta in enumerate(nlags):
    signal[q+1:] += theta * noise[:-(q+1)]
  # autoregressive: signal lag
  for t, time in enumerate(pivot):
    for p, phi in enumerate(slags):
      if t > p:
        signal[t] += phi * signal[t - p - 1]
  if plot:
    ax.plot(pivot, noise, 'k:')
    ax.plot(pivot, signal, 'g.-')
    plt.show()
  return signal, noise

###################
#  Simple         #
###################
# generate signal
# X_t = c + \epsilon_t                           # constant and white noise
#       + \sum_{i=1}^P \phi_i * X_{t-i}          # signal lag, autoregressive term
#       + \sum_{i=1}^Q \theta_i * \epsilon_{t-i} # noise lag, moving average term
slags = [0.4, 0.2]      # \phi_1, \phi_2
nlags = [0.3, 0.2, 0.1] # \theta_1, \theta_2, \theta_3
order = [len(slags), 0, len(nlags)] # ARIMA(P, d, Q)

hist_len = []
errs_s1 = []
errs_s2 = []
errs_n1 = []
errs_n2 = []
errs_n3 = []
errs_l2 = []
errs_max = []
# note:
# the MLE convergence of the ARIMA fit on the training signal
# can fail randomly, frequently for the short histories. This
# changes each time I run this; I have not tried fixing the
# numpy seed to see if it's an under-the-hood regressor behavior.
# -> when this failure occurs, there is a warning thrown to stdout,
# and the errors are unusually large.
for i in range(4, 17):
  print(f'Starting 2^{i} ...')
  N = int(2**i)
  pivot = np.linspace(0, 100, N)
  hist_len.append(N)
  # fitting signal, noise
  signal, noise = createARMASignal(slags, nlags, pivot)
  t_arima = statsmodels.tsa.arima.model.ARIMA(signal, order=order)
  res = t_arima.fit()
  t_ar = res.arparams
  errs_s1.append(abs(slags[0] - t_ar[0])/slags[0])
  errs_s2.append(abs(slags[1] - t_ar[1])/slags[1])
  t_ma = res.maparams
  errs_n1.append(abs(nlags[0] - t_ma[0])/nlags[0])
  errs_n2.append(abs(nlags[1] - t_ma[1])/nlags[1])
  errs_n3.append(abs(nlags[2] - t_ma[2])/nlags[2])
  # recreate
  synth, _ = createARMASignal(t_ar, t_ma, pivot, noise=noise)
  diff = synth - signal
  errs_l2.append(np.linalg.norm(diff) / N)
  errs_max.append(np.max(np.abs(diff)))

fig, ax = plt.subplots()
ax.semilogx(hist_len, errs_l2, 'k.:', label='signal error L2/N', alpha=0.7)
ax.semilogx(hist_len, errs_max, 'kx:', label='signal error max', alpha=0.7)
ax.semilogx(hist_len, errs_s1, 'g-', marker='o', label=r'err $\phi_1$ (AR)', alpha=0.7)
ax.semilogx(hist_len, errs_s2, 'g-', marker='+', label=r'err $\phi_2$ (AR)', alpha=0.7)
ax.semilogx(hist_len, errs_n1, 'b-', marker='o', label=r'err $\theta_1$ (MA)', alpha=0.7)
ax.semilogx(hist_len, errs_n2, 'b-', marker='+', label=r'err $\theta_2$ (MA)', alpha=0.7)
ax.semilogx(hist_len, errs_n3, 'b-', marker='x', label=r'err $\theta_3$ (MA)', alpha=0.7)
ax.text(100, 6, rf'$\phi = {slags}, \theta = {nlags}$')
ax.legend()
ax.set_xlabel('History Length')
ax.set_ylabel('Rel. Error')
fig.savefig('ARMA_converge.pdf')


plt.show()
