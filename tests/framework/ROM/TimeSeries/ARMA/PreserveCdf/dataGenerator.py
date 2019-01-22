"""
  Generates a synthetic signal for testing ARMA operations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# script options
## number of elements in history
N = 100

## noise scale
noise_scale = 10.0

## make plots?
plot = True

## Fourier
periods = (1, 10, 100)
A = (1, 2, 3)
B = (2, 4, 6)

def fourier(freq, a, b, x):
  """ evaluates Fourier expression for a given frequency, amplitudes, and time series """
  sig = 2.0 * np.pi * freq * x
  return a * np.sin(sig) + b * np.cos(sig)


if plot:
  fig, ax = plt.subplots(figsize=(12, 10))

# generate Fourier signal
t = np.linspace(0, 100, N)
signal = np.zeros(N)
for p, period in enumerate(periods):
  signal += fourier(1.0 / period, A[p], B[p], t)

if plot:
  ax.plot(t, signal, '.-', label='Fourier')

# add some random noise
## pure noise
noise = np.random.rand(N) * noise_scale
## time-dependence
noise *= fourier(0.01, 1.0, 0.0, t)**2
signal += noise

if plot:
  ax.plot(t, noise, '.-', label='Noise')
  ax.plot(t, signal, '.-', label='Full')


if plot:
  ax.set_title('Signal Construction')
  ax.legend(loc=0)
  ax.set_ylabel('Value')
  ax.set_xlabel('Time')

idx = pd.Index(t, name='Time')
df = pd.DataFrame(signal, index=idx, columns=['Signal'])
print df
df.to_csv('signal.csv')

if plot:
  fig.savefig('generated_signal.png')
  plt.show()
