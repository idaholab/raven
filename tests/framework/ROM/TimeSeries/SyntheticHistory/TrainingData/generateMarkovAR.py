import numpy as np
from generators import toFile


np.random.seed(12345)


def generateMarkovAR(lags: int,
                     regimes: int,
                     transition: np.ndarray,
                     sigma2: np.ndarray,
                     intercept: np.ndarray,
                     pivot: np.ndarray,
                     burn: int = 100) -> np.ndarray:
  """
    Generates a signal according to a Markov-switching autoregressive process.
    @ In, lags, int, number of lags to use
    @ In, regimes, int, number of regimes to use
    @ In, transition, np.ndarray, transition matrix
    @ In, pivot, np.ndarray, time-like array
    @ In, sigma2, float, optional, variance of noise
    @ In, intercept, float, optional, nominal level of signal
    @ In, burn, int, optional, number of burn-in samples to use
    @ Out, signal, np.ndarray, generated signal
  """
  possibleStates = np.arange(regimes, dtype=int)
  arOrder = lags.shape[1]
  n = len(pivot) + burn
  noiseScale = np.sqrt(sigma2)

  states = np.zeros(n, dtype=int)
  signal = np.zeros(n)
  for i in range(arOrder):
    # Start off in state 0 generating a noise values to get things going
    signal[i] = intercept[0] + np.random.normal(loc=0, scale=noiseScale[0])
  for i in range(arOrder, n):
    states[i] = np.random.choice(possibleStates, p=transition[states[i-1]])
    signal[i] += intercept[states[i]] \
                  + np.random.normal(loc=0, scale=noiseScale[states[i]]) \
                  + np.sum([lags[states[i], j] * (signal[i-j-1] - intercept[states[i-j-1]]) for j in range(arOrder)])
  states = states[burn:]
  signal = signal[burn:]
  return signal, states


regimes = 2
lags = np.array([[0.5],
                 [-0.5]])
intercept = np.array([0.0, 0.0])
sigma2 = np.array([2.0, 0.5])
transition = np.array([[0.9, 0.1],
                       [0.1, 0.9]])

pivot = np.arange(1000)
signal, states = generateMarkovAR(lags, regimes, transition, sigma2, intercept, pivot)

out = np.zeros((len(pivot), 2))
out[:, 0] = pivot
out[:, 1] = signal
toFile(out, 'MarkovAR', pivotName='pivot')
