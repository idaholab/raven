import numpy as np
from generators import toFile

np.random.seed(12345)

pivot = np.arange(200)
burnin = 100
N = burnin + len(pivot)

# 2 correlated variables with AR(1) noise
mean = np.array([2, -2])
cov = np.array([[1.0, 0.8],
                [0.8, 1.0]])
ar = np.array([[0.5, 0.3],
            [-0.5, 0.8]])

signal = np.zeros((N, len(mean)))
noise = np.random.multivariate_normal(np.zeros(2), cov, size=N)

signal[0] = mean + noise[0]
for t in range(1, N):
  signal[t] = mean + ar @ signal[t-1] + noise[t]

signal = signal[burnin:]

# toss in a couple zeros
# A ZeroFilter is applied in the ROM so we can test missing value handling
signal[10, 0] = 0
signal[20, 1] = 0

signal = np.hstack((pivot.reshape(-1, 1), signal))

toFile(signal, 'VARMA', pivotName='seconds')
