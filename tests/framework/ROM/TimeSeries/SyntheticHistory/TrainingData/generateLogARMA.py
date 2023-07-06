import numpy as np
from generators import arma, toFile

##########
# ARMA A #
##########
# normally-distributed noise with 0 loc, 1 scale

seconds = np.arange(100)

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
signal0, _ = arma(slags, nlags, seconds)
slags = [0.5, 0.3]
nlags = [0.1, 0.05, 0.01]
signal1, _ = arma(slags, nlags, seconds)

out = np.zeros((len(seconds), 3))
out[:, 0] = seconds
out[:, 1] = signal0
out[:, 2] = np.exp(signal1)
toFile(out, 'LogARMA', pivotName='seconds')
