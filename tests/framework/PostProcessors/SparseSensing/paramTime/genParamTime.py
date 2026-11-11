"""Parameter+time synthetic field with alpha controlling spatial mode frequency.

T(x, t; alpha) = exp(-t) * sin((3 + alpha) * pi * x)

Rank-1 in (x, t) for each alpha, but different alphas shift the dominant
spatial mode (alpha=0.1 -> sin(3.1 pi x), alpha=1.0 -> sin(4 pi x)), so
stacking 5 different alphas produces a (5*NT, NX) snapshot matrix whose
SVD modes depend on the sampled set -- distinct from the fixed-alpha
transient test's data.
"""
import numpy as np

NX = 20
NT = 8
X = np.linspace(0.0, 1.0, NX)
TIMES = np.linspace(0.0, 1.0, NT)

def run(self, Input):
  alpha = float(getattr(Input, 'alpha', 0.5))
  T = np.exp(-TIMES[:, None]) * np.sin((3.0 + alpha) * np.pi * X)[None, :]
  self.time = TIMES
  self.pointID = np.arange(NX)
  self.x = X.copy()
  self.T = T
  self._indexMap = {'x': ['pointID'], 'T': ['time', 'pointID']}
