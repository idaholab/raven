"""Parameter+time synthetic field with two well-separated Gaussian peaks in space.

T(x, t; alpha) = (1 + 0.5*alpha) * exp(-t) *
                 (2.0 * exp(-((x - 0.25)/0.08)^2) + 1.0 * exp(-((x - 0.75)/0.08)^2))

The two spatial peaks at x=0.25 and x=0.75 have amplitudes 2.0 and 1.0, giving
column norms with a clear, large-margin ordering: x near 0.25 dominates first,
x near 0.75 second, all other x columns are exponentially smaller. QR-based
sensor selection produces the same two sensors regardless of the BLAS backend
(OpenBLAS / MKL / Apple Accelerate), making this test reproducible across
platforms. alpha modulates global amplitude only, preserving the column ordering.
"""
import numpy as np

NX = 20
NT = 8
X = np.linspace(0.0, 1.0, NX)
TIMES = np.linspace(0.0, 1.0, NT)

def run(self, Input):
  alpha = float(getattr(Input, 'alpha', 0.5))
  spatial = (2.0 * np.exp(-((X - 0.25) / 0.08) ** 2)
             + 1.0 * np.exp(-((X - 0.75) / 0.08) ** 2))
  T = (1.0 + 0.5 * alpha) * np.exp(-TIMES[:, None]) * spatial[None, :]
  self.time = TIMES
  self.pointID = np.arange(NX)
  self.x = X.copy()
  self.T = T
  self._indexMap = {'x': ['pointID'], 'T': ['time', 'pointID']}
