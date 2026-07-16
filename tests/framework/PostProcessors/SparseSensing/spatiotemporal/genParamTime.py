"""Parameter+time synthetic field with two well-separated Gaussian peaks in space.

T(x, t; alpha) = exp(-t) *
                 ((2.0 + 0.1*alpha) * exp(-((x - 0.25)/0.08)^2)
                  + (0.4 + 1.8*alpha) * exp(-((x - 0.75)/0.08)^2))

The two spatial peaks at x=0.25 and x=0.75 have distinct alpha dependence, so
the sample matrix is rank 2. QR then has a clear, large-margin second pivot
instead of choosing among numerically-zero residual columns after the first
sensor.
"""
import numpy as np

NX = 20
NT = 8
X = np.linspace(0.0, 1.0, NX)
TIMES = np.linspace(0.0, 1.0, NT)

def run(self, Input):
  rawAlpha = Input.get('alpha', 0.5) if isinstance(Input, dict) else getattr(Input, 'alpha', 0.5)
  alpha = float(np.asarray(rawAlpha).reshape(-1)[0])
  left = np.exp(-((X - 0.25) / 0.08) ** 2)
  right = np.exp(-((X - 0.75) / 0.08) ** 2)
  spatial = (2.0 + 0.1 * alpha) * left + (0.4 + 1.8 * alpha) * right
  T = np.exp(-TIMES[:, None]) * spatial[None, :]
  self.time = TIMES
  self.pointID = np.arange(NX)
  self.x = X.copy()
  self.T = T
  self._indexMap = {'x': ['pointID'], 'T': ['time', 'pointID']}
