"""Synthetic transient field T(x,t) = exp(-alpha*t)*sin(3 pi x) + cos(omega*t)*sin(6 pi x).
Single trajectory for the transient test (nSamples=1, fixed alpha/omega)."""
import numpy as np

NX = 20
NT = 8
X = np.linspace(0.0, 1.0, NX)
TIMES = np.linspace(0.0, 1.0, NT)
DEFAULT_ALPHA = 0.5
OMEGA = 2.0 * np.pi

def run(self, Input):
  alpha = float(getattr(Input, 'alpha', DEFAULT_ALPHA))
  T = (np.exp(-alpha * TIMES[:, None]) * np.sin(3*np.pi*X)[None, :]
       + np.cos(OMEGA * TIMES[:, None]) * np.sin(6*np.pi*X)[None, :])
  self.time = TIMES
  self.pointID = np.arange(NX)
  self.x = X.copy()
  self.T = T            # shape (NT, NX)
  self._indexMap = {'x': ['pointID'],
                    'T': ['time', 'pointID']}
