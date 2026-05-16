
import numpy as np

NX = 18
NY = 12
NT = 8
X_AXIS = np.linspace(0.0, 1.0, NX)
Y_AXIS = np.linspace(0.0, 1.0, NY)
TIMES = np.linspace(0.0, 1.0, NT)
XX, YY = np.meshgrid(X_AXIS, Y_AXIS, indexing='xy')
X = XX.ravel()
Y = YY.ravel()

def read_alpha(Input, default=0.5):
  if isinstance(Input, dict):
    raw = Input.get('alpha', default)
  else:
    raw = getattr(Input, 'alpha', default)
  return float(np.asarray(raw).reshape(-1)[0])

def two_dimensional_field(alpha):
  left = np.exp(-(((X - (0.25 + 0.10 * alpha)) / 0.16) ** 2 + ((Y - 0.35) / 0.18) ** 2))
  right = np.exp(-(((X - 0.72) / 0.18) ** 2 + ((Y - (0.68 - 0.08 * alpha)) / 0.16) ** 2))
  wave = np.sin((2.0 + alpha) * np.pi * X) * np.cos(2.0 * np.pi * Y)
  return (
      np.exp(-0.85 * TIMES[:, None]) * ((2.0 + 0.25 * alpha) * left[None, :] + (0.6 + 1.35 * alpha) * right[None, :])
      + 0.28 * np.cos(2.0 * np.pi * TIMES[:, None]) * wave[None, :]
  )

ALPHAS = np.linspace(0.1, 1.0, 5)

def full_tensor():
  tensor = np.stack([two_dimensional_field(alpha) for alpha in ALPHAS])
  extra_mode = (
      0.18
      * np.linspace(-1.0, 1.0, len(ALPHAS))[:, None, None]
      * np.sin(np.pi * TIMES)[None, :, None]
      * np.sin(3.0 * np.pi * X)[None, None, :]
  )
  return tensor + extra_mode

def run(self, Input):
  alpha = read_alpha(Input)
  alpha_index = int(np.argmin(np.abs(ALPHAS - alpha)))
  self.time = TIMES
  self.pointID = np.arange(X.size)
  self.x = X.copy()
  self.y = Y.copy()
  self.T = full_tensor()[alpha_index]
  self._indexMap = {'x': ['pointID'], 'y': ['pointID'], 'T': ['time', 'pointID']}
