import numpy as np
from scipy.special import eval_legendre as EL
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,10))

xx = np.linspace(-1,1,200)

for p in range(5):
  yy = EL(p, xx)
  ax.plot(xx, yy, '-', label=f'O({p})')

ax.legend()

fig.savefig('legendres.png')
