import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,10))

xx = np.linspace(0,1,100)

coeffs = [
  (1, 0, 0.5),
  (0, 1, 1)
]

tot = np.zeros(len(xx))
for (A, B, k) in coeffs:
  c = 2 * np.pi / k
  yy = A*np.sin(c * xx) + B*np.cos(c * xx)
  ax.plot(xx, yy, ':', lw=5, label=f'({A},{B},{k})')
  tot += yy

ax.plot(xx, tot, 'k-', lw=10, label='total')

ax.legend()
#plt.show()
fig.savefig('fourier.png')
