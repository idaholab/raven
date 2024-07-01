import numpy as np
import matplotlib.pyplot as plt


arc_x = np.linspace(0,1,100)
arc_y = np.sqrt(1 - arc_x**2)

# Monte Carlo
xx = np.random.rand(500)
yy = np.random.rand(500)

fig, ax = plt.subplots(figsize=(12,10))

plt.plot(arc_x, arc_y, 'k-', lw=10)
plt.scatter(xx, yy, color='r')

fig.savefig('arc_mc.png')

# Grid
xx = np.linspace(0,1,10)
yy = np.linspace(0,1,10)

fig, ax = plt.subplots(figsize=(12,10))

plt.plot(arc_x, arc_y, 'k-', lw=10)
for x in xx:
  for y in yy:
    plt.scatter(x, y, color='g')

fig.savefig('arc_grid.png')

# LHS
from pyDOE3 import lhs
lhd = lhs(2, samples=10)
xx = lhd[:, 0]
yy = lhd[:, 1]

fig, ax = plt.subplots(figsize=(12,10))

plt.plot(arc_x, arc_y, 'k-', lw=10)
plt.scatter(xx, yy, color='b')

fig.savefig('arc_lhs.png')
