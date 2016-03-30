from numpy import array, zeros, argmin, inf
import math
from matplotlib import pyplot as plt

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

from sklearn.metrics.pairwise import *

#x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
#y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
#dist_fun = manhattan_distances
timex=np.linspace(0,6.28,80)
x=np.zeros(80)
for i in range(80):
  x[i] = math.sin(timex[i]/3.141*2)

timey=np.linspace(0,5.78,80)  
y=np.zeros(80)
for i in range(80):
  y[i] = math.sin(1.0*timey[i]/3.141*2)*1.2

#x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2]]
#y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
dist_fun = euclidean_distances

#x = np.array([1, 1, 2, 3, 2, 0])
#y = np.array([0, 1, 1, 2, 3, 2, 1])

#idx = np.linspace(0, 6.28, 100)
#x = np.sin(idx)
#y = np.cos(idx)

dist, cost, acc, path = dtw(x, y, dist_fun)

plt.imshow(acc.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
plt.plot(path[0], path[1], '-o') # relation
plt.xlabel('x')
plt.ylabel('y')
plt.axis('tight')
plt.show()

plt.plot(timex, x, 'r',linewidth=3)
plt.plot(timey, y, 'g',linewidth=3)

#plt.plot(idx, x, 'r',linewidth=3)
#plt.plot(idx, y, 'g',linewidth=3)

xCoor = np.zeros(2)
yCoor = np.zeros(2)
pathLength=path[0].size
for i in range(pathLength):
  xCoor[0] = timex[path[0][i]]
  xCoor[1] = timey[path[1][i]]
  yCoor[0] = x[path[0][i]]
  yCoor[1] = y[path[1][i]]
  plt.plot(xCoor,yCoor,'k')
plt.show()
