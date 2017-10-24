import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import sys
import subprocess

######
# This is kept for the future potential metric included.  Written by maljdan, Sept 2017
#####

inputFile = sys.argv[1]
file1 = sys.argv[2]
file2 = sys.argv[3]
eps = float(sys.argv[4])

## TODO: Find a better way to execute raven
retCode = subprocess.call(['python','../../../../framework/Driver.py',inputFile])

if retCode == 0:
  # START metric included here
  points1 = np.loadtxt(file1, delimiter=',', skiprows=1)
  points2 = np.loadtxt(file2, delimiter=',', skiprows=1)

  N = min(points1.shape[0],points2.shape[0])

  C = cdist(points1, points2)

  _, assignment = linear_sum_assignment(C)

  minMaxDistance = max(np.linalg.norm(points1[:N] - points2[assignment], axis=1))
  print(minMaxDistance)
  ## Return zero if passed, non-zero if failed
  retCode = not (minMaxDistance < eps)
  # END metric included here

  ## I am leaving this here as it can potentially be useful for debugging
  if retCode:
    import matplotlib.pyplot as plt
    plt.plot(points1[:,0], points1[:,1],'bo', markersize = 10)
    plt.plot(points2[:,0], points2[:,1],'rs',  markersize = 7)
    for p in range(N):
      plt.plot([points1[p,0], points2[assignment[p],0]], [points1[p,1], points2[assignment[p],1]], 'k')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.axes().set_aspect('equal')
    plt.show()

sys.exit(retCode)
