import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import sys
import subprocess

######
# Run by "tests" to test adaptive batch running
#####

inputFile = sys.argv[1]
file1 = sys.argv[2]
eps = float(sys.argv[3])

## TODO: Find a better way to execute raven
retCode = subprocess.call([os.path.join('..','..','..','..','raven_framework'),inputFile])
#retCode = subprocess.call(['python','../../../../framework/Driver.py',inputFile])

if retCode == 0:

  points = np.loadtxt(file1, delimiter=',', skiprows=1)
  maxDistance = np.max(np.abs(points[:,0] - points[:,1]))

  retCode = not (maxDistance < eps)

  ## I am leaving this here as it can potentially be useful for debugging
  if retCode:
    print(maxDistance)
    import matplotlib.pyplot as plt
    plt.plot(points[:,0], points[:,1],'bo', markersize = 10)
    plt.xlim(-1.0,1.0)
    plt.ylim(-1.0,1.0)
    plt.axes().set_aspect('equal')
    plt.show()

sys.exit(retCode)
