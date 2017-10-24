import matplotlib.pyplot as plt
import sys
import numpy as np

for arg in sys.argv[1:]:
  data = np.loadtxt(arg, delimiter=',', skiprows=1)

  plt.scatter(data[:,0], data[:,1])
  plt.title(arg)
  plt.show()
