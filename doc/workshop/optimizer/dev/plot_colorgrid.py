import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import cPickle as pk

import sys
sys.path.append('../../forDevelopers')
import projectile as f

samps = 500

angles = np.linspace(5,85,samps)
vels = np.linspace(1,60,samps)
X,Y = np.meshgrid(angles,vels)
Z = np.zeros((samps,samps))
for i in range(samps):
  for j in range(samps):
    Z[i,j] = f.prange(vels[j],angles[i])

pk.dump((X,Y,Z),file('response.pk','w'))
