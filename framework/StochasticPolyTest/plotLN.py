
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

L=[1,2,3,4,5]
N=[2,4,6,8]
X,Y = np.meshgrid(L,N)

fig=plt.figure()

def doPlot(Z):
  ax = fig.gca(projection='3d')
  #ax.set_zscale('log')
  surf = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm, linewidth=1, antialiased=False)
  ax.zaxis.set_major_locator(LinearLocator(10))
  plt.xlabel('L')
  plt.ylabel('N')

CCTP=[[9,25,81,289,1089],
      [81,625,6561,83521,1185921],
      [729,15625,531441, 24137569, 1291467969],
      [6561, 390625, 43046721, 6975757441, 1406408618241]]

CCTD=[[5, 13, 29, 65, 145],
      [9,41, 137,401,1105],
      [13, 85, 389,1457, 4865],
      [17, 145,849,3937, 15713]]

CCHC=[[5,9,21, 37, 77],
[9,17, 57, 89, 201],
[13, 25, 109,157,373],
[17, 33, 177,241,593]]

doPlot(CCHC)

plt.show()
