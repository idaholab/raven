import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import cPickle as pk

samps = 500

def plotFunction(title,method,low,high,log=True):
  print 'plotting',title
  fig = plt.figure(title)
  ax = fig.add_subplot(111)#,projection='3d')
  xs = np.linspace(low,high,samps)
  ys = np.linspace(low,high,samps)
  X,Y = np.meshgrid(xs,ys)
  Z = method(X,Y)
  if log:
    norm = colors.LogNorm(vmin=np.min(Z),vmax=np.max(Z))
  else:
    norm = colors.Normalize()
  ax.pcolormesh(X,Y,Z,norm=norm)#,cmap=cm.rainbow,norm=norm)
  #ax.plot_surface(X,Y,Z,
  #             cmap=cm.rainbow,
  #             linewidth=0,
  #             antialiased=False,
  #             norm=norm)
  plt.title(title)
  return X,Y,Z

# Beale
from beale import evaluate as beale
x,y,z = plotFunction('Beale',beale,-4.5,4.5)
pk.dump((x,y,z),file('beale_plotdata.pk','w'))

# Rosenbrock
from rosenbrock import evaluate2d as rosenbrock
x,y,z = plotFunction('Rosenbrock',rosenbrock,-2,2)
pk.dump((x,y,z),file('rosenbrock_plotdata.pk','w'))

# Goldstein-Price
from goldstein_price import evaluate as goldstein_price
x,y,z = plotFunction('Goldstein-Price',goldstein_price,-2,2)
pk.dump((x,y,z),file('goldsteinprice_plotdata.pk','w'))

# McCormick
from mccormick import evaluate as mccormick
x,y,z = plotFunction('McCormick',mccormick,-3,4,log=False)
pk.dump((x,y,z),file('mccormick_plotdata.pk','w'))

plt.show()
