"""
  Module for plotting the various 2d optimization functions included
  in this folder, particularly for obtaining plottable values. Mostly
  used for debugging processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import cPickle as pk

samps = 500

def plotFunction(title,method,low,high,cscale=None,log=True):
  """
    Plots a 2D function as a colormap.  Returns parameters suitable to plotting in a pcolormesh call.
    @ In, title, string, title name for figure
    @ In, method, function, method to call with x,y to get z result
    @ In, low, float, low value for x and y
    @ In, high, float, high value for x and y
    @ In, cscale, tuple(float), optional, low and high values for the color map
    @ In, log, bool, optional, if False will not lognormalize the color map
    @ Out, X, np.array(np.array(float)), mesh grid of X values
    @ Out, Y, np.array(np.array(float)), mesh grid of Y values
    @ Out, Z, np.array(np.array(float)), mesh grid of Z (response) values
  """
  print 'plotting',title,
  fig = plt.figure(title)
  ax = fig.add_subplot(111)#,projection='3d')
  xs = np.linspace(low,high,samps)
  ys = np.linspace(low,high,samps)
  X,Y = np.meshgrid(xs,ys)
  Z = method(X,Y)
  print 'min: {}, max:{}'.format(np.min(Z),np.max(Z))
  if log:
    if cscale is None:
      vmin,vmax = np.min(Z),np.max(Z)
    else:
      vmin,vmax = cscale
    norm = colors.LogNorm(vmin=vmin,vmax=vmax)
  else:
    norm = colors.Normalize()
  ax.pcolormesh(X,Y,Z,norm=norm)
  plt.title(title)
  return X,Y,Z

from sine_valley import evaluate as sine
x,y,z = plotFunction('Sine',sine,-4.5,4.5,cscale=(1e-2,1e1))
pk.dump((x,y,z),file('sine_plotdata.pk','w'))
plt.axes().set_aspect('equal')

# Beale
from beale import evaluate as beale
x,y,z = plotFunction('Beale',beale,-4.5,4.5,cscale=(1e-1,1e3))
pk.dump((x,y,z),file('beale_plotdata.pk','w'))
plt.savefig('Beale_grid.png')

# Rosenbrock
from rosenbrock import evaluate2d as rosenbrock
x,y,z = plotFunction('Rosenbrock',rosenbrock,-2,2,cscale=(1e-1,1e3))
pk.dump((x,y,z),file('rosenbrock_plotdata.pk','w'))

# Goldstein-Price
from goldstein_price import evaluate as goldstein_price
x,y,z = plotFunction('Goldstein-Price',goldstein_price,-2,2,cscale=(3,1e3))
pk.dump((x,y,z),file('goldsteinprice_plotdata.pk','w'))

# McCormick
from mccormick import evaluate as mccormick
x,y,z = plotFunction('McCormick',mccormick,-3,4,log=False)
pk.dump((x,y,z),file('mccormick_plotdata.pk','w'))

plt.show()
