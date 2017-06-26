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

def plotFunction(title,method,constraint,xscale,yscale,cscale=None,log=True):
  """
    Plots a 2D function as a colormap.  Returns parameters suitable to plotting in a pcolormesh call.
    @ In, title, string, title name for figure
    @ In, method, function, method to call with x,y to get z result
    @ In, constraint, function, boolean method that determines acceptability
    @ In, xscale, tuple(float), low/hi value for x
    @ In, yscale, tuple(float), low/hi value for y
    @ In, cscale, tuple(float), optional, low and high values for the color map
    @ In, log, bool, optional, if False will not lognormalize the color map
    @ Out, X, np.array(np.array(float)), mesh grid of X values
    @ Out, Y, np.array(np.array(float)), mesh grid of Y values
    @ Out, Z, np.array(np.array(float)), mesh grid of Z (response) values
  """
  print 'plotting',title,
  fig = plt.figure(title)
  ax = fig.add_subplot(111)#,projection='3d')
  xs = np.linspace(xscale[0],xscale[1],samps)
  ys = np.linspace(yscale[0],yscale[1],samps)
  X,Y = np.meshgrid(xs,ys)
  Z = method(X,Y)
  #Z[np.where(not constraint(X,Y))] = np.nan
  for i,x in enumerate(xs):
    for j,y in enumerate(ys):
      if not constraint(x,y):
        Z[j][i] = np.nan
      #else:
      #  print i,x,'|',j,y,'|',Z[i][j]
  Zm = np.ma.masked_where(np.isnan(Z),Z)
  print 'min: {}, max:{}'.format(np.nanmin(Z),np.nanmax(Z))
  if log:
    if cscale is None:
      vmin,vmax = np.nanmin(Z),np.nanmax(Z)
    else:
      vmin,vmax = cscale
    norm = colors.LogNorm(vmin=vmin,vmax=vmax)
  else:
    norm = colors.Normalize()
  ax.pcolormesh(X,Y,Zm)#,norm=norm)
  plt.title(title)
  return X,Y,Zm

# Beale
from townsend import evaluate as townsend
from townsend import constraint as townsend_c
x,y,z = plotFunction('Townsend',townsend,townsend_c,(-2.25,2.5),(-2.5,1.75),cscale=(-3.5,1.9),log=False)
pk.dump((x,y,z),file('townsend_plotdata.pk','w'))

from mishra_bird import evaluate as mishra
from mishra_bird import constraint as mishra_c
x,y,z = plotFunction('Mishra Bird',mishra,mishra_c,(-10,0),(-7,0),log=False)
pk.dump((x,y,z),file('mishra_plotdata.pk','w'))

plt.show()
