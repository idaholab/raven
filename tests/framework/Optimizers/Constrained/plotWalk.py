"""
  Module for producing an animation of optimization histories
  for 2D problems given pickled mesh grid data.  For examples
  of the mesh grid data, see raven/tests/framework/AnalyticModels/optimizing/plot_functions.py.
"""

import cPickle as pk
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import numpy as np

# load function data
bX,bY,bZ = pk.load(file('mishra_plotdata.pk','r'))
fig = plt.figure()
ax = fig.add_subplot(111)
norm = colors.Normalize()#LogNorm(vmin=1e-2,vmax=1e3)
im = ax.pcolormesh(bX,bY,bZ,norm=norm)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im)

# load walk data
cases = range(5)
data = {}
for c,case in enumerate(cases):
  try:
    with open('opt_export_{}.csv'.format(case),'r') as infile:
      data[case] = {'x':[],'y':[]}
      for l,line in enumerate(infile):
        line = line.strip().split(',')
        if l==0:
          ix = line.index('x')
          iy = line.index('y')
          continue
        data[case]['x'].append(float(line[ix]))
        data[case]['y'].append(float(line[iy]))
  except IOError:
    cases = cases[:c]
    break

# point the first dot
points = []
trails = []
clr = ax._get_lines.prop_cycler
for case in cases:
  c = next(clr)['color']
  point, = ax.plot([data[case]['x'][0]],[data[case]['y'][0]],color=c,marker='${}$'.format(case))
  trail, = ax.plot([data[case]['x'][0]],[data[case]['y'][0]],'-',color=c,alpha=0.5)
  points.append(point)
  trails.append(trail)

def update_point(n,data,points,trails):
  """
    Function to be called to update the animation points, one iteration at a time.
    @ In, n, int, the iteration to use
    @ In, data, dict, all the data collected from the RAVEN output
    @ In, points, list, plotted points in the animation
    @ In, trails, list, currently unused, finite number of trailing points to track in animation
    @ Out, point, matplotlib.pyplot line, last plotted point object
  """
  for c,case in enumerate(cases):
    point = points[c]
    trail = trails[c]
    try:
      point.set_data(np.array([data[case]['x'][n],data[case]['y'][n]]))
      if True:#n < 5:
        trail.set_data(np.array([data[case]['x'][:n+1],data[case]['y'][:n+1]]))
    except IndexError:
      continue
  ax.set_title('iteration {}'.format(n))
  return point

ani=animation.FuncAnimation(fig,update_point,max(len(data[case]['x']) for case in cases),fargs=(data,points,trails),interval=100,repeat_delay=3000)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15,bitrate=1800)
ani.save('path.mp4',writer=writer)
