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
bX,bY,bZ = pk.load(file('../dev/response.pk','r'))
print bX.shape
print bY.shape
print bZ.shape
fig = plt.figure()
ax = fig.add_subplot(111)
#norm = colors.LogNorm(vmin=1e-2,vmax=1e3)
im = ax.pcolormesh(bX,bY,bZ)#,norm=norm)
ax.set_xlabel('angle')
ax.set_ylabel('velocity')
fig.colorbar(im)

# load walk data
cases = range(5)
data = {}
for c,case in enumerate(cases):
  try:
    with open('to_file_{}.csv'.format(case+1),'r') as infile:
      data[case] = {'v0':[],'angle':[]}
      for l,line in enumerate(infile):
        line = line.strip().split(',')
        if l==0:
          ix = line.index('angle')
          iy = line.index('v0')
          continue
        data[case]['angle'].append(float(line[ix]))
        data[case]['v0'].append(float(line[iy]))
  except IOError:
    cases = cases[:c]
    break

print data

# point the first dot
points = []
trails = []
clr = ax._get_lines.prop_cycler
for case in cases:
  c = next(clr)['color']
  point, = ax.plot([data[case]['angle'][0]],[data[case]['v0'][0]],color='k',marker='${}$'.format(case))
  trail, = ax.plot([data[case]['angle'][0]],[data[case]['v0'][0]],'-',color='k',alpha=0.5)
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
      point.set_data(np.array([data[case]['angle'][n],data[case]['v0'][n]]))
      if True:#n < 5:
        trail.set_data(np.array([data[case]['angle'][:n+1],data[case]['v0'][:n+1]]))
    except IndexError:
      continue
  ax.set_title('iteration {}'.format(n))
  return point

ani=animation.FuncAnimation(fig,update_point,max(len(data[case]['angle']) for case in cases),fargs=(data,points,trails),interval=100,repeat_delay=3000)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15,bitrate=1800)
ani.save('path.mp4',writer=writer)
