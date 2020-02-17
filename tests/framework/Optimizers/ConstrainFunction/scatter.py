"""
  Module for producing an animation of optimization histories
  for 2D problems given pickled mesh grid data.  For examples
  of the mesh grid data, see raven/tests/framework/AnalyticModels/optimizing/plot_functions.py.
"""

import pickle as pk
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import numpy as np

# load function data
bX,bY,bZ = pk.load(open('offset_parabola_plotdata.pk','rb'))
fig = plt.figure()
ax = fig.add_subplot(111)
norm = colors.Normalize(vmin=1e-6,vmax=1e0)
#norm = colors.LogNorm(vmin=1e-8,vmax=2e0)
im = ax.pcolormesh(bX,bY,bZ,norm=norm, cmap='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im)

acceptanceMap = {'first': True,
                 'accepted': True,
                 'rerun': True,
                 'rejected': False,
                 'search': False,
                }

# add constraint
import matplotlib.patches as mpatches
circ = mpatches.Circle((0,0), 0.2, fill=False, color='y')
ax.add_artist(circ)
ax.plot([0.25, 0.75,0.75, 0.25, 0.25], [0, 0, 1, 1, 0], 'k-')

# load walk data
cases = range(1)
data = {}
for c,case in enumerate(cases):
  try:
    with open('opt_export_{}.csv'.format(case),'r') as infile:
      data[case] = {'x':[],'y':[],'a':[]}
      for l,line in enumerate(infile):
        line = line.strip().split(',')
        if l==0:
          ix = line.index('x')
          iy = line.index('y')
          ia = line.index('accepted')
          ii = line.index('iteration')
          continue
        x = float(line[ix])
        y = float(line[iy])
        i = int(float(line[ii]))
        a = line[ia]
        if a in ['first', 'accepted']:
          m = '${}$'.format(i)
          accepted = i
        elif a == 'search':
          m = '$s{}$'.format(accepted)
        elif a == 'rerun':
          m = '.'
        elif a == 'rejected':
          m = '$x{}$'.format(accepted)
        else:
          print(a)
          raise NotImplementedError
        ax.plot(x, y, marker=m, ms=12)
  except IOError:
    cases = cases[:c]
    break


plt.show()
# point the first dot
# points = []
# trails = []
# rejects = []
# clr = ax._get_lines.prop_cycler
# for case in cases:
#   c = next(clr)['color']
#   point, = ax.plot([data[case]['x'][0]],[data[case]['y'][0]],color=c,marker='${}$'.format(case))
#   trail, = ax.plot([data[case]['x'][0]],[data[case]['y'][0]],'.-',color=c,alpha=0.5)
#   reject, = ax.plot([],[],'x',color=c,alpha=0.5)
#   points.append(point)
#   trails.append(trail)
#   rejects.append(reject)


# def update_point(n,data,points,trails,rejects):
#   """
#     Function to be called to update the animation points, one iteration at a time.
#     @ In, n, int, the iteration to use
#     @ In, data, dict, all the data collected from the RAVEN output
#     @ In, points, list, plotted points in the animation
#     @ In, trails, list, currently unused, finite number of trailing points to track in animation
#     @ In, rejects, list, rejected samples from evaluations
#     @ Out, point, matplotlib.pyplot line, last plotted point object
#   """
#   print('Animating iteration',n)
#   for c,case in enumerate(cases):
#     point = points[c]
#     trail = trails[c]
#     reject = rejects[c]
#     N = len(data[case]['x'])
#     # truncate data
#     x = np.array(data[case]['x'][:n+1] if n+1 < N else data[case]['x'])
#     y = np.array(data[case]['y'][:n+1] if n+1 < N else data[case]['y'])
#     a = np.array(data[case]['a'][:n+1] if n+1 < N else data[case]['a'])
#     # split data into accepted, rejected points
#     x_a = np.atleast_1d(x[a])
#     y_a = np.atleast_1d(y[a])
#     x_r = np.atleast_1d(x[np.logical_not(a)])
#     y_r = np.atleast_1d(y[np.logical_not(a)])
#     try:
#       point.set_data([x_a[-1]],[y_a[-1]])
#       trail.set_data(x_a,y_a)
#       reject.set_data(x_r,y_r)
#     except IndexError:
#       continue
#   ax.set_title('iteration {}'.format(n))
#   return point

# ani=animation.FuncAnimation(fig,update_point,max(len(data[case]['x']) for case in cases),fargs=(data,points,trails,rejects),interval=100,repeat_delay=3000)
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15,bitrate=1800)
# ani.save('path.mp4',writer=writer)
