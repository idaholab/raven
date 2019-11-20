# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Module for producing an animation of optimization histories
  for 3D problems given pickled mesh grid data.  For examples
  of the mesh grid data, see raven/tests/framework/AnalyticModels/optimizing/plot_functions.py.
"""

import pickle as pk
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

from matplotlib import animation
import numpy as np

# load function data
bX,bY,bZ = pk.load(open('dvalley_plotdata.pk','rb'))
norm = plt.Normalize(bZ.min()-1, bZ.max()+5)
colors = cm.BuGn(norm(bZ))
rcount, ccount, _ = colors.shape
fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')
ax.view_init(70, 0)


surf = ax.plot_surface(bX, bY, bZ, rcount=rcount, ccount=ccount,
                       facecolors=colors,alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('y')

# load walk data
cases = range(5)
data = {}
for c,case in enumerate(cases):
  try:
    with open('opt_export_{}.csv'.format(case+1),'r', encoding = "utf-8-sig") as infile:
      data[case] = {'x':[],'y':[],'z':[],'a':[]}
      for l,line in enumerate(infile):
        line = line.strip().split(',')
        if l==0:
          ix = line.index('x')
          iy = line.index('y')
          iz = line.index('ans')
          ia = line.index('accepted')
          continue
        data[case]['x'].append(float(line[ix]))
        data[case]['y'].append(float(line[iy]))
        data[case]['z'].append(float(line[iz]))
        data[case]['a'].append(bool(float(line[ia])))
  except IOError:
    cases = cases[:c]
    break

# point the first dot
points = []
trails = []
rejects = []
clr = ax._get_lines.prop_cycler
for case in cases:
  c = next(clr)['color']
  point, = ax.plot3D([data[case]['x'][0]],[data[case]['y'][0]],[data[case]['z'][0]],color=c,alpha=0.9,marker='${}$'.format(case))
  trail, = ax.plot3D([data[case]['x'][0]],[data[case]['y'][0]],[data[case]['z'][0]],'.-',color=c,alpha=0.9)
  reject, = ax.plot3D([],[],[],'x',color=c,alpha=0.9)
  points.append(point)
  trails.append(trail)
  rejects.append(reject)


def updatePoint(n,data,points,trails,rejects):
  """
    Function to be called to update the animation points, one iteration at a time.
    @ In, n, int, the iteration to use
    @ In, data, dict, all the data collected from the RAVEN output
    @ In, points, list, plotted points in the animation
    @ In, trails, list, currently unused, finite number of trailing points to track in animation
    @ In, rejects, list, rejected samples from evaluations
    @ Out, point, matplotlib.pyplot line, last plotted point object
  """
  print('Animating iteration',n)
  for c,case in enumerate(cases):
    point = points[c]
    trail = trails[c]
    reject = rejects[c]
    N = len(data[case]['x'])
    # truncate data
    x = np.array(data[case]['x'][:n+1] if n+1 < N else data[case]['x'])
    y = np.array(data[case]['y'][:n+1] if n+1 < N else data[case]['y'])
    z = np.array(data[case]['z'][:n+1] if n+1 < N else data[case]['z'])

    a = np.array(data[case]['a'][:n+1] if n+1 < N else data[case]['a'])
    # split data into accepted, rejected points
    xA = np.atleast_1d(x[a])
    yA = np.atleast_1d(y[a])
    zA = np.atleast_1d(z[a])

    xR = np.atleast_1d(x[np.logical_not(a)])
    yR = np.atleast_1d(y[np.logical_not(a)])
    zR = np.atleast_1d(z[np.logical_not(a)])

    try:
      point.set_data([xA[-1]],[yA[-1]])
      point.set_3d_properties(zA[-1])
      trail.set_data(xA,yA)
      trail.set_3d_properties(zA)
      reject.set_data(xR,yR)
      reject.set_3d_properties(zR)
    except IndexError:
      continue

  ax.set_title('iteration {}'.format(n),loc='center',pad=20)

  return point

ani=animation.FuncAnimation(fig,updatePoint,max(len(data[case]['x']) for case in cases),fargs=(data,points,trails,rejects),interval=100,repeat_delay=3000)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15,bitrate=1800)
ani.save('path3d.mp4',writer=writer)
