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
  Module for plotting the various 3d optimization functions included
  in this folder, particularly for obtaining plottable values. Mostly
  used for debugging processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pk
from mpl_toolkits.mplot3d import axes3d, Axes3D
from diagonal_valley import main
coeffs = [1, 1, 0]

samps = 30
low = -1
high = 1

xs = np.linspace(low, high, samps)
ys = np.linspace(low, high, samps)
X,Y = np.meshgrid(xs, ys)
Z = main(coeffs, X, Y)

print('min: {}, max:{}'.format(np.min(Z), np.max(Z)))

norm = plt.Normalize(Z.min(), Z.max())
colors = cm.twilight(norm(Z))
rcount, ccount, _ = colors.shape

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(70, 0)
surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False)

surf.set_facecolor((0,0,0,0))
plt.show()
pk.dump((X,Y,Z), open('dvalley_plotdata.pk','wb'))
