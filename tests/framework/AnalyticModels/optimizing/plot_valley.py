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
  Module for plotting the various 2d optimization functions included
  in this folder, particularly for obtaining plottable values. Mostly
  used for debugging processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import pickle as pk

from diagonal_valley import main
coeffs = [1, 1, 0]

samps = 500
low = -1
high = 1
log = True


fig, ax = plt.subplots()
xs = np.linspace(low, high, samps)
ys = np.linspace(low, high, samps)
X,Y = np.meshgrid(xs, ys)
Z = main(coeffs, X, Y)
print('min: {}, max:{}'.format(np.min(Z), np.max(Z)))
if log:
  vmin, vmax = np.min(Z), np.max(Z)
  #vmin, vmax = 1e-14, 2
  norm = colors.LogNorm(vmin=vmin,vmax=vmax)
else:
  norm = colors.Normalize()
im = ax.pcolormesh(X, Y, Z, norm=norm)
fig.colorbar(im, ax=ax)
print(Z)
plt.title('Diagonal Valley')

pk.dump((X,Y,Z), open('dvalley_plotdata.pk','wb'))
#plt.axes().set_aspect('equal')


plt.show()
