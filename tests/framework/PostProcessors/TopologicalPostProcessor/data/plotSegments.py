'''
Created on 6/24/15

@author: maljdan

Will plot the results of test_topology_persistence for visually inspection that
the new persistence computations perform as expected

'''
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

## Paul Tol's colorblind safe colors
colors1 = ['#88CCEE', '#DDCC77', '#AA4499', '#117733', '#332288', '#999933',
           '#44AA99', '#882255', '#CC6677']
## Alternative Color Lists from Color Brewer
colors2 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02',
           '#a6761d', '#666666']
colors3 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
           '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']
colors4 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
           '#a65628', '#f781bf', '#999999']
colors5 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
           '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
colors6 = ['#e41a1c', '#88CCEE', '#377eb8', '#DDCC77', '#4daf4a', '#AA4499',
           '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999',
           '#117733', '#332288', '#999933', '#44AA99', '#882255']

colorList = colors6

fileList = ['Count','Value','Prob','Full']

fig = plt.figure()
axes = []
for i in xrange(len(fileList)):
  axes.append(fig.add_subplot(2,len(fileList),i+1, projection='3d'))
for i in xrange(len(fileList)):
  axes.append(fig.add_subplot(2,len(fileList),i+1+len(fileList)))

for i,name in enumerate(fileList):
  f = 'seg'+name+'.csv'
  dataFile = open(f)
  dataFile.readline()

  allX = []
  allY = []
  allZ = []

  allMins = set()
  allMaxs = set()

  segColors = {}
  idxs = {}

  nextColorIdx = 0
  idx = 0
  for line in dataFile:
    tokens = line.strip().split(',')
    label = (int(float(tokens[3])),int(float(tokens[4])))

    if label not in idxs:
      idxs[label] = []

    if idx != int(float(tokens[3])) and idx != int(float(tokens[4])):
      idxs[label].append(idx)

    allMins.add(int(float(tokens[3])))
    allMaxs.add(int(float(tokens[4])))

    allY.append(float(tokens[0]))
    allX.append(float(tokens[1]))
    allZ.append(float(tokens[2]))

    if label not in segColors:
      segColors[label] = colorList[nextColorIdx]
      nextColorIdx += 1
      if nextColorIdx >= len(colorList):
        nextColorIdx = 0

    idx += 1

  allX = np.array(allX)
  allY = np.array(allY)
  allZ = np.array(allZ)
  axes[i].set_title(name)
  for seg in idxs.keys():
    # axes[i].scatter(X[seg], Y[seg], Z[seg], c=segColors[seg], lw=0)
    if len(idxs[seg]) >= 10:
      axes[i].plot_trisurf(allX[idxs[seg]], allY[idxs[seg]], allZ[idxs[seg]],
                           color=segColors[seg], lw=0)
    axes[i+len(fileList)].scatter(allX[idxs[seg]], allY[idxs[seg]],
                                  c=segColors[seg], lw=0)
    axes[i+len(fileList)].scatter(allX[list(allMaxs)], allY[list(allMaxs)], c='red')
    axes[i+len(fileList)].scatter(allX[list(allMins)], allY[list(allMins)], c='blue')
  dimSize=math.sqrt(len(allX))
  axes[i+len(fileList)].contour(np.reshape(allX,(dimSize,dimSize)),
                                np.reshape(allY,(dimSize,dimSize)),
                                np.reshape(allZ,(dimSize,dimSize)),colors="k")

plt.show()
