'''
Created on 6/24/15

@author: maljdan

Will plot the results of test_batch_adaptive for visually inspecting the
iterations of adaptive sampling

'''
import numpy as np
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

colorList = colors1

for i in xrange(1):
  lsFile = open('limitSurfaceDump.csv')
  lsFile.readline()
  limitX = []
  limitY = []
  samplesX = []
  samplesY = []
  limitColor = []
  for line in lsFile:
    tokens = line.strip().split(',')
    limitX.append(float(tokens[0]))
    limitY.append(float(tokens[1]))
    if float(tokens[2]) > 0:
      limitColor.append(colorList[0])
    else:
      limitColor.append(colorList[1])
  lsFile.close()

  sFile = open('samplesDump.csv')
  sFile.readline()
  for line in sFile:
    tokens = line.strip().split(',')
    samplesX.append(float(tokens[0]))
    samplesY.append(float(tokens[1]))
  sFile.close()

  plt.scatter(limitX,limitY,c=limitColor,linewidths=0)
  plt.scatter(samplesX,samplesY,c=colorList[2],linewidths=0,marker='^')
  plt.xlim(-1,1)
  plt.ylim(-1,1)
  plt.savefig('limitSurface'+str(i)+'.png')