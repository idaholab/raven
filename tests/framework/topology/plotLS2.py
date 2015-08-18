'''
Created on 6/24/15

@author: maljdan

Will plot the results of test_batch_adaptive for visually inspecting the
iterations of adaptive sampling

'''
import numpy as np
import matplotlib.pyplot as plt
import math
from glob import glob
import os
from matplotlib import cm, colors

myColormap = cm.coolwarm

samplesX = []
samplesY = []
sFile = open('samplesDump.csv')
sFile.readline()
for line in sFile:
  tokens = line.strip().split(',')
  samplesX.append(float(tokens[1]))
  samplesY.append(float(tokens[0]))
sFile.close()
print(len(samplesX))

samples = zip(samplesX,samplesY)

inputs = glob('scores_*.csv')
for fin in inputs:
  i = int(fin.split('_')[-1].replace('.csv',''))
  if i < len(samplesX):
    lsFile = open(fin)
    lsFile.readline()
    limitX = []
    limitY = []
    limitColor = []
    for line in lsFile:
      tokens = line.strip().split(',')
      limitX.append(float(tokens[0]))
      limitY.append(float(tokens[1]))
      limitColor.append(float(tokens[2]))
    lsFile.close()

    plt.scatter(limitX, limitY, c=limitColor, cmap=myColormap, linewidths=0)
    idx = np.argmax(limitColor)
    if (limitX[idx],limitY[idx]) not in samples:
      print(i,'nope')
    # else:
    #   print(i,samples.index((limitX[idx],limitY[idx])))

    plt.scatter(samplesX[0:(i-2)],samplesY[0:(i-2)],c='#DDCC77',linewidths=0,marker='^')
    plt.scatter(samplesX[i-1],samplesY[i-1],c='#117733',linewidths=1,marker='^')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.title(i)
    plt.savefig('example/limitSurface'+str(i)+'.png')
    plt.clf()
    plt.cla()

#Comment next two lines out if you don't have imagemagick installed
os.system('convert -delay 100 -loop 0 example/limitSurface*.png example/candidates.gif')
os.system('rm example/limitSurface*.png')