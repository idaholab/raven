from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

files = glob('dummy_*.csv')

fptr = open(files[0])
fptr.readline()
numTimeSteps = len([1 for line in fptr])
fptr.close()

numPoints = len(files)

xLabel = 'x1'
yLabel = 'x2'
label = 'labels'

X = np.zeros(shape=(numTimeSteps,numPoints))
Y = np.zeros(shape=(numTimeSteps,numPoints))
C = np.zeros(shape=(numTimeSteps,numPoints))

for filename in files:
  counter = int(filename.split('_')[1].split('.')[0])
  fptr = open(filename)
  labels = fptr.readline().split(',')
  for idx,line in enumerate(fptr):
    tokens = line.split(',')
    X[idx,counter] = float(tokens[labels.index(xLabel)])
    Y[idx,counter] = float(tokens[labels.index(yLabel)])
    C[idx,counter] = float(tokens[labels.index(label)])
  fptr.close()

mapper = cm.ScalarMappable(cmap=cm.get_cmap('Set1'))
mapper.set_clim(np.min(C),np.max(C))

def init_plot():
  pass

def update_plot(idx):
  fig.clf()
  if idx == numTimeSteps:
    for i in xrange(numPoints):
      ## The line below fails as the plot function cannot figure out how to take
      ## a list of rgba values
      # myScatter = plt.plot(X[:,i], Y[:,i], c=mapper.cmap(C[:,i]))
      myScatter = plt.plot(X[:,i], Y[:,i], c='k')
    for i in xrange(numTimeSteps):
      myScatter = plt.scatter(x=X[i,:], y=Y[i,:], c=C[i,:], cmap=mapper.cmap)
  else:
    myScatter = plt.scatter(x=X[idx,:], y=Y[idx,:], c=C[idx,:], cmap=mapper.cmap)
  plt.xlim(-2,2)
  plt.ylim(-0.5,3.5)
  return None

fig = plt.figure()
im_ani = animation.FuncAnimation(fig, update_plot, frames=range(numTimeSteps+1),
                                 interval=500, init_func=init_plot,
                                 repeat_delay=5000, repeat=True)

im_ani.save('test.gif', writer='imagemagick',fps=10)
im_ani.save('test.mp4', fps=10)
# plt.show()
