import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib import pyplot as pp

data = np.loadtxt('testPrintTimePointSet_dump.csv', delimiter=',',skiprows=1)
m1, m2 = data[:,0], data[:,1]
xmin, xmax = min(m1), max(m1)
ymin, ymax = min(m2), max(m2)

points = np.vstack([m1, m2])
kde = gaussian_kde(points)

x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

z = kde((x.ravel(),y.ravel())).reshape(*x.shape)

# plot
fig,ax = pp.subplots(1,1)
ax.hold(True)
pc = ax.pcolor(x,y,z)
cb = pp.colorbar(pc)
cb.ax.set_ylabel('Probability density')
ax.plot(points[0],points[1],'o',mfc='w',mec='k')

pp.show()