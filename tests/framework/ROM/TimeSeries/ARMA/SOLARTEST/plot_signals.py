import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
data = pd.read_csv('workingdir/debugg_varma.csv',index_col=0, header=None).T

day1 = slice(0,168)
day180 = slice(4320,4488)

def plotDay(data,sl,label):
  print 'Plotting:',label
  fig,ax = plt.subplots(figsize=(12,10))
  ax.plot(data.index.values[sl], data['GHI_original'].values[sl], '.-', label='original')
  ax.plot(data.index.values[sl], data['GHI_fourier'].values[sl], '-', label='fourier', alpha=0.3)
  ax.plot(data.index.values[sl], data['GHI_nofourier'].values[sl], '.-', label='remaining')
  #ax.plot(data.index.values[sl], data['GHI_nofourier2'].values[sl], '.-', label='remaining 2')
  #ax.plot(data.index.values[sl], data['GHI_fourier2'].values[sl], '-', label='fourier 2', alpha=0.3)
  ax.plot(data.index.values[sl], data['GHI_zerofilter'].values[sl], '-', label='zfilter')
  print '  Fourier mean:',np.average(data['GHI_fourier'])
  print '          mag :',np.max(data['GHI_fourier']) - np.average(data['GHI_fourier'])
  #print '  Fourier2 mean:',np.average(data['GHI_fourier2'])
  #print '  Fourier2 mag :',np.max(data['GHI_fourier2']) - np.average(data['GHI_fourier2'])
  ax.set_ylabel('GHI')
  ax.set_xlabel('Time (h)')
  ax.set_title('GHI on {} Week 2007'.format(label))
  ax.set_ylim(-500,1000)
  plt.legend(loc=0)
  plt.savefig('GHI_fourier_{}.pdf'.format(label))

plotDay(data,day1,'Jan')
plotDay(data,day180,'July')


plt.show()
