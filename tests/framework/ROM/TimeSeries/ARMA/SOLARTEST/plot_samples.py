import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

day1 = slice(0,168)
day180 = slice(4320,4488)

fig1, axs1 = plt.subplots(2,1,figsize=(12,10))
fig180, axs180 = plt.subplots(2,1,figsize=(12,10))

# load original data
orig = pd.read_csv('workingdir/debugg_varma.csv',index_col=0,header=None).T
orig_index = orig.index.values
orig_dem = orig['Demand_original']
orig_ghi = orig['GHI_original']

# load sampled data
for i in range(10):
  df = pd.read_csv('workingdir/samples_{}.csv'.format(i))
  axs1[0].plot(df.index.values[day1],df['Demand'][day1],'o:',label=str(i),alpha=0.5)
  axs1[1].plot(df.index.values[day1],df['GHI'   ][day1],'o:',label=str(i),alpha=0.5)
  axs180[0].plot(df.index.values[day1],df['Demand'][day180],'o:',label=str(i),alpha=0.5)
  axs180[1].plot(df.index.values[day1],df['GHI'   ][day180],'o:',label=str(i),alpha=0.5)
  fig,ax = plt.subplots(2,1,figsize=(12,10))
  ax[0].plot(orig_index[day180],orig_dem[day180],'k-',label='original')
  ax[0].plot(df.index.values[day180],df['Demand'][day180],'o:',label='sythetic')
  ax[0].set_title('July Demand, Sample {}'.format(i))
  ax[1].plot(orig_index[day180],orig_ghi[day180],'k-',label='original')
  ax[1].plot(df.index.values[day180],df['GHI'][day180],'o:',label='sythetic')
  ax[1].set_title('July GHI, Sample {}'.format(i))
  fig.savefig('synthetic_{}_july07.pdf'.format(i))
  plt.close(fig)

axs1[0].plot(orig_index[day1],orig_dem[day1],'k-',label='orig')
axs1[1].plot(orig_index[day1],orig_ghi[day1],'k-',label='orig')
axs180[0].plot(orig_index[day1],orig_dem[day180],'k-',label='orig')
axs180[1].plot(orig_index[day1],orig_ghi[day180],'k-',label='orig')

axs1[0].set_title('January 2007: Demand')
axs1[1].set_title('January 2007: GHI')
axs180[0].set_title('July 2007: Demand')
axs180[1].set_title('July 2007: GHI')
axs1[1].set_xlabel('Time (h)')
axs180[1].set_xlabel('Time (h)')
axs1[0].set_ylabel('Demand (W)')
axs1[1].set_ylabel('GHI (W/m2)')
axs180[0].set_ylabel('Demand (W)')
axs180[1].set_ylabel('GHI (W/m2)')

fig1.savefig('synthetic_all_jan07.pdf')
fig180.savefig('synthetic_all_july07.pdf')

plt.show()
