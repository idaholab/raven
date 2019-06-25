import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('synthetic.csv')
#df = df[['RAVEN_sample_ID', 'Year', 'Time', 'Signal']]
df = df.sort_values(['RAVEN_sample_ID', 'Year', 'Time'])
df = df.set_index(['RAVEN_sample_ID', 'Year', 'Time'])

samples = df.index.levels[0]
years = df.index.levels[1]
times = df.index.levels[2]
#print('DEBUGG samples:', list(s for s in samples))
#print('DEBUGG years:', list(y for y in years))

# one plot per year
#figs = []
#axs = []
#for y in years:
#  fig, ax = plt.subplots(figsize=(12,10))
#  figs.append(fig)
#  axs.append(ax)

fig, ax = plt.subplots(figsize=(12, 10))

for y, year in enumerate(years):
  for s in samples:
    signal = df.loc[s, year]['Signal'].values
    label = 'year {}'.format(year) if s == 0 else None
    #ax.plot(range(len(signal)), signal, color='C{}'.format(y), label=label, alpha=1.0/len(samples))
  xs = df.xs(year, level=1)['Signal']
  #print('DEBUGG xs:\n', xs)
  mean = xs.mean(axis=0, level=1).values
  #print('DEBUGG mean:', mean)
  mmax = xs.max(axis=0, level=1).values
  mmin = xs.min(axis=0, level=1).values
  label = 'year {}'.format(y)
  ax.plot(range(len(mean)), mean, color='C{}'.format(y), label=label)
  ax.plot(range(len(mmax)), mmax, ':', color='C{}'.format(y))
  ax.plot(range(len(mmin)), mmin, ':', color='C{}'.format(y))
  #print('\n\n\n\n')

leg = ax.legend(loc=0)
for l in leg.legendHandles:
  l.set_alpha(1)

plt.show()
