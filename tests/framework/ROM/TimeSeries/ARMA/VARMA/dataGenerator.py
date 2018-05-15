

"""
  Generates two correlated time series with a base underlying syignal
"""

import matplotlib.pyplot as plt
import numpy as np

##### CONTROL OPTIONS #####
# what periods for squared cosine cycles (seconds)
periods = [3600, 21600] #1 hour, 6 hours

# what weight for base signal addition, set to None for even weights
weights = [0.5, 1.0] # 3/4 from 6 hour, 1/4 from 1 hour

# how long is the signal (sample every second)
signalLength = 3600*24

# time correlation length; i.e. how many past signals to consider at 1/(seconds between) weight
timeCorrelation = 20

# plot signals? set to True to visualize data
plot = True

# file name for output csv
fname = 'correlated_series' + '.csv'


##### METHOD DEFINITIONS #####
def A_t(t,periods,weights=None):
  """
    Generates a deterministic wave signal
    @ In, t, float or np.array(float), point or points in time to evaluate
    @ In, periods, list, list of periods to consider
    @ In, weights, list, optional, weights to give to each period.  Must have len equal to periods
    @ Out, val, np.array(float), signal with same length as "t"
  """
  if weights is None:
    weights = np.ones(len(periods))
  val = 0
  for p,period in enumerate(periods):
    val += weights[p]*np.cos(t*np.pi/period)**2
  return val

def r():
  """
    Provides a random value on [-0.5,0.5]
    @ In, None
    @ Out, None
  """
  return np.random.rand() - 0.5

def AB(t,periods,weights=None):
  """
    Produces stochastic signals A and B.
    A is cosine in time plus random on [-0.5,0.5]
    B is same cosine in time plus average of A's random and a separate random
    @ In, t, float, time point to evaluate
    @ In, periods, list, list of periods to consider
    @ In, weights, list, optional, weights to give to each period.  Must have len equal to periods
    @ Out, A, np.array, signal
    @ Out, B, np.array, signal
  """
  # A
  Ar = r()
  At = A_t(t,periods,weights)
  A = At + Ar
  # B
  Br = 0.5*(Ar + r())
  B = At + Br
  return A,B


ts = np.arange(signalLength)
As = []
Bs = []
for i,t in enumerate(ts):
  print ' {:2.2f}% complete ...\r'.format(100.*float(i+1)/float(signalLength)),
  A,B = AB(t,periods,weights)
  # linear time correlation, with preceedings up to 5
  if i == 0:
    As.append(A)
    Bs.append(B)
    continue
  x = 0
  avals = [A]
  bvals = [B]
  coef_tot = 1.0
  while i+x >= 0 and x > -1 * timeCorrelation:
    x -= 1
    coef = 1.0/float(abs(x)+1)
    coef_tot += coef
    avals.append(coef * As[i+x])
    bvals.append(coef * Bs[i+x])
  As.append(np.sum(avals)/coef_tot)
  Bs.append(np.sum(bvals)/coef_tot)

if plot:
  print 'Plotting ...                '
  fig,ax = plt.subplots()
  ax.plot(ts,As,'-',label='A')
  ax.plot(ts,Bs,'-',label='B')
  plt.legend()
  plt.show()

# write out
print 'Writing to',fname,'...'
with open(fname,'w') as out:
  out.writelines('Time,A,B\n')
  for i,t in enumerate(ts):
    out.writelines('{},{},{}\n'.format(t,As[i],Bs[i]))

print 'Done.'
