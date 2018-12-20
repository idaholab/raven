import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of samples (units of "t")
N = 100
# pivot param
t = np.arange(N)

A = np.zeros(N)
B = np.zeros(N)
C = np.zeros(N)
D = np.zeros(N)

# Fourer Moments: A sin(2pi/k * t)
# A and B share the same frequencies, as do C and D
fA = {100.0 : 1.0,
        20.0 : 1.0}
fB = {100.0 : 4.0,
        20.0 : 2.0}
fC = { 50.0 : 1.0,
        10.0 : 1.0}
fD = { 50.0 : 4.0,
        10.0 : 2.0}
for freq,magn in fA.items():
  A += magn * np.sin(2.0*np.pi*t/freq)
for freq,magn in fB.items():
  B += magn * np.sin(2.0*np.pi*t/freq)
for freq,magn in fC.items():
  C += magn * np.sin(2.0*np.pi*t/freq)
for freq,magn in fD.items():
  D += magn * np.sin(2.0*np.pi*t/freq)

mn = zip(A,B,C,D)

idx = pd.Index(t,name='Time')
df = pd.DataFrame(mn,index=idx, columns=['A','B','C','D'])

fig,ax = plt.subplots()
df.plot(y='A',label='A',marker='.',ax=ax)
df.plot(y='B',label='B',marker='.',ax=ax)
df.plot(y='C',label='C',marker='.',ax=ax)
df.plot(y='D',label='D',marker='.',ax=ax)
plt.show()
df.to_csv('correlated_0.csv')
