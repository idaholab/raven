import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 1000

mean = np.array([42,42,42])
#         A     B     C
cov = [ [ 25.0, 22.0, -12.0], # A
        [ 22.0, 25.0, -6.0], # B
        [-12.0, -6.0, 16.0]] # C

mn = np.random.multivariate_normal(mean,cov,size=N)

# sample
A,B,C = zip(*mn)
D = np.random.rand(N)+30.

##### OLD #####
#A = np.random.rand(N) - 0.5
#
#B = np.random.rand(N) - 0.5
#B = 0.9*A + 0.1*B

t = np.arange(N)

idx = pd.Index(t,name='Time')
df = pd.DataFrame(mn,index=idx, columns=['A','B','C'])
df['D'] = D

print df
fig,ax = plt.subplots()
df.plot(y='A',label='A',marker='.',ax=ax)
df.plot(y='B',label='B',marker='.',ax=ax)
df.plot(y='C',label='C',marker='.',ax=ax)
df.plot(y='D',label='D',marker='.',ax=ax)
plt.show()
df.to_csv('correlated_0.csv')
