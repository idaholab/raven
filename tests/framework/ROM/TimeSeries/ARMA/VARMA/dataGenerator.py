import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 100

A = np.random.rand(N) - 0.5

B = np.random.rand(N) - 0.5
B = 0.9*A + 0.1*B

t = np.arange(N)

idx = pd.Index(t,name='Time')
df = pd.DataFrame(zip(A,B),index=idx, columns=['A','B'])

print df
fig,ax = plt.subplots()
df.plot(y='A',label='A',marker='.',ax=ax)
df.plot(y='B',label='A',marker='.',ax=ax)
plt.show()
df.to_csv('correlated_nofourier.csv')
