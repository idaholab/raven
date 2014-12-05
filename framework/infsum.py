import numpy as np

'''
Simulates the infinite sum of Poisson (or adjust for other) infinite sum to tol.
'''

tol = 0#1e-10

r=4
prev = 0
cur = 0
k = 1.0
lda = 4.
a = lda*np.exp(-lda)
tot = a + 0**r*np.exp(-lda)

while tot-prev>tol:
  #print tot-prev>tol
  prev=tot
  k+=1.0
  a*=lda/k
  tot+=k**r*a
  #print 'tot,k,dtot: %1.2e, %i %1.4e ' %(tot,k,tot-prev)

print 'tot,k:',tot,k

