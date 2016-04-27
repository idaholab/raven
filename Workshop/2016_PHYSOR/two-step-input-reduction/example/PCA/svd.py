import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

data = np.loadtxt('covariance.txt')
data = data.reshape((308,308))

U,S,V= linalg.svd(data,full_matrices=True)

plt.semilogy(S/S[0],marker='x')
plt.title('PCA for Demonstration Case')
plt.xlabel('Latent Dimension Number')
plt.ylabel('Expansion Coefficient')
plt.savefig('pca.pdf')
plt.show()
