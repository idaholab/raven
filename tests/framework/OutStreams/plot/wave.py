import numpy as np

def run(self,Inputs):
  self.mag = evaluate(self.A,self.k)

def evaluate(A,k):
  return A * np.cos(k * 2.*np.pi)

if __name__=='__main__':
  xs = np.linspace(0,2,1000)
  ys = evaluate(1.0,xs)
  import matplotlib.pyplot as plt
  plt.plot(xs,ys)
  plt.show()
