

"""
Optimizing function.  Has minimum at x=0, (t-y)=0 for each value of t.
"""

import numpy as np

def evaluate(x,y,t):
  return x*x + np.sum((t-y)**2*np.exp(-t))

def run(self,Input):
  # "x" is scalar, "ans" and "y" depend on vector "t"
  self.t = np.linspace(-5,5,11)
  self.ans = evaluate(self.x,self.y,self.t)

