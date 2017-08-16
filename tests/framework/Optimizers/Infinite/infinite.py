import numpy as np

def run(self,Inputs):
  if self.x != 0.0:
    self.ans = self.y/self.x
  else:
    self.ans = np.array([float('inf')])

