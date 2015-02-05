import numpy as np

def eval(inp):
  return np.exp(-sum(inp)/len(inp))

def run(self,Input):
  self.ans = eval((self.x1,self.x2))
