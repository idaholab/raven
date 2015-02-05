import numpy as np

def eval(inp):
  exp=1
  return sum(n**exp for n in inp)
  #return np.exp(-sum(inp)/len(inp))

def run(self,Input):
  self.ans = eval((self.x1))
