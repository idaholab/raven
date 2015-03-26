import numpy as np

def eval(inp,exp):
  return sum(n**exp for n in inp)
  #return np.exp(-sum(inp)/len(inp))

def run(self,Input):
  self.ans = eval((self.x1,self.x2),1)
  self.ans2 = eval((self.x1,self.x2),2)
