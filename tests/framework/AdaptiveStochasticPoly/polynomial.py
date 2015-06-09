#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
import numpy as np

def evaluate(inp):
  return np.prod(list(1.+n for n in inp))
  #return np.exp(-sum(inp)/len(inp))

def run(self,Input):
  self.ans = self.x1**2*self.x2 + self.x1**2 + self.x1*self.x2 + self.x1 + self.x2 + 1.
  self.ans2 = self.x1*self.x2 + self.x1 + self.x2 + 1.
  #self.ans = (1.+self.x1**3) * (1.+self.x2**2) * (1.+self.x3**1)
  #self.ans = evaluate((self.x1,self.x2,self.x3,self.x4))
  #self.ans =self.x1*self.x2 + self.x3
  #self.ans=eval((self.x1,self.x2,self.x3),1)
  #self.ans2 = eval((self.x1,self.x2,self.x3),2)
