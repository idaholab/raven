#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Evaluates ans=x+y and ans2=x^2+y^2.
import numpy as np

def eval(inp,exp):
  return sum(n**exp for n in inp)
  #return np.exp(-sum(inp)/len(inp))

def run(self,Input):
  self.ans = eval((self.x1,self.x2),1)
  self.ans2 = eval((self.x1,self.x2),2)
