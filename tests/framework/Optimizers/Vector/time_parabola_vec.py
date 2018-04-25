

"""
Optimizing function.  Has minimum at x=0, (t-y)=0 for each value of t.
"""

import numpy as np

def evaluate(x,y,t):
  print 'TIME PARABOLA Y CONTRIB:',(t-y)**2*np.exp(-t)
  print 'TIME PARABOLA X CONTRIB:',x*x
  print 'TIME PARABOLA TOTAL    :',x*x + np.sum((t-y)**2*np.exp(-t))
  return x*x + np.sum((t-y)**2*np.exp(-t))

def run(self,Input):
  # "x" is scalar, "ans" and "y" depend on vector "t"
  self.t = np.linspace(-5,5,11)
  ys = np.array([float(x) for x in [self.yA,self.yB,self.yC,self.yD,self.yE,self.yF,self.yG,self.yH,self.yI,self.yJ,self.yK]])
  self.ans = evaluate(self.x,ys,self.t)

class A:
  def __init__(self,x=0,y=0):
    self.x = 0
    self.y = np.zeros(11)
    self.ans = None

if __name__=='__main__':
  a = A()
  run(a,None)
  print 'ans:',a.ans
